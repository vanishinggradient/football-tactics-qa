"""Main RAG pipeline: query rewrite -> hybrid retrieve -> re-rank -> generate."""

import time

from app.llm_client import get_llm_client, get_model_name, estimate_cost
from app.search import get_es_client, get_embedding_model, hybrid_search_rrf
from app.reranker import Reranker
from app.query_rewriter import rewrite_query
from app.judge import evaluate_relevance
from app.db import save_conversation


PROMPT_TEMPLATES = {
    "basic": """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:""",

    "expert": """You are a football tactics analyst with deep knowledge of formations,
pressing systems, build-up play, and defensive structures. Answer the question
using the provided context. Use proper tactical terminology. If you're not sure
about something, say so rather than guessing.

Context:
{context}

Question: {question}

Answer:""",

    "structured": """You are a football tactics analyst. Answer the question using
the provided context. Structure your response with these sections where relevant:

- Tactical Overview: brief summary of the concept or system
- How It Works: the mechanics and player roles involved
- Key Examples: teams or managers known for using this approach

Use proper tactical terminology. Only use information from the context.

Context:
{context}

Question: {question}

Answer:""",
}

DEFAULT_TEMPLATE = "expert"


class FootballTacticsRAG:
    def __init__(self):
        self.llm_client = get_llm_client()
        self.model_name = get_model_name()
        self.es_client = get_es_client()
        self.embedding_model = get_embedding_model()
        self.reranker = Reranker()

    def answer(self, question, prompt_template=DEFAULT_TEMPLATE, top_k=5):
        """Full RAG pipeline. Returns a result dict with all metadata."""
        start_time = time.time()

        total_prompt_tokens = 0
        total_completion_tokens = 0

        # 1. Query rewriting
        rewritten_query, rewrite_usage = rewrite_query(
            self.llm_client, self.model_name, question
        )
        total_prompt_tokens += rewrite_usage["prompt_tokens"]
        total_completion_tokens += rewrite_usage["completion_tokens"]

        # 2. Hybrid search with RRF
        retrieved = hybrid_search_rrf(
            self.es_client, self.embedding_model, rewritten_query, k=20
        )

        # 3. Re-rank with cross-encoder
        reranked = self.reranker.rerank(rewritten_query, retrieved, top_k=top_k)

        # 4. Build context and generate answer
        context = "\n\n---\n\n".join(
            f"Source: {doc['title']}\n{doc['content']}" for doc in reranked
        )

        template = PROMPT_TEMPLATES.get(prompt_template, PROMPT_TEMPLATES[DEFAULT_TEMPLATE])
        prompt = template.format(context=context, question=question)

        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        answer_text = response.choices[0].message.content.strip()
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
        total_tokens = total_prompt_tokens + total_completion_tokens

        response_time = time.time() - start_time
        cost = estimate_cost(self.model_name, total_prompt_tokens, total_completion_tokens)

        # 5. Auto-judge relevance
        try:
            relevance, relevance_explanation = evaluate_relevance(
                self.llm_client, self.model_name, question, answer_text
            )
        except Exception:
            relevance = None
            relevance_explanation = None

        # 6. Log to database
        try:
            conversation_id = save_conversation(
                question=question,
                rewritten_query=rewritten_query,
                answer=answer_text,
                model=self.model_name,
                prompt_template=prompt_template,
                retrieval_method="hybrid_rrf_reranked",
                num_sources=len(reranked),
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                response_time=response_time,
                cost=cost,
                relevance=relevance,
                relevance_explanation=relevance_explanation,
            )
        except Exception:
            conversation_id = None

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer_text,
            "sources": reranked,
            "model": self.model_name,
            "prompt_template": prompt_template,
            "response_time": response_time,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "relevance": relevance,
            "relevance_explanation": relevance_explanation,
            "conversation_id": conversation_id,
        }
