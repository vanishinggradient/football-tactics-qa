"""Evaluate RAG quality: compare prompt templates and models using LLM-as-judge.

Tests 3 prompt styles x 2 models, judges each response, and reports
the percentage of RELEVANT answers for each combination.
"""

import csv
import json
import os
import time

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from openai import OpenAI
from app.search import get_es_client, get_embedding_model, hybrid_search_rrf
from app.reranker import Reranker
from app.judge import evaluate_relevance
from app.rag import PROMPT_TEMPLATES


GT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "ground_truth.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "rag_eval_results.csv"
)

# Models to test (all via Groq free tier)
MODELS = [
    ("llama-3.1-8b-instant", "groq"),
    ("llama-3.3-70b-versatile", "groq"),
]


def get_client(provider):
    if provider == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(client, model, question, context, template_name):
    """Generate an answer using the given model and prompt template."""
    template = PROMPT_TEMPLATES[template_name]
    prompt = template.format(context=context, question=question)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def main():
    with open(GT_PATH) as f:
        ground_truth = list(csv.DictReader(f))

    # Use a subset for evaluation (first 30 to stay within rate limits)
    eval_set = ground_truth[:30]
    print(f"Evaluating on {len(eval_set)} questions")

    es = get_es_client()
    emb_model = get_embedding_model()
    reranker = Reranker()

    # Pre-retrieve context for each question (same retrieval for all model/prompt combos)
    print("Retrieving contexts...")
    contexts = {}
    for gt in tqdm(eval_set):
        q = gt["question"]
        results = hybrid_search_rrf(es, emb_model, q, k=20)
        reranked = reranker.rerank(q, results, top_k=5)
        ctx = "\n\n---\n\n".join(
            f"Source: {doc['title']}\n{doc['content']}" for doc in reranked
        )
        contexts[q] = ctx

    # Evaluate each (model, prompt) combination
    template_names = list(PROMPT_TEMPLATES.keys())
    results = []

    # Use a single judge model (the better one)
    judge_client = get_client("groq")
    judge_model = "llama-3.3-70b-versatile"

    for model_name, provider in MODELS:
        client = get_client(provider)

        for template_name in template_names:
            print(f"\n--- {model_name} / {template_name} ---")
            verdicts = {"RELEVANT": 0, "PARTLY_RELEVANT": 0, "NON_RELEVANT": 0}

            for gt in tqdm(eval_set, desc=f"{model_name}/{template_name}"):
                q = gt["question"]
                ctx = contexts[q]

                try:
                    answer = generate_answer(client, model_name, q, ctx, template_name)
                    relevance, _ = evaluate_relevance(
                        judge_client, judge_model, q, answer
                    )
                    verdicts[relevance] = verdicts.get(relevance, 0) + 1
                except Exception as e:
                    print(f"  Error: {e}")
                    verdicts["NON_RELEVANT"] += 1

                time.sleep(0.3)  # rate limiting

            total = sum(verdicts.values())
            pct_relevant = round(100 * verdicts["RELEVANT"] / total, 1) if total else 0

            result = {
                "model": model_name,
                "prompt": template_name,
                "relevant": verdicts["RELEVANT"],
                "partly_relevant": verdicts["PARTLY_RELEVANT"],
                "non_relevant": verdicts["NON_RELEVANT"],
                "pct_relevant": pct_relevant,
            }
            results.append(result)
            print(f"  RELEVANT: {verdicts['RELEVANT']}/{total} ({pct_relevant}%)")

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Prompt':<15} {'% Relevant':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<30} {r['prompt']:<15} {r['pct_relevant']:>10.1f}%")
    print("=" * 70)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "prompt", "relevant", "partly_relevant",
                "non_relevant", "pct_relevant",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
