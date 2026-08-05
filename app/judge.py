"""LLM-as-judge for auto-evaluating answer relevance."""

import json
from pydantic import BaseModel
from typing import Literal


class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


JUDGE_INSTRUCTIONS = """You are an expert evaluator for a football tactics Q&A system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer directly addresses the question with accurate tactical information
- PARTLY_RELEVANT: the answer partially addresses the question or is vaguely related
- NON_RELEVANT: the answer does not address the question at all

Respond with JSON: {"relevance": "...", "explanation": "..."}"""

JUDGE_PROMPT = """Question: {question}
Generated Answer: {answer}"""


def evaluate_relevance(client, model, question, answer):
    """Auto-judge the relevance of an answer to a question.

    Returns (relevance_label, explanation).
    """
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    verdict = RelevanceVerdict(**parsed)

    return verdict.relevance, verdict.explanation
