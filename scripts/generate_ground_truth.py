"""Generate ground truth Q&A pairs from document chunks using an LLM.

Reads data/processed/documents.json, samples chunks, asks the LLM to
generate questions that each chunk can answer, and saves the result to
data/ground_truth.csv.
"""

import csv
import json
import os
import random
import time

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from app.llm_client import get_llm_client, get_model_name
from ingestion.chunk import chunk_documents


GENERATE_PROMPT = """Based on the following text about football tactics, generate
exactly 3 questions that this text can answer. The questions should be the kind
a football fan or student of the game would ask.

Text:
{text}

Source title: {title}

Return ONLY the questions, one per line, numbered 1-3. No other text."""

DOCS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "processed", "documents.json"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "ground_truth.csv"
)


def generate_questions(client, model, chunk):
    """Ask the LLM to generate questions for a chunk."""
    prompt = GENERATE_PROMPT.format(text=chunk["content"][:1500], title=chunk["title"])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )

    text = response.choices[0].message.content.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    questions = []
    for line in lines:
        # Strip numbering like "1. " or "1) "
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned and len(cleaned) > 10:
            questions.append(cleaned)

    return questions[:3]


def main():
    with open(DOCS_PATH) as f:
        documents = json.load(f)

    chunks = chunk_documents(documents, chunk_size=500, overlap=50)
    print(f"Total chunks: {len(chunks)}")

    # Sample chunks to generate questions from (not all, to save API calls)
    # Aim for ~200 Q&A pairs from ~80 chunks (3 questions each)
    sample_size = min(80, len(chunks))
    random.seed(42)
    sampled = random.sample(chunks, sample_size)

    client = get_llm_client()
    model = get_model_name()

    ground_truth = []

    for chunk in tqdm(sampled, desc="Generating questions"):
        try:
            questions = generate_questions(client, model, chunk)
            for q in questions:
                ground_truth.append({
                    "question": q,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                })
        except Exception as e:
            print(f"  Error for {chunk['chunk_id']}: {e}")

        # Rate limiting for Groq free tier
        time.sleep(0.5)

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "chunk_id", "doc_id", "source", "title"]
        )
        writer.writeheader()
        writer.writerows(ground_truth)

    print(f"\nGenerated {len(ground_truth)} Q&A pairs -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
