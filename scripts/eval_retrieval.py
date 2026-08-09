"""Evaluate retrieval approaches: text, vector, hybrid linear, hybrid RRF, RRF + reranking.

Reads data/ground_truth.csv and compares hit_rate and MRR across 5 retrieval methods.
Outputs a comparison table and saves results to data/retrieval_eval_results.csv.
"""

import csv
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from app.search import (
    get_es_client,
    get_embedding_model,
    text_only_search,
    vector_only_search,
    hybrid_search_linear,
    hybrid_search_rrf,
)
from app.reranker import Reranker


GT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "ground_truth.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "retrieval_eval_results.csv"
)


def hit_rate(relevance_total):
    cnt = sum(1 for line in relevance_total if True in line)
    return cnt / len(relevance_total) if relevance_total else 0


def mrr(relevance_total):
    score = 0.0
    for line in relevance_total:
        for rank, val in enumerate(line):
            if val:
                score += 1 / (rank + 1)
                break
    return score / len(relevance_total) if relevance_total else 0


def evaluate(ground_truth, search_fn, k=5):
    """Run a search function over ground truth and compute hit_rate + MRR."""
    relevance_total = []

    for gt in tqdm(ground_truth, desc="Evaluating"):
        results = search_fn(gt["question"])
        relevance = [d.get("chunk_id") == gt["chunk_id"] for d in results[:k]]
        relevance_total.append(relevance)

    return {
        "hit_rate": round(hit_rate(relevance_total), 4),
        "mrr": round(mrr(relevance_total), 4),
    }


def main():
    # Load ground truth
    with open(GT_PATH) as f:
        ground_truth = list(csv.DictReader(f))
    print(f"Ground truth: {len(ground_truth)} Q&A pairs")

    es = get_es_client()
    model = get_embedding_model()
    reranker = Reranker()

    # Define search functions
    def text_search(q):
        return text_only_search(es, q, k=10)

    def vec_search(q):
        return vector_only_search(es, model, q, k=10)

    def hybrid_lin(q):
        return hybrid_search_linear(es, model, q, k=10)

    def hybrid_rrf(q):
        return hybrid_search_rrf(es, model, q, k=10)

    def hybrid_rrf_reranked(q):
        results = hybrid_search_rrf(es, model, q, k=20)
        return reranker.rerank(q, results, top_k=5)

    approaches = [
        ("text_only", text_search),
        ("vector_only", vec_search),
        ("hybrid_linear", hybrid_lin),
        ("hybrid_rrf", hybrid_rrf),
        ("hybrid_rrf_reranked", hybrid_rrf_reranked),
    ]

    results = []
    for name, fn in approaches:
        print(f"\n--- {name} ---")
        metrics = evaluate(ground_truth, fn)
        metrics["approach"] = name
        results.append(metrics)
        print(f"  hit_rate: {metrics['hit_rate']}")
        print(f"  mrr:      {metrics['mrr']}")

    # Print comparison table
    print("\n" + "=" * 50)
    print(f"{'Approach':<25} {'Hit Rate':>10} {'MRR':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['approach']:<25} {r['hit_rate']:>10.4f} {r['mrr']:>10.4f}")
    print("=" * 50)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["approach", "hit_rate", "mrr"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
