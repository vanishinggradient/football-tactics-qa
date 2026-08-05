"""Elasticsearch search functions: text, vector, hybrid, and RRF."""

import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

INDEX_NAME = "football-tactics"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"


def get_es_client(url=None):
    url = url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    return Elasticsearch(url)


def get_embedding_model(model_name=EMBEDDING_MODEL):
    return SentenceTransformer(model_name)


def text_only_search(es_client, query, k=10, index_name=INDEX_NAME):
    """BM25 keyword search only."""
    search_query = {
        "multi_match": {
            "query": query,
            "fields": ["title^2", "content"],
            "type": "best_fields",
        }
    }

    results = es_client.search(
        index=index_name,
        query=search_query,
        size=k,
        _source=["chunk_id", "doc_id", "source", "title", "content"],
    )

    return [hit["_source"] for hit in results["hits"]["hits"]]


def vector_only_search(es_client, model, query, k=10, index_name=INDEX_NAME):
    """kNN cosine similarity search only."""
    query_vector = model.encode(query).tolist()

    knn_query = {
        "field": "content_vector",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": 10000,
    }

    results = es_client.search(
        index=index_name,
        knn=knn_query,
        size=k,
        _source=["chunk_id", "doc_id", "source", "title", "content"],
    )

    return [hit["_source"] for hit in results["hits"]["hits"]]


def hybrid_search_linear(es_client, model, query, k=10, index_name=INDEX_NAME):
    """Hybrid search: BM25 + kNN combined with linear boost weights."""
    query_vector = model.encode(query).tolist()

    keyword_query = {
        "multi_match": {
            "query": query,
            "fields": ["title^2", "content"],
            "type": "best_fields",
            "boost": 0.5,
        }
    }

    knn_query = {
        "field": "content_vector",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": 10000,
        "boost": 0.5,
    }

    results = es_client.search(
        index=index_name,
        query=keyword_query,
        knn=knn_query,
        size=k,
        _source=["chunk_id", "doc_id", "source", "title", "content"],
    )

    return [hit["_source"] for hit in results["hits"]["hits"]]


def compute_rrf(rank, k=60):
    """Reciprocal Rank Fusion score."""
    return 1 / (k + rank)


def hybrid_search_rrf(es_client, model, query, k=10, index_name=INDEX_NAME):
    """Hybrid search with custom RRF (Reciprocal Rank Fusion).

    Runs BM25 and kNN as separate queries, then merges results
    using RRF scoring. This avoids needing ES's paid RRF feature.
    """
    query_vector = model.encode(query).tolist()

    # BM25 search
    keyword_query = {
        "multi_match": {
            "query": query,
            "fields": ["title^2", "content"],
            "type": "best_fields",
        }
    }
    keyword_results = es_client.search(
        index=index_name,
        query=keyword_query,
        size=k,
    )["hits"]["hits"]

    # Vector search
    knn_query = {
        "field": "content_vector",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": 10000,
    }
    knn_results = es_client.search(
        index=index_name,
        knn=knn_query,
        size=k,
    )["hits"]["hits"]

    # Merge with RRF
    rrf_scores = {}
    doc_data = {}

    for rank, hit in enumerate(knn_results):
        doc_id = hit["_id"]
        rrf_scores[doc_id] = compute_rrf(rank + 1)
        doc_data[doc_id] = hit["_source"]

    for rank, hit in enumerate(keyword_results):
        doc_id = hit["_id"]
        if doc_id in rrf_scores:
            rrf_scores[doc_id] += compute_rrf(rank + 1)
        else:
            rrf_scores[doc_id] = compute_rrf(rank + 1)
        doc_data[doc_id] = hit["_source"]

    # Sort by RRF score, take top k
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in sorted_docs[:k]:
        doc = doc_data[doc_id]
        results.append({
            "chunk_id": doc.get("chunk_id", ""),
            "doc_id": doc.get("doc_id", ""),
            "source": doc.get("source", ""),
            "title": doc.get("title", ""),
            "content": doc.get("content", ""),
        })

    return results
