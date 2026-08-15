"""Search functions: Elasticsearch (local/docker) or minsearch (cloud/lightweight)."""

import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

INDEX_NAME = "football-tactics"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
DOCUMENTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "documents.json"


def is_elasticsearch_available():
    """Check if Elasticsearch is reachable."""
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(url)
        es.info()
        return True
    except Exception:
        return False


def get_search_backend():
    """Return 'elasticsearch' or 'minsearch' based on availability."""
    if os.getenv("SEARCH_BACKEND"):
        return os.getenv("SEARCH_BACKEND")
    return "elasticsearch" if is_elasticsearch_available() else "minsearch"


def get_es_client(url=None):
    from elasticsearch import Elasticsearch
    url = url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    return Elasticsearch(url)


def get_embedding_model(model_name=EMBEDDING_MODEL):
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# Minsearch backend (lightweight, no external services needed)
# ---------------------------------------------------------------------------

class MinsearchBackend:
    """In-memory text search using minsearch + sentence-transformers for vector."""

    def __init__(self, embedding_model):
        import minsearch
        self.index = minsearch.Index(
            text_fields=["title", "content"],
            keyword_fields=["chunk_id", "doc_id", "source"],
        )
        self.embedding_model = embedding_model
        self.docs = []
        self.vectors = []
        self._load_documents()

    def _load_documents(self):
        """Load pre-built documents.json, chunk, embed, and index."""
        from ingestion.chunk import chunk_documents

        with open(DOCUMENTS_PATH) as f:
            documents = json.load(f)

        chunks = chunk_documents(documents)
        self.docs = chunks
        self.index.fit(chunks)

        # Pre-compute vectors for cosine similarity search
        texts = [c["content"] for c in chunks]
        self.vectors = self.embedding_model.encode(texts)

    def text_search(self, query, k=10):
        results = self.index.search(
            query,
            boost_dict={"title": 2.0, "content": 1.0},
            num_results=k,
        )
        return self._format(results)

    def vector_search(self, query, k=10):
        import numpy as np
        query_vec = self.embedding_model.encode(query)
        scores = np.dot(self.vectors, query_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        )
        top_indices = np.argsort(scores)[::-1][:k]
        return self._format([self.docs[i] for i in top_indices])

    def hybrid_search_rrf(self, query, k=10, rrf_k=60):
        text_results = self.text_search(query, k=k)
        vector_results = self.vector_search(query, k=k)

        rrf_scores = {}
        doc_data = {}

        for rank, doc in enumerate(vector_results):
            cid = doc["chunk_id"]
            rrf_scores[cid] = 1 / (rrf_k + rank + 1)
            doc_data[cid] = doc

        for rank, doc in enumerate(text_results):
            cid = doc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rrf_k + rank + 1)
            doc_data[cid] = doc

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [doc_data[cid] for cid in sorted_ids[:k]]

    def _format(self, docs):
        return [
            {
                "chunk_id": d.get("chunk_id", ""),
                "doc_id": d.get("doc_id", ""),
                "source": d.get("source", ""),
                "title": d.get("title", ""),
                "content": d.get("content", ""),
            }
            for d in docs
        ]


# ---------------------------------------------------------------------------
# Elasticsearch backend functions (unchanged from original)
# ---------------------------------------------------------------------------

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
