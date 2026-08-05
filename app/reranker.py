"""Cross-encoder re-ranking for retrieved documents."""

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):
        """Score each (query, doc) pair and return the top_k highest-scoring docs."""
        if not documents:
            return []

        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )

        return [doc for _, doc in scored_docs[:top_k]]
