"""Tests for cross-encoder re-ranking."""

import numpy as np
from unittest.mock import patch, MagicMock
from app.reranker import Reranker


def _make_docs(n):
    """Create n test documents."""
    return [
        {"chunk_id": f"c{i}", "title": f"Doc {i}", "content": f"Content about topic {i}"}
        for i in range(n)
    ]


@patch("app.reranker.CrossEncoder")
def test_rerank_returns_top_k(mock_ce_cls):
    """Should return exactly top_k documents."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    mock_ce_cls.return_value = mock_model

    reranker = Reranker()
    docs = _make_docs(5)
    result = reranker.rerank("query", docs, top_k=3)
    assert len(result) == 3


@patch("app.reranker.CrossEncoder")
def test_rerank_orders_by_score(mock_ce_cls):
    """Documents should be ordered by descending score."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
    mock_ce_cls.return_value = mock_model

    reranker = Reranker()
    docs = _make_docs(3)
    result = reranker.rerank("query", docs, top_k=3)
    # Doc with score 0.9 (index 1) should be first
    assert result[0]["chunk_id"] == "c1"
    assert result[1]["chunk_id"] == "c2"
    assert result[2]["chunk_id"] == "c0"


@patch("app.reranker.CrossEncoder")
def test_rerank_empty_documents(mock_ce_cls):
    """Empty document list should return empty list."""
    mock_ce_cls.return_value = MagicMock()
    reranker = Reranker()
    result = reranker.rerank("query", [], top_k=5)
    assert result == []


@patch("app.reranker.CrossEncoder")
def test_rerank_fewer_docs_than_top_k(mock_ce_cls):
    """When fewer docs than top_k, return all of them."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5, 0.8])
    mock_ce_cls.return_value = mock_model

    reranker = Reranker()
    docs = _make_docs(2)
    result = reranker.rerank("query", docs, top_k=5)
    assert len(result) == 2


@patch("app.reranker.CrossEncoder")
def test_rerank_preserves_doc_structure(mock_ce_cls):
    """Reranked documents should keep all original fields."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5])
    mock_ce_cls.return_value = mock_model

    reranker = Reranker()
    docs = [{"chunk_id": "c1", "title": "T", "content": "C", "extra_field": "keep"}]
    result = reranker.rerank("q", docs, top_k=1)
    assert result[0]["extra_field"] == "keep"


@patch("app.reranker.CrossEncoder")
def test_rerank_creates_correct_pairs(mock_ce_cls):
    """Should create (query, content) pairs for the cross-encoder."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5, 0.3])
    mock_ce_cls.return_value = mock_model

    reranker = Reranker()
    docs = [
        {"chunk_id": "c1", "content": "about pressing"},
        {"chunk_id": "c2", "content": "about formations"},
    ]
    reranker.rerank("gegenpressing", docs, top_k=2)

    pairs = mock_model.predict.call_args[0][0]
    assert pairs == [
        ("gegenpressing", "about pressing"),
        ("gegenpressing", "about formations"),
    ]
