"""Tests for search functions (RRF scoring, backend detection)."""

import os
from unittest.mock import patch
from app.search import compute_rrf, get_search_backend, MinsearchBackend


# --- RRF scoring tests ---

def test_compute_rrf_rank_1():
    """RRF score for rank 1 with k=60 should be 1/61."""
    score = compute_rrf(1, k=60)
    assert abs(score - 1 / 61) < 1e-10


def test_compute_rrf_rank_0():
    """RRF score for rank 0 with k=60 should be 1/60."""
    score = compute_rrf(0, k=60)
    assert abs(score - 1 / 60) < 1e-10


def test_compute_rrf_decreasing():
    """Higher ranks should produce lower RRF scores."""
    scores = [compute_rrf(i, k=60) for i in range(1, 11)]
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1]


def test_compute_rrf_always_positive():
    """RRF scores should always be positive."""
    for rank in range(0, 100):
        assert compute_rrf(rank, k=60) > 0


def test_compute_rrf_custom_k():
    """Different k values should change scores."""
    score_k10 = compute_rrf(1, k=10)
    score_k60 = compute_rrf(1, k=60)
    # Smaller k means higher score for same rank
    assert score_k10 > score_k60


# --- Backend detection tests ---

def test_get_search_backend_env_override():
    """SEARCH_BACKEND env var should override auto-detection."""
    with patch.dict(os.environ, {"SEARCH_BACKEND": "minsearch"}):
        assert get_search_backend() == "minsearch"


def test_get_search_backend_env_elasticsearch():
    with patch.dict(os.environ, {"SEARCH_BACKEND": "elasticsearch"}):
        assert get_search_backend() == "elasticsearch"


def test_get_search_backend_fallback_to_minsearch():
    """When ES is unavailable and no env var set, should fall back to minsearch."""
    env = os.environ.copy()
    env.pop("SEARCH_BACKEND", None)
    with patch.dict(os.environ, env, clear=True):
        with patch("app.search.is_elasticsearch_available", return_value=False):
            assert get_search_backend() == "minsearch"


# --- MinsearchBackend._format tests ---

def test_minsearch_format():
    """_format should normalize document structure."""
    # Use the static method without instantiating (avoid loading documents)
    formatted = MinsearchBackend._format(None, [
        {"chunk_id": "c1", "doc_id": "d1", "source": "wiki", "title": "T1", "content": "text1", "extra": "ignored"},
        {"chunk_id": "c2", "doc_id": "d2", "source": "yt", "title": "T2", "content": "text2"},
    ])
    assert len(formatted) == 2
    assert set(formatted[0].keys()) == {"chunk_id", "doc_id", "source", "title", "content"}
    assert formatted[0]["chunk_id"] == "c1"
    assert formatted[1]["source"] == "yt"


def test_minsearch_format_missing_keys():
    """_format should handle missing keys gracefully with empty strings."""
    formatted = MinsearchBackend._format(None, [{"some_field": "value"}])
    assert formatted[0]["chunk_id"] == ""
    assert formatted[0]["content"] == ""
