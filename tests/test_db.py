"""Tests for database layer using SQLite backend."""

import os
import tempfile
from unittest.mock import patch
from pathlib import Path


def _force_sqlite_backend(tmp_path):
    """Force SQLite backend with a temporary database file."""
    import app.db as db_module
    db_module._backend = "sqlite"
    db_module.SQLITE_PATH = tmp_path / "test.db"
    return db_module


def test_save_and_retrieve_conversation(tmp_path):
    db = _force_sqlite_backend(tmp_path)

    conv_id = db.save_conversation(
        question="What is gegenpressing?",
        rewritten_query="high pressing counter-press tactics",
        answer="Gegenpressing is an immediate counter-press after losing the ball.",
        model="llama-3.1-8b-instant",
        prompt_template="expert",
        retrieval_method="hybrid_rrf_reranked",
        num_sources=5,
        prompt_tokens=500,
        completion_tokens=200,
        total_tokens=700,
        response_time=3.5,
        cost=0.000041,
        relevance="RELEVANT",
        relevance_explanation="Directly addresses pressing.",
    )

    assert conv_id is not None
    assert isinstance(conv_id, int)

    conversations = db.get_recent_conversations(limit=10)
    assert len(conversations) == 1
    assert conversations[0]["question"] == "What is gegenpressing?"
    assert conversations[0]["relevance"] == "RELEVANT"


def test_save_feedback(tmp_path):
    db = _force_sqlite_backend(tmp_path)

    conv_id = db.save_conversation(
        question="test", rewritten_query="test", answer="test",
        model="m", prompt_template="basic", retrieval_method="rrf",
        num_sources=1, prompt_tokens=10, completion_tokens=5,
        total_tokens=15, response_time=1.0, cost=0.0,
    )

    db.save_feedback(conv_id, score=1, comment="Great answer!")
    # No assertion needed beyond "doesn't crash" since there's no get_feedback


def test_save_negative_feedback(tmp_path):
    db = _force_sqlite_backend(tmp_path)

    conv_id = db.save_conversation(
        question="q", rewritten_query="q", answer="a",
        model="m", prompt_template="basic", retrieval_method="rrf",
        num_sources=1, prompt_tokens=10, completion_tokens=5,
        total_tokens=15, response_time=1.0, cost=0.0,
    )

    db.save_feedback(conv_id, score=-1, comment="Wrong answer")


def test_dashboard_stats_empty_db(tmp_path):
    db = _force_sqlite_backend(tmp_path)
    stats = db.get_dashboard_stats()
    assert stats["total_queries"] == 0
    assert stats["total_cost"] == 0


def test_dashboard_stats_with_data(tmp_path):
    db = _force_sqlite_backend(tmp_path)

    db.save_conversation(
        question="q1", rewritten_query="q1", answer="a1",
        model="llama-3.1-8b-instant", prompt_template="expert",
        retrieval_method="rrf", num_sources=5,
        prompt_tokens=500, completion_tokens=200, total_tokens=700,
        response_time=3.0, cost=0.0001, relevance="RELEVANT",
        relevance_explanation="ok",
    )
    db.save_conversation(
        question="q2", rewritten_query="q2", answer="a2",
        model="llama-3.1-8b-instant", prompt_template="expert",
        retrieval_method="rrf", num_sources=3,
        prompt_tokens=400, completion_tokens=150, total_tokens=550,
        response_time=5.0, cost=0.0002, relevance="PARTLY_RELEVANT",
        relevance_explanation="partial",
    )

    stats = db.get_dashboard_stats()
    assert stats["total_queries"] == 2
    assert stats["avg_response_time"] == 4.0
    assert stats["total_cost"] > 0
    assert stats["relevance_pct"] == 50.0  # 1 out of 2 is RELEVANT


def test_chart_data_structure(tmp_path):
    db = _force_sqlite_backend(tmp_path)

    db.save_conversation(
        question="q", rewritten_query="q", answer="a",
        model="m", prompt_template="basic", retrieval_method="rrf",
        num_sources=1, prompt_tokens=10, completion_tokens=5,
        total_tokens=15, response_time=1.0, cost=0.0, relevance="RELEVANT",
    )

    chart_data = db.get_chart_data()
    assert "response_times" in chart_data
    assert "relevance_dist" in chart_data
    assert "feedback_dist" in chart_data
    assert "token_usage" in chart_data
    assert "model_usage" in chart_data
    assert len(chart_data["response_times"]) == 1
    assert chart_data["relevance_dist"]["RELEVANT"] == 1


def test_recent_conversations_limit(tmp_path):
    db = _force_sqlite_backend(tmp_path)

    for i in range(5):
        db.save_conversation(
            question=f"q{i}", rewritten_query=f"q{i}", answer=f"a{i}",
            model="m", prompt_template="basic", retrieval_method="rrf",
            num_sources=1, prompt_tokens=10, completion_tokens=5,
            total_tokens=15, response_time=1.0, cost=0.0,
        )

    conversations = db.get_recent_conversations(limit=3)
    assert len(conversations) == 3


def test_conversation_without_relevance(tmp_path):
    """Conversations can be saved without relevance (judge failure)."""
    db = _force_sqlite_backend(tmp_path)

    conv_id = db.save_conversation(
        question="q", rewritten_query="q", answer="a",
        model="m", prompt_template="basic", retrieval_method="rrf",
        num_sources=1, prompt_tokens=10, completion_tokens=5,
        total_tokens=15, response_time=1.0, cost=0.0,
        relevance=None, relevance_explanation=None,
    )

    assert conv_id is not None
    conversations = db.get_recent_conversations()
    assert conversations[0]["relevance"] is None
