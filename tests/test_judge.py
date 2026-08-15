"""Tests for the LLM-as-judge relevance evaluator."""

import json
from unittest.mock import MagicMock
from app.judge import evaluate_relevance, RelevanceVerdict


def _mock_llm_response(content):
    """Create a mock LLM response with given JSON content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(content)
    return response


def test_evaluate_relevance_relevant():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response({
        "relevance": "RELEVANT",
        "explanation": "Directly addresses the question about pressing."
    })

    relevance, explanation = evaluate_relevance(
        client, "test-model", "What is gegenpressing?", "Gegenpressing is..."
    )
    assert relevance == "RELEVANT"
    assert "pressing" in explanation


def test_evaluate_relevance_non_relevant():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response({
        "relevance": "NON_RELEVANT",
        "explanation": "Answer discusses cooking, not football."
    })

    relevance, _ = evaluate_relevance(client, "m", "tactics?", "recipe for cake")
    assert relevance == "NON_RELEVANT"


def test_evaluate_relevance_partly_relevant():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response({
        "relevance": "PARTLY_RELEVANT",
        "explanation": "Mentions formation but not the specific system asked about."
    })

    relevance, _ = evaluate_relevance(client, "m", "4-3-3?", "formations vary")
    assert relevance == "PARTLY_RELEVANT"


def test_evaluate_relevance_calls_llm_correctly():
    """Verify the LLM is called with correct parameters."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response({
        "relevance": "RELEVANT", "explanation": "ok"
    })

    evaluate_relevance(client, "my-model", "question?", "answer text")

    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "my-model"
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["response_format"] == {"type": "json_object"}
    # Should have system + user messages
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"


def test_relevance_verdict_valid_values():
    """Pydantic model should accept all three valid relevance values."""
    for val in ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]:
        v = RelevanceVerdict(relevance=val, explanation="test")
        assert v.relevance == val


def test_relevance_verdict_rejects_invalid():
    """Pydantic model should reject invalid relevance values."""
    import pytest
    with pytest.raises(Exception):
        RelevanceVerdict(relevance="INVALID", explanation="test")
