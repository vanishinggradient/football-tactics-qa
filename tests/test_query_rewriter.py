"""Tests for query rewriting."""

from unittest.mock import MagicMock
from app.query_rewriter import rewrite_query


def _mock_rewrite_response(rewritten_text, prompt_tokens=50, completion_tokens=20):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = rewritten_text
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


def test_rewrite_returns_text_and_usage():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_rewrite_response(
        "What is the role of a central defensive midfielder in a 4-3-3 formation?"
    )

    rewritten, usage = rewrite_query(client, "model", "What does a CDM do?")
    assert "central defensive midfielder" in rewritten
    assert usage["prompt_tokens"] == 50
    assert usage["completion_tokens"] == 20


def test_rewrite_strips_whitespace():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_rewrite_response(
        "  some query with spaces  "
    )

    rewritten, _ = rewrite_query(client, "m", "q")
    assert rewritten == "some query with spaces"


def test_rewrite_calls_llm_with_low_temperature():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_rewrite_response("rewritten")

    rewrite_query(client, "my-model", "original question")

    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["max_tokens"] == 150
    assert call_kwargs["model"] == "my-model"


def test_rewrite_prompt_contains_original_question():
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_rewrite_response("rewritten")

    rewrite_query(client, "m", "What is tiki-taka?")

    call_kwargs = client.chat.completions.create.call_args[1]
    user_message = call_kwargs["messages"][0]["content"]
    assert "tiki-taka" in user_message
