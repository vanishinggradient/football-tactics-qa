"""Tests for the LLM client abstraction."""

import os
from unittest.mock import patch
from app.llm_client import estimate_cost, get_model_name, MODEL_COSTS


# --- Cost estimation tests ---

def test_estimate_cost_llama_8b():
    """Cost for llama-3.1-8b-instant should match known rates."""
    cost = estimate_cost("llama-3.1-8b-instant", 1_000_000, 1_000_000)
    expected = 0.05 + 0.08  # $0.05 input + $0.08 output per 1M tokens
    assert abs(cost - expected) < 0.001


def test_estimate_cost_llama_70b():
    cost = estimate_cost("llama-3.3-70b-versatile", 1_000_000, 1_000_000)
    expected = 0.59 + 0.79
    assert abs(cost - expected) < 0.001


def test_estimate_cost_gpt4o_mini():
    cost = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    expected = 0.15 + 0.60
    assert abs(cost - expected) < 0.001


def test_estimate_cost_unknown_model():
    """Unknown model should return 0.0 cost."""
    cost = estimate_cost("unknown-model", 1000, 1000)
    assert cost == 0.0


def test_estimate_cost_zero_tokens():
    cost = estimate_cost("llama-3.1-8b-instant", 0, 0)
    assert cost == 0.0


def test_estimate_cost_scales_linearly():
    """Doubling tokens should double cost."""
    cost_1 = estimate_cost("llama-3.1-8b-instant", 500, 500)
    cost_2 = estimate_cost("llama-3.1-8b-instant", 1000, 1000)
    assert abs(cost_2 - 2 * cost_1) < 1e-10


def test_estimate_cost_typical_query():
    """A typical query with ~500 prompt + ~200 completion tokens."""
    cost = estimate_cost("llama-3.1-8b-instant", 500, 200)
    # (500/1M) * 0.05 + (200/1M) * 0.08 = 0.000025 + 0.000016 = 0.000041
    assert cost > 0
    assert cost < 0.001  # should be very cheap


# --- Model name tests ---

def test_get_model_name_groq_default():
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}, clear=False):
        # Remove LLM_MODEL if set
        env = os.environ.copy()
        env.pop("LLM_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}):
                name = get_model_name()
                assert name == "llama-3.1-8b-instant"


def test_get_model_name_openai_default():
    with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=False):
        env = os.environ.copy()
        env.pop("LLM_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
                name = get_model_name()
                assert name == "gpt-4o-mini"


def test_get_model_name_custom_override():
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "LLM_MODEL": "custom-model"}):
        name = get_model_name()
        assert name == "custom-model"


def test_model_costs_has_all_known_models():
    """All documented models should have cost entries."""
    expected_models = {"llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gpt-4o-mini"}
    assert expected_models == set(MODEL_COSTS.keys())


def test_model_costs_structure():
    """Each model cost entry should have input and output rates."""
    for model, costs in MODEL_COSTS.items():
        assert "input" in costs, f"Missing 'input' for {model}"
        assert "output" in costs, f"Missing 'output' for {model}"
        assert costs["input"] >= 0
        assert costs["output"] >= 0
