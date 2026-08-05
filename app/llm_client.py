"""LLM client abstraction for Groq and OpenAI."""

import os
from openai import OpenAI


def get_llm_client():
    """Returns an OpenAI-compatible client for the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "groq")
    if provider == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_model_name():
    """Returns the model name based on provider config."""
    provider = os.getenv("LLM_PROVIDER", "groq")
    if provider == "groq":
        return os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


# Cost per 1M tokens (rough estimates for monitoring)
MODEL_COSTS = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model, prompt_tokens, completion_tokens):
    """Estimate cost in USD for a completion."""
    costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (prompt_tokens / 1_000_000) * costs["input"]
    output_cost = (completion_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost
