"""LLM-based query rewriting for better retrieval."""

REWRITE_PROMPT = """You are a football tactics expert. Rewrite the user's question
to be more specific and searchable. Expand abbreviations (e.g. "CDM" -> "central
defensive midfielder"), add relevant tactical terms, and clarify vague references.

Keep it to one sentence. Don't answer the question, just rewrite it.

Original question: {question}

Rewritten question:"""


def rewrite_query(client, model, question):
    """Use LLM to expand and normalize the user's query.

    Returns (rewritten_query, usage_dict).
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": REWRITE_PROMPT.format(question=question)},
        ],
        temperature=0.0,
        max_tokens=150,
    )

    rewritten = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

    return rewritten, usage
