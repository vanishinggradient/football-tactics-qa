"""PostgreSQL logging for conversations and feedback."""

import os
import psycopg
from datetime import datetime, timezone


def get_db_connection():
    """Create a PostgreSQL connection from environment variables."""
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "football_qa"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


def save_conversation(
    question,
    rewritten_query,
    answer,
    model,
    prompt_template,
    retrieval_method,
    num_sources,
    prompt_tokens,
    completion_tokens,
    total_tokens,
    response_time,
    cost,
    relevance=None,
    relevance_explanation=None,
):
    """Save a conversation to the database. Returns the conversation ID."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversations
                (question, rewritten_query, answer, model, prompt_template,
                 retrieval_method, num_sources, prompt_tokens, completion_tokens,
                 total_tokens, response_time, cost, relevance,
                 relevance_explanation, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (
                    question,
                    rewritten_query,
                    answer,
                    model,
                    prompt_template,
                    retrieval_method,
                    num_sources,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    response_time,
                    cost,
                    relevance,
                    relevance_explanation,
                    datetime.now(timezone.utc),
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
        return conversation_id
    finally:
        conn.close()


def save_feedback(conversation_id, score, comment=None):
    """Save user feedback (thumbs up/down) for a conversation."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO feedback (conversation_id, score, comment, created_at)
                VALUES (%s, %s, %s, %s)""",
                (conversation_id, score, comment, datetime.now(timezone.utc)),
            )
        conn.commit()
    finally:
        conn.close()


def get_recent_conversations(limit=20):
    """Fetch recent conversations for the dashboard."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, question, answer, model, response_time, relevance,
                          total_tokens, cost, created_at
                FROM conversations ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def get_dashboard_stats():
    """Fetch aggregate stats for the Streamlit dashboard."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT
                    COUNT(*) AS total_queries,
                    COALESCE(AVG(response_time), 0) AS avg_response_time,
                    COALESCE(AVG(total_tokens), 0) AS avg_tokens,
                    COALESCE(SUM(cost), 0) AS total_cost
                FROM conversations"""
            )
            row = cur.fetchone()
            columns = [desc[0] for desc in cur.description]
            stats = dict(zip(columns, row))

            cur.execute(
                """SELECT
                    COALESCE(SUM(CASE WHEN relevance = 'RELEVANT' THEN 1 ELSE 0 END), 0) AS relevant,
                    COALESCE(COUNT(*), 0) AS total
                FROM conversations WHERE relevance IS NOT NULL"""
            )
            row = cur.fetchone()
            if row[1] > 0:
                stats["relevance_pct"] = round(100 * row[0] / row[1], 1)
            else:
                stats["relevance_pct"] = 0.0

            return stats
    finally:
        conn.close()
