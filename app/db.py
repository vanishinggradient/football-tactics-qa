"""Database logging for conversations and feedback.

Uses PostgreSQL when available (docker-compose), falls back to SQLite (cloud/local).
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SQLITE_PATH = Path(__file__).parent.parent / "data" / "football_qa.db"


def _use_postgres():
    """Check if PostgreSQL is configured and reachable."""
    if os.getenv("DB_BACKEND") == "sqlite":
        return False
    if os.getenv("DB_BACKEND") == "postgres":
        return True
    try:
        import psycopg
        conn = psycopg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "football_qa"),
            user=os.getenv("POSTGRES_USER", "user"),
            password=os.getenv("POSTGRES_PASSWORD", "password"),
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


# Cache the backend choice for the session
_backend = None


def _get_backend():
    global _backend
    if _backend is None:
        _backend = "postgres" if _use_postgres() else "sqlite"
    return _backend


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_sqlite_conn():
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        rewritten_query TEXT,
        answer TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_template TEXT NOT NULL,
        retrieval_method TEXT NOT NULL DEFAULT 'hybrid_rrf',
        num_sources INTEGER NOT NULL DEFAULT 0,
        prompt_tokens INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        response_time REAL NOT NULL DEFAULT 0.0,
        cost REAL NOT NULL DEFAULT 0.0,
        relevance TEXT,
        relevance_explanation TEXT,
        created_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER REFERENCES conversations(id),
        score INTEGER NOT NULL,
        comment TEXT,
        created_at TEXT
    )""")
    return conn


# ---------------------------------------------------------------------------
# PostgreSQL helpers
# ---------------------------------------------------------------------------

def _get_pg_conn():
    import psycopg
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "football_qa"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------

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
    now = datetime.now(timezone.utc).isoformat()

    if _get_backend() == "postgres":
        conn = _get_pg_conn()
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
                        question, rewritten_query, answer, model, prompt_template,
                        retrieval_method, num_sources, prompt_tokens, completion_tokens,
                        total_tokens, response_time, cost, relevance,
                        relevance_explanation, now,
                    ),
                )
                conversation_id = cur.fetchone()[0]
            conn.commit()
            return conversation_id
        finally:
            conn.close()
    else:
        conn = _get_sqlite_conn()
        try:
            cur = conn.execute(
                """INSERT INTO conversations
                (question, rewritten_query, answer, model, prompt_template,
                 retrieval_method, num_sources, prompt_tokens, completion_tokens,
                 total_tokens, response_time, cost, relevance,
                 relevance_explanation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    question, rewritten_query, answer, model, prompt_template,
                    retrieval_method, num_sources, prompt_tokens, completion_tokens,
                    total_tokens, response_time, cost, relevance,
                    relevance_explanation, now,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def save_feedback(conversation_id, score, comment=None):
    """Save user feedback (thumbs up/down) for a conversation."""
    now = datetime.now(timezone.utc).isoformat()

    if _get_backend() == "postgres":
        conn = _get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO feedback (conversation_id, score, comment, created_at)
                    VALUES (%s, %s, %s, %s)""",
                    (conversation_id, score, comment, now),
                )
            conn.commit()
        finally:
            conn.close()
    else:
        conn = _get_sqlite_conn()
        try:
            conn.execute(
                """INSERT INTO feedback (conversation_id, score, comment, created_at)
                VALUES (?, ?, ?, ?)""",
                (conversation_id, score, comment, now),
            )
            conn.commit()
        finally:
            conn.close()


def get_recent_conversations(limit=20):
    """Fetch recent conversations for the dashboard."""
    if _get_backend() == "postgres":
        conn = _get_pg_conn()
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
    else:
        conn = _get_sqlite_conn()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, question, answer, model, response_time, relevance,
                          total_tokens, cost, created_at
                FROM conversations ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_dashboard_stats():
    """Fetch aggregate stats for the Streamlit dashboard."""
    if _get_backend() == "postgres":
        conn = _get_pg_conn()
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
                stats["relevance_pct"] = round(100 * row[0] / row[1], 1) if row[1] > 0 else 0.0
                return stats
        finally:
            conn.close()
    else:
        conn = _get_sqlite_conn()
        try:
            row = conn.execute(
                """SELECT
                    COUNT(*) AS total_queries,
                    COALESCE(AVG(response_time), 0) AS avg_response_time,
                    COALESCE(AVG(total_tokens), 0) AS avg_tokens,
                    COALESCE(SUM(cost), 0) AS total_cost
                FROM conversations"""
            ).fetchone()
            stats = {
                "total_queries": row[0],
                "avg_response_time": row[1],
                "avg_tokens": row[2],
                "total_cost": row[3],
            }

            row = conn.execute(
                """SELECT
                    COALESCE(SUM(CASE WHEN relevance = 'RELEVANT' THEN 1 ELSE 0 END), 0),
                    COALESCE(COUNT(*), 0)
                FROM conversations WHERE relevance IS NOT NULL"""
            ).fetchone()
            stats["relevance_pct"] = round(100 * row[0] / row[1], 1) if row[1] > 0 else 0.0
            return stats
        finally:
            conn.close()

