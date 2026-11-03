"""Durable Agent session storage with a small in-memory fallback."""

from __future__ import annotations

import json
from threading import RLock
from typing import Any, Optional


def psycopg_connection_string(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1).replace(
        "postgres://",
        "postgresql://",
        1,
    )


class MemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            value = self._sessions.get(session_id)
            return json.loads(json.dumps(value)) if value is not None else None

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._sessions[session_id] = json.loads(json.dumps(state))


class PostgresSessionStore:
    def __init__(self, connection_string: str) -> None:
        self.connection_string = psycopg_connection_string(connection_string)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        import psycopg

        with psycopg.connect(self.connection_string, connect_timeout=5) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_memo_agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    state JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '30 days'
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS voice_memo_agent_sessions_expires_at
                ON voice_memo_agent_sessions (expires_at)
                """
            )

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        import psycopg

        with psycopg.connect(self.connection_string, connect_timeout=5) as connection:
            connection.execute(
                "DELETE FROM voice_memo_agent_sessions WHERE expires_at <= NOW()"
            )
            row = connection.execute(
                """
                SELECT state
                FROM voice_memo_agent_sessions
                WHERE session_id = %s AND expires_at > NOW()
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row[0])

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        import psycopg

        with psycopg.connect(self.connection_string, connect_timeout=5) as connection:
            connection.execute(
                """
                INSERT INTO voice_memo_agent_sessions (
                    session_id,
                    state,
                    updated_at,
                    expires_at
                )
                VALUES (%s, %s::jsonb, NOW(), NOW() + INTERVAL '30 days')
                ON CONFLICT (session_id)
                DO UPDATE SET
                    state = EXCLUDED.state,
                    updated_at = NOW(),
                    expires_at = NOW() + INTERVAL '30 days'
                """,
                (session_id, json.dumps(state, ensure_ascii=False)),
            )
