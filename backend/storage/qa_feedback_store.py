from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class QaFeedbackStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'legacy',
                    question TEXT NOT NULL,
                    original_question TEXT NOT NULL DEFAULT '',
                    rewrite_query TEXT NOT NULL DEFAULT '',
                    final_query_source TEXT NOT NULL DEFAULT '',
                    rewrite_mode TEXT NOT NULL DEFAULT '',
                    rewrite_strategy TEXT NOT NULL DEFAULT '',
                    rewrite_warning TEXT NOT NULL DEFAULT '',
                    answer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    warning TEXT NOT NULL DEFAULT '',
                    evidence_json TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'ragagent_ui',
                    created_at TEXT NOT NULL,
                    feedback TEXT NULL,
                    feedback_updated_at TEXT NULL
                )
                """
            )
            self._migrate_schema(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_logs_created_at ON qa_logs(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_logs_user_created_at ON qa_logs(user_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_logs_feedback ON qa_logs(feedback)"
            )
            conn.commit()

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(qa_logs)").fetchall()}
        columns = {
            "user_id": "TEXT NOT NULL DEFAULT 'legacy'",
            "original_question": "TEXT NOT NULL DEFAULT ''",
            "rewrite_query": "TEXT NOT NULL DEFAULT ''",
            "final_query_source": "TEXT NOT NULL DEFAULT ''",
            "rewrite_mode": "TEXT NOT NULL DEFAULT ''",
            "rewrite_strategy": "TEXT NOT NULL DEFAULT ''",
            "rewrite_warning": "TEXT NOT NULL DEFAULT ''",
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE qa_logs ADD COLUMN {name} {definition}")
        conn.execute(
            """
            UPDATE qa_logs
            SET original_question = question
            WHERE original_question = ''
            """
        )
        conn.execute(
            """
            UPDATE qa_logs
            SET final_query_source = CASE
                WHEN source = 'ragagent_ui_rewrite' THEN 'rewrite 结果'
                WHEN source = 'ragagent_ui_original' THEN '原始 query'
                WHEN source = 'ragagent_ui' THEN '原始 query'
                ELSE final_query_source
            END
            WHERE final_query_source = ''
              AND source IN ('ragagent_ui_rewrite', 'ragagent_ui_original', 'ragagent_ui')
            """
        )
        conn.execute(
            """
            UPDATE qa_logs
            SET rewrite_query = question
            WHERE rewrite_query = ''
              AND source = 'ragagent_ui_rewrite'
            """
        )
        conn.execute(
            """
            UPDATE qa_logs
            SET rewrite_mode = 'legacy'
            WHERE rewrite_mode = ''
              AND source = 'ragagent_ui_rewrite'
            """
        )
        conn.execute(
            """
            UPDATE qa_logs
            SET rewrite_strategy = 'legacy_rewrite'
            WHERE rewrite_strategy = ''
              AND source = 'ragagent_ui_rewrite'
            """
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def insert_qa_log(
        self,
        *,
        user_id: str = "legacy",
        question: str,
        original_question: str = "",
        rewrite_query: str = "",
        final_query_source: str = "",
        rewrite_mode: str = "",
        rewrite_strategy: str = "",
        rewrite_warning: str = "",
        answer: str,
        status: str,
        warning: str,
        evidence_json: str,
        source: str = "ragagent_ui",
    ) -> int:
        safe_user_id = self._normalize_user_id(user_id)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO qa_logs(
                    user_id, question, original_question, rewrite_query, final_query_source,
                    rewrite_mode, rewrite_strategy, rewrite_warning, answer, status, warning,
                    evidence_json, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    safe_user_id,
                    question,
                    original_question or question,
                    rewrite_query,
                    final_query_source,
                    rewrite_mode,
                    rewrite_strategy,
                    rewrite_warning,
                    answer,
                    status,
                    warning,
                    evidence_json,
                    source,
                    self._now_iso(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_feedback(self, record_id: int, feedback: str, user_id: str | None = None) -> bool:
        if feedback not in {"useful", "useless"}:
            raise ValueError(f"invalid feedback: {feedback}")
        with self._connect() as conn:
            if user_id is None:
                cur = conn.execute(
                    """
                    UPDATE qa_logs
                    SET feedback = ?, feedback_updated_at = ?
                    WHERE id = ?
                    """,
                    (feedback, self._now_iso(), record_id),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE qa_logs
                    SET feedback = ?, feedback_updated_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (feedback, self._now_iso(), record_id, self._normalize_user_id(user_id)),
                )
            conn.commit()
            return cur.rowcount > 0

    def list_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        safe_user_id = self._normalize_user_id(user_id)
        safe_limit = min(max(int(limit), 1), 200)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    id, question, original_question, status, warning, source,
                    created_at, feedback, final_query_source
                FROM qa_logs
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (safe_user_id, safe_limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_history_record(self, user_id: str, record_id: int) -> Dict[str, Any] | None:
        safe_user_id = self._normalize_user_id(user_id)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT
                    id, user_id, question, original_question, rewrite_query, final_query_source,
                    rewrite_mode, rewrite_strategy, rewrite_warning, answer, status, warning,
                    evidence_json, source, created_at, feedback, feedback_updated_at
                FROM qa_logs
                WHERE id = ? AND user_id = ?
                """,
                (record_id, safe_user_id),
            ).fetchone()
            return dict(row) if row is not None else None

    @staticmethod
    def _normalize_user_id(user_id: str) -> str:
        clean = (user_id or "").strip()
        return clean[:128] if clean else "legacy"
