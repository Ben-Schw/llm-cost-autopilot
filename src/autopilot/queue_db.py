"""
Database-backed verification queue: The channel between the
API process and the worker process. The API enqueues a job after answering;
the worker claims jobs and verifies. They only ever talk through this table.

Claiming is atomic: a worker flips a row from 'pending' to 'in_progress' in a
single guarded UPDATE, so two workers can't grab the same job.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .logging_db import DEFAULT_DB

_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS verify_queue (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending',
    payload       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON verify_queue(status);
"""

@dataclass
class VerifyJob:
    """One unit of work for the worker. Mirrors what the router produced."""

    prompt: str
    chosen_model: str
    output: str
    task_type: str
    tier: str
    cost_usd: float
    reference_cost_usd: float
    confidence: float


class VerifyQueue:
    """Enqueue/claim/complete jobs in the verify_queue table."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_QUEUE_SCHEMA)
        
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        # WAL lets the API and worker access the file concurrently.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    
    # Producer site
    def enqueue(self, job: VerifyJob) -> int:
        payload = json.dumps(job.__dict__)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO verify_queue (created_at, status, payload) "
                "VALUES (?, 'pending', ?)",
                (datetime.now(timezone.utc).isoformat(), payload),
            )
            return int(cur.lastrowid)
        
    # Consumer site
    def claim_one(self) -> tuple[int, VerifyJob] | None:
        """Atomically claim the oldest pending job, or None if empty."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, payload FROM verify_queue "
                "WHERE status = 'pending' ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            updated = conn.execute(
                "UPDATE verify_queue SET status = 'in_progress' "
                "WHERE id = ? AND status = 'pending'",
                (row["id"],),
            )
            if updated.rowcount == 0:
                return None
            job = VerifyJob(**json.loads(row["payload"]))
            return int(row["id"]), job


    def complete(self, job_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE verify_queue SET status = 'done' WHERE id = ?",
                (job_id,),
            )

    def pending_count(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM verify_queue WHERE status = 'pending'"
            ).fetchone()[0]                                