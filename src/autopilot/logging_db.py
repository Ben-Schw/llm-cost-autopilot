"""
Logs everything. 

Every request becomes one row in a SQLite database — the full audit trail the
plan asks for: timestamp, prompt hash, complexity tier, routed model, cost,
latency, the verifier's quality score, and whether it was escalated.

I use SQLite because it's zero-infrastructure, file-based, and inspectable
(you can open the .db with any SQLite browser). The logger is deliberately
decoupled from the router: the router produces RoutedResponse objects, and
`log_routed_response` turns one into a row. That separation means logging can't
break request handling, and the router doesn't need to know about storage.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = _ROOT / "data" / "autopilot.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    prompt_hash     TEXT    NOT NULL,
    task_type       TEXT    NOT NULL,
    tier            TEXT    NOT NULL,
    routed_model    TEXT    NOT NULL,
    cost_usd        REAL    NOT NULL,
    reference_cost_usd REAL NOT NULL,
    latency_ms      REAL    NOT NULL,
    quality_score   REAL,
    verified        INTEGER NOT NULL,
    escalated       INTEGER NOT NULL,
    escalation_cost_usd REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(routed_model);
"""

def _prompt_hash(prompt: str) -> str:
    """Stable short hash so we never store raw prompts (privacy) but can still
    group identical requests."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

@dataclass
class RequestLog:
    """One logged request. Mirrors the requests table columns."""

    timestamp: str
    prompt_hash: str
    task_type: str
    tier: str
    routed_model: str
    cost_usd: float
    reference_cost_usd: float
    latency_ms: float
    quality_score: float | None
    verified: bool
    escalated: bool
    escalation_cost_usd: float = 0.0

class RequestLogger:
    """Writes request rows to SQLite and exposes a couple of read helpers.

    `reference_model_cost` is injected so the logger can record what the top
    model would have cost for each request without importing pricing logic.
    """

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)       
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # Writing

    def log(self, entry: RequestLog) -> int:
        """Insert one row, return its id."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO requests
                   (timestamp, prompt_hash, task_type, tier, routed_model,
                    cost_usd, reference_cost_usd, latency_ms, quality_score,
                    verified, escalated, escalation_cost_usd)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (entry.timestamp, entry.prompt_hash, entry.task_type, entry.tier,
                 entry.routed_model, entry.cost_usd, entry.reference_cost_usd,
                 entry.latency_ms, entry.quality_score, int(entry.verified),
                 int(entry.escalated), entry.escalation_cost_usd),
            )
            return int(cur.lastrowid)
        
    def log_routed_response(self, routed, reference_cost_usd: float, latency_ms: float = 0.0) -> int:
        """Convenience: turn a router RoutedResponse into a row.

        `reference_cost_usd` is what the top-tier model would have cost for this
        prompt (the dashboard's savings baseline). When the request was
        verified, the verifier already paid this; otherwise we estimate it from
        the caller. `latency_ms` is the user-facing latency of the cheap call."""

        verification = getattr(routed, "verification", None)
        quality = verification.agreement if verification is not None else None
        escalated = getattr(routed, "escalation", None) is not None
        esc_cost = routed.escalation.cost_delta_usd if escalated else 0.0
        entry = RequestLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_hash=_prompt_hash(routed.prompt),
            task_type=getattr(routed, "task_type", "default"),
            tier=routed.tier,
            routed_model=routed.chosen_model,
            cost_usd=routed.cost_usd,
            reference_cost_usd=reference_cost_usd,
            latency_ms=latency_ms,
            quality_score=quality,
            verified=verification is not None,
            escalated=escalated,
            escalation_cost_usd=esc_cost,
        )
        return self.log(entry)

    # Reading

    def all_rows(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute("SELECT * FROM requests ORDER BY id"))

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    def recent_failure_rate(self, limit: int = 200) -> float:
        """Share of escalations among the most recent verified requests.

        This is how I reconstruct the sampler's adaptation state after a
        restart. The log IS the persistent state."""

        with self._connect() as conn:
            rows = conn.execute(
                """SELECT escalated FROM requests
                   WHERE verified = 1 ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        if not rows:
            return 0.0
        return sum(r["escalated"] for r in rows) / len(rows)                                   
