"""
Tests for the verification queue (the API<->worker channel). No network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.queue_db import VerifyJob, VerifyQueue


def _job(prompt="hello"):
    return VerifyJob(
        prompt=prompt, chosen_model="claude-haiku-4.5", output="x",
        task_type="classification", tier="low", cost_usd=0.001,
        reference_cost_usd=0.013, confidence=0.9,
    )


def test_enqueue_increases_pending(tmp_path):
    q = VerifyQueue(tmp_path / "q.db")
    assert q.pending_count() == 0
    q.enqueue(_job())
    q.enqueue(_job())
    assert q.pending_count() == 2


def test_claim_returns_job_and_marks_in_progress(tmp_path):
    q = VerifyQueue(tmp_path / "q.db")
    q.enqueue(_job("route me"))
    claimed = q.claim_one()
    assert claimed is not None
    job_id, job = claimed
    assert job.prompt == "route me"
    assert q.pending_count() == 0


def test_claim_is_fifo(tmp_path):
    q = VerifyQueue(tmp_path / "q.db")
    q.enqueue(_job("first"))
    q.enqueue(_job("second"))
    _, j1 = q.claim_one()
    _, j2 = q.claim_one()
    assert j1.prompt == "first"
    assert j2.prompt == "second"


def test_claim_empty_returns_none(tmp_path):
    q = VerifyQueue(tmp_path / "q.db")
    assert q.claim_one() is None


def test_complete_removes_from_pipeline(tmp_path):
    q = VerifyQueue(tmp_path / "q.db")
    q.enqueue(_job())
    job_id, _ = q.claim_one()
    q.complete(job_id)
    assert q.pending_count() == 0
    assert q.claim_one() is None


def test_job_roundtrips_all_fields(tmp_path):
    q = VerifyQueue(tmp_path / "q.db")
    original = _job("roundtrip")
    q.enqueue(original)
    _, restored = q.claim_one()
    assert restored == original