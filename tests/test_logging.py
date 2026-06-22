"""
Tests for Logger : the SQLite request logger and cost analytics.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.analytics import CostAnalytics, format_summary
from autopilot.logging_db import RequestLog, RequestLogger, _prompt_hash

def _entry(model="claude-haiku-4.5", cost=0.001, ref_cost=0.013,
           tier="low", verified=False, escalated=False, quality=None,
           esc_cost=0.0):
    return RequestLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_hash=_prompt_hash("hello"),
        task_type="classification", tier=tier, routed_model=model,
        cost_usd=cost, reference_cost_usd=ref_cost, latency_ms=900.0,
        quality_score=quality, verified=verified, escalated=escalated,
        escalation_cost_usd=esc_cost,
    )

# Logger

def test_log_and_count(tmp_path):
    logger = RequestLogger(tmp_path / "t.db")
    assert logger.count() == 0
    logger.log(_entry())
    logger.log(_entry())
    assert logger.count() == 2

def test_prompt_is_hashed_not_stored_raw(tmp_path):
    logger = RequestLogger(tmp_path / "t.db")
    logger.log(_entry())
    row = logger.all_rows()[0]
    assert row["prompt_hash"] == _prompt_hash("hello")
    assert "hello" not in row["prompt_hash"]

def test_recent_failure_rate_counts_only_verified(tmp_path):
    logger = RequestLogger(tmp_path / "t.db")
    logger.log(_entry(verified=True, escalated=True, quality=0.3))
    logger.log(_entry(verified=True, escalated=False, quality=0.9))
    logger.log(_entry(verified=False))   # not verified -> ignored
    assert logger.recent_failure_rate() == 0.5

# Analytics

def test_summary_empty_db_is_safe(tmp_path):
    summary = CostAnalytics(tmp_path / "t.db").summary()
    assert summary.total_requests == 0
    assert summary.cost_reduction_pct == 0.0


def test_cost_reduction_percentage(tmp_path):
    db = tmp_path / "t.db"
    logger = RequestLogger(db)
    for _ in range(10):
        logger.log(_entry(cost=0.001, ref_cost=0.013))
    summary = CostAnalytics(db).summary()
    # actual = 0.01, baseline = 0.13, savings = 0.12 -> ~92.3%
    assert abs(summary.actual_cost_usd - 0.01) < 1e-9
    assert abs(summary.baseline_cost_usd - 0.13) < 1e-9
    assert 92.0 < summary.cost_reduction_pct < 92.5
    

def test_escalation_cost_counts_against_savings(tmp_path):
    db = tmp_path / "t.db"
    logger = RequestLogger(db)
    logger.log(_entry(cost=0.001, ref_cost=0.013, verified=True,
                      escalated=True, quality=0.2, esc_cost=0.013))
    summary = CostAnalytics(db).summary()
    assert abs(summary.actual_cost_usd - 0.014) < 1e-9
    assert summary.escalation_count == 1
    assert summary.escalation_rate == 1.0


def test_routing_distribution_shares_sum_to_one(tmp_path):
    db = tmp_path / "t.db"
    logger = RequestLogger(db)
    for _ in range(3):
        logger.log(_entry(model="claude-haiku-4.5"))
    for _ in range(1):
        logger.log(_entry(model="claude-opus-4.8"))
    summary = CostAnalytics(db).summary()
    assert abs(sum(summary.routing_distribution.values()) - 1.0) < 1e-9
    assert summary.routing_distribution["claude-haiku-4.5"] == 0.75


def test_format_summary_contains_money_shot(tmp_path):
    db = tmp_path / "t.db"
    logger = RequestLogger(db)
    logger.log(_entry(cost=0.001, ref_cost=0.013))
    text = format_summary(CostAnalytics(db).summary())
    assert "COST REDUCTION" in text
