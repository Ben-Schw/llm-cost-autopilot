"""
Cost analytics and the money-shot metric.

Reads the request log and computes everything the dashboard shows:
  - total actual cost vs. the baseline cost of sending EVERYTHING to the top
    model ("you saved $X")
  - the headline cost-reduction percentage (the money shot)
  - routing distribution (which model handled what share of traffic)
  - quality score distribution (from the verifier)
  - escalation rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .logging_db import DEFAULT_DB, RequestLogger

@dataclass
class CostSummary:
    """The numbers behind the dashboard's headline."""

    total_requests: int
    actual_cost_usd: float
    baseline_cost_usd: float
    savings_usd: float              
    cost_reduction_pct: float 
    escalation_count: int
    escalation_rate: float
    routing_distribution: dict = field(default_factory=dict)
    quality_distribution: dict = field(default_factory=dict)
    avg_quality_score: float | None = None

class CostAnalytics:
    """Computes dashboard metrics from the request log."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self._logger = RequestLogger(db_path)

    def summary(self) -> CostSummary:
        rows = self._logger.all_rows()
        n = len(rows)
        if n == 0:
            return CostSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        
        actual = sum(r["cost_usd"] + r["escalation_cost_usd"] for r in rows)
        baseline = sum(r["reference_cost_usd"] for r in rows)

        savings = baseline - actual
        reduction_pct = (savings / baseline * 100.0) if baseline > 0 else 0.0

        escalations = sum(r["escalated"] for r in rows)
        esc_rate = escalations / n

        dist: dict[str, int] = {}
        for r in rows:
            dist[r["routed_model"]] = dist.get(r["routed_model"], 0) + 1
        routing_distribution = {m: c / n for m, c in dist.items()}

        scored = [r["quality_score"] for r in rows if r["quality_score"] is not None]
        quality_distribution = _bucket_scores(scored)
        avg_quality = sum(scored) / len(scored) if scored else None

        return CostSummary(
            total_requests=n,
            actual_cost_usd=actual,
            baseline_cost_usd=baseline,
            savings_usd=savings,
            cost_reduction_pct=reduction_pct,
            escalation_count=escalations,
            escalation_rate=esc_rate,
            routing_distribution=routing_distribution,
            quality_distribution=quality_distribution,
            avg_quality_score=avg_quality,
        )


def _bucket_scores(scores: list[float]) -> dict[str, int]:
    """Group quality scores into readable buckets for the distribution chart."""
    buckets = {"0.0-0.5": 0, "0.5-0.8": 0, "0.8-1.0": 0}
    for s in scores:
        if s < 0.5:
            buckets["0.0-0.5"] += 1
        elif s < 0.8:
            buckets["0.5-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    return buckets


def format_summary(summary: CostSummary) -> str:
    """Render the summary as a plain-text dashboard for the CLI / README."""
    lines = [
        "=" * 48,
        "  LLM COST AUTOPILOT — COST DASHBOARD",
        "=" * 48,
        f"  Requests handled:      {summary.total_requests}",
        f"  Actual cost:           ${summary.actual_cost_usd:.4f}",
        f"  Baseline (all-top):    ${summary.baseline_cost_usd:.4f}",
        f"  Savings:               ${summary.savings_usd:.4f}",
        "",
        f"  COST REDUCTION:    {summary.cost_reduction_pct:.1f}%",
        "",
        f"  Escalation rate:       {summary.escalation_rate:.1%} "
        f"({summary.escalation_count} of {summary.total_requests})",
    ]
    if summary.avg_quality_score is not None:
        lines.append(f"  Avg quality score:     {summary.avg_quality_score:.2f}")
    lines.append("-" * 48)
    lines.append("  Routing distribution:")
    for model, share in sorted(summary.routing_distribution.items(),
                               key=lambda kv: kv[1], reverse=True):
        bar = "#" * int(share * 20)
        lines.append(f"    {model:<20} {share:5.1%} {bar}")
    lines.append("=" * 48)
    return "\n".join(lines)