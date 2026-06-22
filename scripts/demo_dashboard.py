"""
Demo: simulate a batch of requests through the logger and print the
cost dashboard. No API calls — uses the real pricing from the registry to make
the savings number realistic.

    python -m scripts.demo_dashboard

Writes to a throwaway DB (data/demo.db) so it never touches your real log.
"""

from __future__ import annotations

import random 
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.analytics import CostAnalytics, format_summary
from autopilot.logging_db import RequestLog, RequestLogger, _prompt_hash
from autopilot.registry import ModelRegistry

DEMO_DB = Path(__file__).resolve().parents[1] / "data" / "demo.db"

# Realistic per-request costs derived from the registry (≈300 in / 200 out).
_REG = ModelRegistry.default()

def _cost(model_name: str) -> float:
    cfg = _REG.get(model_name)
    return cfg.estimate_cost(300, 200)

def main() -> int:
    if DEMO_DB.exists():
        DEMO_DB.unlink()
    logger = RequestLogger(DEMO_DB)
    rng = random.Random(42)

    top_cost = _cost("claude-opus-4.8")
    tiers = [("low", "claude-haiku-4.5", 0.55),
             ("medium", "claude-sonnet-4.6", 0.30),
             ("high", "claude-opus-4.8", 0.15)]

    N = 500
    for _ in range(N):
        roll = rng.random()
        cum = 0.0
        for tier, model, share in tiers:
            cum += share
            if roll <= cum:
                break

        verified = rng.random() < 0.06
        escalated = verified and rng.random() < 0.15
        quality = None
        esc_cost = 0.0
        if verified:
            quality = rng.uniform(0.82, 1.0) if not escalated else rng.uniform(0.2, 0.75)
            if escalated:
                esc_cost = top_cost

        logger.log(RequestLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_hash=_prompt_hash(f"req-{rng.random()}"),
            task_type="classification",
            tier=tier, routed_model=model,
            cost_usd=_cost(model),
            reference_cost_usd=top_cost,
            latency_ms=rng.uniform(800, 3000),
            quality_score=quality,
            verified=verified, escalated=escalated,
            escalation_cost_usd=esc_cost,
        ))

    summary = CostAnalytics(DEMO_DB).summary()
    print(format_summary(summary))
    print(f"\n(simulated {N} requests into {DEMO_DB.name})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
          