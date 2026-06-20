"""
Auto-escalation.

When the verifier catches a routing failure (the cheap model's output diverged
too far from the reference), we re-run the request on the higher-tier model and
return the better result. Every escalation is logged with the cost delta and
the quality gap that triggered it, so you can show how often and how
expensively the router was wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .registry import ModelRegistry
from .verifier import VerificationResult

@dataclass
class EscalationEvent:
    """Record for one escaltion."""

    timestamp: str
    prompt: str
    original_model: str
    escalated_model: str
    original_output: str
    escalated_output: str
    original_cost_usd: float
    escalated_cost_usd: float
    cost_delta_usd: float        # extra cost incurred by escalating
    quality_gap: float           # threshold - agreement (how far short it fell)
    task_type: str

class Escalator:
    """Re-runs prompts that failed on a higher-tier model and records the event."""

    def __init__(self, send_request, registry: ModelRegistry | None = None):
        self._send = send_request
        self._registry = registry or ModelRegistry.default()

    def escalate(
            self,
            result: VerificationResult,
            original_cost_usd: float,
            *,
            target_model_name: str | None = None,
    ) -> EscalationEvent:
        """Produces escalated answer for failed verifaction.
        By default we escalate to the reference model the verifier already
        used, reusing its output so we don't pay for a second top-tier call.
        If `target_model_name` differs from the reference, we make a fresh call
        to that model instead."""

        if target_model_name is None or target_model_name == result.reference_model:
            escalated_model = result.reference_model
            escalated_output = result.reference_output
            escalated_cost = result.reference_cost_usd
        else:
            cfg = self._registry.get(target_model_name)
            resp = self._send(result.prompt, cfg, max_tokens=512)
            escalated_model = target_model_name
            escalated_output = resp.text
            escalated_output = resp.cost_usd

        return EscalationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt=result.prompt,
            original_model=result.cheap_model,
            escalated_model=escalated_model,
            original_output=result.cheap_output,
            escalated_output=escalated_output,
            original_cost_usd=original_cost_usd,
            escalated_cost_usd=escalated_cost,
            cost_delta_usd=escalated_cost,  # the reference call is the extra spend
            quality_gap=max(0.0, result.threshold - result.agreement),
            task_type=result.task_type,
        )