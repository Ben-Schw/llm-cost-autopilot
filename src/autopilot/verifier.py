"""
Async quality verifier.

After the cheap model's response is already returned to the user, we queue a
background job that re-runs the same prompt on the highest-tier (reference)
model and scores how well the two agree. If agreement is below the per-task
threshold, the request is flagged as a routing failure.

The "async" part matters: verification must not add latency to the user-facing
response. For V1 we use asyncio in-process (the leanest option that's still
genuinely concurrent). Phase 5 will move this to a separate worker process.

Design:
    VerificationResult  - the outcome (score, pass/fail, both outputs)
    Verifier            - holds config + the send_request callable, exposes
                          verify() (one job) and a small asyncio queue runner
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .registry import ModelRegistry
from .scoring import score_agreement

_ROOT = Path(__file__).resolve().parents[2]
VERIFICATION_CONFIG = _ROOT / "config" / "verification.yaml"

@dataclass
class VerificationResult:
    """Outcome of verifying one request against the reference model."""

    prompt: str
    task_type: str
    cheap_model: str
    reference_model: str
    cheap_output: str
    reference_output: str
    agreement: float
    threshold: float
    passed: bool
    reference_cost_usd: float = 0.0
    # Populated only when an escalation happens (Phase 3, step 3).
    escalated: bool = False
    extra: dict = field(default_factory=dict)

class Verifier:
    """Runs quality verification for completed requests.

    `send_request` is injected so this module never imports providers directly and stays unit-testable with a fake sender.
    """

    def __init__(self, send_request, registry: ModelRegistry | None = None,
                 config_path: Path = VERIFICATION_CONFIG):
        self._send = send_request
        self._registry = registry or ModelRegistry.default()
        cfg = yaml.safe_load(Path(config_path).read_text())
        self._thresholds: dict = cfg["thresholds"]
        self._reference_model_name: str = cfg["reference_model"]
        self._judge_prompt: str = cfg.get("judge_prompt", "")

    # Threshold lookup
    def threshold_for(self, task_type: str) -> float:
        return float(self._thresholds.get(task_type, self._thresholds["default"]))
        
    # LLM-as-judge callable for summarization
    def _make_judge(self):
        """Return a judge(candidate, reference) -> str backed by the reference
        model. Used only for summarization scoring."""
        ref_cfg = self._registry.get(self._reference_model_name)

        def judge(candidate: str, reference: str) -> str:
            prompt = self._judge_prompt.format(candidate=candidate, reference=reference)
            resp = self._send(prompt, ref_cfg, max_tokens=8)
            return resp.text
        
        return judge
    
    # One verification job

    def verify(self, prompt: str, cheap_model_name: str, cheap_output: str,
               task_type: str = "default") -> VerificationResult:
        """Re-run `prompt` on the reference model, score agreement vs the cheap
        output, and decide pass/fail. Synchronous core; the async wrappers below
        call this off the user's critical path."""
        ref_cfg = self._registry.get(self._reference_model_name)
        ref_resp = self._send(prompt, ref_cfg, max_tokens=512)

        judge = self._make_judge() if task_type == "summarization" else None
        agreement = score_agreement(
            cheap_output, ref_resp.text, task_type, judge=judge
        )
        threshold = self.threshold_for(task_type)

        return VerificationResult(
            prompt=prompt,
            task_type=task_type,
            cheap_model=cheap_model_name,
            reference_model=self._reference_model_name,
            cheap_output=cheap_output,
            reference_output=ref_resp.text,
            agreement=agreement,
            threshold=threshold,
            passed=agreement >= threshold,
            reference_cost_usd=ref_resp.cost_usd,
        )
    
    # Async queue

    async def verify_async(self, *args, **kwargs) -> VerificationResult:
        """Run verify() in a thread so a blocking provider call doesn't stall
        the event loop."""
        return await asyncio.to_thread(self.verify, *args, **kwargs)

    async def run_queue(self, jobs: list[dict]) -> list[VerificationResult]:
        """Verify many completed requests concurrently.

        Each job is a dict: {prompt, cheap_model_name, cheap_output, task_type}.
        Returns results in the same order. This models the background worker
        draining a queue of finished requests.
        """
        tasks = [
            self.verify_async(
                j["prompt"], j["cheap_model_name"], j["cheap_output"],
                j.get("task_type", "default"),
            )
            for j in jobs
        ]
        return await asyncio.gather(*tasks)