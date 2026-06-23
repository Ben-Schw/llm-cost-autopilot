"""The router — the chain that ties everything together.

    classify -> route to cheap model -> respond to user (fast path)
        -> sampler decides whether to verify (adaptive sampling)
            -> verifier scores agreement vs the reference model
                -> on failure: escalate + record feedback
            -> record outcome back into the sampler (adaptation)

The user-facing call (`handle`) returns immediately with the cheap response and
a decision record. Verification/escalation run via `verify_and_maybe_escalate`,
which in production is the async background job. Keeping them as two
methods makes the fast path and the background path explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .classifier import ComplexityClassifier
from .escalation import Escalator, EscalationEvent
from .feedback import record_failure
from .registry import ModelRegistry
from .sampling import AdaptiveSampler
from .verifier import Verifier
_ROOT = Path(__file__).resolve().parents[2]
ROUTING_CONFIG = _ROOT / "config" / "routing.yaml"

@dataclass
class RoutedResponse:
    """What the router returns on the fast path."""

    prompt: str
    tier: str
    confidence: float
    chosen_model: str
    output: str
    cost_usd: float
    will_verify: bool
    verify_reason: str
    task_type: str = "default"
    verification: object = None
    escalation: EscalationEvent | None = None
    extra: dict = field(default_factory=dict)

class Router:
    """Routes each request to the cheapest capable model and runs the
    sampled, async verification + escalation + feedback chain."""

    def __init__(self, send_request, classifier: ComplexityClassifier, 
                 registry: ModelRegistry | None = None,
                 sampler: AdaptiveSampler | None = None,
                 verifier: Verifier | None = None,
                 escalator: Escalator | None = None,
                 routing_config: Path = ROUTING_CONFIG):
        self._send = send_request
        self._registry = registry or ModelRegistry.default()
        self._classifier = classifier
        self._sampler = sampler or AdaptiveSampler()
        self._verifier = verifier or Verifier(send_request, self._registry)
        self._escalator = escalator or Escalator(send_request, self._registry)

        cfg = yaml.safe_load(Path(routing_config).read_text())
        self._routing_map: dict = cfg["routing"]

    # Fast path
    _LOW_PROVIDER_MODELS = {"ollama": "llama3.1-8b", "haiku": "claude-haiku-4.5"}

    def handle(self, prompt: str, task_type: str = "default",
               low_provider: str | None = None) -> RoutedResponse:
        """Classify, route to the cheap model, return immediately. Also decides
        (but does not run) whether this request will be verified.

        `low_provider` ('ollama' | 'haiku') overrides the model only when the
        request routes to the low tier; medium/high routing is untouched.
        """
        tier, confidence = self._classifier.predict_with_confidence(prompt)
        model_name = self._routing_map[tier]

        if tier == "low" and low_provider is not None:
            override = self._LOW_PROVIDER_MODELS.get(low_provider)
            if override is None:
                raise ValueError(
                    f"Unknown low_provider '{low_provider}'. "
                    f"Use one of {list(self._LOW_PROVIDER_MODELS)}."
                )
            model_name = override

        cfg = self._registry.get(model_name)

        resp = self._send(prompt, cfg, max_tokens=512)
        decision = self._sampler.should_verify(confidence=confidence)

        return RoutedResponse(
            prompt=prompt, tier=tier, confidence=confidence,
            chosen_model=model_name, output=resp.text, cost_usd=resp.cost_usd,
            will_verify=decision.verify, verify_reason=decision.reason,
            task_type=task_type,
        )
    
    # Background path
    def verify_and_maybe_escalate(
            self, routed: RoutedResponse
    ) -> RoutedResponse:
        """Run verification for a routed response (only call when
        routed.will_verify is True). On failure, escalate and record feedback.
        Feeds the outcome back into the sampler so the rate adapts."""
        result = self._verifier.verify(
            routed.prompt, routed.chosen_model, routed.output, routed.task_type
        )
        routed.verification = result

        if not result.passed:
            event = self._escalator.escalate(result, routed.cost_usd)
            routed.escalation = event
            routed.output = event.escalated_output
            record_failure(event)

        self._sampler.record_outcome(was_failure=not result.passed)
        return routed