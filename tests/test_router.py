"""Tests for escalation, feedback loop, and the router chain.

No network. A fake send_request drives the cheap model, the reference model,
and the judge. The classifier is trained in-process (fast).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.classifier import ComplexityClassifier, train_and_evaluate
from autopilot.escalation import Escalator, EscalationEvent
from autopilot.feedback import (
    corrected_tier_for, load_failures, record_failure, TrainingExample,
)
from autopilot.router import Router
from autopilot.sampling import AdaptiveSampler
from autopilot.verifier import VerificationResult


@dataclass
class _FakeResp:
    text: str
    cost_usd: float = 0.001
    input_tokens: int = 10
    output_tokens: int = 20

def _make_sender(cheap_text: str, ref_text: str):
    def send(prompt, cfg, max_tokens=512, **kwargs):
        if max_tokens == 8:
            return _FakeResp(text="5", cost_usd=0.0001)
        if cfg.name == "claude-opus-4.8":         # reference / top tier
            return _FakeResp(text=ref_text, cost_usd=0.013)
        return _FakeResp(text=cheap_text, cost_usd=0.001)
    return send

# Tests for escalation

def _failed_result():
    return VerificationResult(
        prompt="Classify ...", task_type="classification",
        cheap_model="claude-haiku-4.5", reference_model="claude-opus-4.8",
        cheap_output="technical", reference_output="billing",
        agreement=0.0, threshold=1.0, passed=False, reference_cost_usd=0.013,
    )

def test_escalation_reuses_reference_output():
    esc = Escalator(send_request=_make_sender("technical", "billing"))
    event = esc.escalate(_failed_result(), original_cost_usd=0.001)
    assert isinstance(event, EscalationEvent)
    assert event.escalated_model == "claude-opus-4.8"
    assert event.escalated_output == "billing"
    assert event.cost_delta_usd == 0.013
    assert event.quality_gap == 1.0

# Feedback

def test_corrected_tier_is_escalated_model_tier():
    event = EscalationEvent(
        timestamp="t", prompt="p", original_model="claude-haiku-4.5",
        escalated_model="claude-opus-4.8", original_output="x",
        escalated_output="y", original_cost_usd=0.001, escalated_cost_usd=0.013,
        cost_delta_usd=0.013, quality_gap=1.0, task_type="classification",
    )
    assert corrected_tier_for(event) == "high"

def test_record_failure_appends_example(tmp_path):
    log = tmp_path / "failures.jsonl"
    event = EscalationEvent(
        timestamp="t", prompt="route me", original_model="claude-haiku-4.5",
        escalated_model="claude-sonnet-4.6", original_output="x",
        escalated_output="y", original_cost_usd=0.001, escalated_cost_usd=0.006,
        cost_delta_usd=0.006, quality_gap=0.3, task_type="default",
    )
    ex = record_failure(event, path=log)
    assert isinstance(ex, TrainingExample)
    assert ex.tier == "medium"
    loaded = load_failures(log)
    assert len(loaded) == 1
    assert loaded[0]["prompt"] == "route me"


# Router chain

def _trained_classifier():
    train_and_evaluate(seed=42)
    return ComplexityClassifier.load()


def test_router_fast_path_returns_cheap_response():
    clf = _trained_classifier()
    router = Router(_make_sender("billing", "billing"), clf, sampler=AdaptiveSampler(rng=random.Random(1)))
    routed = router.handle("What is the capital of Japam? One word.")
    assert routed.chosen_model in {"claude-haiku-4.5", "llama3.1-8b"}
    assert routed.output == "billing"
    assert routed.cost_usd == 0.001


def test_router_escalates_on_failure(tmp_path, monkeypatch):
    import autopilot.feedback as fb
    monkeypatch.setattr(fb, "FAILURE_LOG", tmp_path / "f.jsonl")

    clf = _trained_classifier()
    router = Router(_make_sender("technical", "billing"), clf,
                    sampler=AdaptiveSampler(rng=random.Random(1)))
    routed = router.handle("Classify this: my invoice is wrong",
                           task_type="classification")
    routed.will_verify = True
    routed = router.verify_and_maybe_escalate(routed)

    assert routed.verification.passed is False
    assert routed.escalation is not None
    assert routed.output == "billing"


def test_router_pass_does_not_escalate():
    clf = _trained_classifier()
    router = Router(_make_sender("billing", "billing"), clf,
                    sampler=AdaptiveSampler(rng=random.Random(1)))
    routed = router.handle("Classify this: my invoice is wrong",
                           task_type="classification")
    routed.will_verify = True
    routed = router.verify_and_maybe_escalate(routed)
    assert routed.verification.passed is True
    assert routed.escalation is None