"""
Tests for scoring strategies, the async verifier, and
adaptive sampling without LLMs and APIs.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.scoring import (
    score_agreement, score_classification, score_extraction,
    score_token_overlap,
)
from autopilot.verifier import Verifier, VerificationResult
from autopilot.sampling import AdaptiveSampler


# Scoring

def test_extraction_full_match_is_one():
    assert score_extraction("billing@acme.com", "billing@acme.com") == 1.0


def test_extraction_partial_match_between_zero_and_one():
    s = score_extraction("billing", "billing@acme.com invoice 42 due monday")
    assert 0.0 < s < 1.0


def test_classification_exact_match():
    assert score_classification("billing", "billing") == 1.0


def test_classification_mismatch():
    assert score_classification("technical", "billing") == 0.0


def test_classification_tolerates_punctuation():
    assert score_classification("billing.", "billing") == 1.0


def test_token_overlap_symmetric_half():
    assert score_token_overlap("the cat sat", "the cat ran") == 0.5


def test_summarization_uses_injected_judge():
    score = score_agreement(
        "candidate summary", "reference summary", "summarization",
        judge=lambda candidate, reference: "4",
    )
    assert score == 0.8


# Verifier

@dataclass
class _FakeResp:
    text: str
    cost_usd: float = 0.013
    input_tokens: int = 10
    output_tokens: int = 20


def _make_sender(reference_text: str):
    def send(prompt, cfg, max_tokens=512, **kwargs):
        if max_tokens == 8:        # the judge call
            return _FakeResp(text="5")
        return _FakeResp(text=reference_text)
    return send


def test_verifier_passes_when_labels_match():
    v = Verifier(send_request=_make_sender("billing"))
    r = v.verify("Classify ...", "claude-haiku-4.5", "billing", "classification")
    assert isinstance(r, VerificationResult)
    assert r.passed is True
    assert r.agreement == 1.0
    assert r.reference_cost_usd == 0.013


def test_verifier_fails_when_labels_differ():
    v = Verifier(send_request=_make_sender("billing"))
    r = v.verify("Classify ...", "claude-haiku-4.5", "technical", "classification")
    assert r.passed is False
    assert r.agreement == 0.0


def test_verifier_threshold_lookup_falls_back_to_default():
    v = Verifier(send_request=_make_sender("x"))
    assert v.threshold_for("something_unknown") == v.threshold_for("default")


def test_async_queue_processes_all_jobs():
    v = Verifier(send_request=_make_sender("billing"))
    jobs = [
        {"prompt": "p1", "cheap_model_name": "claude-haiku-4.5",
         "cheap_output": "billing", "task_type": "classification"},
        {"prompt": "p2", "cheap_model_name": "claude-haiku-4.5",
         "cheap_output": "technical", "task_type": "classification"},
    ]
    results = asyncio.run(v.run_queue(jobs))
    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False


# Adaptive sampling

def test_sampling_is_reproducible_with_seed():
    a = AdaptiveSampler()
    b = AdaptiveSampler()
    seq_a = [a.should_verify(confidence=0.9).verify for _ in range(30)]
    seq_b = [b.should_verify(confidence=0.9).verify for _ in range(30)]
    assert seq_a == seq_b


def test_low_confidence_is_always_verified():
    s = AdaptiveSampler()
    decision = s.should_verify(confidence=0.40)
    assert decision.verify is True
    assert "low confidence" in decision.reason


def test_rate_climbs_when_failures_rise():
    s = AdaptiveSampler()
    base = s.effective_rate()
    for _ in range(100):
        s.record_outcome(was_failure=True)
    assert s.effective_rate() > base


def test_rate_stays_low_without_failures():
    s = AdaptiveSampler()
    for _ in range(100):
        s.record_outcome(was_failure=False)
    assert s.effective_rate() <= 0.05 + 1e-9


def test_effective_rate_respects_ceiling():
    s = AdaptiveSampler()
    for _ in range(200):
        s.record_outcome(was_failure=True)
    assert s.effective_rate() <= 0.50 + 1e-9