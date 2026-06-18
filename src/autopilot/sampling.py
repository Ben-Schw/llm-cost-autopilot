"""
Adaptive Sampling, deciding which requests get verified.

Two signals drive whether a request is sampled for verification:
  1. A base rate (e.g. verify 5% of traffic at random).
  2. Classifier confidence: low-confidence routing decisions are verified more
     often than confident ones — we spend our verification budget where the
     routing is most likely wrong.

The rate is adaptive: as observed failures stay low, trust rises and the
effective rate drops toward a floor; when failures rise, it climbs toward a
ceiling.

Everything is config-driven (config/sampling.yaml) so the policy changes
without code edits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
SAMPLING_CONFIG = _ROOT / "config" / "sampling.yaml"

@dataclass
class SamplingDecision:
    """Why a request was selected for verification."""

    verify: bool
    effective_rate: float
    reason: str

class AdaptiveSampler:
    """Decides whether a completed request should be verified."""

    def __init__(self, config_path: Path = SAMPLING_CONFIG, rng: random.Random | None = None):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self._base_rate: float = float(cfg["base_rate"])
        self._min_rate: float = float(cfg["min_rate"])
        self._max_rate: float = float(cfg["max_rate"])
        # Below this classifier confidence, always verify
        self._low_conf_threshold: float = float(cfg["low_confidence_threshold"])
        # How strongly the observed failure rate pushes the effective rate up.
        self._failure_sensitivity: float = float(cfg["failure_sensitivity"])   

        if rng is not None:
            self._rng = rng
        else:
            self._rng = random.Random(cfg.get("seed"))

        # Rolling failure tracking
        self._window = int(cfg.get("window", 200))
        self._recent: list[bool] = []

    def observed_failure_rate(self) -> float:
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)

    def record_outcome(self, was_failure: bool) -> None:
        """Feed a verification outcome back so the rate can adapt. Called after each verification completes."""
        self._recent.append(was_failure)
        if len(self._recent) > self._window:
            self._recent.pop(0)

    def effective_rate(self) -> float:
        """Current base rate, nudged up by the observed failure rate and clamped to [min_rate, max_rate]. When failures are ~0 the rate sits
        near base/min; as failures rise it climbs toward max."""
        bump = self.observed_failure_rate() * self._failure_sensitivity
        rate = self._base_rate + bump
        return max(self._min_rate, min(self._max_rate, rate))
    
    # Decision making

    def should_verify(self, confidence: float | None = None) -> SamplingDecision:
        """Decide whether to verify one request.

        confidence: the classifier's confidence in its tier (0-1) if available.
        Low-confidence requests are always verified; otherwise we sample at the
        current effective rate.
        """
        if confidence is not None and confidence < self._low_conf_threshold:
            return SamplingDecision(
                verify=True,
                effective_rate=1.0,
                reason=f"low confidence ({confidence:.2f} < "
                       f"{self._low_conf_threshold})",
            )
        rate = self.effective_rate()
        roll = self._rng.random()
        if roll < rate:
            return SamplingDecision(
                verify=True,
                effective_rate=rate,
                reason=f"sampled (roll {roll:.3f} < rate {rate:.3f})",
            )
        return SamplingDecision(
            verify=False,
            effective_rate=rate,
            reason=f"not sampled (roll {roll:.3f} >= rate {rate:.3f})",
        )