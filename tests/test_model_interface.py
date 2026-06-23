"""
Tests for the unified model interface. No network, no API keys:
they verify the registry, the cost math, the tier helpers, and that
send_request dispatches and normalizes correctly (provider call monkeypatched).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.models import ModelConfig, Provider, QualityTier, Response
from autopilot.registry import ModelRegistry
from autopilot import providers


def test_cost_estimate_per_token():
    cfg = ModelConfig(
        name="x", provider=Provider.ANTHROPIC, model_id="x",
        cost_per_input_token=1.0 / 1_000_000,
        cost_per_output_token=5.0 / 1_000_000,
        avg_latency_ms=100, quality_tier=QualityTier.LOW,
    )
    # 1M input + 1M output at $1/$5 per M => $6 exactly.
    assert cfg.estimate_cost(1_000_000, 1_000_000) == pytest.approx(6.0)   

def test_registry_has_expected_models():
    names = {m.name for m in ModelRegistry.default().all()}
    assert {"claude-haiku-4.5", "claude-sonnet-4.6",
            "claude-opus-4.8", "llama3.1-8b"} <= names

def test_yaml_registry_matches_code():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    reg = ModelRegistry.from_yaml(cfg_path)
    haiku = reg.get("claude-haiku-4.5")
    assert haiku.cost_per_input_token == pytest.approx(1.0 / 1_000_000)
    assert haiku.quality_tier is QualityTier.LOW

def test_send_request_normalizes_response(monkeypatch):
    cfg = ModelRegistry.default().get("claude-haiku-4.5")

    def fake_anthropic(prompt, c, max_tokens, temperature):
        return "hello world", 100, 20

    monkeypatch.setattr(providers, "_send_anthropic", fake_anthropic)

    resp = providers.send_request("hi", cfg)
    assert isinstance(resp, Response)
    assert resp.text == "hello world"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 20
    # 100*$1/M + 20*$5/M
    assert resp.cost_usd == pytest.approx(100 * (1/1e6) + 20 * (5/1e6))
    assert resp.latency_ms >= 0


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        ModelRegistry.default().get("gpt-does-not-exist")