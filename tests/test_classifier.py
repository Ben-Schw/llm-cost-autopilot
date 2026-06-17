"""
Testing the classifier and the routing map.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.classifier import ComplexityClassifier, train_and_evaluate

_ROOT = Path(__file__).resolve().parents[1]

def test_accuracy_meets_bar():
    model = train_and_evaluate(seed=43)
    assert model["accuracy"] >= 0.8

def test_low_prompt_routes_low():
    train_and_evaluate(seed=43)
    clf = ComplexityClassifier.load()
    tier = clf.predict("What is the capital of Japan? Answer in one word.")
    assert tier == "low"

def test_high_prompt_routes_high():
    train_and_evaluate(seed=43)
    clf = ComplexityClassifier.load()
    tier = clf.predict(
        "Design a caching strategy for an API gateway under heavy load. "
        "Compare two approaches, reason through the trade-offs, and justify "
        "your recommendation step by step."
    )
    assert tier == "high"

def test_confidence_is_a_probability():
    train_and_evaluate(seed=42)
    clf = ComplexityClassifier.load()
    tier, conf = clf.predict_with_confidence("Is 7 a prime number? Yes or no.")
    assert tier in {"low", "medium", "high"}
    assert 0.0 <= conf <= 1.0

def test_routing_map_covers_all_tiers():
    routing = yaml.safe_load((_ROOT / "config" / "routing.yaml").read_text())
    mapping = routing["routing"]
    assert set(mapping) == {"low", "medium", "high"}
    from autopilot.registry import ModelRegistry
    reg = ModelRegistry.default()
    for model_name in mapping.values():
        assert reg.get(model_name) is not None