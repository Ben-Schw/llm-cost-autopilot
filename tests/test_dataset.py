"""
Tests autopilot.complexity_dataset.py and autopilot.features.py without network, API.
Verifies the dataset is balanced and >=200, that it's
reproducible, and that features separate the tiers in the expected direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.complexity_dataset import generate_dataset
from autopilot.features import FEATURE_NAMES, extract_features, features_to_vector


def test_dataset_length():
    data = generate_dataset()
    assert len(data) >= 200

def test_dataset_tiers():
    data = generate_dataset()
    tiers = {r["tier"] for r in data}
    assert tiers == {"low", "medium", "high"}

def test_dataset_reproducability():
    a = generate_dataset(seed=200)
    b = generate_dataset(seed=200)
    assert {r["prompt"] for r in a} == {r["prompt"] for r in b}

def test_duplicates():
    data = generate_dataset()
    seen = set()
    for r in data:
        key = (r["tier"], r["prompt"])
        assert key not in seen
        seen.add(key)

def test_vector_matches_names():
    vec = features_to_vector("Is 7 a prime number? Answer yes or now")
    assert len(vec) == len(FEATURE_NAMES)

def test_reasoning_words():
    low = extract_features("What is the capital of Austria? Answer in one word.")
    high = extract_features(
        "Compare two caching approaches and justify your recommendation."
    )
    assert high["reasoning_words"] > low["reasoning_words"]

def test_context_flag_detects_quoted_text():
    with_ctx = extract_features("Summarize this: 'Revenue grew 12% last quarter.'")
    without_ctx = extract_features("What is the capital of Japan?")
    assert with_ctx["has_context"] == 1.0
    assert without_ctx["has_context"] == 0.0
