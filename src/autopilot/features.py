"""
Here we turn prompts into numeric strings.
This includes:  - token count
                - presence of instruction words like "analyze" / "compare"
                - number of constraints
                - whether context is provided
                - output-format complexity.
We keep the features cheap and interpretable on purpose.
"""

from __future__ import annotations

import re

# Words that signal reasoning/analysis
_REASONING_WORDS = [
    "analyze", "compare", "design", "reason", "justify", "critique",
    "evaluate", "recommend", "trade-off", "tradeoff", "explain why",
    "step by step", "pros and cons", "propose", "assess",
]

# Words that signal a constrained / structured output.
_FORMAT_WORDS = [
    "json", "bulleted", "bullet", "list", "table", "one sentence",
    "one word", "yes or no", "only the", "fields", "structured", "iso 8601",
]

# === Functions to quantize a string ===

def _count_tokens(text: str) -> int:
    return len(text.split())

def _count_constraints(text: str) -> int:
    lowered = text.lower()

    constraint_markers = [
        "return only", "only the", "in one sentence", "one word", "yes or no",
        "as a bulleted list", "step by step", "and return", "justify",
        "with fields",
    ]
    count = sum(lowered.count(m) for m in constraint_markers)

    for group in re.findall(r"\[([^\]]+)\]", text):
        count += group.count(",")
    return count

def _has_quote(text: str) -> int:
    quote = "'" in text or '"' in text
    return 1 if quote else 0

def _format_complexity(text: str) -> int:
    lowered = text.lower()
    return sum(1 for w in _FORMAT_WORDS if w in lowered)

def _reasoning_signal(text: str) -> int:
    lowered = text.lower()
    return sum(1 for w in _REASONING_WORDS if w in lowered)  

# Stable feature order. The classifier relies on this.
FEATURE_NAMES = [
    "token_count",
    "reasoning_words",
    "constraint_count",
    "has_context",
    "format_complexity",
    "char_count",
    "avg_word_length",
]


def extract_features(prompt: str) -> dict[str, float]:
    """Extract the ordered numeric feature dict for one prompt."""
    tokens = _count_tokens(prompt)
    chars = len(prompt)
    return {
        "token_count": float(tokens),
        "reasoning_words": float(_reasoning_signal(prompt)),
        "constraint_count": float(_count_constraints(prompt)),
        "has_context": float(_has_quote(prompt)),
        "format_complexity": float(_format_complexity(prompt)),
        "char_count": float(chars),
        "avg_word_length": float(chars / tokens) if tokens else 0.0,
    }


def features_to_vector(prompt: str) -> list[float]:
    """Feature dict -> ordered vector matching FEATURE_NAMES."""
    feats = extract_features(prompt)
    return [feats[name] for name in FEATURE_NAMES]


if __name__ == "__main__":
    # Quick manual sanity check across the three tiers.
    samples = {
        "low": "Is 17 a prime number? Answer yes or no.",
        "medium": "Classify into [billing, technical, account, general] and "
                  "return only the category: 'My invoice is wrong.'",
        "high": "Design a caching strategy for an API gateway under heavy load. "
                "Compare two approaches and justify your recommendation.",
    }
    for tier, prompt in samples.items():
        print(f"[{tier}] {extract_features(prompt)}")