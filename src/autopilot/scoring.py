"""
Quality scoring strategies.

Every prompt is re-run on the highest-tier model and score how the "cheap" model holds against it.
"Good enough" is different for every task.
    extraction     -> field overlap (heuristic, no extra LLM call)
    classification -> exact label match (heuristic, no extra LLM call)
    summarization  -> LLM-as-judge 1-5 (needs one extra API call)
    default        -> token overlap (heuristic fallback)

Every scorer returns a float in [0, 1]. The caller compares it to the
per-task threshold in config/verification.yaml.
"""

from __future__ import annotations

import re

# Heuristic scores

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def score_extraction(candidate: str, reference: str) -> float:
    """Field overlap: how many fields does the cheap model produce that are also found in the reference?
    We approximate fields as distinct content tokens in the reference. 
    Returns |cand ∩ ref| / |ref|."""
    ref_tokens = _tokens(reference)
    if not ref_tokens:
        return 1.0
    cand_tokens = _tokens(candidate)
    hit = len(ref_tokens & cand_tokens)
    return hit / len(ref_tokens)

def score_classification(candidate: str, reference: str) -> float:
    """Exact label match. Classification outputs are short labels, so we
    compare the normalized strings. 1.0 if they match, else 0.0. We also accept
    the case where one is contained in the other (e.g. 'billing' vs
    'billing.')."""
    c, r = _normalize(candidate), _normalize(reference)
    if c == r:
        return 1.0
    if r and (r in c or c in r):
        return 1.0
    return 0.0

def score_token_overlap(candidate: str, reference: str) -> float:
    """Generic fallback: Jaccard overlap of content tokens."""
    c, r = _tokens(candidate), _tokens(reference)
    if not c and not r:
        return 1.0
    if not c or not r:
        return 0.0
    return len(c & r) / len(c | r)

# LLM-as-judge scorer

def score_summarization(
        candidate: str,
        reference: str,
        *,
        judge,
) -> float:
    """Ask a model as a judge to rate agreement 1-5, map [0, 1].
    
    `judge` is a callable (candidate, reference) -> str so the LLM call is
    injected. This keeps scoring testable (pass a fake judge) and avoids
    importing the provider layer here."""
    raw = judge(candidate=candidate, reference=reference)
    match = re.search(r"[1-5]", str(raw))
    if not match:
        return 0.0
    score_1_5 = int(match.group())
    return score_1_5 / 5.0


# Dispatch by task type

TASK_TYPES = ("extraction", "classification", "summarization", "default")

def score_agreement(
        candidate: str,
        reference: str,
        task_type: str = "default",
        *,
        judge=None
) -> float:
    """Score how well `candidate` (cheap model) agrees with `reference`
    (top-tier model) for the given task type. Returns a float in [0, 1].

    For summarization a `judge` callable must be provided; if it isn't, we fall
    back to token overlap so the verifier never crashes for lack of a judge.
    """
    if task_type == "extraction":
        return score_extraction(candidate, reference)
    if task_type == "classification":
        return score_classification(candidate, reference)
    if task_type == "summarization":
        if judge is None:
            return score_token_overlap(candidate, reference)
        return score_summarization(candidate, reference, judge=judge)
    return score_token_overlap(candidate, reference)