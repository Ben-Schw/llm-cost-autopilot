"""
Feeding failures back to the classifier.

Every routing failure is a labeled example the classifier got wrong: the prompt
was routed to a tier that turned out too weak, so its TRUE tier is at least the
reference tier. We accumulate these corrected examples and periodically retrain
the complexity classifier on the original dataset PLUS the accumulated
failures. This is the flywheel that makes routing smarter over time.

The failures are appended to data/failure_examples.jsonl (one JSON per line) so
they persist across restarts and can be inspected by hand. retrain_with_feedback
merges them with the synthetic dataset and retrains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from .escalation import EscalationEvent

_ROOT = Path(__file__).resolve().parents[2]
FAILURE_LOG = _ROOT / "data" / "failure_examples.jsonl"

_MODEL_TIER = {
    "claude-haiku-4.5": "low",
    "llama3.1-8b": "low",
    "claude-sonnet-4.6": "medium",
    "claude-opus-4.8": "high",
}

@dataclass
class TrainingExample:
    """A corrected (prompt, tier) pair derived from routing failure."""

    prompt: str
    tier: str
    source: str = "failure"

def corrected_tier_for(event: EscalationEvent) -> str:
    """Tier the request should have been routed to."""
    return _MODEL_TIER.get(event.escalated_model, "high")

def record_failure(event: EscalationEvent, path: Path = FAILURE_LOG) -> TrainingExample:
    """Append a corrected training example"""
    example = TrainingExample(
        prompt=event.prompt,
        tier=corrected_tier_for(event),
        source="failure"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps({"prompt": example.prompt, "tier": example.tier,
                             "source": example.source}) + "\n")
    return example

def load_failures(path: Path = FAILURE_LOG) -> list[dict]:
    """Load accumulated failure examples (empty list if none yet)."""
    if not path.exists():
        return []
    examples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples
    

def retrain_with_feedback(seed: int = 42, failure_path: Path = FAILURE_LOG) -> dict:
    """Retrain the classifier on the sythetic dataset and failures.
    Returns the same metrics dict as the base trainer, plus how many failure
    examples were folded in. Imports are local so this module stays light when
    only recording (not retraining)."""
    from .complexity_dataset import generate_dataset
    from .features import features_to_vector
    from .classifier import ComplexityClassifier, TIERS
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    base = generate_dataset(seed=seed)
    failures = load_failures(failure_path)
    combined = base + [{"prompt": f["prompt"], "tier": f["tier"]} for f in failures]

    X = [features_to_vector(r["prompt"]) for r in combined]
    y = [r["tier"] for r in combined]

    # Create a train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=TIERS)

    ComplexityClassifier(model).save()

    return {"accuracy": acc, "confusion_matrix": cm,
            "n_failures_added": len(failures), "n_total": len(combined)}

