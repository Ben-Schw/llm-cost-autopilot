"""
Random Forest trained on extracted features to predict a prompt's complexity tier.
>80% accuracy is enough for the routing skeleton.

The model is trained and evaluated. The module loads a saved model for inference called by the router.

Saves:
    models/complexity_clf.joblib                 # the trained pipeline
    data/confusion_matrix.txt                    # human-readable eval
"""

from __future__ import annotations

from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .complexity_dataset import generate_dataset
from .features import FEATURE_NAMES, features_to_vector

_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _ROOT / "models" / "complexity_clf.joblib"
TIERS = ["low", "medium", "high"]

class ComplexityClassifier:
    """Wrapper around a Random Forest. Maps: prompt -> tier"""

    def __init__(self, model: RandomForestClassifier):
        self._model = model

    def predict(self, prompt: str) -> str:
        vec = [features_to_vector(prompt)]
        return self._model.predict(vec)[0]
    
    def predict_with_confidence(self, prompt: str) -> tuple[str, float]:
        """Return (tier, probability_of_that_tier). The router uses the
        probability against min_confidence in routing.yaml."""
        vec = [features_to_vector(prompt)]
        proba = self._model.predict_proba(vec)[0]
        classes = list(self._model.classes_)
        best_idx = int(proba.argmax())
        return classes[best_idx], float(proba[best_idx])
    
    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ComplexityClassifier":
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run: python -m src.autopilot.classifier"
            )
        return cls(joblib.load(path))

def train_and_evaluate(seed: int = 42) -> dict:
    """Train on the generated dataset, evaluate on a held-out split, save the
    model. Returns a small metrics dict."""
    data = generate_dataset(seed=seed)
    X = [features_to_vector(r["prompt"]) for r in data]
    y = [r["tier"] for r in data]

    # Create a train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=TIERS)
    report = classification_report(y_test, y_pred, labels=TIERS, zero_division=0)  

    # Persist model + a readable confusion matrix.
    clf = ComplexityClassifier(model)
    clf.save()

    cm_lines = ["Confusion matrix (rows = true, cols = predicted)",
                "          " + "  ".join(f"{t:>7}" for t in TIERS)]
    for tier, row in zip(TIERS, cm):
        cm_lines.append(f"{tier:>8}  " + "  ".join(f"{v:7d}" for v in row))
    cm_text = "\n".join(cm_lines)
    (_ROOT / "data").mkdir(exist_ok=True)
    (_ROOT / "data" / "confusion_matrix.txt").write_text(
        f"Held-out accuracy: {acc:.3f}\n\n{cm_text}\n\n{report}\n"
    )

    # Obtains feature importances
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda p: p[1], reverse=True,
    )

    return {"accuracy": acc, "confusion_matrix": cm, "report": report,
            "importances": importances, "n_train": len(X_train),
            "n_test": len(X_test)}   

def main() -> int:
    m = train_and_evaluate()
    print(f"Trained on {m['n_train']} examples, tested on {m['n_test']}.")
    print(f"Held-out accuracy: {m['accuracy']:.1%}  "
          f"(target for V1: >80%)\n")

    print("Confusion matrix (rows = true, cols = predicted):")
    header = "          " + "  ".join(f"{t:>7}" for t in TIERS)
    print(header)
    for tier, row in zip(TIERS, m["confusion_matrix"]):
        print(f"{tier:>8}  " + "  ".join(f"{v:7d}" for v in row))

    print("\nTop feature importances:")
    for name, imp in m["importances"]:
        print(f"  {name:<18} {imp:.3f}")

    print(f"\nSaved model -> {MODEL_PATH.relative_to(_ROOT)}")
    print("Wrote data/confusion_matrix.txt")
    status = "PASS" if m["accuracy"] > 0.80 else "BELOW TARGET"
    print(f"\nV1 quality gate (>80%): {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())