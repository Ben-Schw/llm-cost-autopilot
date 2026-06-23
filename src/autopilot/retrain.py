"""Close the feedback loop: retrain the classifier on the base dataset PLUS the
routing failures collected in production.

    python -m scripts.retrain
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.feedback import FAILURE_LOG, load_failures, retrain_with_feedback


def main() -> int:
    failures = load_failures()
    if not failures:
        print(f"No failures recorded yet ({FAILURE_LOG} is empty or missing).")
        print("Nothing to retrain on — the base model already covers this.")
        return 0

    print(f"Found {len(failures)} routing failure(s) to fold into training.")
    print("Retraining classifier on base dataset + failures ...\n")

    metrics = retrain_with_feedback()

    print(f"  Failures folded in:    {metrics['n_failures_added']}")
    print(f"  Total training rows:   {metrics['n_total']}")
    print(f"  Held-out accuracy:     {metrics['accuracy']:.1%}")
    print("\nUpdated model saved. Restart the service to pick it up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())