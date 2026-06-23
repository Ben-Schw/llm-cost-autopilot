"""
The background verification worker.

Polls the verify_queue, claims one job at a time, rebuilds the RoutedResponse,
runs the background path (verify -> maybe escalate -> feedback), and logs the
outcome. The API never blocks on any of this.

    python -m scripts.worker
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.classifier import ComplexityClassifier
from autopilot.logging_db import RequestLogger
from autopilot.queue_db import VerifyQueue
from autopilot.registry import ModelRegistry
from autopilot.router import Router, RoutedResponse
from autopilot.providers import send_request

POLL_INTERVAL_S = 2.0


def _process_one(router: Router, logger: RequestLogger, queue: VerifyQueue) -> bool:
    claimed = queue.claim_one()
    if claimed is None:
        return False
    job_id, job = claimed

    routed = RoutedResponse(
        prompt=job.prompt, tier=job.tier, confidence=job.confidence,
        chosen_model=job.chosen_model, output=job.output, cost_usd=job.cost_usd,
        will_verify=True, verify_reason="claimed from queue",
        task_type=job.task_type,
    )

    routed = router.verify_and_maybe_escalate(routed)
    logger.log_routed_response(routed, reference_cost_usd=job.reference_cost_usd)
    queue.complete(job_id)

    status = "ESCALATED" if routed.escalation else "ok"
    print(f"  [job {job_id}] {job.chosen_model} -> verified ({status})")
    return True


def main() -> int:
    registry = ModelRegistry.default()
    classifier = ComplexityClassifier.load()
    router = Router(send_request, classifier, registry)
    logger = RequestLogger()
    queue = VerifyQueue(logger.db_path)

    print("Verification worker started. Polling queue ... (Ctrl+C to stop)")
    try:
        while True:
            if not _process_one(router, logger, queue):
                time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nWorker stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())