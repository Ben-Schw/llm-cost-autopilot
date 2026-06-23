"""
The FastAPI service.

POST /v1/completions     - router picks the model; returns answer + metadata.
                           Verification is handed to the worker via the queue.
GET  /v1/models          - list registered models and their costs.
GET  /v1/stats           - the cost-savings summary.
PUT  /v1/routing-config  - change the tier->model map without a redeploy.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException

from .analytics import CostAnalytics
from .api_models import (CompletionRequest, CompletionResponse, ModelInfo,
                         RoutingConfigUpdate, RoutingMetadata)
from .classifier import ComplexityClassifier
from .logging_db import RequestLogger
from .queue_db import VerifyJob, VerifyQueue
from .registry import ModelRegistry
from .router import Router, ROUTING_CONFIG
from .providers import send_request

app = FastAPI(title="LLM Cost Autopilot", version="1.0")

class _State:
    registry: ModelRegistry
    classifier: ComplexityClassifier
    router: Router
    logger: RequestLogger
    queue: VerifyQueue
    reference_model: str

state = _State()


@app.on_event("startup")
def _startup() -> None:
    state.registry = ModelRegistry.default()
    state.classifier = ComplexityClassifier.load()   # must be trained first
    state.router = Router(send_request, state.classifier, state.registry)
    state.logger = RequestLogger()
    state.queue = VerifyQueue(state.logger.db_path)
    vcfg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "config" / "verification.yaml")
        .read_text()
    )
    state.reference_model = vcfg["reference_model"]


def _log_unverified(routed) -> None:
    """Log a request that was NOT sampled for verification."""
    ref_cfg = state.registry.get(state.reference_model)
    reference_cost = ref_cfg.estimate_cost(300, 200)
    state.logger.log_routed_response(
        routed, reference_cost_usd=reference_cost,
        latency_ms=routed.extra.get("latency_ms", 0.0),
    )   

@app.post("/v1/completions", response_model=CompletionResponse)
def completions(req: CompletionRequest, background: BackgroundTasks):
    start = time.perf_counter()

    try:
        routed = state.router.handle(req.prompt, task_type=req.task_type, low_provider=req.low_provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:         # provider failure, etc.
        raise HTTPException(status_code=502, detail=str(e))
    routed.extra["latency_ms"] = (time.perf_counter() - start) * 1000.0

    if routed.will_verify:
        # Hand off to the worker process via the DB queue.
        ref_cfg = state.registry.get(state.reference_model)
        state.queue.enqueue(VerifyJob(
            prompt=routed.prompt, chosen_model=routed.chosen_model,
            output=routed.output, task_type=routed.task_type, tier=routed.tier,
            cost_usd=routed.cost_usd,
            reference_cost_usd=ref_cfg.estimate_cost(300, 200),
            confidence=routed.confidence,
        ))
    else:
        background.add_task(_log_unverified, routed)

    return CompletionResponse(
        output=routed.output,
        routing=RoutingMetadata(
            tier=routed.tier, confidence=routed.confidence,
            chosen_model=routed.chosen_model, cost_usd=routed.cost_usd,
            will_verify=routed.will_verify, verify_reason=routed.verify_reason,
        ),
    )
    

@app.get("/v1/models", response_model=list[ModelInfo])
def list_models():
    return [
        ModelInfo(
            name=m.name, provider=m.provider.value, model_id=m.model_id,
            cost_per_input_token=m.cost_per_input_token,
            cost_per_output_token=m.cost_per_output_token,
            quality_tier=m.quality_tier.value,
        )
        for m in state.registry.all()
    ]


@app.get("/v1/stats")
def stats():
    summary = CostAnalytics(state.logger.db_path).summary()
    return {
        "total_requests": summary.total_requests,
        "actual_cost_usd": round(summary.actual_cost_usd, 6),
        "baseline_cost_usd": round(summary.baseline_cost_usd, 6),
        "savings_usd": round(summary.savings_usd, 6),
        "cost_reduction_pct": round(summary.cost_reduction_pct, 2),
        "escalation_rate": round(summary.escalation_rate, 4),
        "routing_distribution": summary.routing_distribution,
        "avg_quality_score": summary.avg_quality_score,
    }


@app.put("/v1/routing-config")
def update_routing(update: RoutingConfigUpdate):
    """Change the tier->model map at runtime (and persist to YAML)."""
    new_map = {"low": update.low, "medium": update.medium, "high": update.high}
    for tier, model_name in new_map.items():
        try:
            state.registry.get(model_name)
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_name}' for tier '{tier}'",
            )

    state.router._routing_map = new_map
    existing = yaml.safe_load(Path(ROUTING_CONFIG).read_text())
    existing["routing"] = new_map
    Path(ROUTING_CONFIG).write_text(yaml.safe_dump(existing, sort_keys=False))
    return {"status": "updated", "routing": new_map}


@app.get("/")
def root():
    return {"service": "LLM Cost Autopilot", "docs": "/docs"}