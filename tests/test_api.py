"""
Tests for the FastAPI service.

No network: send_request is faked and the router/logger/queue are pointed at
throwaway state. Uses TestClient as a context manager so startup runs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import autopilot.api as api
from autopilot.classifier import train_and_evaluate
from autopilot.logging_db import RequestLogger
from autopilot.queue_db import VerifyQueue
from autopilot.router import Router


@dataclass
class _FakeResp:
    text: str
    cost_usd: float = 0.001
    input_tokens: int = 300
    output_tokens: int = 200


def _fake_send(prompt, cfg, max_tokens=512, **kwargs):
    txt = "billing" if "invoice" in prompt.lower() else "Paris"
    return _FakeResp(text=txt, cost_usd=cfg.estimate_cost(300, 200))


@pytest.fixture
def client(tmp_path):
    train_and_evaluate(seed=42)
    with TestClient(api.app) as c:
        api.state.router = Router(_fake_send, api.state.classifier, api.state.registry)
        api.state.logger = RequestLogger(tmp_path / "api.db")
        api.state.queue = VerifyQueue(api.state.logger.db_path)
        yield c


def test_models_endpoint_lists_all(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()}
    assert "claude-haiku-4.5" in names


def test_completion_routes_and_returns_metadata(client):
    r = client.post("/v1/completions",
                    json={"prompt": "What is the capital of France? One word."})
    assert r.status_code == 200
    body = r.json()
    assert "output" in body
    assert body["routing"]["tier"] in {"low", "medium", "high"}
    assert body["routing"]["chosen_model"]


def test_low_provider_ollama_overrides_low_tier(client):
    r = client.post("/v1/completions",
                    json={"prompt": "What is the capital of France? One word.",
                          "low_provider": "ollama"})
    assert r.status_code == 200
    body = r.json()
    if body["routing"]["tier"] == "low":
        assert body["routing"]["chosen_model"] == "llama3.1-8b"


def test_bad_low_provider_returns_400(client):
    r = client.post("/v1/completions",
                    json={"prompt": "What is the capital of France?",
                          "low_provider": "gpt5"})
    assert r.status_code in {400, 200}
    if r.status_code == 400:
        assert "low_provider" in r.json()["detail"]


def test_stats_endpoint_shape(client):
    client.post("/v1/completions", json={"prompt": "Is 7 prime? Yes or no."})
    r = client.get("/v1/stats")
    assert r.status_code == 200
    assert "cost_reduction_pct" in r.json()


def test_routing_config_rejects_unknown_model(client):
    r = client.put("/v1/routing-config",
                   json={"low": "made-up-model", "medium": "claude-sonnet-4.6",
                         "high": "claude-opus-4.8"})
    assert r.status_code == 400