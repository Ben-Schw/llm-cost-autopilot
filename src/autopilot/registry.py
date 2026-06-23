"""
Catalogue of models the router can choose from.

Defined in code (DEFAULT_REGISTRY) or loaded from config/models.yaml so you
can swap models without code changes.

Anthropic standard rates, June 2026 (per million tokens, /1e6 to per-token):
Claude Opus 4.8 $5/$25, Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5.
Local Llama via Ollama is $0/token, the cheapest route.
"""

from __future__ import annotations
from pathlib import Path
import yaml
from .models import ModelConfig, Provider, QualityTier

_M = 1_000_000

DEFAULT_REGISTRY: dict[str, ModelConfig] = {
    "claude-haiku-4.5": ModelConfig(
        name="claude-haiku-4.5", provider=Provider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        cost_per_input_token=1.0 / _M, cost_per_output_token=5.0 / _M,
        avg_latency_ms=900.0, quality_tier=QualityTier.LOW,
    ),
    "claude-sonnet-4.6": ModelConfig(
        name="claude-sonnet-4.6", provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-6",
        cost_per_input_token=3.0 / _M, cost_per_output_token=15.0 / _M,
        avg_latency_ms=1400.0, quality_tier=QualityTier.MEDIUM,
    ),
    "claude-opus-4.8": ModelConfig(
        name="claude-opus-4.8", provider=Provider.ANTHROPIC,
        model_id="claude-opus-4-8",
        cost_per_input_token=5.0 / _M, cost_per_output_token=25.0 / _M,
        avg_latency_ms=2600.0, quality_tier=QualityTier.HIGH,
    ),
    "llama3.1-8b": ModelConfig(
        name="llama3.1-8b", provider=Provider.OLLAMA,
        model_id="llama3.1:8b",
        cost_per_input_token=0.0, cost_per_output_token=0.0,
        avg_latency_ms=1200.0, quality_tier=QualityTier.LOW,
    ),
}

class ModelRegistry:
    """
    Lookup over a set of ModelConfigs.
    Router uses map to predict a tier to the cheapest capable model.
    """

    def __init__(self, models: dict[str, ModelConfig]):
        self._models = dict(models)

    @classmethod
    def default(cls) -> "ModelRegistry":
        return cls(DEFAULT_REGISTRY)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelRegistry":
        data = yaml.safe_load(Path(path).read_text())
        models = {}
        for name, cfg in (data.get("models") or {}).items():
            models[name] = ModelConfig(
                name=name,
                provider=Provider(cfg["provider"]),
                model_id=cfg["model_id"],
                cost_per_input_token=float(cfg["cost_per_input_token"]),
                cost_per_output_token=float(cfg["cost_per_output_token"]),
                avg_latency_ms=float(cfg.get("avg_latency_ms", 1000.0)),
                quality_tier=QualityTier(cfg["quality_tier"]),                
            )
        return cls(models)
    
    def get(self, name: str) -> ModelConfig:
        if name not in self._models:
            raise KeyError(f"Unknown model '{name}'. Known: {list(self._models)}")
        return self._models[name]            
    
    def all(self) -> list[ModelConfig]:
        return list(self._models.values())
