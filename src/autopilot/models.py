"""
Core data models: Enums, ModelConfig (a registry entry) and Response (what every provider call returns).
The tiers get predicted by a classifier, and a routing map sends it to a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class QualityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass(frozen=True)
class ModelConfig:
    """
    Router chooses one model.
    Costs are USD per single token.
    """

    name: str
    provider: Provider
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_ms: float
    quality_tier: QualityTier

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.cost_per_input_token + output_tokens * self.cost_per_output_token)
    

@dataclass
class Response:
    """
    The standardized return type for provider calls.
    """

    text: str
    model_id: str
    provider: Provider
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float