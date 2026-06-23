"""
API request/response schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

class CompletionRequest(BaseModel):
    """What a caller POSTs to /v1/completions. The caller may not pass a model (router). `low_provider` ('ollama'|'haiku') optionally
    overrides the low-tier model for this one request."""

    prompt: str = Field(..., min_length=1, description="The user prompt.")
    task_type: str = Field("default", description="extraction|classification|summarization|default")
    max_tokens: int = Field(512, ge=1, le=4096)
    max_tokens: int = Field(1024, ge=1, le=4096)
    low_provider: str | None = Field(
        None, description="Override the low-tier model: 'ollama' or 'haiku'. "
        "Only applies when the request routes to the low tier.")
    

class RoutingMetadata(BaseModel):
    tier: str
    confidence: float
    chosen_model: str
    cost_usd: float
    will_verify: bool
    verify_reason: str


class CompletionResponse(BaseModel):
    output: str
    routing: RoutingMetadata


class ModelInfo(BaseModel):
    name: str
    provider: str
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    quality_tier: str


class RoutingConfigUpdate(BaseModel):
    low: str
    medium: str
    high: str

