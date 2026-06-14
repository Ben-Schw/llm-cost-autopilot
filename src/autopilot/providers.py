"""
The abstraction layer: one send_request() function hides provider-specific
details and always returns a standardized Response. Add a provider by writing
one adapter and adding it to the dispatch dict.

Latency is measured here; cost is computed from the ModelConfig so pricing
lives in one place. API keys come from the environment.
"""

from __future__ import annotations

import os
import time

from .models import ModelConfig, Provider, Response

class ProviderError(RuntimeError):
    pass

def send_request(prompt: str, model_config: ModelConfig, max_tokens: int = 1024, temperature: float = 1.0) -> Response:
    """
    Send prompt to a given model.
    Returns standardized Response.
    ProviderError on Failure.
    """
    dispatch = {Provider.ANTHROPIC: _send_anthropic, Provider.OLLAMA: _send_ollama}
    handler = dispatch.get(model_config.provider)
    if handler is None:
        raise ProviderError(f"No adapter for provider {model_config.provider}")
    
    start = time.perf_counter()
    try:
        text, in_tok, out_tok = handler(prompt, model_config, max_tokens, temperature)
    except Exception as e:
        raise ProviderError(
            f"{model_config.provider.value}/{model_config.model_id} failed: {e}"
        ) from e
    latency_ms = (time.perf_counter() - start) * 1000.0

    return Response(
        text=text,
        model_id=model_config.model_id,
        provider=model_config.provider,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        cost_usd=model_config.estimate_cost(in_tok, out_tok),
    )

def _send_anthropic(prompt, cfg, max_tokens, temperature):
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ProviderError("Anthropic API key is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.message.create(
        model=cfg.model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return text, msg.usage.input_tokens, msg.usage.output_tokens

def _send_ollama(prompt, cfg, max_tokens, temperature):
    import requests

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": cfg.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "")
    in_tok = data.get("prompt_eval_count") or _estimate_tokens(prompt)
    out_tok = data.get("eval_count") or _estimate_tokens(text)
    return text, in_tok, out_tok

def _estimate_tokens(text: str) -> int:
    """Fallback when Ollama does not return token usage."""
    return int(max(1, len(text.split())) * 1.33)