"""
Send 10 Prompts to every model log outputs, costs, latencies.
Proves abstraction layer works and gives basline data to the router.
    python -m scripts.run_baseline
    python -m scripts.run_baseline --skip-ollama
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autopilot.baseline_prompts import BASELINE_PROMPTS
from autopilot.models import Provider
from autopilot.providers import ProviderError, send_request
from autopilot.registry import ModelRegistry

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama models.")
    args = parser.parse_args()

    models = [m for m in ModelRegistry.default().all() if not (args.skip_ollama and m.provider is Provider.OLLAMA)]

    records = []
    for model in models:
        print(f"{model.name}")
        for baseline in BASELINE_PROMPTS:
            try:
                resp = send_request(baseline["prompt"], model, max_tokens=512)
                records.append({
                    "prompt_id": baseline["id"],
                    "tier": baseline["tier"],
                    "model": model.name,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "latency_ms": round(resp.latency_ms, 2),
                    "cost_usd": round(resp.cost_usd, 8),
                    "output_text": resp.text.strip(),
                })
                print(f"  [ok]  {baseline['id']:<20}; {resp.latency_ms:6.0f}ms;  ${resp.cost_usd:.6f}")

            except ProviderError as exc:
                print(f"  [ERR] {baseline['id']:<20} {exc}")
            
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "baseline_results.json").write_text(json.dumps(records, indent=2))
    print(f"\nWrote data/baseline_results.json ({len(records)} records)")
    return 0

if __name__=="__main__":
    raise SystemExit(main())