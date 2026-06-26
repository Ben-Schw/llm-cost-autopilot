# LLM Cost Autopilot

Automatically routes LLM requests to the cheapest model capable of handling them — and verifies quality in the background. Sends simple prompts to cheap models, complex ones to expensive models, and escalates when quality fails.

## How It Works

```
Incoming prompt
      │
      ▼
 Classifier  ──── predicts complexity tier (low / medium / high)
      │
      ▼
   Router  ──── picks cheapest model for that tier
      │
      ▼
 Provider  ──── calls the model, returns answer immediately
      │
      ├── 5% of requests ──► Queue ──► Worker (background)
      │                                    │
      │                              Verifier: compares
      │                              output to reference model
      │                                    │
      │                              Failure? ──► Escalator
      │                                          (calls high-tier model)
      │                                          + records failure
      ▼
  Dashboard  ──── tracks cost savings, routing distribution, quality
```

**The feedback loop:** Every routing failure gets saved as a corrected training example. Running `retrain` folds these into the next classifier — the model gets smarter with each iteration.

---

## Models

| Tier | Default Model | Use case |
|------|--------------|----------|
| low | `claude-haiku-4.5` | Simple factual queries, one-word answers |
| medium | `claude-sonnet-4.6` | Summarization, extraction, moderate reasoning |
| high | `claude-opus-4.8` | Complex reasoning, multi-step analysis |

Low-tier can optionally be swapped to `llama3.1-8b` (local via Ollama, free).

---

## Requirements

- Python 3.11+
- Anthropic API key
- (Optional) [Ollama](https://ollama.com) for local inference

---

## Setup

**1. Clone and create virtual environment**

```powershell
git clone <repo>
cd llm-cost-autopilot
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```

**2. Configure API key**

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_HOST=http://localhost:11434   # optional
```

---

## Running the Project

### Step 1 — Train the classifier (once)

```powershell
.venv\Scripts\python.exe -m src.autopilot.classifier
```

Trains a Random Forest on synthetic prompts labelled by complexity tier. Saves the model to `models/complexity_clf.joblib`. Must run before starting the API.

### Step 2 — Start the API (Terminal 1)

```powershell
.venv\Scripts\uvicorn.exe autopilot.api:app --reload --app-dir src
```

API runs at `http://localhost:8000`. The dashboard is at `http://localhost:8000/`.

### Step 3 — Start the verification worker (Terminal 2)

```powershell
.venv\Scripts\python.exe -m scripts.worker
```

Polls the queue every 2 seconds, runs verification, logs results. Must run in parallel with the API for the feedback loop to work.

### Step 4 — (Optional) Collect baseline data

```powershell
.venv\Scripts\python.exe -m scripts.run_baseline --skip-ollama
```

Sends 10 prompts to every model and writes results to `data/baseline_results.json`. Useful for comparing model performance before/after routing.

---

## Retraining

Once the worker has logged real routing failures (stored in `data/failure_examples.jsonl`), retrain the classifier:

```powershell
.venv\Scripts\python.exe -m scripts.retrain
```

Then restart the API to pick up the improved model:

```powershell
.venv\Scripts\uvicorn.exe autopilot.api:app --reload --app-dir src
```

The retraining merges the original synthetic dataset with the collected failures. The printed accuracy shows whether the model improved.

---

## API Endpoints

### `POST /v1/completions`

Route a prompt through the autopilot.

```json
{
  "prompt": "Summarize this article in one sentence.",
  "task_type": "summarization",
  "max_tokens": 512,
  "low_provider": "haiku"
}
```

`task_type` options: `default`, `classification`, `extraction`, `summarization`  
`low_provider` options: `haiku`, `ollama` (overrides the low-tier model only)

Response:
```json
{
  "output": "...",
  "routing": {
    "tier": "low",
    "confidence": 0.91,
    "chosen_model": "claude-haiku-4.5",
    "cost_usd": 0.000023,
    "will_verify": false,
    "verify_reason": "not sampled (roll 0.812 >= rate 0.050)"
  }
}
```

### `GET /v1/stats`

Returns the cost savings dashboard data.

```json
{
  "total_requests": 142,
  "actual_cost_usd": 0.0031,
  "baseline_cost_usd": 0.0189,
  "savings_usd": 0.0158,
  "cost_reduction_pct": 83.6,
  "escalation_rate": 0.021,
  "routing_distribution": {
    "claude-haiku-4.5": 0.71,
    "claude-sonnet-4.6": 0.22,
    "claude-opus-4.8": 0.07
  }
}
```

### `GET /v1/models`

Lists all registered models with pricing.

### `PUT /v1/routing-config`

Change the tier→model mapping at runtime without redeploying.

```json
{
  "low": "llama3.1-8b",
  "medium": "claude-sonnet-4.6",
  "high": "claude-opus-4.8"
}
```

---

## Configuration

All config lives in `config/` and can be changed without touching code.

### `config/routing.yaml`

Which model handles each tier and the minimum classifier confidence threshold.

### `config/sampling.yaml`

Controls how many requests get verified:

| Key | Default | Meaning |
|-----|---------|---------|
| `base_rate` | 0.05 | 5% of confident requests are verified |
| `min_rate` | 0.01 | Never drop below 1% |
| `max_rate` | 0.50 | Never verify more than 50% |
| `low_confidence_threshold` | 0.55 | Always verify if classifier confidence < 55% |
| `failure_sensitivity` | 2.0 | How strongly failures raise the sample rate |

### `config/verification.yaml`

Quality thresholds per task type. A request "passes" when agreement with the reference model meets the threshold.

| Task | Threshold | Method |
|------|-----------|--------|
| `extraction` | 0.80 | Field overlap |
| `classification` | 1.0 | Exact label match |
| `summarization` | 0.80 | LLM-as-judge (1–5 score) |
| `default` | 0.70 | Token overlap |

---

## Project Structure

```
llm-cost-autopilot/
├── src/autopilot/
│   ├── api.py               # FastAPI app and endpoints
│   ├── router.py            # Fast path + background verification chain
│   ├── classifier.py        # Random Forest complexity classifier
│   ├── providers.py         # Anthropic + Ollama adapters
│   ├── registry.py          # Model registry loaded from models.yaml
│   ├── sampling.py          # Adaptive verification sampling
│   ├── verifier.py          # Compares cheap output vs. reference model
│   ├── escalation.py        # Calls high-tier model on failure
│   ├── feedback.py          # Records failures, drives retraining
│   ├── logging_db.py        # SQLite request log
│   ├── queue_db.py          # SQLite-backed verification queue
│   ├── analytics.py         # Dashboard metrics
│   ├── features.py          # Prompt feature extraction
│   ├── complexity_dataset.py# Synthetic training data generator
│   └── static/index.html    # Dashboard + playground UI
├── scripts/
│   ├── worker.py            # Background verification worker
│   ├── retrain.py           # Retraining with collected failures
│   └── run_baseline.py      # Baseline data collection
├── config/
│   ├── models.yaml          # Model registry and pricing
│   ├── routing.yaml         # Tier → model mapping
│   ├── sampling.yaml        # Verification sampling policy
│   └── verification.yaml    # Quality thresholds
├── data/                    # Generated (gitignored)
│   ├── autopilot.db         # Request log (SQLite)
│   ├── baseline_results.json
│   └── failure_examples.jsonl
├── models/                  # Generated (gitignored)
│   └── complexity_clf.joblib
├── tests/
└── requirements.txt
```

---

## Tests

```powershell
.venv\Scripts\pytest.exe tests/
```
