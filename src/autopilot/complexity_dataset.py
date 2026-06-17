"""
Create a labeled dataset (3 tiers: low, medium, high) for different prompts.
We use prompt templates with slot vatiation (210 examples)
Running this file creates a dataset in data/complexity_dataset.json and prints tier distribution."""


from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42

# Slot vocabularies

_EMAILS = ["billing@acme-corp.com", "support@globex.io", "jane.doe@example.org",
           "info@northwind.co", "team@initech.dev", "hello@umbrella.com",
           "accounts@hooli.net", "no-reply@piedpiper.io", "sales@wonka.co"]
_DATES = ["March 5, 2026", "12th of January 2025", "2024/07/19", "Aug 1 2026",
          "31 December 2023", "2nd of February 2027", "10/15/2024",
          "July 4, 2025", "29 Feb 2024"]
_COUNTRIES = ["Austria", "Japan", "Brazil", "Canada", "Kenya", "Portugal",
              "Norway", "Vietnam", "Chile", "Morocco", "Iceland", "Peru"]
_NUMBERS = ["17", "100", "49", "7", "256", "1000", "23", "81", "97", "144", "53"]
_TEXTS = [
    "Contact us at {email} for any billing questions.",
    "Our office moved to a new address last {month}.",
    "The meeting is scheduled for {date} at noon.",
    "Please find the invoice attached to this message.",
    "Reach out to {email} if you need a copy of the report.",
    "The warranty period ends on {date}.",
    "We will review your request within one {month}.",
]
_MONTHS = ["spring", "quarter", "month", "year", "week", "season"]
_TICKETS = [
    "My invoice shows a charge I do not recognize and I want a refund.",
    "The app crashes every time I open the settings page.",
    "I cannot log into my account after the password reset.",
    "Do you offer discounts for non-profit organizations?",
    "My subscription renewed but I was charged twice this month.",
    "The export button does nothing when I click it.",
    "How do I change the email address on my account?",
    "Your latest update deleted all my saved templates.",
    "I was promised a callback yesterday and never heard back.",
]
_PARAGRAPHS = [
    "Q3 revenue grew 12% on enterprise subscriptions while small-business "
    "churn rose slightly due to competition.",
    "The new release improves load times by 30% but introduces a regression "
    "in the export feature that affects large files.",
    "Customer satisfaction climbed after the support team expanded, though "
    "average response time during peak hours still lags the target.",
    "The pilot program reduced onboarding time by half, but adoption stalled "
    "in regions where the local language was unsupported.",
    "Marketing spend doubled year over year, driving strong top-of-funnel "
    "growth while conversion rates remained essentially flat.",
    "The migration cut infrastructure costs by 20% yet caused two brief "
    "outages that eroded trust among enterprise customers.",
]
_REVIEWS = [
    "The product works but setup took far longer than the docs suggested.",
    "Great value for the price, though shipping was slower than expected.",
    "Beautiful design, but the battery drains too quickly for daily use.",
    "Solid performance overall, yet the mobile app feels like an afterthought.",
    "Customer service was excellent, but the return process was a nightmare.",
    "Does exactly what it promises, though the learning curve is steep.",
]
_TOPICS = ["microservices", "rate limiting", "database sharding",
           "caching strategies", "message queues", "API versioning",
           "load balancing", "service discovery", "schema migration",
           "circuit breaking", "event sourcing", "connection pooling"]
_DECISIONS = [
    "hiring two mid-level engineers versus one senior engineer for the same budget",
    "building an internal tool versus buying a SaaS product",
    "migrating to microservices versus keeping the current monolith",
    "investing in automated tests versus shipping features faster",
    "open-sourcing a core library versus keeping it proprietary",
    "adopting a new framework versus deepening expertise in the current one",
    "centralizing data in one warehouse versus federating across teams",
]
_ROLES = ["a backend service", "a data pipeline", "an API gateway",
          "a recommendation engine", "a billing system", "a search index",
          "a notification service", "an authentication layer",
          "a real-time analytics dashboard", "a document storage system"]


def _pick(rng: random.Random, seq: list[str]) -> str:
    return rng.choice(seq)


# templates per tier
def _low_templates(rng: random.Random) -> list[str]:
    text = _pick(rng, _TEXTS).format(
        email=_pick(rng, _EMAILS), date=_pick(rng, _DATES),
        month=_pick(rng, _MONTHS))
    return [
        f"Extract the email address and return only it: '{text}'",
        f"Convert this date to ISO 8601 (YYYY-MM-DD), return only the date: "
        f"'{_pick(rng, _DATES)}'.",
        f"What is the capital of {_pick(rng, _COUNTRIES)}? Answer in one word.",
        f"Is {_pick(rng, _NUMBERS)} a prime number? Answer yes or no.",
        f"Reformat this text to uppercase: '{text}'",
        f"How many words are in this sentence: '{text}'? Return only the number.",
    ]

def _medium_templates(rng: random.Random) -> list[str]:
    return [
        f"Classify this support email into one of [billing, technical, account, "
        f"general] and return only the category: '{_pick(rng, _TICKETS)}'",
        f"Summarize the following in one sentence: '{_pick(rng, _PARAGRAPHS)}'",
        f"Analyze the sentiment (positive/negative/neutral) and give one reason: "
        f"'{_pick(rng, _REVIEWS)}'",
        f"Extract the three key points from this text as a bulleted list: "
        f"'{_pick(rng, _PARAGRAPHS)}'",
        f"Compare the pros and cons described here and state which side is "
        f"stronger: '{_pick(rng, _PARAGRAPHS)}'",
        f"Rewrite this customer review as a polite one-sentence summary for a "
        f"report: '{_pick(rng, _REVIEWS)}'",
        f"Categorize this review as a complaint, praise, or mixed feedback and "
        f"justify briefly: '{_pick(rng, _REVIEWS)}'",
        f"Turn this support ticket into a short structured note with fields "
        f"'issue' and 'urgency': '{_pick(rng, _TICKETS)}'",
        f"Identify the main trade-off described in this text in one sentence: "
        f"'{_pick(rng, _PARAGRAPHS)}'",
    ]

def _high_templates(rng: random.Random) -> list[str]:
    return [
        f"A train leaves City A at 9:00 at {_pick(rng, ['60','50','75'])} km/h. "
        f"Another leaves City B at 9:30 at {_pick(rng, ['90','80','100'])} km/h "
        f"toward A. The cities are {_pick(rng, ['300','250','400'])} km apart. "
        f"At what time do they meet? Show your reasoning step by step.",
        f"A startup must choose between {_pick(rng, _DECISIONS)}. Lay out the key "
        f"trade-offs across speed, risk, cost, and long-term scalability, then "
        f"give a reasoned recommendation.",
        f"Design a {_pick(rng, _TOPICS)} strategy for {_pick(rng, _ROLES)} that "
        f"must stay reliable under heavy load. Discuss at least two approaches "
        f"and their failure modes.",
        f"Write a short, persuasive product announcement for a new feature in "
        f"{_pick(rng, _ROLES)}, balancing excitement with technical credibility.",
        f"Given conflicting requirements (low latency, strong consistency, low "
        f"cost) for {_pick(rng, _ROLES)}, reason through the trade-offs and "
        f"propose an architecture, justifying each decision.",
        f"Critique the following plan and identify hidden assumptions and risks: "
        f"'We will {_pick(rng, _DECISIONS)} and expect to double output in a month.'",
    ]

TIER_GENERATORS = {
    "low": _low_templates,
    "medium": _medium_templates,
    "high": _high_templates,
}


def generate_dataset(n_per_tier: int = 70, seed: int = SEED) -> list[dict]:
    """Generate a dataset."""
    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for tier, generator in TIER_GENERATORS.items():
        produced = 0
        attempts = 0
        while produced < n_per_tier and attempts < n_per_tier * 40:
            attempts += 1
            prompt = _pick(rng, generator(rng))
            key = (tier, prompt)
            if key in seen:
                continue
            seen.add(key)
            records.append({"id": f"{tier}_{produced:03d}",
                            "prompt": prompt, "tier": tier})
            produced += 1

    rng.shuffle(records)
    return records

def main() -> int:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(exist_ok=True)
    dataset = generate_dataset()    

    out = data_dir / "complexity_dataset.json"
    out.write_text(json.dumps(dataset, indent=2))

    # shows the count of different tiers
    counts: dict[str, int] = {}
    for r in dataset:
        counts[r["tier"]] = counts.get(r["tier"], 0) + 1

    print(f"Generated {len(dataset)} labeled prompts -> {out}")
    print("Tier distribution:")
    for tier, c in sorted(counts.items()):
        print(f"  {tier:<8} {c}")
    print("\nExamples:")
    for tier in ("low", "medium", "high"):
        ex = next(r for r in dataset if r["tier"] == tier)
        print(f"  [{tier}] {ex['prompt'][:90]}...")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())