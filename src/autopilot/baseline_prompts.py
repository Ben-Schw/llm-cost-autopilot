"""
Ten prompts across the three complexity tiers, used to validate the
abstraction layer against every provider.
"""

BASELINE_PROMPTS = [
    # Tier 1: simple (reformat / extract / basic Q&A)
    {"id": "t1_extract_email", "tier": "low",
     "prompt": "Extract the email address and return only it: "
               "'Contact us at billing@acme-corp.com for questions.'"},
    {"id": "t1_reformat_date", "tier": "low",
     "prompt": "Convert to ISO 8601 (YYYY-MM-DD), return only the date: 'March 5, 2026'."},
    {"id": "t1_capital", "tier": "low",
     "prompt": "What is the capital of Austria? Answer in one word."},
    {"id": "t1_yesno", "tier": "low",
     "prompt": "Is 17 a prime number? Answer yes or no."},

    # Tier 2: moderate (summarize / classify / structured analysis)
    {"id": "t2_classify_ticket", "tier": "medium",
     "prompt": "Classify into [billing, technical, account, general], return only the category: "
               "'My invoice shows a charge I do not recognize and I want a refund.'"},
    {"id": "t2_summarize", "tier": "medium",
     "prompt": "Summarize in one sentence: 'Q3 revenue grew 12% on enterprise subscriptions, "
               "while small-business churn rose slightly due to competition.'"},
    {"id": "t2_sentiment", "tier": "medium",
     "prompt": "Sentiment (positive/negative/neutral) and one reason: "
               "'The product works but setup took far longer than the docs suggested.'"},

    # Tier 3: complex (multi-step reasoning / nuanced judgment)
    {"id": "t3_reasoning", "tier": "high",
     "prompt": "A train leaves City A at 9:00 at 60 km/h. Another leaves City B at 9:30 at "
               "90 km/h toward A. Cities are 300 km apart. When do they meet? Show your steps."},
    {"id": "t3_tradeoff", "tier": "high",
     "prompt": "Two mid-level engineers vs one senior for the same budget: lay out the "
               "trade-offs (speed, mentorship, risk, scalability) and give a recommendation."},
    {"id": "t3_edgecase", "tier": "high",
     "prompt": "Design a rate-limiting strategy for an API fair across thousands of tenants of "
               "different sizes while protecting a shared database. Discuss two algorithms and "
               "their failure modes."},
]