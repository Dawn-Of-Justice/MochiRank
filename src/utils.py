"""
Shared constants, helpers, and data loaders for the runtime pipeline.
Reference date: 2026-06-10.
"""

import json
import math
from datetime import date
from pathlib import Path
from typing import Iterator

REFERENCE_DATE = date(2026, 6, 10)

# --------------------------------------------------------------------------- #
# Company lists
# --------------------------------------------------------------------------- #

IT_SERVICES: set[str] = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "hcl technologies", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "mindtree", "persistent", "persistent systems",
    "coforge", "birlasoft", "niit technologies", "l&t infotech",
    "lti", "mastech", "kpit", "sonata software",
}

# Fictional companies used as honeypot markers (with founding year)
FICTIONAL_COMPANIES: dict[str, int] = {
    "pied piper": 2014,
    "initech": 1999,
    "wayne enterprises": 1939,
    "stark industries": 1940,
    "dunder mifflin": 1949,
    "umbrella corporation": 1968,
    "soylent": 2010,
    "globex": 1996,
}

# --------------------------------------------------------------------------- #
# JD skill lists (used by feature_engineering.py)
# --------------------------------------------------------------------------- #

JD_CORE_SKILLS: list[str] = [
    "embeddings", "retrieval", "vector", "ranking", "search",
    "sentence-transformers", "sentence_transformers", "faiss", "elasticsearch",
    "pinecone", "weaviate", "ndcg", "evaluation", "a/b", "python",
    "llm", "fine-tuning", "finetuning", "lora", "recommendation",
    "nlp", "ir", "information retrieval",
]

JD_NICE_SKILLS: list[str] = [
    "xgboost", "learning-to-rank", "ltr", "distributed systems",
    "inference optimization", "open source", "hr tech", "recruitment",
    "qdrant", "milvus", "opensearch",
]

# City names mapped for preferred-city matching
PREFERRED_CITIES: list[str] = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
    "gurgaon", "gurugram",
]

# Production-action words used for production_signal_count
PRODUCTION_KEYWORDS: list[str] = [
    "deployed", "deploy", "shipped", "ship", "scaled", "scale",
    "served", "serving", "production", "prod", "users", "latency",
    "throughput", "real-time", "realtime", "live",
]

# --------------------------------------------------------------------------- #
# Education tier mapping
# --------------------------------------------------------------------------- #

EDU_TIER_SCORES: dict[str, int] = {
    "tier_1": 4,
    "tier_2": 3,
    "tier_3": 2,
    "tier_4": 1,
}

STEM_FIELDS: set[str] = {
    "computer science", "cs", "cse", "information technology", "it",
    "electronics", "ece", "electrical", "eee", "mathematics", "statistics",
    "physics", "data science", "machine learning", "artificial intelligence",
    "ai", "ml", "information systems", "software engineering", "software",
}

# --------------------------------------------------------------------------- #
# Company-size ordinal encoding
# --------------------------------------------------------------------------- #

COMPANY_SIZE_ORDINAL: dict[str, int] = {
    "1-10": 1,
    "11-50": 2,
    "51-200": 3,
    "201-500": 4,
    "501-1000": 5,
    "1001-5000": 6,
    "5001-10000": 7,
    "10001+": 8,
}

# --------------------------------------------------------------------------- #
# Date / duration helpers
# --------------------------------------------------------------------------- #

def days_since(date_str: str, ref: date = REFERENCE_DATE) -> int:
    """Return days between date_str (ISO) and REFERENCE_DATE. Negative = future."""
    try:
        d = date.fromisoformat(date_str)
        return (ref - d).days
    except (ValueError, TypeError):
        return 0


def parse_year(date_str: str | None) -> int | None:
    """Extract year from an ISO date string or bare year string."""
    if not date_str:
        return None
    try:
        return int(str(date_str)[:4])
    except (ValueError, TypeError):
        return None


def notice_penalty(notice_days: int) -> float:
    """
    0 if notice_period_days <= 30.
    Linear decay: 0.5 at 90d, 0.0 at 180d+.
    """
    if notice_days <= 30:
        return 0.0
    if notice_days >= 180:
        return 1.0  # max penalty
    # linear between 30→90: 0→0.5; 90→180: 0.5→1.0
    return min(1.0, (notice_days - 30) / 150.0)


def yoe_fit_score(yoe: float) -> float:
    """
    Gaussian-like score peaked at [6, 8] years.
    - Below 4: steep penalty
    - 4–6: ramp up
    - 6–8: peak (1.0)
    - 8–12: soft decay
    - Above 12: harder decay
    """
    if yoe < 2:
        return 0.0
    if yoe <= 6:
        return max(0.0, (yoe - 2) / 4.0)
    if yoe <= 8:
        return 1.0
    if yoe <= 12:
        return max(0.5, 1.0 - (yoe - 8) / 8.0)
    return max(0.2, 0.5 - (yoe - 12) / 20.0)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def career_text(candidate: dict) -> str:
    """Concatenate all career description text, lowercased."""
    parts = [candidate["profile"].get("summary", ""),
             candidate["profile"].get("headline", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
    return " ".join(parts).lower()


def is_it_services_company(company_name: str) -> bool:
    name_lower = company_name.lower()
    return any(svc in name_lower for svc in IT_SERVICES)


def company_size_ordinal(size_str: str | None) -> int:
    return COMPANY_SIZE_ORDINAL.get(size_str or "", 0)


# --------------------------------------------------------------------------- #
# JSONL streaming loader (memory-efficient for 100K candidates)
# --------------------------------------------------------------------------- #

def stream_candidates(path: str | Path) -> Iterator[dict]:
    """Yield one candidate dict per line without loading the full file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates_json(path: str | Path) -> list[dict]:
    """Load sample_candidates.json (list format, not JSONL)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
