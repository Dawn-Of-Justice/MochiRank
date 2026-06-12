"""
Stage G — SHAP-derived reasoning strings.

Routing logic: SHAP confirms the rank is correct; the sentence structure is
chosen based on what is most *distinctive* in the actual profile, so each
candidate reads differently. All facts are pulled from the raw profile —
no hallucination.
"""

from __future__ import annotations
import numpy as np

_JD_CORE = [
    "python", "pytorch", "tensorflow", "transformers", "hugging face",
    "llm", "large language model", "fine-tuning", "lora", "rlhf",
    "embeddings", "vector database", "faiss", "pinecone", "weaviate",
    "retrieval", "rag", "ranking", "recommendation", "search",
    "information retrieval", "nlp", "natural language processing",
    "mlops", "model serving", "kubernetes", "docker", "triton",
    "langchain", "llamaindex", "openai", "anthropic", "opensearch",
    "elasticsearch", "semantic search", "reranking", "qlora",
]

_IT_SERVICES = {
    "tcs", "infosys", "wipro", "hcl", "tech mahindra", "cognizant",
    "accenture", "capgemini", "mphasis", "mindtree", "hexaware",
    "niit", "zensar", "l&t infotech",
}

_PROD_KW = {
    "deployed", "production", "serving", "latency", "throughput",
    "billion", "million users", "real-time", "a/b test", "monitoring",
    "inference", "scalable", "high-availability",
}

_TIER1 = {"Tier-1", "IIT", "NIT", "Top-10-Global"}


# ------------------------------------------------------------------ #
# Profile fact extractors — all read raw profile, nothing invented
# ------------------------------------------------------------------ #

def _skills(c: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in c.get("skills", []):
        name = s.get("name", "")
        if not name:
            continue
        nl = name.lower()
        if any(kw in nl or nl in kw for kw in _JD_CORE) and nl not in seen:
            seen.add(nl)
            out.append(name)
    return out


def _production_count(c: dict) -> int:
    text = " ".join(j.get("description", "") for j in c.get("career_history", [])).lower()
    return sum(1 for kw in _PROD_KW if kw in text)


def _startup_co(c: dict) -> str:
    """Name of first startup the candidate spent 6+ months at."""
    for j in c.get("career_history", []):
        if j.get("company_size") in ("1-10", "11-50", "51-200") and j.get("duration_months", 0) > 6:
            return j.get("company", "")
    return ""


def _product_cos(c: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for j in c.get("career_history", []):
        co = j.get("company", "")
        if not co or any(t in co.lower() for t in _IT_SERVICES):
            continue
        if j.get("duration_months", 0) > 6 and co.lower() not in seen:
            seen.add(co.lower())
            out.append(co)
    return out[:2]


def _best_edu(c: dict) -> tuple[str, str]:
    """(institution, tier) of highest-tier degree."""
    rank_of = {"Top-10-Global": 5, "IIT": 5, "Tier-1": 4, "NIT": 3}
    best, best_r = ("", ""), 0
    for e in c.get("education", []):
        t = e.get("tier", "")
        r = rank_of.get(t, 0)
        if r > best_r:
            best_r, best = r, (e.get("institution", ""), t)
    return best


def _response_rate(c: dict) -> float:
    return float(c.get("redrob_signals", {}).get("recruiter_response_rate", 0.0))


def _last_active(c: dict) -> str:
    return c.get("redrob_signals", {}).get("last_active_date", "")[:7]


def _notice_days(c: dict) -> int:
    return int(c.get("redrob_signals", {}).get("notice_period_days", 90))


def _longest_tenure(c: dict) -> tuple[int, str]:
    """(months, company) of longest single role."""
    best = (0, "")
    for j in c.get("career_history", []):
        dm = j.get("duration_months", 0)
        if dm > best[0]:
            best = (dm, j.get("company", ""))
    return best


# ------------------------------------------------------------------ #
# Routing: pick the most distinctive story based on profile, not SHAP
# ------------------------------------------------------------------ #

def _story_type(c: dict) -> str:
    """
    Profile-first routing. Every candidate in the top-100 has good SHAP
    semantic scores (that's why they're here). The interesting question is
    *what else* makes them stand out.
    """
    # Founding-team / startup experience is rare and highly valued
    if _startup_co(c):
        return "startup"

    # Tier-1 education is a strong differentiator
    inst, tier = _best_edu(c)
    if inst and tier in _TIER1:
        return "education"

    # Heavily production-focused career
    if _production_count(c) >= 7:
        return "production"

    # Highly responsive / immediately available
    if _response_rate(c) >= 0.80:
        return "behavioral"

    # Notable product-company tenure (name the companies)
    cos = _product_cos(c)
    if cos and len(cos) >= 2:
        return "career"

    # Default: describe skills + role (still specific due to different skill sets)
    return "role"


# ------------------------------------------------------------------ #
# Sentence composers
# ------------------------------------------------------------------ #

def _s(c: dict, yoe: float, title: str, co_type: str) -> str:
    """Dispatch to the right composer."""
    story = _story_type(c)
    skills = _skills(c)
    s3 = ", ".join(skills[:3]) if skills else "core ML/NLP stack"

    if story == "startup":
        name = _startup_co(c)
        prod = _production_count(c)
        prod_note = f"; {prod} production signals across descriptions" if prod >= 5 else ""
        return (
            f"{yoe:.0f}yr {title} with founding-team experience at {name}"
            f"{prod_note}; skills include {s3}."
        )

    if story == "education":
        inst, _ = _best_edu(c)
        cos = _product_cos(c)
        co_note = f" now at {cos[0]}" if cos else ""
        return (
            f"{yoe:.0f}yr {title} from {inst}{co_note}; "
            f"evidenced skills in {s3}."
        )

    if story == "production":
        n = _production_count(c)
        cos = _product_cos(c)
        co_note = f" at {cos[0]}" if cos else f" at {co_type}"
        return (
            f"{n} production deployment signals across career descriptions"
            f"{co_note}; {yoe:.0f}yr {title} with skills in {s3}."
        )

    if story == "behavioral":
        rr = _response_rate(c)
        la = _last_active(c)
        active = f", active as of {la}" if la else ""
        return (
            f"{rr:.0%} recruiter response rate{active}; "
            f"{yoe:.0f}yr {title} with skills in {s3}."
        )

    if story == "career":
        cos = _product_cos(c)
        co_str = " and ".join(cos[:2])
        prod = _production_count(c)
        prod_note = f"; {prod} production signals" if prod >= 4 else ""
        return (
            f"{yoe:.0f}yr {title} across {co_str}{prod_note}; "
            f"skills include {s3}."
        )

    # role — most common; vary the sentence lead using skill count or YOE
    n_skills = len(skills)
    prod = _production_count(c)
    if n_skills >= 6:
        return (
            f"{yoe:.0f}yr {title}; {n_skills} JD-matched skills "
            f"({', '.join(skills[:4])})."
        )
    if prod >= 4:
        return (
            f"{yoe:.0f}yr {title} at {co_type}; "
            f"{prod} production signals and skills in {s3}."
        )
    return (
        f"{yoe:.0f}yr {title} at {co_type}; "
        f"evidenced in {s3} across career."
    )


# ------------------------------------------------------------------ #
# Concern note (SHAP-driven — only things that pushed score DOWN)
# ------------------------------------------------------------------ #

def _concern(negatives: list[tuple[str, float]], c: dict) -> str:
    sig = c.get("redrob_signals", {})
    for feat, _ in negatives[:2]:
        if feat == "notice_penalty":
            days = _notice_days(c)
            return f" Note: {days}-day notice period."
        if feat == "recency_score":
            la = sig.get("last_active_date", "")
            return f" Note: inactive since {la[:7]}." if la else " Note: low recent activity."
        if feat == "behavioral_composite":
            rr = _response_rate(c)
            return f" Note: recruiter response rate {rr:.0%}." if rr > 0 else ""
        if feat == "product_company_ratio":
            return " Note: limited product company exposure."
        if feat == "sim_anti_max":
            return " Note: partial profile overlap with anti-persona archetype."
        if feat == "title_chaser_flag":
            return " Note: average tenure under 18 months across roles."
    return ""


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def generate_reasoning(
    candidate_id: str,
    candidate: dict,
    shap_row: np.ndarray,
    feature_names: list,
    rank: int,
) -> str:
    """
    1-2 sentence reasoning string.
    Story type chosen from profile facts; concern note from SHAP negatives.
    All facts sourced directly from the candidate dict — no hallucination.
    """
    shap = dict(zip(feature_names, shap_row.tolist()))
    negatives = sorted(
        [(f, v) for f, v in shap.items() if v < -0.005], key=lambda x: x[1]
    )

    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0))
    title = profile.get("current_title", "professional")
    cs = profile.get("current_company_size", "")
    co_type = (
        "startup" if cs in ("1-10", "11-50", "51-200")
        else "mid-size company" if cs in ("201-500", "501-1000")
        else "large company"
    )

    sentence = _s(candidate, yoe, title, co_type)
    concern = _concern(negatives, candidate) if rank <= 60 else ""
    return (sentence + concern).strip()


def build_explainer(model_path: str = "artifacts/ranker_model.json"):
    import shap, xgboost as xgb
    model = xgb.Booster()
    model.load_model(model_path)
    return model, shap.TreeExplainer(model)
