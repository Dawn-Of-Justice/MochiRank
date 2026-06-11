"""
Stage G — SHAP-derived reasoning strings.

Builds 1-2 sentence explanations from the model's actual decision weights.
No hallucination: all facts are pulled directly from the candidate profile.
"""

from __future__ import annotations

import numpy as np

FEATURE_LABELS: dict[str, str] = {
    "semantic_contrastive":           "semantic match to role",
    "sim_ideal_max":                  "strong similarity to ideal profile",
    "per_req_coverage_mean":          "JD requirement coverage",
    "core_skill_count_evidenced":     "corroborated core skills",
    "skill_evidence_ratio":           "skill credibility",
    "product_company_ratio":          "product company experience",
    "product_company_months":         "product company tenure",
    "yoe_fit_score":                  "experience level fit",
    "production_signal_count":        "production deployment evidence",
    "startup_experience":             "startup / founding-team experience",
    "behavioral_composite":           "engagement and availability signals",
    "recency_score":                  "recent platform activity",
    "bm25_score":                     "keyword match to JD",
    "best_edu_tier":                  "education tier",
    "github_activity_score":          "open-source / GitHub activity",
    "notice_penalty":                 "notice period",
    "ever_at_it_services_only":       "consulting-only background",
    "title_chaser_flag":              "frequent title changes",
}

_CONCERN_MESSAGES: dict[str, str] = {
    "notice_penalty":            "notice period may delay start",
    "recency_score":             "lower recent platform activity",
    "behavioral_composite":      "engagement signals below average",
    "product_company_ratio":     "limited product company tenure",
    "ever_at_it_services_only":  "career concentrated in IT services",
    "title_chaser_flag":         "frequent role changes",
    "sim_anti_max":              "profile partially resembles anti-persona",
}


def generate_reasoning(
    candidate_id: str,
    candidate: dict,
    shap_row: np.ndarray,
    feature_names: list,
    rank: int,
) -> str:
    """
    Returns a 1-2 sentence reasoning string derived from SHAP values.
    """
    feature_shap = dict(zip(feature_names, shap_row.tolist()))

    top_positive = sorted(
        [(f, v) for f, v in feature_shap.items() if v > 0.01],
        key=lambda x: -x[1],
    )[:2]

    top_concern = sorted(
        [(f, v) for f, v in feature_shap.items() if v < -0.01],
        key=lambda x: x[1],
    )[:1]

    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "professional")
    company_size = profile.get("current_company_size", "")
    if company_size in ("1-10", "11-50", "51-200"):
        company_type = "startup"
    elif company_size in ("201-500", "501-1000"):
        company_type = "mid-size company"
    else:
        company_type = "large company"

    positive_labels = [
        FEATURE_LABELS.get(f, f.replace("_", " ")) for f, _ in top_positive
    ]

    if positive_labels:
        pos_str = " and ".join(positive_labels[:2])
        positive_sentence = f"{yoe}yr {title} at {company_type}; strong {pos_str}."
    else:
        positive_sentence = f"{yoe}yr {title} at {company_type}."

    concern_sentence = ""
    if top_concern and rank <= 50:
        feat, _ = top_concern[0]
        msg = _CONCERN_MESSAGES.get(feat)
        if msg:
            if feat == "notice_penalty":
                nd = candidate.get("redrob_signals", {}).get("notice_period_days", "?")
                concern_sentence = f" Note: {nd}-day notice period."
            else:
                concern_sentence = f" Note: {msg}."

    return (positive_sentence + concern_sentence).strip()


def build_explainer(model_path: str = "artifacts/ranker_model.json"):
    """Load XGBoost model and build SHAP TreeExplainer. Call once at startup."""
    import shap
    import xgboost as xgb

    model = xgb.Booster()
    model.load_model(model_path)
    explainer = shap.TreeExplainer(model)
    return model, explainer
