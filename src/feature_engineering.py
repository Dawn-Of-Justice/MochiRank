"""
Feature engineering — all ~48 features for a candidate profile.

Interface contracts (frozen after Sync 2):
    FEATURE_NAMES : list[str]          module-level, column order is the contract
    compute_features(candidate, precomputed, violation_count) -> np.ndarray
    compute_features_dict(candidate, precomputed)             -> dict[str, float]
    load_precomputed(artifacts_dir)                           -> dict

Semantic features (groups 6.1, 6.2) return 0.0 stubs until Sync 1 delivers
artifacts. All other feature groups are fully operational from day 1.
"""

import math
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import (
    REFERENCE_DATE,
    IT_SERVICES,
    JD_CORE_SKILLS,
    JD_NICE_SKILLS,
    PREFERRED_CITIES,
    PRODUCTION_KEYWORDS,
    EDU_TIER_SCORES,
    STEM_FIELDS,
    COMPANY_SIZE_ORDINAL,
    career_text,
    company_size_ordinal,
    days_since,
    is_it_services_company,
    notice_penalty,
    yoe_fit_score,
)

# --------------------------------------------------------------------------- #
# Feature name list — column order is frozen after Sync 2
# Append-only from that point; never reorder.
# --------------------------------------------------------------------------- #

FEATURE_NAMES: list[str] = [
    # 6.1 Semantic (dense) — stubs until Sync 1
    "sim_ideal_max",            # 0
    "sim_ideal_mean",           # 1
    "sim_anti_max",             # 2
    "semantic_contrastive",     # 3
    "sim_jd_summary",           # 4
    "per_req_coverage_mean",    # 5
    "per_req_coverage_min",     # 6
    "per_req_coverage_std",     # 7
    # 6.2 Lexical (BM25) — stubs until Sync 1
    "bm25_score",               # 8
    "bm25_rank_pct",            # 9
    # 6.3 Evidence-gated skill match
    "core_skill_count_raw",     # 10
    "core_skill_count_evidenced",  # 11
    "skill_evidence_ratio",     # 12
    "nice_skill_count_evidenced",  # 13
    "max_assessment_score_jd",  # 14
    "assessment_coverage",      # 15
    "github_activity_score",    # 16
    # 6.4 Career quality
    "years_of_experience",      # 17
    "yoe_fit_score",            # 18
    "product_company_months",   # 19
    "product_company_ratio",    # 20
    "current_role_relevance",   # 21
    "ever_at_it_services_only", # 22
    "longest_tenure_months",    # 23
    "avg_tenure_months",        # 24
    "title_chaser_flag",        # 25
    "max_company_size",         # 26
    "startup_experience",       # 27
    "production_signal_count",  # 28
    # 6.5 Education
    "best_edu_tier",            # 29
    "stem_degree",              # 30
    # 6.6 Logistics
    "india_location",           # 31
    "preferred_city_match",     # 32
    "willing_to_relocate",      # 33
    "notice_penalty",           # 34
    "salary_in_range",          # 35
    "work_mode_match",          # 36
    # 6.7 Behavioral signals
    "recency_score",            # 37
    "open_to_work",             # 38
    "recruiter_response_rate",  # 39
    "interview_completion_rate",# 40
    "applications_30d",         # 41
    "profile_completeness",     # 42
    "saved_by_recruiters_30d",  # 43
    "verified_contact",         # 44
    "behavioral_composite",     # 45
    # 6.8 Consistency
    "consistency_violation_count",  # 46
    "is_honeypot",              # 47
]

assert len(FEATURE_NAMES) == 48


# --------------------------------------------------------------------------- #
# Artifact loader
# --------------------------------------------------------------------------- #

def load_precomputed(artifacts_dir: Path) -> dict:
    """
    Load all precomputed artifacts from artifacts/.
    Missing files are silently skipped — stubs kick in for those features.
    """
    pre: dict[str, Any] = {"feature_names": FEATURE_NAMES}

    emb_path = artifacts_dir / "candidate_embeddings.npy"
    ids_path = artifacts_dir / "candidate_ids.json"
    jd_vec_path = artifacts_dir / "jd_query_vectors.npy"
    hyp_path = artifacts_dir / "hypothetical_resumes.json"
    bm25_path = artifacts_dir / "bm25_index.pkl"

    if emb_path.exists() and ids_path.exists():
        import json
        pre["candidate_embeddings"] = np.load(emb_path)
        with open(ids_path) as f:
            pre["candidate_ids"] = json.load(f)
        # Build a fast cid → row-index lookup
        pre["cid_to_idx"] = {cid: i for i, cid in enumerate(pre["candidate_ids"])}

    if jd_vec_path.exists():
        pre["jd_query_vectors"] = np.load(jd_vec_path)  # shape (n_queries, dim)

    if hyp_path.exists():
        import json
        with open(hyp_path) as f:
            pre["hypothetical_resumes"] = json.load(f)

    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            pre["bm25_index"] = pickle.load(f)

    return pre


# --------------------------------------------------------------------------- #
# Feature group implementations
# --------------------------------------------------------------------------- #

def _semantic_features(candidate: dict, precomputed: dict) -> list[float]:
    """Group 6.1 — returns stubs (0.0) until Sync 1 artifacts arrive."""
    cid = candidate["candidate_id"]
    cid_to_idx = precomputed.get("cid_to_idx", {})
    embeddings = precomputed.get("candidate_embeddings")
    jd_vecs = precomputed.get("jd_query_vectors")

    if embeddings is None or cid not in cid_to_idx:
        return [0.0] * 8

    # candidate embedding row(s) — may be multiple chunks per candidate
    # cid_to_idx maps to the first chunk; handle multi-chunk via candidate_ids list
    candidate_ids_list = precomputed.get("candidate_ids", [])
    rows = [i for i, c in enumerate(candidate_ids_list) if c == cid]
    if not rows:
        return [0.0] * 8

    cand_vecs = embeddings[rows].astype(np.float32)  # (n_chunks, dim)

    hyp = precomputed.get("hypothetical_resumes", [])
    ideal_vecs = jd_vecs[[i for i, h in enumerate(hyp) if h.get("type") == "ideal"]]
    anti_vecs  = jd_vecs[[i for i, h in enumerate(hyp) if h.get("type") == "anti"]]
    jd_summary_vec = jd_vecs[0:1]  # first vector is the JD summary

    # Per-requirement query vectors (last 6 in jd_query_vectors by convention)
    req_vecs = jd_vecs[-6:] if len(jd_vecs) >= 6 else jd_vecs

    def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return a_n @ b_n.T

    sim_ideal = cosine_matrix(cand_vecs, ideal_vecs) if len(ideal_vecs) else np.zeros((len(cand_vecs), 1))
    sim_anti  = cosine_matrix(cand_vecs, anti_vecs)  if len(anti_vecs)  else np.zeros((len(cand_vecs), 1))
    sim_jd    = cosine_matrix(cand_vecs, jd_summary_vec)

    sim_ideal_max  = float(sim_ideal.max())
    sim_ideal_mean = float(sim_ideal.mean())
    sim_anti_max   = float(sim_anti.max())
    semantic_contrastive = sim_ideal_max - 0.4 * sim_anti_max
    sim_jd_summary = float(sim_jd.max())

    # Per-requirement coverage
    sim_req = cosine_matrix(cand_vecs, req_vecs)  # (n_chunks, 6)
    per_req_max = sim_req.max(axis=0)             # (6,) best chunk per requirement
    per_req_coverage_mean = float(per_req_max.mean())
    per_req_coverage_min  = float(per_req_max.min())
    per_req_coverage_std  = float(per_req_max.std())

    return [
        sim_ideal_max, sim_ideal_mean, sim_anti_max, semantic_contrastive,
        sim_jd_summary, per_req_coverage_mean, per_req_coverage_min, per_req_coverage_std,
    ]


def _bm25_features(candidate: dict, precomputed: dict) -> list[float]:
    """Group 6.2 — stubs until Sync 1 delivers bm25_index."""
    bm25 = precomputed.get("bm25_index")
    if bm25 is None:
        return [0.0, 0.0]

    # BM25 score against JD core query tokens
    jd_query_tokens = (
        "production embeddings retrieval vector database hybrid search "
        "NDCG evaluation LLM fine-tuning LoRA Python senior AI engineer "
        "ranking recommendation NLP information retrieval startup"
    ).lower().split()

    ctext = career_text(candidate).split()
    score = float(bm25.get_scores(jd_query_tokens)[
        precomputed.get("cid_to_idx", {}).get(candidate["candidate_id"], 0)
    ])

    all_scores = precomputed.get("bm25_all_scores")
    if all_scores is not None:
        rank_pct = float(np.mean(all_scores <= score))
    else:
        rank_pct = 0.0

    return [score, rank_pct]


def _skill_features(candidate: dict) -> list[float]:
    """Group 6.3 — evidence-gated skill match."""
    skills: list[dict] = candidate.get("skills", [])
    career: list[dict] = candidate.get("career_history", [])
    sig: dict = candidate.get("redrob_signals", {})
    assessment_scores: dict = sig.get("skill_assessment_scores", {})

    all_career_text = " ".join(
        j.get("description", "") for j in career
    ).lower()

    def skill_name_lower(s: dict) -> str:
        return s.get("name", "").lower()

    def is_corroborated(skill: dict) -> bool:
        name = skill_name_lower(skill)
        in_career = name in all_career_text
        long_duration = skill.get("duration_months", 0) > 6
        many_endorsements = skill.get("endorsements", 0) > 5
        has_assessment = skill.get("name", "") in assessment_scores and \
                         assessment_scores[skill["name"]] > 60
        senior_proficiency = skill.get("proficiency") in ("advanced", "expert")
        return sum([in_career, long_duration, many_endorsements,
                    has_assessment, senior_proficiency]) >= 1

    def matches_jd_core(skill: dict) -> bool:
        name = skill_name_lower(skill)
        return any(kw.lower() in name or name in kw.lower()
                   for kw in JD_CORE_SKILLS)

    def matches_jd_nice(skill: dict) -> bool:
        name = skill_name_lower(skill)
        return any(kw.lower() in name or name in kw.lower()
                   for kw in JD_NICE_SKILLS)

    core_raw = sum(1 for s in skills if matches_jd_core(s))
    core_evidenced = sum(1 for s in skills if matches_jd_core(s) and is_corroborated(s))
    skill_evidence_ratio = core_evidenced / core_raw if core_raw > 0 else 0.0
    nice_evidenced = sum(1 for s in skills if matches_jd_nice(s) and is_corroborated(s))

    # Best assessment score on any JD-relevant skill
    jd_assessment_scores = [
        v for k, v in assessment_scores.items()
        if any(kw.lower() in k.lower() for kw in JD_CORE_SKILLS + JD_NICE_SKILLS)
    ]
    max_assessment = max(jd_assessment_scores) / 100.0 if jd_assessment_scores else 0.0

    # Fraction of JD core skills that have any assessment score
    core_skill_names = {s.get("name", "") for s in skills if matches_jd_core(s)}
    assessed_core = sum(1 for n in core_skill_names if n in assessment_scores)
    assessment_coverage = assessed_core / len(JD_CORE_SKILLS)

    github = sig.get("github_activity_score", -1)
    github_score = max(0.0, float(github)) / 100.0 if github >= 0 else 0.0

    return [
        float(core_raw),
        float(core_evidenced),
        skill_evidence_ratio,
        float(nice_evidenced),
        max_assessment,
        assessment_coverage,
        github_score,
    ]


def _career_features(candidate: dict) -> list[float]:
    """Group 6.4 — career quality."""
    profile: dict = candidate.get("profile", {})
    career: list[dict] = candidate.get("career_history", [])

    yoe = float(profile.get("years_of_experience", 0.0))

    # Product company months / ratio
    total_months = sum(j.get("duration_months", 0) for j in career)
    product_months = sum(
        j.get("duration_months", 0) for j in career
        if not is_it_services_company(j.get("company", ""))
    )
    product_ratio = product_months / total_months if total_months > 0 else 0.0

    # Ever exclusively at IT services
    ever_at_it_only = (
        len(career) > 0
        and all(is_it_services_company(j.get("company", "")) for j in career)
    )

    # Tenure stats
    durations = [j.get("duration_months", 0) for j in career if j.get("duration_months", 0) > 0]
    longest_tenure = max(durations) if durations else 0
    avg_tenure = sum(durations) / len(durations) if durations else 0.0

    # Title chaser: avg tenure < 18mo across 3+ consecutive jobs
    title_chaser = (len(durations) >= 3 and avg_tenure < 18.0)

    # Max company size (ordinal)
    max_size = max(
        (company_size_ordinal(j.get("company_size")) for j in career),
        default=0,
    )

    # Startup experience: any role at small company
    startup_exp = any(
        j.get("company_size") in ("1-10", "11-50", "51-200") for j in career
    )

    # Production signal count: action words in all descriptions
    all_desc = " ".join(j.get("description", "") for j in career).lower()
    production_signals = sum(kw in all_desc for kw in PRODUCTION_KEYWORDS)

    # Current role relevance — keyword proxy until Sync 1 delivers embeddings
    AI_ENG_KEYWORDS = {
        "ml", "machine learning", "ai", "artificial intelligence", "nlp",
        "deep learning", "llm", "retrieval", "ranking", "recommendation",
        "embeddings", "vector", "search", "data scientist", "research",
        "engineer", "senior",
    }
    current_title_lower = profile.get("current_title", "").lower()
    matched = sum(1 for kw in AI_ENG_KEYWORDS if kw in current_title_lower)
    current_role_relevance = min(1.0, matched / 3.0)

    return [
        yoe,
        yoe_fit_score(yoe),
        float(product_months),
        product_ratio,
        current_role_relevance,
        float(ever_at_it_only),
        float(longest_tenure),
        avg_tenure,
        float(title_chaser),
        float(max_size),
        float(startup_exp),
        float(production_signals),
    ]


def _education_features(candidate: dict) -> list[float]:
    """Group 6.5 — education."""
    edu: list[dict] = candidate.get("education", [])

    best_tier = max(
        (EDU_TIER_SCORES.get(e.get("tier", ""), 1) for e in edu),
        default=1,
    )

    stem = any(
        any(sf in (e.get("field_of_study", "") + " " + e.get("degree", "")).lower()
            for sf in STEM_FIELDS)
        for e in edu
    )

    return [float(best_tier), float(stem)]


def _logistics_features(candidate: dict) -> list[float]:
    """Group 6.6 — location / logistics."""
    profile: dict = candidate.get("profile", {})
    sig: dict = candidate.get("redrob_signals", {})

    country = profile.get("country", "")
    location = (profile.get("location", "") + " " + country).lower()

    india = country == "India"
    city_match = any(city in location for city in PREFERRED_CITIES)
    relocate = bool(sig.get("willing_to_relocate", False))

    np_days = int(sig.get("notice_period_days", 90))
    np_penalty = notice_penalty(np_days)

    salary_range = sig.get("expected_salary_range_inr_lpa", {})
    sal_min = salary_range.get("min", 0)
    sal_max = salary_range.get("max", 0)
    # JD budget estimate: 25–60 LPA
    salary_ok = not (sal_max < 25 or sal_min > 60)

    mode = sig.get("preferred_work_mode", "").lower()
    work_mode = 1.0 if mode in ("hybrid", "flexible", "onsite") else 0.6

    return [
        float(india),
        float(city_match),
        float(relocate),
        np_penalty,
        float(salary_ok),
        work_mode,
    ]


def _behavioral_features(candidate: dict) -> list[float]:
    """Group 6.7 — Redrob behavioral signals."""
    sig: dict = candidate.get("redrob_signals", {})

    last_active = sig.get("last_active_date", "")
    if last_active:
        inactive_days = days_since(last_active)
        recency = max(0.0, 1.0 - inactive_days / 180.0)
    else:
        recency = 0.0

    open_flag = float(sig.get("open_to_work_flag", False))
    response_rate = float(sig.get("recruiter_response_rate", 0.0))
    interview_rate = float(sig.get("interview_completion_rate", 0.0))
    apps_30d = math.log1p(float(sig.get("applications_submitted_30d", 0)))
    completeness = float(sig.get("profile_completeness_score", 0.0)) / 100.0
    saved = math.log1p(float(sig.get("saved_by_recruiters_30d", 0)))
    verified = int(sig.get("verified_email", False)) + int(sig.get("verified_phone", False))

    # Multiplicative composite — all three must be good for a high score
    open_multiplier = 1.1 if sig.get("open_to_work_flag") else 0.9
    composite = (
        (recency ** 0.5)
        * (response_rate ** 0.3)
        * (interview_rate ** 0.2)
        * open_multiplier
    )
    composite = float(np.clip(composite, 0.0, 1.1))

    return [
        recency,
        open_flag,
        response_rate,
        interview_rate,
        apps_30d,
        completeness,
        saved,
        float(verified),
        composite,
    ]


def _consistency_features(violation_count: int, is_honeypot: bool) -> list[float]:
    """Group 6.8 — passed in from Stage A results."""
    return [float(violation_count), float(is_honeypot)]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def compute_features(
    candidate: dict,
    precomputed: dict,
    violation_count: int = 0,
    is_honeypot: bool = False,
) -> np.ndarray:
    """
    Returns a float32 vector of length len(FEATURE_NAMES) == 48.
    Column order matches FEATURE_NAMES — never reorder after Sync 2.
    """
    feats: list[float] = []
    feats.extend(_semantic_features(candidate, precomputed))   # 8
    feats.extend(_bm25_features(candidate, precomputed))       # 2
    feats.extend(_skill_features(candidate))                   # 7
    feats.extend(_career_features(candidate))                  # 12
    feats.extend(_education_features(candidate))               # 2
    feats.extend(_logistics_features(candidate))               # 6
    feats.extend(_behavioral_features(candidate))              # 9
    feats.extend(_consistency_features(violation_count, is_honeypot))  # 2

    assert len(feats) == 48, f"Feature count mismatch: {len(feats)}"
    return np.array(feats, dtype=np.float32)


def compute_features_dict(
    candidate: dict,
    precomputed: dict,
    violation_count: int = 0,
    is_honeypot: bool = False,
) -> dict[str, float]:
    """Dict form of compute_features — used by Stage F disqualifier checks."""
    vec = compute_features(candidate, precomputed, violation_count, is_honeypot)
    return dict(zip(FEATURE_NAMES, vec.tolist()))
