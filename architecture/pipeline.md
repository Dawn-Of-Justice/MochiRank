# Ranking Pipeline Components

> Implementation detail for the runtime stages of `rank.py` (Stages A, B, E, F, G).
> Stage C/D feature detail is in [features.md](features.md); overall flow in [spec.md](spec.md).

## Consistency Engine (src/consistency_checks.py) — Stage A

Detects honeypots and impossible profiles. Run as hard gates — candidates with
`is_honeypot=True` are removed from the top-100 before writing the CSV.

```python
def check_consistency(c: dict) -> tuple[bool, int, list[str]]:
    """
    Returns (is_honeypot, violation_count, reasons).
    Any one of the first 3 checks alone is honeypot-level evidence.
    """
    violations = []
    yoe = c["profile"]["years_of_experience"]
    edu = c.get("education", [])
    career = c.get("career_history", [])

    # Check 1: YOE impossible given education end date
    if edu:
        latest_grad = max(e["end_year"] for e in edu)
        max_possible_yoe = 2026 - latest_grad
        if yoe > max_possible_yoe + 1.5:  # +1.5 yr buffer for estimation
            violations.append(f"yoe={yoe} impossible: graduated {latest_grad}")

    # Check 2: Company founded date vs duration
    # (Use known founding dates for fictional companies in the dataset)
    COMPANY_FOUNDED = {
        "pied piper": 2014,     # Silicon Valley fictional
        "initech": 1999,        # Office Space
        "wayne enterprises": 1939,
        "stark industries": 1940,
        "dunder mifflin": 1949,
    }
    for job in career:
        company_lower = job["company"].lower()
        for name, founded in COMPANY_FOUNDED.items():
            if name in company_lower:
                latest_possible_start = 2026
                if job.get("start_date"):
                    start_year = int(job["start_date"][:4])
                    if start_year < founded:
                        violations.append(
                            f"worked at {job['company']} before it was founded ({founded})"
                        )

    # Check 3: "expert" skill with 0 duration
    for skill in c.get("skills", []):
        if skill["proficiency"] == "expert" and skill.get("duration_months", 0) == 0:
            violations.append(f"expert {skill['name']} with 0 months experience")

    # Check 4: Overlapping employment dates
    active_roles = [j for j in career if not j.get("is_current")]
    for i, j1 in enumerate(active_roles):
        for j2 in active_roles[i+1:]:
            if j1.get("start_date") and j1.get("end_date") and \
               j2.get("start_date") and j2.get("end_date"):
                if j1["start_date"] < j2["end_date"] and j2["start_date"] < j1["end_date"]:
                    violations.append(f"overlapping roles: {j1['title']} and {j2['title']}")

    # Check 5: Total career months >> YOE * 12
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if total_career_months > yoe * 12 + 24:  # 24mo buffer
        violations.append(f"career months ({total_career_months}) >> YOE*12 ({yoe*12})")

    # Check 6: 10+ "expert" skills (keyword inflation)
    expert_count = sum(1 for s in c.get("skills", []) if s["proficiency"] == "expert")
    if expert_count >= 10:
        violations.append(f"suspiciously many expert skills: {expert_count}")

    # Check 7: signup_date > last_active_date
    sig = c["redrob_signals"]
    if sig["signup_date"] > sig["last_active_date"]:
        violations.append("signup_date after last_active_date")

    is_honeypot = len([v for v in violations if "impossible" in v or "before it was founded" in v or "expert" in v and "0 months" in v]) > 0
    return is_honeypot, len(violations), violations
```

## JD Disqualifier Engine — Stage F

Applied as hard gates BEFORE the top-100 is finalized. Saves NDCG@10 from being
polluted by candidates the JD explicitly says to reject.

```python
def apply_jd_disqualifiers(candidate: dict, features: dict) -> tuple[bool, str]:
    """
    Returns (disqualified, reason).
    """
    c = candidate
    career = c.get("career_history", [])

    # 1. Title-function mismatch: current role is not engineering/tech
    NON_TECH_TITLES = [
        "marketing", "sales", "accountant", "operations manager",
        "customer support", "hr ", "human resource", "finance",
        "supply chain", "civil engineer", "mechanical engineer",
        "electrical engineer", "procurement",
    ]
    current_title = c["profile"]["current_title"].lower()
    if any(t in current_title for t in NON_TECH_TITLES):
        return True, f"non-technical current title: {c['profile']['current_title']}"

    # 2. Consulting-only career (all career at IT services, no product company)
    if career and features.get("ever_at_it_services_only"):
        return True, "entire career at IT services/consulting, no product company experience"

    # 3. Zero years relevant experience
    if c["profile"]["years_of_experience"] < 2.0:
        return True, f"insufficient experience: {c['profile']['years_of_experience']} years"

    # 4. No India location + not willing to relocate
    sig = c["redrob_signals"]
    if c["profile"]["country"] != "India" and not sig.get("willing_to_relocate"):
        return True, "outside India, not willing to relocate (no visa sponsorship)"

    # 5. CV/speech/robotics domain (no NLP/IR)
    CV_ONLY = ["computer vision", "object detection", "image segmentation",
               "speech recognition", "text to speech", "robotics", "ros "]
    NLP_IR = ["nlp", "retrieval", "ranking", "search", "recommendation",
              "text", "language model", "embedding", "information retrieval"]

    all_text = " ".join([
        c["profile"].get("summary", ""),
        c["profile"].get("headline", ""),
        *[j.get("description", "") for j in career]
    ]).lower()

    has_cv_only = any(t in all_text for t in CV_ONLY)
    has_nlp_ir = any(t in all_text for t in NLP_IR)
    if has_cv_only and not has_nlp_ir:
        return True, "CV/speech/robotics domain without NLP/IR exposure"

    return False, ""
```

## Hybrid Retrieval (src/retrieval.py) — Stage B

### BM25
```python
from rank_bm25 import BM25Okapi
import pickle

def build_bm25_index(candidates: list) -> BM25Okapi:
    corpus = []
    for c in candidates:
        text = " ".join([
            c["profile"].get("summary", ""),
            c["profile"].get("headline", ""),
            " ".join(s["name"] for s in c.get("skills", [])),
            " ".join(j.get("description", "") for j in c.get("career_history", [])),
        ])
        corpus.append(text.lower().split())
    return BM25Okapi(corpus)
```

BM25 query: concatenate JD's "Things you absolutely need" section + "How to read between the lines"
section. These are the recruiter's actual intent, not boilerplate.

### RRF Fusion
```python
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    rankings: list of ordered candidate_id lists (each from a different retriever)
    k: smoothing constant (default 60, from original Cormack et al. 2009 paper)
    """
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores  # higher = better

# Usage at retrieval time:
# bm25_ranking = [cid for cid, _ in sorted(bm25_scores.items(), key=lambda x: -x[1])]
# dense_ranking = [cid for cid, _ in sorted(dense_scores.items(), key=lambda x: -x[1])]
# rrf_scores = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
# top_2000 = sorted(rrf_scores, key=lambda c: -rrf_scores[c])[:2000]
```

Run multi-query: one dense search per ideal resume (3-5 queries) + one per JD section.
Merge all rankings into one RRF call. This gives much better recall than a single JD embedding.

## Optional Cross-Encoder Re-rank (src/reranker.py) — Stage E

Apply only to top-200 candidates. Dramatically improves NDCG@10 (the 50% metric).

```python
from sentence_transformers import CrossEncoder

def rerank_top_n(
    top_candidates: list[dict],
    jd_text: str,
    n: int = 200,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    backend: str = "onnx",
) -> list[tuple[str, float]]:
    """
    Returns (candidate_id, ce_score) sorted descending.
    ONNX backend: 2-3x faster on CPU. model_name is 22.7M params.
    """
    model = CrossEncoder(model_name, backend=backend,
                         model_kwargs={"provider": "CPUExecutionProvider"})
    pairs = []
    for c in top_candidates[:n]:
        candidate_text = " ".join([
            c["profile"].get("summary", ""),
            *[j.get("description", "") for j in c.get("career_history", [])[:3]]
        ])[:512]  # truncate to avoid slow inference
        pairs.append((jd_text[:512], candidate_text))

    scores = model.predict(pairs)
    results = [(top_candidates[i]["candidate_id"], float(scores[i]))
               for i in range(len(scores))]
    return sorted(results, key=lambda x: -x[1])
```

**Budget check:** 200 pairs × avg 300 tokens × quantized MiniLM-L6 ≈ 30-60 seconds on CPU.
This fits inside the 5-min window after the ~2 min for BM25+dense retrieval and feature computation.
If tight, reduce to top-100.

**Alternative — FlashRank (no torch, smaller):**
```python
from flashrank import Ranker
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")  # ~34MB, no torch required
```

## SHAP-Based Reasoning Generator (src/reasoning_generator.py) — Stage G

This is the Stage 4 survival mechanism. Derive reasoning from the model's actual decision.

```python
import xgboost as xgb
import shap
import numpy as np

# Load model and explainer once at startup (outside the per-candidate loop)
model = xgb.Booster()
model.load_model("artifacts/ranker_model.json")
explainer = shap.TreeExplainer(model)

FEATURE_LABELS = {
    "semantic_contrastive":       "semantic match to role",
    "core_skill_count_evidenced": "corroborated core skills",
    "skill_evidence_ratio":       "skill credibility",
    "product_company_ratio":      "product company experience",
    "yoe_fit_score":              "experience fit",
    "production_signal_count":    "production deployment evidence",
    "behavioral_composite":       "engagement/availability",
    "recency_score":              "recent platform activity",
    "per_req_coverage_mean":      "JD requirement coverage",
    "notice_penalty":             "notice period",
    "ever_at_it_services_only":   "consulting-only background",
}

POSITIVE_TEMPLATES = [
    "{yoe}yr {domain} background with {evidence} in career history",
    "Strong {req_area} experience at product company ({company_type}); {other_positive}",
    "{evidence} with {skill_count} corroborated core skills; good engagement signals",
]

CONCERN_TEMPLATES = [
    "Concern: {concern} may require additional assessment",
    "Note: {concern} — monitor at interview stage",
]

def generate_reasoning(candidate_id: str, candidate: dict, shap_values: np.ndarray,
                        feature_names: list, rank: int) -> str:
    """
    Returns a 1-2 sentence reasoning string.
    Built from SHAP attributions — hallucination-free by construction.
    """
    feature_shap = dict(zip(feature_names, shap_values))
    top_positive = sorted(
        [(f, v) for f, v in feature_shap.items() if v > 0],
        key=lambda x: -x[1]
    )[:2]
    top_concern = sorted(
        [(f, v) for f, v in feature_shap.items() if v < 0],
        key=lambda x: x[1]
    )[:1]

    c = candidate["profile"]
    yoe = c["years_of_experience"]

    # Build positive part
    positive_parts = []
    for feat, _ in top_positive:
        label = FEATURE_LABELS.get(feat, feat.replace("_", " "))
        positive_parts.append(label)

    # Add specific facts (no hallucination: pulled directly from profile)
    current_company_size = c.get("current_company_size", "")
    company_type = "startup" if current_company_size in ["1-10", "11-50", "51-200"] \
                   else "mid-size" if current_company_size in ["201-500", "501-1000"] \
                   else "large company"

    positive_sentence = (
        f"{yoe}yr {c['current_title']} at {company_type}; "
        f"strong {' and '.join(positive_parts[:2])}."
    )

    # Build concern part
    concern_sentence = ""
    if top_concern and rank <= 50:  # only add concern for high-ranked candidates
        feat, _ = top_concern[0]
        if feat == "notice_penalty":
            nd = candidate["redrob_signals"]["notice_period_days"]
            concern_sentence = f" Concern: {nd}-day notice period."
        elif feat == "recency_score":
            concern_sentence = f" Concern: lower recent platform activity."
        elif feat == "behavioral_composite":
            concern_sentence = f" Concern: engagement signals below average."
        elif feat == "product_company_ratio":
            concern_sentence = f" Concern: limited product company tenure."

    return (positive_sentence + concern_sentence).strip()
```
