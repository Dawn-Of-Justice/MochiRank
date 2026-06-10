# Feature Engineering — Full ~40-Feature List

> All features are computed in `src/feature_engineering.py`.
> Function signature: `compute_features(candidate: dict, precomputed: dict) -> np.ndarray`
> See [spec.md](spec.md) for where features fit in the pipeline (Stage C/D).

## 6.1 Semantic Features (dense)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `sim_ideal_max` | max cosine(candidate_chunks, ideal_resumes) |
| 2 | `sim_ideal_mean` | mean cosine(candidate_chunks, ideal_resumes) |
| 3 | `sim_anti_max` | max cosine(candidate_chunks, anti_personas) |
| 4 | `semantic_contrastive` | `sim_ideal_max - 0.4 * sim_anti_max` |
| 5 | `sim_jd_summary` | cosine(candidate_global_vec, JD_summary_vec) |
| 6 | `per_req_coverage_mean` | mean of per-JD-requirement max-sim scores |
| 7 | `per_req_coverage_min` | min — how well candidate covers their *weakest* requirement |
| 8 | `per_req_coverage_std` | std — uneven coverage (strong in some, weak in others) |

**Per-requirement coverage detail:**
```python
# JD requirements decomposed into ~6 query vectors:
JD_REQUIREMENT_QUERIES = [
    "production embeddings-based retrieval systems sentence-transformers deployed real users",
    "vector database hybrid search infrastructure Pinecone Weaviate Elasticsearch FAISS",
    "evaluation frameworks ranking systems NDCG MRR MAP A/B testing offline-to-online",
    "LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost neural ranker",
    "production ML system deployed scale Python code quality engineering",
    "startup product company shipped recommendation search ranking real users",
]
# For each candidate: max-sim per requirement → 6-vector, then stat-pool
```

## 6.2 Lexical (BM25) Features

| # | Feature | Description |
|---|---------|-------------|
| 9 | `bm25_score` | BM25 score of candidate's career text against JD |
| 10 | `bm25_rank_pct` | rank percentile in BM25 ranking (0=bottom, 1=top) |

## 6.3 Evidence-Gated Skill Match

**Core idea:** a skill counts ONLY when corroborated. Defeats keyword stuffers.

Corroboration signals for each skill:
- Mentioned in any `career_history.description` (text match)
- `duration_months > 6`
- `endorsements > 5`
- Skill assessment score exists and > 60
- `proficiency` is `advanced` or `expert`

```python
JD_CORE_SKILLS = [
    "embeddings", "retrieval", "vector", "ranking", "search",
    "sentence-transformers", "FAISS", "Elasticsearch", "Pinecone", "Weaviate",
    "NDCG", "evaluation", "A/B", "Python", "LLM", "fine-tuning", "LoRA",
    "recommendation", "NLP", "IR", "information retrieval",
]
JD_NICE_SKILLS = [
    "XGBoost", "learning-to-rank", "distributed systems", "inference optimization",
    "open source", "HR tech", "recruitment",
]
```

| # | Feature | Description |
|---|---------|-------------|
| 11 | `core_skill_count_raw` | count of JD_CORE_SKILLS in skills list (naive) |
| 12 | `core_skill_count_evidenced` | count of corroborated core skills |
| 13 | `skill_evidence_ratio` | evidenced / raw (low = stuffer signal) |
| 14 | `nice_skill_count_evidenced` | count of corroborated nice-to-have skills |
| 15 | `max_assessment_score_jd` | highest assessment score on any JD-relevant skill |
| 16 | `assessment_coverage` | fraction of JD core skills with an assessment score |
| 17 | `github_activity_score` | raw value from redrob_signals (-1 = not linked → 0) |

## 6.4 Career Quality

| # | Feature | Description |
|---|---------|-------------|
| 18 | `years_of_experience` | raw float |
| 19 | `yoe_fit_score` | gaussian-like score, peak at 6-8yr, penalty below 4 and above 12 |
| 20 | `product_company_months` | months at non-IT-services, non-consulting companies |
| 21 | `product_company_ratio` | product_months / total_months |
| 22 | `current_role_relevance` | semantic sim of current_title to "Senior AI Engineer ML" |
| 23 | `ever_at_it_services_only` | bool: entire career at TCS/Infosys/Wipro/Accenture/etc |
| 24 | `longest_tenure_months` | longest single role duration |
| 25 | `avg_tenure_months` | average role duration |
| 26 | `title_chaser_flag` | bool: avg_tenure < 18mo across 3+ consecutive jobs |
| 27 | `max_company_size` | largest company size worked at (as ordinal 1–8) |
| 28 | `startup_experience` | any role at company_size ≤ "51-200" → bool |
| 29 | `production_signal_count` | count of action words in descriptions: "deployed", "shipped", "scaled", "served", "production", "users", "latency", "throughput" |

**IT Services company list:**
```python
IT_SERVICES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "mindtree", "persistent", "coforge", "birlasoft", "niit technologies",
}
```

## 6.5 Education

| # | Feature | Description |
|---|---------|-------------|
| 30 | `best_edu_tier` | ordinal encoding: tier_1=4, tier_2=3, tier_3=2, tier_4=1, unknown=1 |
| 31 | `stem_degree` | bool: CS/ECE/Math/Stats/Physics degree |

Note: education has a soft weight — JD explicitly says "5-9 years is a range not a requirement."
Don't let education dominate. Apply a monotonic constraint floor.

## 6.6 Logistics

| # | Feature | Description |
|---|---------|-------------|
| 32 | `india_location` | bool: country == "India" |
| 33 | `preferred_city_match` | bool: location contains Pune/Noida/Hyderabad/Mumbai/Delhi/Bangalore |
| 34 | `willing_to_relocate` | raw bool |
| 35 | `notice_penalty` | 0 if notice_period_days ≤ 30, linear decay to 0.5 at 90d, 0.0 at 180d |
| 36 | `salary_in_range` | bool: expected_salary overlaps with [25, 60] LPA (estimate) |
| 37 | `work_mode_match` | preferred_work_mode in {hybrid, flexible, onsite} → 1.0; remote → 0.6 |

## 6.7 Behavioral Signals (Redrob-specific, critical)

These are the "availability multipliers." The JD explicitly calls out dead accounts.

| # | Feature | Description |
|---|---------|-------------|
| 38 | `recency_score` | days since last_active_date, normalized: 0d→1.0, 180d→0.0, clamp |
| 39 | `open_to_work` | raw bool (hard positive signal) |
| 40 | `recruiter_response_rate` | raw float 0–1 |
| 41 | `interview_completion_rate` | raw float 0–1 |
| 42 | `applications_30d` | log1p(applications_submitted_30d) |
| 43 | `profile_completeness` | raw float 0–100, normalized /100 |
| 44 | `saved_by_recruiters_30d` | log1p value — proxy for market-validated interest |
| 45 | `verified_contact` | int: verified_email + verified_phone (0, 1, or 2) |
| 46 | `behavioral_composite` | weighted product of recency, response_rate, interview_completion |

**The behavioral composite formula:**
```python
def behavioral_composite(sig: dict, reference_date="2026-06-10") -> float:
    from datetime import date
    last = date.fromisoformat(sig["last_active_date"])
    today = date.fromisoformat(reference_date)
    days_inactive = (today - last).days
    recency = max(0.0, 1.0 - days_inactive / 180.0)

    response = sig["recruiter_response_rate"]  # 0–1
    interview = sig["interview_completion_rate"]  # 0–1
    open_flag = 1.1 if sig["open_to_work_flag"] else 0.9  # small boost/penalty

    # Multiplicative: all three must be good for a high composite
    composite = (recency ** 0.5) * (response ** 0.3) * (interview ** 0.2) * open_flag
    return float(np.clip(composite, 0.0, 1.1))
```

**Monotonic constraints for XGBoost:** behavioral signals should only HELP a candidate, never
hurt a good profile. Set monotone constraints to +1 for features 38–46. This prevents the
GBDT from learning spurious negative correlations from label noise.

## 6.8 Consistency / Honeypot Score

| # | Feature | Description |
|---|---------|-------------|
| 47 | `consistency_violation_count` | total number of logical violations detected |
| 48 | `is_honeypot` | bool (hard gate in Stage F, not a soft feature) |

See [pipeline.md](pipeline.md) for the consistency engine implementation.
