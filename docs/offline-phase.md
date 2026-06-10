# Offline Phase — Step by Step

> Everything here runs with internet on, no time limit. Output artifacts feed the
> 5-minute ranking step. See [architecture/spec.md](../architecture/spec.md) for the big picture.

## 5.1 Embedding (precompute_embeddings.py)

**Model:** `BAAI/bge-small-en-v1.5` — 33M params, 384 dims, ~4x faster than MiniLM-L6 on CPU,
MTEB scores competitive. Ships in ~130MB. Alternatively `minishlab/potion-base-8M` (Model2Vec)
for a pure-numpy fallback that embeds 100K in ~30 seconds.

**What to embed — per-role utterances, NOT whole profiles:**
Each candidate gets a list of text chunks, not one blob. Research finding from Malt (2026):
embedding per-utterance enables late-interaction scoring and per-requirement coverage computation.

```python
def candidate_to_chunks(c: dict) -> list[str]:
    chunks = []
    # Role descriptions — most signal-rich field
    for job in c.get("career_history", []):
        if job.get("description"):
            chunks.append(f"{job['title']} at {job['company']}: {job['description']}")
    # Summary if present
    if c["profile"].get("summary"):
        chunks.append(c["profile"]["summary"])
    # Headline
    if c["profile"].get("headline"):
        chunks.append(c["profile"]["headline"])
    return chunks or [c["profile"].get("current_title", "")]
```

Save:
- `candidate_embeddings.npy` — shape `(N_chunks_total, 384)`, float16 to halve disk
- `candidate_ids.json` — list of candidate_ids, one per chunk row (allows grouping back)
- At load time, group by candidate_id and pool (max or mean) for global candidate vector

**Runtime estimate:** bge-small + batch_size=512 → ~8–12 min for 100K on 8-core CPU.
This is pre-computation; no time limit.

## 5.2 Hypothetical Resume Generation (generate_hypothetical.py)

**Why:** ConFit v2 (ACL 2025) showed 17.5% nDCG improvement by embedding a hypothetical
ideal resume alongside the JD before scoring. The format asymmetry (discursive JD vs structured
resume) kills naive cosine similarity. Generating ideal resumes bridges the gap.

**Novel addition: anti-persona resumes.** Not in any paper. Generates negative anchors
matching exactly the failure modes Redrob planted in the dataset.

**Prompt for ideal resumes:**
```python
IDEAL_PERSONA_PROMPT = """
You are generating a realistic candidate profile for the following job description.
Create a profile that would be a STRONG FIT — a real person with consistent career history,
specific accomplishments, and natural language (no keyword stuffing).

Generate {n} different profiles. Each should be a different archetype:
1. The IR veteran — 8yr, built search/ranking at product company pre-LLM era, now adding modern ML
2. The startup ML shipper — 6yr, 2-3 startups, shipped RAG/rec-sys to real users, scrappy
3. The platform engineer — 7yr, vector DB + hybrid search infra, scale-focused
4. The applied researcher — 5yr, MSc/PhD but in industry, eval frameworks, A/B testing mindset
5. The product-ML hybrid — 6yr, ex-PM turned engineer, retrieval + ranking + product instincts

For each profile, write:
- headline (1 line)
- summary (3-4 sentences, natural language, no buzzwords)
- 2-3 job roles with title, company type, duration, description (50-100 words each)
- 5-7 skills with realistic experience durations

JOB DESCRIPTION:
{jd_text}

Return JSON array of profiles.
"""
```

**Prompt for anti-persona resumes (our novel contribution):**
```python
ANTI_PERSONA_PROMPT = """
Generate {n} profiles that would seem relevant on surface but are explicitly
disqualified by the following job description.

The JD explicitly says these are NOT wanted:
1. Keyword stuffer — Marketing Manager with every AI keyword in skills, but career is marketing
2. Pure researcher — academic lab career, never shipped to production users
3. Consulting lifer — entire career at TCS/Infosys/Wipro/Accenture, no product company
4. Framework enthusiast — only LangChain/OpenAI wrapper projects, no pre-LLM ML experience
5. Title chaser — avg tenure <18 months across 4+ jobs, optimizing for "Senior" → "Staff"

Each profile should look superficially plausible but fail the actual JD requirements.

JOB DESCRIPTION:
{jd_text}

Return JSON array of profiles.
"""
```

**Final query vectors:**
- Embed each ideal+anti-persona resume as chunks
- `jd_query_vectors.npy` — shape `(n_ideals + n_anti, 384)` with metadata flag (positive/negative)
- At retrieval time: `sim_positive = max(cosine(candidate, ideals))`,
  `sim_negative = max(cosine(candidate, anti_personas))`,
  `semantic_score = sim_positive - 0.4 * sim_negative`

## 5.3 Stratified Sampler (stratified_sampler.py)

Claude teacher labels are expensive (API cost) and take time. Sample smartly.
Want coverage across the full relevance spectrum, including hard cases.

```python
def stratified_sample(candidates, embeddings, n=2500):
    strata = {
        "top_retrieval_bm25":    200,  # top BM25 hits — high relevance likely
        "top_retrieval_dense":   200,  # top dense hits — captures plain-language Tier 5s
        "top_anti_persona_sim":  150,  # high sim to anti-personas — keyword stuffers
        "title_match_strong":    200,  # current_title contains engineer/ML/AI
        "title_mismatch":        150,  # high skill-match, wrong title (stuffer detection)
        "consulting_only":       100,  # all career at big 5 IT services
        "honeypot_flagged":      100,  # caught by consistency engine
        "high_behavioral":       150,  # top redrob_signals scores
        "low_behavioral":        150,  # poor behavioral signals, maybe good profile
        "tier1_education":       100,  # IIT/IIM/NIT tier_1 candidates
        "random":                1000, # uniform random for distributional coverage
    }
    # Returns list of (candidate_id, stratum_label)
```

## 5.4 Teacher Labeling (teacher_label.py)

**Model:** Claude (via Anthropic API). `claude-sonnet-4-6` or `claude-haiku-4-5-20251001`
for cost efficiency. Haiku is ~20x cheaper and sufficient for labeling.

**Malt's two techniques (both required for label quality):**

1. **Semantic rubric anchoring** — fixed scale baked into the prompt so scores mean
   the same thing across all batches:
```
0.0 — No relevant skills or experience. Completely unable to perform the job.
0.2 — Minor relevance. Some adjacent skills but fundamentally wrong profile.
0.4 — Moderate match. Some relevant skills, significant gaps on core requirements.
0.6 — Good match. Mostly relevant, can perform with some ramp-up. Meets most requirements.
0.8 — Strong match. Highly relevant skills and experience. Ready to perform well.
1.0 — Perfect match. Skills and experience fully aligned. Expert on the topic.
```

2. **Anchored batching** — always include 1 obvious 0.0 and 1 obvious 1.0 as anchors
   in every batch of 12 candidates. Forces consistent calibration across batches.

**Batch prompt structure:**
```python
TEACHER_PROMPT = """
You are an objective evaluator for a recruiting platform.

JOB DESCRIPTION:
{jd_text}

SCORING RUBRIC (use ONLY these values):
0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0
{rubric_text}

Below are {n} candidate profiles. Score each independently.
Profile 1 is a known PERFECT FIT (score must be 0.9-1.0).
Profile {n} is a known NON-FIT (score must be 0.0-0.1).
Score profiles 2 through {n-1} based solely on the rubric.

For each candidate, provide:
- score: float (0.0, 0.2, 0.4, 0.6, 0.8, or 1.0)
- rationale: 1 sentence citing specific evidence from their profile

Return JSON array with fields: candidate_id, score, rationale

CANDIDATES:
{candidate_profiles_json}
"""
```

**What to include per candidate for the teacher (keep minimal to save tokens):**
```python
def candidate_for_teacher(c: dict) -> dict:
    return {
        "candidate_id": c["candidate_id"],
        "current_title": c["profile"]["current_title"],
        "years_of_experience": c["profile"]["years_of_experience"],
        "summary": c["profile"]["summary"][:400],  # truncated
        "career": [
            {
                "title": j["title"],
                "company": j["company"],
                "industry": j["industry"],
                "company_size": j["company_size"],
                "duration_months": j["duration_months"],
                "description": j["description"][:200]
            }
            for j in c["career_history"][:4]
        ],
        "skills_top5": [
            {"name": s["name"], "proficiency": s["proficiency"],
             "endorsements": s["endorsements"], "duration_months": s.get("duration_months", 0)}
            for s in sorted(c.get("skills", []),
                           key=lambda x: x["endorsements"], reverse=True)[:5]
        ],
        "education": [
            {"degree": e["degree"], "field": e["field_of_study"],
             "institution": e["institution"], "tier": e.get("tier")}
            for e in c.get("education", [])[:2]
        ]
    }
```

Save `teacher_labels.csv` with columns: `candidate_id, score, rationale, stratum`.

**Quality check before training:** compute self-consistency by double-labeling 100 candidates
with a fresh prompt. If Pearson correlation of scores < 0.85, the rubric needs tightening.

## 5.5 Train LambdaMART (train_ranker.py)

**Why LambdaMART:** Directly optimizes NDCG (the competition metric). LinkedIn production
talent search uses LTR with embedding features. XGBoost `rank:ndcg` is the standard implementation.

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group_sizes_train)  # required for LTR

params = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 500,
    "tree_method": "hist",
    # Monotonic constraints: behavioral signals should have monotone positive effect
    # Feature index must match column order in X
    "monotone_constraints": "(0,0,0,...,1,1,1,...)",  # fill after feature list is finalized
}

model = xgb.train(params, dtrain, evals=[(dval, "val")], early_stopping_rounds=30)
model.save_model("artifacts/ranker_model.json")
```

**Hold-out eval:** use 20% of teacher-labeled candidates as validation set.
Compute NDCG@10 locally. Run ablations (no behavioral signals, no anti-persona, etc.) —
this directly becomes your Stage 4 methodology and Stage 5 interview material.
