# Pipeline Explained

**The problem:** 100,000 candidate profiles. 1 job description. Pick the best 100. In under 5 minutes, no internet, no AI calls at runtime.

---

## Stage A — Honeypot Detection

**What:** Flag profiles with impossible data (e.g. claims 13 years experience but earliest job started 2 years ago).

**Why:** The dataset has ~80 fake/trap profiles deliberately planted. If >10% of your top-100 are honeypots, you're disqualified. So we hard-exclude them before anything else.

**Signals used:**
- `duration_months` > actual date span + 12m → impossible tenure
- 3+ expert skills each with 0 months experience → keyword inflation
- YOE > (current year − earliest career start + 2) → impossible career span

---

## Stage B — Hybrid Retrieval → top 2,000

**What:** Two searches running in parallel, then merged:
- **BM25** (keyword matching) — finds candidates whose text literally contains relevant words like "LLM", "ranking", "Python"
- **Dense retrieval** (embeddings) — finds candidates who are *semantically* similar to the JD even if they use different words

Then **RRF (Reciprocal Rank Fusion)** combines both ranked lists into one.

**Why:** Running full feature engineering + ML on all 100K would be too slow. We cut to 2,000 plausible candidates first. BM25 alone misses good candidates who don't keyword-match. Embeddings alone miss exact skill matches. Together they're more robust.

**Model:** `minishlab/potion-base-8M` (model2vec) — 256d static embeddings, 23K chunks/sec on CPU.

---

## Stage C — Feature Engineering on top 2,000

**What:** For each of the 2,000 candidates, compute ~48 numbers: years of experience, skill match scores, recency, semantic similarity to JD, behavioral signals (response rate, profile completeness), etc.

**Why:** The ML model can't read text — it needs numbers. This converts each profile into a vector the model understands.

**Key feature groups:**
- Semantic similarity (dense cosine, BM25 score, anti-persona contrast)
- Experience signals (YOE, seniority, top-company flag)
- Skill signals (matched skills count, AI/ML depth)
- Behavioral signals (recency, response rate, profile completeness)
- Consistency signals (violation count, is_honeypot)

---

## Stage D — XGBoost Scoring

**What:** Feed the 48-feature vectors into a trained XGBoost model. It outputs a score (0–1) for each candidate.

**Why:** This is the core intelligence. The model was trained on 2,500 candidates labeled by Claude (0.0 = terrible fit → 1.0 = perfect fit). It learned which combinations of features actually predict a good senior AI engineer, including subtle things like "has NLP + systems experience + recent activity" being worth more than just "many skills listed."

**Training detail:** Pointwise regression (`reg:squarederror`) on a 6-point rubric. LambdaMART was tried but failed — single-query groups collapse pairwise gradients to ~0. Regression on the same labels achieves identical ranking behaviour with dense gradients.

---

## Stage E — Cross-encoder Reranking on top 200

**What:** Take the top 200 by XGBoost score and run a cross-encoder (FlashRank) that reads the actual text of each profile against the JD.

**Why:** XGBoost only sees numbers — it can't read a candidate's actual project descriptions. The cross-encoder reads the real text and can catch nuance. It's too slow to run on all 2,000, so we only apply it to the already-promising top 200.

**Fallback:** If FlashRank is unavailable, Stage E is a passthrough (XGBoost order preserved).

---

## Stage F — Hard Gates → final 100

**What:** Walk down the reranked list and skip anyone who is:
1. A detected honeypot
2. Non-technical current title (marketing, HR, sales, etc.)
3. Outside India and not willing to relocate
4. Pure CV/speech/robotics background with no NLP/IR exposure
5. Under 2 years experience
6. Entire career at IT services only (no product company)

Keep going until we have exactly 100.

**Why:** The model scores candidates on general quality, but these are absolute disqualifiers for *this specific job*. Keeping them as a post-filter means the model doesn't need to learn them — cleaner separation of concerns.

---

## Stage G — SHAP Reasoning

**What:** For each of the 100 finalists, use XGBoost's native SHAP (`pred_contribs=True`) to find which features pushed the score up or down. Generate a 1–2 sentence explanation from those.

**Why:** The submission requires a `reasoning` column. It must be *honest* — derived from what the model actually used, not made up. SHAP tells us "this candidate ranked #3 primarily because of high semantic similarity + 8yr YOE + strong behavioral signals."

**Implementation note:** Uses `model.predict(dmat, pred_contribs=True)` (XGBoost native) instead of the `shap` library, which has a version incompatibility with XGBoost 2.x (`base_score` stored as a string).

---

## The Big Picture

```
100K candidates
    ↓ [Stage A] remove ~65 honeypots
    ↓ [Stage B] keyword + semantic search → 2,000
    ↓ [Stage C] compute 48 features per candidate
    ↓ [Stage D] XGBoost score everything → ranked list
    ↓ [Stage E] deep text rerank → top 200 reordered
    ↓ [Stage F] hard JD filters → exactly 100
    ↓ [Stage G] SHAP → reasoning strings
submission.csv (100 rows)
```

**Why this order?** Each stage narrows the funnel cheaply before the next, more expensive stage. BM25 is microseconds. Embeddings are fast (model2vec). XGBoost is ~10ms for 2,000 rows. Cross-encoder only touches 200. Total wall-clock: ~18 seconds.

---

## Offline Phase (precomputed before rank.py runs)

Everything in `artifacts/` is built once and reused:

| Artifact | Built by | Used in |
|----------|----------|---------|
| `candidate_embeddings.npy` | `precompute_embeddings.py` | Stage B dense retrieval |
| `bm25_index.pkl` | `build_bm25_index.py` | Stage B BM25 retrieval |
| `ranker_model.json` | `train_ranker.py` | Stage D scoring |
| `feature_names.json` | `train_ranker.py` | Stage C/D feature contract |
| `hypothetical_resumes.json` | `generate_hypothetical.py` | Stage B query vector |

Teacher labels (`data/teacher_labels.csv`) were generated by Claude (2,500 candidates, 6-point rubric) and used to train the XGBoost model. Claude is not involved at ranking time.
