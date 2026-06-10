# Build Plan — Two-Person Workstreams

> Who builds what, in what order, and where the two halves meet.
> Design background: [architecture/spec.md](../architecture/spec.md).

The project splits along the runtime/offline boundary that already exists in the architecture.
The two workstreams only touch through **artifact files** and **one function signature**
(see [Interface contracts](#interface-contracts)) — so each person can build and test
independently against `dataset/sample_candidates.json`.

| | Workstream A — Runtime Ranker | Workstream B — Offline ML Pipeline |
|---|---|---|
| **Owner** | _(Surya)_ | _(Salo)_ |
| **Owns dirs** | `src/`, `rank.py`, `tests/` | `offline/`, `artifacts/`, `sandbox/` |
| **Reads** | [architecture/pipeline.md](../architecture/pipeline.md), [architecture/features.md](../architecture/features.md) | [docs/offline-phase.md](offline-phase.md) |
| **Needs from other** | artifacts (embeddings, query vectors, BM25 index, model) | feature matrix builder (`compute_features`) + labels for training |
| **Skills focus** | pure-Python logic, perf budget (5 min / 16 GB), SHAP | embeddings, Claude API prompting, XGBoost training |

## Workstream A — Runtime Ranker (`src/` + `rank.py`)

Everything that runs inside the 5-minute window. No torch, no anthropic, no network.

### A1 — Foundation (Day 1)
1. `src/consistency_checks.py` — honeypot + impossible-profile detection ([pipeline.md §Consistency](../architecture/pipeline.md))
2. `src/utils.py` — IT-services list, skill lists, JSONL streaming loader, date parsing
3. Sanity-check the consistency engine against all 50 sample candidates; eyeball every flagged profile

### A2 — Features (Day 2–3)
4. `src/feature_engineering.py` — all ~48 features per [features.md](../architecture/features.md), unit test each group
   - Non-semantic features (groups 6.3–6.8) need **no artifacts** — build these first
   - Semantic features (6.1) + BM25 features (6.2) stub against fake vectors until **Sync 1** delivers real artifacts
5. `src/retrieval.py` — dense retrieval + RRF fusion (BM25 loading once B ships the index)

### A3 — Gates + Reasoning (Day 4–5)
6. JD disqualifier engine (in `src/feature_engineering.py` or `src/utils.py`) per [pipeline.md §Disqualifiers](../architecture/pipeline.md)
7. `src/reasoning_generator.py` — SHAP → natural language (needs `ranker_model.json` from **Sync 2**; develop against a dummy XGBoost model trained on random labels until then)
8. `src/reranker.py` — optional ONNX cross-encoder (skippable if the time budget gets tight)

### A4 — Integration (Day 5–6, joint)
9. `rank.py` — wire Stages A→G per [spec.md](../architecture/spec.md)
10. Run on 50-sample → `dataset/validate_submission.py` must pass
11. Full 100K run: measure wall-clock + peak RAM; tune if >4 min (drop cross-encoder first)

## Workstream B — Offline ML Pipeline (`offline/` + `artifacts/`)

Everything with no time limit: internet, GPU, and Claude API all allowed.

### B1 — Embeddings (Day 1–2)
1. `offline/precompute_embeddings.py` — chunk per role, embed 100K with bge-small, save
   `candidate_embeddings.npy` (float16) + `candidate_ids.json` ([offline-phase.md §5.1](offline-phase.md))
2. `offline/build_bm25_index.py` — build + pickle `bm25_index.pkl`
3. Smoke-test both on the 50-sample first; then kick off the ~10-min 100K embedding run

### B2 — Hypothetical Resumes (Day 2)
4. `offline/generate_hypothetical.py` — 5 ideal + 5 anti-persona resumes via Claude
   ([offline-phase.md §5.2](offline-phase.md)); **manually review** — they must read human, not AI-obvious
5. Embed them → `jd_query_vectors.npy` + `hypothetical_resumes.json`
6. → **Sync 1**: hand A the four artifacts

### B3 — Teacher Labels (Day 3–4)
7. `offline/stratified_sampler.py` — 2,500 candidates across the strata table ([offline-phase.md §5.3](offline-phase.md))
   (the `honeypot_flagged` stratum needs A's `consistency_checks.py` from A1 — import it, don't rewrite it)
8. `offline/teacher_label.py` — Claude labeling with rubric anchoring + anchored batching;
   run the self-consistency check (double-label 100, Pearson r ≥ 0.85) → `data/teacher_labels.csv`

### B4 — Training (Day 4–5)
9. `offline/train_ranker.py` — build X via A's `compute_features` (**Sync 2** input), train
   XGBoost LambdaMART with monotonic constraints, validate NDCG@10 on 20% hold-out
10. Run ablations (table below) → save results in `docs/ablations.md`
11. → **Sync 2**: hand A `ranker_model.json` + the frozen `feature_names` list

### B5 — Submission Infra (Day 6–7)
12. `sandbox/app.py` — Streamlit app on HF Spaces (≤100 candidates in, CSV out)
13. Fill `submission_metadata.yaml` ([submission.md](submission.md) has the declarations)

## Interface contracts

Agree on these **before Day 2** and treat them as frozen — breaking one blocks the other person.

1. **Artifact files** (B produces, A consumes) — exact formats in [spec.md §Repository Structure](../architecture/spec.md):
   - `artifacts/candidate_embeddings.npy` — float16, one row per *chunk*
   - `artifacts/candidate_ids.json` — candidate_id per chunk row, same order
   - `artifacts/jd_query_vectors.npy` + `artifacts/hypothetical_resumes.json` — with positive/negative flags
   - `artifacts/bm25_index.pkl` — pickled `BM25Okapi` + the corpus's candidate_id order
   - `artifacts/ranker_model.json` — XGBoost model
2. **Feature function** (A produces, B consumes for training):
   `compute_features(candidate: dict, precomputed: dict) -> np.ndarray` plus a module-level
   `FEATURE_NAMES: list[str]`. **Column order is the contract** — monotonic constraints,
   SHAP labels, and the trained model all depend on it. Append-only after Sync 2.
3. **Consistency checker** (A produces, B consumes for the sampler):
   `check_consistency(c: dict) -> tuple[bool, int, list[str]]`

## Sync points

| When | What happens | Blocks |
|------|-------------|--------|
| **Day 1 end** | Agree contracts above; freeze `FEATURE_NAMES` draft | everything |
| **Sync 1** (~Day 3) | B → A: embeddings, query vectors, BM25 index. A swaps stubs for real artifacts, runs retrieval end-to-end on 50-sample | A2 semantic features |
| **Sync 2** (~Day 4–5) | A → B: `compute_features` + frozen feature order. B trains, then B → A: `ranker_model.json` + `feature_names` | B4 training, A3 reasoning |
| **Day 5–6** | Pair on `rank.py` integration + full 100K timing run | submission |
| **Day 7** | Joint: validate, fill metadata, submit (3 attempts max — save 2 for corrections) | — |

## Git workflow

- `git init` on `main`; commit the docs as the first commit (judges check for real history).
- Each person works on their own branch (`runtime/...`, `offline/...`) and merges to `main`
  at least daily — small frequent merges, since `src/` and `offline/` barely overlap.
- Never commit `dataset/candidates.jsonl` or the large artifacts (`.gitignore` handles it);
  share `candidate_embeddings.npy` / `bm25_index.pkl` via Drive or a USB stick.
- Commit after each numbered step above.

## Local Validation Approach

No live leaderboard. Build your own eval. (Owner: **A** builds the harness, **B** supplies labels.)

**Step 1:** Hand-label the 50 sample candidates into tiers 0–5 (use Claude to assist,
then review yourself). That's your mini ground truth.

**Step 2:** Run your ranker on the 50. Compute NDCG@10 against your labels.

**Step 3:** Ablations — remove one component at a time, measure NDCG@10 drop:
```
baseline:                  NDCG@10 = X.XX
- no behavioral composite: NDCG@10 = ?  (should drop — behavioral twins test)
- no anti-persona:         NDCG@10 = ?  (should drop — stuffer test)
- no consistency filter:   NDCG@10 = ?  (should drop — honeypot test)
- only BM25 retrieval:     NDCG@10 = ?  (should drop — plain-language Tier 5 test)
- only dense retrieval:    NDCG@10 = ?  (may drop on exact-term queries)
```

Save this table — it's your Stage 4 methodology summary and your Stage 5 interview script.
