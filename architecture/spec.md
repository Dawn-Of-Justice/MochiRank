# Architecture Spec

> System design for the Redrob Track 1 candidate ranker. Background: [docs/problem.md](../docs/problem.md).
> Detail docs: [features.md](features.md), [pipeline.md](pipeline.md), [docs/offline-phase.md](../docs/offline-phase.md).

## Architecture Overview

```
OFFLINE PHASE (days/weeks, internet on, AI tools allowed)
├── precompute_embeddings.py    — embed 100K candidates, save .npy
├── generate_hypothetical.py    — LLM generates ideal + anti-persona resumes (HyRe)
├── stratified_sampler.py       — pull ~2500 candidates for teacher labeling
├── teacher_label.py            — Claude API labels them 0.0–1.0 with rubric
└── train_ranker.py             — XGBoost LambdaMART on ~40 features → model.json

RANKING STEP (5-min window, CPU, no network)
├── rank.py                     — single entry point, produces submission CSV
│   ├── Stage 0: build dense (model2vec) + BM25 indexes from the UPLOADED candidates
│   │            (src/runtime_index.py — handles unseen judge datasets; ~30s/100K)
│   ├── Stage A: consistency engine (honeypot filter)
│   ├── Stage B: hybrid retrieval BM25 + dense → RRF → top ~2000
│   ├── Stage C: full feature matrix on top-2000
│   ├── Stage D: XGBoost inference → scores
│   ├── Stage E: ONNX int8 cross-encoder re-rank of top 200 (optional)
│   ├── Stage F: hard gates (honeypots + disqualifiers) on final top-100
│   └── Stage G: SHAP-derived reasoning → CSV
└── validate_submission.py      — already provided, run before upload
```

The key insight: **Claude is a labeling workforce, not the ranker.**
Its judgments get baked into XGBoost weights during training.
`rank.py` has zero LLM calls — just numpy, scipy, xgboost, and the torch-free
model2vec embedder.

> **Runtime indexing (handles unseen judge datasets).** The dense embeddings and
> BM25 index are NOT loaded from by-`candidate_id` artifacts — they are rebuilt at
> rank time from whatever candidates file is passed in (`src/runtime_index.py`).
> An earlier design precomputed them keyed by id, which silently dropped any
> candidate whose id wasn't in the original dataset. Only **dataset-independent**
> artifacts stay precomputed: the trained ranker, `jd_query_vectors`,
> `hypothetical_resumes`, and the vendored `potion-base-8M/` embedder. Chunking and
> tokenization live in `src/utils.py` and are shared by offline precompute and the
> runtime builder, so features match what the model was trained on.

## Repository Structure (target layout)

```
redrob-ranker/
├── README.md                          # setup + reproduce_command
├── submission_metadata.yaml           # filled from template
├── requirements.txt
│
├── data/                              # gitignored except small files
│   ├── candidates.jsonl.gz            # original dataset (not committed)
│   ├── candidates.jsonl               # extracted (not committed)
│   ├── sample_candidates.json         # 50-candidate sample (commit this)
│   ├── job_description.md             # JD (commit this)
│   └── teacher_labels.csv            # Claude-generated labels (commit this)
│
├── artifacts/                         # only dataset-independent files are used at rank time
│   ├── potion-base-8M/                # vendored model2vec embedder (256d), committed — loaded
│   │                                  #   locally at runtime, zero network
│   ├── hypothetical_resumes.json      # ideal + anti-persona resumes (committed)
│   ├── jd_query_vectors.npy           # multi-query JD embeddings, ideals+anti (committed)
│   ├── ranker_model.json              # trained XGBoost model (committed)
│   ├── candidate_embeddings.npy       # OFFLINE TRAINING ONLY (256d model2vec), gitignored —
│   ├── candidate_ids.json             #   rank.py rebuilds the dense index at runtime instead
│   └── bm25_index.pkl                 # OFFLINE TRAINING ONLY, gitignored — rebuilt at runtime
│
├── offline/
│   ├── precompute_embeddings.py
│   ├── generate_hypothetical.py
│   ├── stratified_sampler.py
│   ├── teacher_label.py
│   └── train_ranker.py
│
├── src/
│   ├── consistency_checks.py          # honeypot + impossible-profile detection
│   ├── feature_engineering.py         # all ~40 features
│   ├── retrieval.py                   # BM25 + dense + RRF
│   ├── reranker.py                    # optional ONNX cross-encoder
│   ├── reasoning_generator.py         # SHAP → natural language
│   └── utils.py
│
├── rank.py                            # THE entry point
├── validate_submission.py             # provided by Redrob
└── sandbox/                           # Streamlit or HF Spaces app
    └── app.py
```

> Note: the provided hackathon bundle (candidates.jsonl, JD, schema, sample data,
> validator) currently lives in `dataset/` at the repo root.

## rank.py — Entry Point

```python
#!/usr/bin/env python3
"""
rank.py — produces submission CSV from candidates.jsonl

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints: ≤5 min, ≤16GB RAM, CPU only, no network.
All heavy computation is in artifacts/ (precomputed).
"""

import argparse
import json
import csv
import time
import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path

from src.consistency_checks import check_consistency
from src.feature_engineering import compute_features, load_precomputed
from src.retrieval import bm25_retrieve, dense_retrieve, reciprocal_rank_fusion
from src.reranker import rerank_top_n
from src.reasoning_generator import generate_reasoning, explainer

ARTIFACTS = Path("artifacts")

def main(candidates_path: str, output_path: str):
    t0 = time.time()
    print(f"[0.0s] Loading candidates...")

    # Load
    candidates = {}
    with open(candidates_path) as f:
        for line in f:
            c = json.loads(line)
            candidates[c["candidate_id"]] = c
    print(f"[{time.time()-t0:.1f}s] Loaded {len(candidates)} candidates")

    # Load artifacts
    precomputed = load_precomputed(ARTIFACTS)
    model = xgb.Booster()
    model.load_model(ARTIFACTS / "ranker_model.json")
    bm25_index = pickle.load(open(ARTIFACTS / "bm25_index.pkl", "rb"))
    print(f"[{time.time()-t0:.1f}s] Artifacts loaded")

    # Stage A: consistency + honeypot flags (run on ALL candidates)
    honeypot_ids = set()
    consistency_violations = {}
    for cid, c in candidates.items():
        is_hp, n_violations, reasons = check_consistency(c)
        consistency_violations[cid] = n_violations
        if is_hp:
            honeypot_ids.add(cid)
    print(f"[{time.time()-t0:.1f}s] Honeypots flagged: {len(honeypot_ids)}")

    # Stage B: hybrid retrieval → top 2000
    bm25_ranking = bm25_retrieve(bm25_index, list(candidates.keys()))
    dense_ranking = dense_retrieve(precomputed, list(candidates.keys()))
    rrf_scores = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
    top_2000_ids = sorted(rrf_scores, key=lambda c: -rrf_scores[c])[:2000]
    print(f"[{time.time()-t0:.1f}s] Retrieval done: {len(top_2000_ids)} candidates")

    # Stage C: feature matrix on top-2000
    feature_rows = []
    cid_order = []
    for cid in top_2000_ids:
        c = candidates[cid]
        feats = compute_features(c, precomputed, consistency_violations.get(cid, 0))
        feature_rows.append(feats)
        cid_order.append(cid)
    X = np.array(feature_rows, dtype=np.float32)
    print(f"[{time.time()-t0:.1f}s] Features computed: shape {X.shape}")

    # Stage D: XGBoost inference
    dmat = xgb.DMatrix(X)
    scores = model.predict(dmat)
    ranked_ids = [cid_order[i] for i in np.argsort(-scores)]
    print(f"[{time.time()-t0:.1f}s] XGBoost scored")

    # Stage E: cross-encoder re-rank on top 200
    top_200 = [candidates[cid] for cid in ranked_ids[:200]]
    jd_text = open("data/job_description.md").read()
    reranked = rerank_top_n(top_200, jd_text, n=200)
    reranked_ids = [cid for cid, _ in reranked]
    reranked_scores = {cid: score for cid, score in reranked}
    print(f"[{time.time()-t0:.1f}s] Cross-encoder done")

    # Stage F: hard gates on final top-100
    final_100 = []
    skipped = []
    for cid in reranked_ids:
        c = candidates[cid]
        if cid in honeypot_ids:
            skipped.append((cid, "honeypot"))
            continue
        from src.feature_engineering import compute_features_dict
        feat_dict = compute_features_dict(c, precomputed)
        disqualified, reason = apply_jd_disqualifiers(c, feat_dict)
        if disqualified:
            skipped.append((cid, reason))
            continue
        final_100.append(cid)
        if len(final_100) == 100:
            break

    # If we filtered too many, backfill from ranked_ids beyond 200
    if len(final_100) < 100:
        for cid in ranked_ids[200:]:
            if cid not in honeypot_ids and cid not in [x[0] for x in skipped]:
                final_100.append(cid)
            if len(final_100) == 100:
                break

    print(f"[{time.time()-t0:.1f}s] Gates applied: {len(skipped)} filtered, {len(final_100)} final")

    # Stage G: SHAP reasoning
    from shap import TreeExplainer
    shap_values = explainer.shap_values(xgb.DMatrix(X[:200]))

    # Build output
    output_rows = []
    for rank_pos, cid in enumerate(final_100, start=1):
        c = candidates[cid]
        score = reranked_scores.get(cid, float(scores[cid_order.index(cid)]))
        cid_idx = cid_order.index(cid) if cid in cid_order else 0
        reasoning = generate_reasoning(cid, c, shap_values[cid_idx],
                                       feature_names=precomputed["feature_names"],
                                       rank=rank_pos)
        output_rows.append({
            "candidate_id": cid,
            "rank": rank_pos,
            "score": round(float(score), 6),
            "reasoning": reasoning,
        })

    # Write CSV (scores must be non-increasing)
    output_rows.sort(key=lambda x: x["rank"])
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[{time.time()-t0:.1f}s] Done. Written to {output_path}")
    print(f"Total wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.candidates, args.out)
```
