#!/usr/bin/env python3
"""
rank.py — Redrob Track 1 candidate ranking entry point.

Usage:
    python rank.py --candidates ./dataset/candidates.jsonl --out ./submission.csv

Constraints: ≤5 min wall-clock, ≤16 GB RAM, CPU only, zero network calls.
All heavy computation is precomputed into artifacts/.
"""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.consistency_checks import check_consistency
from src.feature_engineering import (
    FEATURE_NAMES,
    compute_features,
    compute_features_dict,
    load_precomputed,
)
from src.reasoning_generator import generate_reasoning
from src.retrieval import bm25_retrieve, dense_retrieve, reciprocal_rank_fusion
from src.reranker import rerank_top_n

ARTIFACTS = Path("artifacts")

NON_TECH_TITLES = (
    "marketing", "sales", "accountant", "account manager",
    "operations manager", "customer support", "hr ", "human resource",
    "finance", "supply chain", "civil engineer", "mechanical engineer",
    "electrical engineer", "procurement", "recruiter",
)

CV_ONLY_TERMS = (
    "computer vision", "object detection", "image segmentation",
    "speech recognition", "text to speech", "robotics", "ros ",
)

NLP_IR_TERMS = (
    "nlp", "retrieval", "ranking", "search", "recommendation",
    "text", "language model", "embedding", "information retrieval",
    "llm", "transformer",
)


def apply_jd_disqualifiers(candidate: dict, features: dict) -> tuple:
    """
    Hard gates derived from the JD. Returns (disqualified: bool, reason: str).
    Applied to finalists only — not to the 100K retrieval stage.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    sig = candidate.get("redrob_signals", {})

    current_title = profile.get("current_title", "").lower()
    if any(t in current_title for t in NON_TECH_TITLES):
        return True, f"non-technical current title: {profile.get('current_title','')}"

    if career and features.get("ever_at_it_services_only", 0) > 0.5:
        return True, "entire career at IT services/consulting, no product company"

    if profile.get("years_of_experience", 0) < 2.0:
        return True, f"insufficient experience: {profile.get('years_of_experience',0)} yrs"

    if profile.get("country") != "India" and not sig.get("willing_to_relocate"):
        return True, "outside India, not willing to relocate"

    all_text = " ".join([
        profile.get("summary", ""),
        profile.get("headline", ""),
        *[j.get("description", "") for j in career],
    ]).lower()

    has_cv_only = any(t in all_text for t in CV_ONLY_TERMS)
    has_nlp_ir  = any(t in all_text for t in NLP_IR_TERMS)
    if has_cv_only and not has_nlp_ir:
        return True, "CV/speech/robotics domain without NLP/IR exposure"

    return False, ""


def main(candidates_path: str, output_path: str) -> None:
    t0 = time.time()

    def tick(msg: str) -> None:
        print(f"[{time.time() - t0:.1f}s] {msg}", flush=True)

    # ------------------------------------------------------------------ #
    # Load candidates
    # ------------------------------------------------------------------ #
    tick("Loading candidates…")
    candidates: dict = {}
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                candidates[c["candidate_id"]] = c
    tick(f"Loaded {len(candidates)} candidates")

    # ------------------------------------------------------------------ #
    # Load artifacts
    # ------------------------------------------------------------------ #
    tick("Loading artifacts…")
    precomputed = load_precomputed(ARTIFACTS)

    model = xgb.Booster()
    model.load_model(ARTIFACTS / "ranker_model.json")

    with open(ARTIFACTS / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    jd_text = ""
    hyp_path = ARTIFACTS / "hypothetical_resumes.json"
    if hyp_path.exists():
        with open(hyp_path, encoding="utf-8") as f:
            jd_text = json.load(f).get("jd_text", "")

    tick("Artifacts loaded")

    # ------------------------------------------------------------------ #
    # Stage A: consistency checks on all 100K candidates
    # ------------------------------------------------------------------ #
    tick("Stage A: consistency + honeypot detection…")
    honeypot_ids: set = set()
    violation_counts: dict = {}
    is_honeypot_map: dict = {}
    for cid, c in candidates.items():
        is_hp, n_v, _ = check_consistency(c)
        violation_counts[cid] = n_v
        is_honeypot_map[cid] = is_hp
        if is_hp:
            honeypot_ids.add(cid)
    tick(f"Stage A done: {len(honeypot_ids)} honeypots detected")

    # ------------------------------------------------------------------ #
    # Stage B: hybrid retrieval → top 2000
    # ------------------------------------------------------------------ #
    tick("Stage B: hybrid retrieval…")
    all_ids = list(candidates.keys())
    bm25_ranking = bm25_retrieve(bm25_data, all_ids, top_n=5000)
    dense_ranking = dense_retrieve(precomputed, all_ids, top_n=5000)
    rrf_scores = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
    top_2000_ids = sorted(rrf_scores, key=lambda c: -rrf_scores[c])[:2000]
    tick(f"Stage B done: {len(top_2000_ids)} candidates retrieved")

    # ------------------------------------------------------------------ #
    # Stage C: feature matrix on top-2000
    # ------------------------------------------------------------------ #
    tick("Stage C: feature engineering on top 2000…")
    feature_rows = []
    cid_order = []
    for cid in top_2000_ids:
        c = candidates[cid]
        feats = compute_features(
            c, precomputed,
            violation_counts.get(cid, 0),
            is_honeypot_map.get(cid, False),
        )
        feature_rows.append(feats)
        cid_order.append(cid)
    X = np.array(feature_rows, dtype=np.float32)
    tick(f"Stage C done: shape {X.shape}")

    # ------------------------------------------------------------------ #
    # Stage D: XGBoost inference
    # ------------------------------------------------------------------ #
    tick("Stage D: XGBoost scoring…")
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    scores = model.predict(dmat)
    ranked_ids = [cid_order[i] for i in np.argsort(-scores)]
    tick("Stage D done")

    # ------------------------------------------------------------------ #
    # Stage E: cross-encoder re-rank on top 200
    # ------------------------------------------------------------------ #
    tick("Stage E: cross-encoder re-rank on top 200…")
    top_200_candidates = [candidates[cid] for cid in ranked_ids[:200] if cid in candidates]
    reranked = rerank_top_n(top_200_candidates, jd_text, n=200)
    reranked_ids = [cid for cid, _ in reranked]
    reranked_scores_map = {cid: score for cid, score in reranked}
    # Append the rest (201+) after reranked
    reranked_tail = [cid for cid in ranked_ids[200:]]
    all_ranked_ids = reranked_ids + reranked_tail
    tick("Stage E done")

    # ------------------------------------------------------------------ #
    # Stage F: hard gates → final top-100
    # ------------------------------------------------------------------ #
    tick("Stage F: hard gates…")
    final_100: list = []
    seen_skipped: set = set()

    for cid in all_ranked_ids:
        if len(final_100) >= 100:
            break
        if cid in honeypot_ids:
            seen_skipped.add(cid)
            continue
        c = candidates[cid]
        feat_dict = compute_features_dict(c, precomputed, violation_counts.get(cid, 0))
        disqualified, reason = apply_jd_disqualifiers(c, feat_dict)
        if disqualified:
            seen_skipped.add(cid)
            continue
        final_100.append(cid)

    tick(f"Stage F done: {len(final_100)} finalists, {len(seen_skipped)} filtered")

    # ------------------------------------------------------------------ #
    # Stage G: SHAP reasoning
    # ------------------------------------------------------------------ #
    tick("Stage G: SHAP reasoning…")
    try:
        import shap as _shap
        explainer = _shap.TreeExplainer(model)
        shap_values = explainer.shap_values(dmat)
    except Exception as e:
        tick(f"SHAP unavailable ({e}), using stub reasoning")
        shap_values = None

    # ------------------------------------------------------------------ #
    # Write output CSV
    # ------------------------------------------------------------------ #
    output_rows = []
    for rank_pos, cid in enumerate(final_100, start=1):
        c = candidates[cid]
        # Score: cross-encoder if available, else XGBoost
        if cid in reranked_scores_map:
            score = reranked_scores_map[cid]
        else:
            idx = cid_order.index(cid) if cid in cid_order else -1
            score = float(scores[idx]) if idx >= 0 else 0.0

        # Reasoning
        if shap_values is not None and cid in cid_order:
            idx = cid_order.index(cid)
            reasoning = generate_reasoning(
                cid, c, shap_values[idx], FEATURE_NAMES, rank_pos
            )
        else:
            title = c["profile"].get("current_title", "professional")
            yoe = c["profile"].get("years_of_experience", 0)
            reasoning = f"{yoe}yr {title}; ranked {rank_pos} by model score."

        output_rows.append({
            "candidate_id": cid,
            "rank": rank_pos,
            "score": round(float(score), 6),
            "reasoning": reasoning,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    elapsed = time.time() - t0
    tick(f"Done. Written {len(output_rows)} rows to {output_path}")
    print(f"Total wall-clock: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank 100K candidates for Redrob Track 1")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()
    main(args.candidates, args.out)
