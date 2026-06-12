"""
B4 — Train XGBoost LambdaMART on teacher-labeled candidates.

Requires (Sync 2 — A must have shipped compute_features + FEATURE_NAMES):
  src/feature_engineering.py  — compute_features(), load_precomputed(), FEATURE_NAMES
  data/teacher_labels.csv     — from teacher_label.py
  artifacts/                  — embeddings, BM25 index, query vectors

Outputs:
  artifacts/ranker_model.json   — trained XGBoost model
  artifacts/feature_names.json  — frozen column order (contract for rank.py)
  docs/ablations.md             — ablation NDCG@10 table

Usage:
  python -m offline.train_ranker
  python -m offline.train_ranker --ablate no_behavioral
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path("artifacts")
DATASET   = Path("dataset")
DATA      = Path("data")
DOCS      = Path("docs")

_BEHAVIORAL_FEATURES = {
    "recency_score", "open_to_work", "recruiter_response_rate",
    "interview_completion_rate", "applications_30d", "profile_completeness",
    "saved_by_recruiters_30d", "verified_contact", "behavioral_composite",
}

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def load_labels(path: Path) -> dict[str, float]:
    labels: dict[str, float] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            labels[row["candidate_id"]] = float(row["score"])
    return labels


def build_feature_matrix(
    candidates: list[dict],
    precomputed: dict,
    zero_out: set[str] | None = None,
) -> np.ndarray:
    from src.consistency_checks import check_consistency
    from src.feature_engineering import FEATURE_NAMES, compute_features
    rows = []
    for c in candidates:
        is_hp, n_v, _ = check_consistency(c)
        rows.append(compute_features(c, precomputed, n_v, is_hp))
    X = np.array(rows, dtype=np.float32)
    if zero_out:
        for j, name in enumerate(FEATURE_NAMES):
            if name in zero_out:
                X[:, j] = 0.0
    return X


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute NDCG@k for a single ranking query."""
    order = np.argsort(-y_score)[:k]
    gains = 2 ** y_true[order] - 1
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains / discounts).sum())
    ideal = np.sort(y_true)[::-1][:k]
    ideal_gains = 2 ** ideal - 1
    ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = float((ideal_gains / ideal_discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0

# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    num_boost_round: int = 500,
) -> "xgb.Booster":
    import xgboost as xgb

    mono = tuple(
        1 if name in _BEHAVIORAL_FEATURES else 0
        for name in feature_names
    )

    # Pointwise regression: predict relevance score in [0,1].
    # LambdaMART (rank:ndcg) needs multiple query groups to compute
    # pairwise gradients; with a single JD query and one big group the
    # gradients collapse to ~0 and the model stalls at round 0.
    # Regression on the 6-point rubric achieves the same ranking objective
    # while giving XGBoost a dense, informative gradient signal.
    dtrain = xgb.DMatrix(X_train, label=y_train.astype(np.float32), feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val.astype(np.float32), feature_names=feature_names)

    params = {
        "objective":            "reg:squarederror",
        "eval_metric":          ["rmse"],
        "eta":                  0.05,
        "max_depth":            6,
        "min_child_weight":     5,
        "subsample":            0.8,
        "colsample_bytree":     0.8,
        "tree_method":          "hist",
        "monotone_constraints": str(mono).replace(" ", ""),
        "seed":                 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )
    return model

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(ablate: str | None = None) -> None:
    import xgboost as xgb
    from src.feature_engineering import FEATURE_NAMES, load_precomputed
    from src.utils import stream_candidates

    DOCS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    print("Loading labels...")
    labels = load_labels(DATA / "teacher_labels.csv")
    labeled_ids = set(labels.keys())
    print(f"  {len(labeled_ids)} labeled candidates")

    print("Loading candidate profiles...")
    candidates: list[dict] = []
    for c in stream_candidates(DATASET / "candidates.jsonl"):
        if c["candidate_id"] in labeled_ids:
            candidates.append(c)
        if len(candidates) == len(labeled_ids):
            break
    print(f"  Loaded {len(candidates)} profiles")

    print("Loading precomputed artifacts...")
    precomputed = load_precomputed(ARTIFACTS)

    # Ablation: zero out specific feature groups
    zero_out: set[str] | None = None
    if ablate == "no_behavioral":
        zero_out = _BEHAVIORAL_FEATURES
        print("Ablation: zeroing behavioral features")
    elif ablate == "no_anti_persona":
        zero_out = {"sim_anti_max", "semantic_contrastive"}
        print("Ablation: zeroing anti-persona features")
    elif ablate == "no_consistency":
        zero_out = {"consistency_violation_count"}
        print("Ablation: zeroing consistency features")
    elif ablate == "bm25_only":
        zero_out = {n for n in FEATURE_NAMES if n.startswith("sim_") or "dense" in n}
        print("Ablation: BM25 only (zeroing dense features)")
    elif ablate == "dense_only":
        zero_out = {"bm25_score", "bm25_rank_pct"}
        print("Ablation: dense only (zeroing BM25 features)")

    print("Building feature matrix...")
    X = build_feature_matrix(candidates, precomputed, zero_out)
    y = np.array([labels[c["candidate_id"]] for c in candidates], dtype=np.float32)
    print(f"  X shape: {X.shape}  y: [{y.min():.1f}, {y.max():.1f}]  grade dist: {dict(zip(*np.unique(np.round(y*5).astype(int), return_counts=True)))}")

    rng = np.random.default_rng(42)
    val_mask = rng.random(len(candidates)) < 0.2
    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val,   y_val   = X[val_mask],  y[val_mask]
    print(f"  Train: {len(y_train)}  Val: {len(y_val)}")

    print("Training LambdaMART...")
    model = train(X_train, y_train, X_val, y_val, FEATURE_NAMES)

    val_scores = model.predict(xgb.DMatrix(X_val, feature_names=FEATURE_NAMES))
    ndcg10 = ndcg_at_k(y_val, val_scores, k=10)
    ndcg50 = ndcg_at_k(y_val, val_scores, k=50)
    print(f"\nVal NDCG@10={ndcg10:.4f}  NDCG@50={ndcg50:.4f}")
    print(f"Best round: {model.best_iteration}")

    if ablate is None:
        model_path = ARTIFACTS / "ranker_model.json"
        model.save_model(model_path)
        print(f"Saved {model_path}")

        fn_path = ARTIFACTS / "feature_names.json"
        with open(fn_path, "w") as f:
            json.dump(FEATURE_NAMES, f)
        print(f"Saved {fn_path}")

        importance = model.get_score(importance_type="gain")
        top15 = sorted(importance.items(), key=lambda x: -x[1])[:15]
        print("\nTop 15 features by gain:")
        for feat, gain in top15:
            print(f"  {feat}: {gain:.2f}")

        # Seed ablations.md with baseline result
        _write_ablation_result("baseline", ndcg10, ndcg50)
    else:
        _write_ablation_result(ablate, ndcg10, ndcg50)


def _write_ablation_result(label: str, ndcg10: float, ndcg50: float) -> None:
    """Append or update a row in docs/ablations.md."""
    abl_path = DOCS / "ablations.md"
    if not abl_path.exists():
        abl_path.write_text(
            "# Ablation Results\n\n"
            "| Condition | NDCG@10 | NDCG@50 | Delta@10 |\n"
            "|-----------|---------|---------|----------|\n"
        )
    content = abl_path.read_text()
    row = f"| {label} | {ndcg10:.4f} | {ndcg50:.4f} | — |\n"
    abl_path.write_text(content + row)
    print(f"Updated {abl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablate",
        choices=["no_behavioral", "no_anti_persona", "no_consistency",
                 "bm25_only", "dense_only"],
        help="Run one ablation study",
    )
    args = parser.parse_args()
    main(args.ablate)
