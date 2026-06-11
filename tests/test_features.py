"""
Unit tests for feature_engineering.py.
Runs on 50 sample candidates with no artifacts (stubs for semantic/BM25).
Checks: correct vector length, no NaN/Inf, value ranges, spot-checks.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import (
    FEATURE_NAMES,
    compute_features,
    compute_features_dict,
    load_precomputed,
)
from src.consistency_checks import check_consistency

SAMPLE_PATH = Path(__file__).parent.parent / "dataset" / "sample_candidates.json"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def main():
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        candidates = json.load(f)

    # Load precomputed (empty — no artifacts yet)
    precomputed = load_precomputed(ARTIFACTS_DIR) if ARTIFACTS_DIR.exists() else {"feature_names": FEATURE_NAMES}

    errors = []
    all_vecs = []

    for c in candidates:
        cid = c["candidate_id"]
        is_hp, n_viol, _ = check_consistency(c)

        try:
            vec = compute_features(c, precomputed, violation_count=n_viol, is_honeypot=is_hp)
        except Exception as e:
            errors.append(f"[{cid}] compute_features raised: {e}")
            continue

        # Length check
        if len(vec) != len(FEATURE_NAMES):
            errors.append(f"[{cid}] vector length {len(vec)} != {len(FEATURE_NAMES)}")

        # NaN / Inf check
        if not np.all(np.isfinite(vec)):
            bad = [FEATURE_NAMES[i] for i, v in enumerate(vec) if not np.isfinite(v)]
            errors.append(f"[{cid}] non-finite values in: {bad}")

        # Range checks on known-bounded features
        fdict = dict(zip(FEATURE_NAMES, vec.tolist()))
        range_checks = {
            "skill_evidence_ratio":     (0.0, 1.0),
            "product_company_ratio":    (0.0, 1.0),
            "notice_penalty":           (0.0, 1.0),
            "recency_score":            (0.0, 1.0),
            "behavioral_composite":     (0.0, 1.1),
            "profile_completeness":     (0.0, 1.0),
            "recruiter_response_rate":  (0.0, 1.0),
            "interview_completion_rate":(0.0, 1.0),
            "work_mode_match":          (0.5, 1.0),
            "best_edu_tier":            (1.0, 4.0),
            "is_honeypot":              (0.0, 1.0),
        }
        for feat, (lo, hi) in range_checks.items():
            v = fdict[feat]
            if not (lo - 1e-6 <= v <= hi + 1e-6):
                errors.append(f"[{cid}] {feat}={v:.4f} out of [{lo}, {hi}]")

        all_vecs.append(vec)

    # Summary stats
    mat = np.array(all_vecs)
    print(f"\n{'='*65}")
    print(f"Feature Engineering Test — {len(candidates)} candidates, {len(FEATURE_NAMES)} features")
    print(f"{'='*65}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nAll checks passed.")

    print(f"\nFeature stats (non-stub rows only):")
    print(f"  {'Feature':<35} {'min':>7} {'mean':>7} {'max':>7} {'nonzero%':>9}")
    print(f"  {'-'*65}")
    for i, name in enumerate(FEATURE_NAMES):
        col = mat[:, i]
        nz_pct = 100.0 * np.mean(col != 0)
        print(f"  {name:<35} {col.min():>7.3f} {col.mean():>7.3f} {col.max():>7.3f} {nz_pct:>8.0f}%")
    print()


if __name__ == "__main__":
    main()
