"""
B5 — Streamlit demo app (HF Spaces or local).
Upload up to 100 candidates as JSON, get a ranked CSV back.

Run:  streamlit run sandbox/app.py
"""

import csv
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import streamlit as st

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

st.set_page_config(page_title="MochiRank", layout="wide")
st.title("MochiRank — Senior AI Engineer Candidate Ranker")
st.caption("Redrob Track 1 · Rank up to 100 candidates against the Founding Team JD")

# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("About")
    st.markdown(
        "This demo runs the full MochiRank pipeline:\n"
        "1. Honeypot / consistency filter\n"
        "2. Feature engineering\n"
        "3. XGBoost LambdaMART scoring\n\n"
        "Accepts the same JSON schema as `dataset/sample_candidates.json`.\n\n"
        "For the full 100 K submission run:\n"
        "```\npython rank.py --candidates candidates.jsonl --out submission.csv\n```"
    )

# ------------------------------------------------------------------ #
# File upload
# ------------------------------------------------------------------ #
uploaded = st.file_uploader(
    "Upload candidates JSON (list of candidate objects, ≤ 100)",
    type=["json"],
)

if uploaded is not None:
    try:
        candidates = json.load(uploaded)
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")
        st.stop()

    if not isinstance(candidates, list):
        st.error("Expected a JSON array at the top level.")
        st.stop()

    if len(candidates) > 100:
        st.warning(f"Truncated to 100 candidates (uploaded {len(candidates)}).")
        candidates = candidates[:100]

    st.success(f"Loaded {len(candidates)} candidates.")

    if st.button("Rank Candidates", type="primary"):
        with st.spinner("Running pipeline…"):
            try:
                import xgboost as xgb
                from src.consistency_checks import check_consistency
                from src.feature_engineering import (
                    FEATURE_NAMES,
                    compute_features,
                    load_precomputed,
                )

                precomputed = load_precomputed(ARTIFACTS)
                model = xgb.Booster()
                model.load_model(ARTIFACTS / "ranker_model.json")

                rows = []
                for c in candidates:
                    is_hp, n_v, reasons = check_consistency(c)
                    feats = compute_features(c, precomputed, n_v, is_hp)
                    rows.append({
                        "candidate_id": c["candidate_id"],
                        "feats":        feats,
                        "is_honeypot":  is_hp,
                    })

                X = np.array([r["feats"] for r in rows], dtype=np.float32)
                dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
                scores = model.predict(dmat)

                results = []
                for row, score in zip(rows, scores):
                    if not row["is_honeypot"]:
                        results.append({
                            "candidate_id": row["candidate_id"],
                            "score": float(score),
                        })

                results.sort(key=lambda x: -x["score"])
                for rank, r in enumerate(results[:100], 1):
                    r["rank"] = rank

                # CSV download
                buf = io.StringIO()
                writer = csv.DictWriter(
                    buf, fieldnames=["candidate_id", "rank", "score"])
                writer.writeheader()
                writer.writerows(results[:100])

                st.download_button(
                    label="⬇ Download ranked_candidates.csv",
                    data=buf.getvalue(),
                    file_name="ranked_candidates.csv",
                    mime="text/csv",
                )

                # Preview table
                st.dataframe(
                    [
                        {
                            "rank": r["rank"],
                            "candidate_id": r["candidate_id"],
                            "score": f"{r['score']:.4f}",
                        }
                        for r in results[:100]
                    ],
                    use_container_width=True,
                    height=600,
                )

            except FileNotFoundError as e:
                st.error(
                    f"Artifact not found: {e}\n\n"
                    "Run the offline pipeline first to generate `artifacts/`."
                )
            except Exception:
                st.error("Ranking failed.")
                st.code(traceback.format_exc())
