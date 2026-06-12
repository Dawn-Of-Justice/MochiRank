"""
B1 — Precompute candidate embeddings with model2vec (potion-base-8M).

Distilled from BAAI/bge-small-en-v1.5, produces 256d normalized static embeddings.
~23K chunks/sec on CPU vs ~14 chunks/sec for bge-small — 1600x faster.

Outputs:
  artifacts/candidate_embeddings[_sample].npy  — float16, one row per chunk (256d)
  artifacts/candidate_ids[_sample].json        — candidate_id per row (same order)

Usage:
  python -m offline.precompute_embeddings           # full 100K run
  python -m offline.precompute_embeddings --sample  # smoke-test on 50-sample
"""

import argparse
import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path("artifacts")
DATASET = Path("dataset")
MODEL_ID = "minishlab/potion-base-8M"


def candidate_to_chunks(c: dict) -> list[str]:
    chunks = []
    for job in c.get("career_history", []):
        if job.get("description"):
            chunks.append(f"{job['title']} at {job['company']}: {job['description']}")
    if c["profile"].get("summary"):
        chunks.append(c["profile"]["summary"])
    if c["profile"].get("headline"):
        chunks.append(c["profile"]["headline"])
    return chunks or [c["profile"].get("current_title", "")]


def main(sample_mode: bool = False) -> None:
    from model2vec import StaticModel
    from src.utils import load_candidates_json, stream_candidates

    ARTIFACTS.mkdir(exist_ok=True)

    print(f"Loading model2vec: {MODEL_ID}")
    model = StaticModel.from_pretrained(MODEL_ID)
    dim = model.dim
    print(f"Model loaded: {dim}d embeddings")

    if sample_mode:
        candidates = load_candidates_json(DATASET / "sample_candidates.json")
        print(f"Sample mode: {len(candidates)} candidates")
    else:
        candidates = list(stream_candidates(DATASET / "candidates.jsonl"))
        print(f"Full dataset: {len(candidates)} candidates")

    all_chunks: list[str] = []
    all_ids: list[str] = []
    for c in candidates:
        for chunk in candidate_to_chunks(c):
            all_chunks.append(chunk)
            all_ids.append(c["candidate_id"])

    print(f"Total chunks: {len(all_chunks)} — encoding...")

    embeddings = model.encode(all_chunks, show_progress_bar=True).astype(np.float16)

    # model2vec outputs L2-normalized vectors by default
    print(f"Embedding shape: {embeddings.shape}")

    suffix = "_sample" if sample_mode else ""
    emb_path = ARTIFACTS / f"candidate_embeddings{suffix}.npy"
    ids_path = ARTIFACTS / f"candidate_ids{suffix}.json"

    np.save(emb_path, embeddings)
    with open(ids_path, "w") as f:
        json.dump(all_ids, f)

    print(f"Saved {emb_path}  ({embeddings.nbytes / 1e6:.1f} MB)")
    print(f"Saved {ids_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Use 50-sample for smoke-test")
    args = parser.parse_args()
    main(args.sample)
