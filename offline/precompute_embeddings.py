"""
B1 — Precompute candidate embeddings with bge-small-en-v1.5.

Outputs:
  artifacts/candidate_embeddings[_sample].npy  — float16, one row per chunk
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
    from sentence_transformers import SentenceTransformer
    from src.utils import load_candidates_json, stream_candidates

    ARTIFACTS.mkdir(exist_ok=True)

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    dim = model.get_sentence_embedding_dimension()
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

    print(f"Total chunks: {len(all_chunks)}")

    embeddings = model.encode(
        all_chunks,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float16)

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
