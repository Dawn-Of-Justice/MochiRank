"""
B1 — Build BM25Okapi index over all candidate career text.

Outputs:
  artifacts/bm25_index[_sample].pkl — pickled dict:
      {"bm25": BM25Okapi, "candidate_ids": list[str]}

Usage:
  python -m offline.build_bm25_index           # full 100K
  python -m offline.build_bm25_index --sample  # smoke-test
"""

import argparse
import pickle
from pathlib import Path

ARTIFACTS = Path("artifacts")
DATASET = Path("dataset")


def main(sample_mode: bool = False) -> None:
    from rank_bm25 import BM25Okapi
    # tokenize / candidate_to_bm25_text are shared with src.runtime_index so the
    # offline index and the rank-time index are built identically.
    from src.utils import (
        candidate_to_bm25_text,
        load_candidates_json,
        stream_candidates,
        tokenize,
    )

    ARTIFACTS.mkdir(exist_ok=True)

    if sample_mode:
        candidates = load_candidates_json(DATASET / "sample_candidates.json")
        print(f"Sample mode: {len(candidates)} candidates")
    else:
        candidates = list(stream_candidates(DATASET / "candidates.jsonl"))
        print(f"Full dataset: {len(candidates)} candidates")

    corpus_tokens: list[list[str]] = []
    corpus_ids: list[str] = []
    for c in candidates:
        corpus_tokens.append(tokenize(candidate_to_bm25_text(c)))
        corpus_ids.append(c["candidate_id"])

    print(f"Building BM25 index over {len(corpus_tokens)} docs...")
    index = BM25Okapi(corpus_tokens)

    suffix = "_sample" if sample_mode else ""
    out_path = ARTIFACTS / f"bm25_index{suffix}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": index, "candidate_ids": corpus_ids}, f)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    main(args.sample)
