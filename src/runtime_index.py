"""
Runtime candidate-side index builder.

The dense embeddings and BM25 index are computed FROM THE UPLOADED candidates
file at rank time — they are NOT loaded from by-candidate_id artifacts. This is
what lets rank.py score an arbitrary judge-supplied dataset correctly, even when
the candidate_ids are unseen (or collide with the original dataset but carry
different content).

Why this is cheap enough to stay inside the 5-min / 16GB / CPU / zero-network
budget: the embedder is model2vec (potion-base-8M), a static torch-free model at
~19K chunks/s on CPU. 100K candidates ≈ 0.5M chunks ≈ ~26s to embed; BM25 build
≈ a few seconds. The model weights (~30MB) are vendored into
artifacts/potion-base-8M, so loading is local — no network call.

Only JD-side artifacts (the trained ranker, jd_query_vectors, hypothetical
resumes) remain precomputed, because those are dataset-independent.

Parity contract: chunking/tokenization come from src.utils, the SAME helpers the
offline precompute uses, and embeddings are stored float16 like the offline
artifact — so runtime features match what the XGBoost model was trained on.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from src.utils import candidate_to_bm25_text, candidate_to_chunks, tokenize

DEFAULT_MODEL_DIR = Path("artifacts") / "potion-base-8M"


def build_dense_index(candidates: Iterable[dict], model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    """
    Embed every candidate's chunks with the local model2vec model.

    Returns the same structure load_precomputed produces for the dense channel:
        candidate_embeddings : np.ndarray (n_chunks, dim) float16
        candidate_ids        : list[str]  (candidate_id per row, chunk order)
        cid_to_idx           : dict[str, int]        first row per candidate
        cid_to_rows          : dict[str, list[int]]  all rows per candidate
    """
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(str(model_dir))

    all_chunks: list[str] = []
    all_ids: list[str] = []
    for c in candidates:
        cid = c["candidate_id"]
        for chunk in candidate_to_chunks(c):
            all_chunks.append(chunk)
            all_ids.append(cid)

    if not all_chunks:
        dim = model.dim
        return {
            "candidate_embeddings": np.zeros((0, dim), dtype=np.float16),
            "candidate_ids": [],
            "cid_to_idx": {},
            "cid_to_rows": {},
        }

    embeddings = model.encode(all_chunks).astype(np.float16)

    cid_to_idx: dict[str, int] = {}
    cid_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(all_ids):
        if cid not in cid_to_idx:
            cid_to_idx[cid] = i
        cid_to_rows[cid].append(i)

    return {
        "candidate_embeddings": embeddings,
        "candidate_ids": all_ids,
        "cid_to_idx": cid_to_idx,
        "cid_to_rows": dict(cid_to_rows),
    }


def build_bm25_index(candidates: Iterable[dict]) -> dict:
    """
    Build a BM25Okapi index over the uploaded candidates.

    Returns:
        bm25_data : {"bm25": BM25Okapi, "candidate_ids": list[str]}  (for retrieval)
        bm25_cid_to_idx : dict[str, int]                             (for features)
    """
    from rank_bm25 import BM25Okapi

    corpus_tokens: list[list[str]] = []
    corpus_ids: list[str] = []
    for c in candidates:
        corpus_tokens.append(tokenize(candidate_to_bm25_text(c)))
        corpus_ids.append(c["candidate_id"])

    if not corpus_tokens:
        corpus_tokens = [[""]]  # BM25Okapi rejects an empty corpus

    index = BM25Okapi(corpus_tokens)
    return {
        "bm25_data": {"bm25": index, "candidate_ids": corpus_ids},
        "bm25_cid_to_idx": {cid: i for i, cid in enumerate(corpus_ids)},
    }


def attach_runtime_index(
    precomputed: dict,
    candidates: dict | Iterable[dict],
    model_dir: Path = DEFAULT_MODEL_DIR,
    tick: Callable[[str], None] | None = None,
) -> dict:
    """
    Compute candidate-side dense + BM25 indexes from `candidates` and inject them
    into `precomputed`, overwriting any stale by-id artifacts. Returns bm25_data
    for src.retrieval.bm25_retrieve.

    `candidates` may be the rank.py dict {cid: candidate} or any iterable of
    candidate dicts.
    """
    cand_iter = candidates.values() if isinstance(candidates, dict) else list(candidates)
    cand_list = list(cand_iter)

    def say(msg: str) -> None:
        if tick:
            tick(msg)

    say("Runtime index: embedding candidates…")
    dense = build_dense_index(cand_list, model_dir)
    precomputed["candidate_embeddings"] = dense["candidate_embeddings"]
    precomputed["candidate_ids"] = dense["candidate_ids"]
    precomputed["cid_to_idx"] = dense["cid_to_idx"]
    precomputed["cid_to_rows"] = dense["cid_to_rows"]
    say(f"Runtime index: dense done, {dense['candidate_embeddings'].shape[0]} chunks")

    say("Runtime index: building BM25…")
    bm25 = build_bm25_index(cand_list)
    precomputed["bm25_index"] = bm25["bm25_data"]["bm25"]
    precomputed["bm25_cid_to_idx"] = bm25["bm25_cid_to_idx"]
    precomputed.pop("bm25_all_scores", None)  # drop any cached scores from a prior dataset
    say("Runtime index: BM25 done")

    return bm25["bm25_data"]
