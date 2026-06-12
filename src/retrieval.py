"""
Stage B — Hybrid retrieval: BM25 + dense embeddings + RRF fusion.

All operations run on precomputed artifacts — no model inference at rank time.
"""

import re

import numpy as np

JD_QUERY_TOKENS = re.findall(
    r"[a-z0-9]+",
    (
        "production embeddings retrieval vector database hybrid search ndcg evaluation "
        "llm fine-tuning lora python senior ai engineer ranking recommendation nlp "
        "information retrieval startup founding team sentence transformers faiss xgboost "
        "learning to rank dense sparse reranker"
    ).lower(),
)


def bm25_retrieve(bm25_data: dict, candidate_ids: list, top_n: int = 5000) -> list:
    """
    Score all candidates against the JD query and return top_n sorted descending.
    bm25_data: {"bm25": BM25Okapi, "candidate_ids": list[str]}
    """
    bm25 = bm25_data["bm25"]
    corpus_ids: list = bm25_data["candidate_ids"]

    all_scores = bm25.get_scores(JD_QUERY_TOKENS)
    cid_score = {cid: float(all_scores[i]) for i, cid in enumerate(corpus_ids)}

    requested = set(candidate_ids)
    ranked = sorted(
        (cid for cid in candidate_ids if cid in requested),
        key=lambda c: -cid_score.get(c, 0.0),
    )
    return ranked[:top_n]


def dense_retrieve(precomputed: dict, candidate_ids: list, top_n: int = 5000) -> list:
    """
    Multi-query dense retrieval using precomputed embeddings.
    Scores candidates against all ideal JD vectors, takes max similarity.
    Fully vectorized — no per-candidate Python loop.
    """
    embeddings = precomputed.get("candidate_embeddings")
    jd_vecs = precomputed.get("jd_query_vectors")
    cid_to_rows: dict = precomputed.get("cid_to_rows", {})

    if embeddings is None or jd_vecs is None or not cid_to_rows:
        return candidate_ids[:top_n]

    hyp_data = precomputed.get("hypothetical_resumes", {})
    resumes = hyp_data.get("resumes", []) if isinstance(hyp_data, dict) else []
    ideal_idxs = [i for i, r in enumerate(resumes) if r.get("is_positive", False)]
    if not ideal_idxs:
        ideal_idxs = list(range(len(jd_vecs)))
    ideal_vecs = jd_vecs[ideal_idxs].astype(np.float32)
    norms = np.linalg.norm(ideal_vecs, axis=1, keepdims=True)
    ideal_vecs_n = ideal_vecs / (norms + 1e-9)  # (n_ideal, dim)

    # Mean-pool chunk embeddings per candidate (vectorized)
    unique_cids = [c for c in candidate_ids if c in cid_to_rows]
    if not unique_cids:
        return candidate_ids[:top_n]

    pooled = np.array(
        [embeddings[cid_to_rows[cid]].mean(axis=0) for cid in unique_cids],
        dtype=np.float32,
    )  # (N, dim)
    p_norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled_n = pooled / (p_norms + 1e-9)

    sims = pooled_n @ ideal_vecs_n.T  # (N, n_ideal)
    max_sims = sims.max(axis=1)       # (N,)

    cid_score = dict(zip(unique_cids, max_sims.tolist()))
    ranked = sorted(unique_cids, key=lambda c: -cid_score.get(c, 0.0))
    return ranked[:top_n]


def reciprocal_rank_fusion(rankings: list, k: int = 60) -> dict:
    """
    RRF fusion over multiple ranked lists. k=60 (Cormack et al. 2009).
    Returns {candidate_id: rrf_score}, higher = better.
    """
    scores: dict = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores
