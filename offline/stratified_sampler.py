"""
B3 — Stratified sampler: select ~2500 candidates for teacher labeling.

Requires (Sync 1 artifacts):
  artifacts/candidate_embeddings.npy
  artifacts/candidate_ids.json
  artifacts/bm25_index.pkl
  artifacts/jd_query_vectors.npy
  artifacts/hypothetical_resumes.json

Requires (from A1):
  src/consistency_checks.check_consistency

Output:
  data/sampled_candidates.json — list of {candidate_id, stratum}

Usage:
  python -m offline.stratified_sampler
"""

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARTIFACTS = Path("artifacts")
DATASET = Path("dataset")
DATA = Path("data")

# Strata targets — total ~2500
STRATA_TARGETS = {
    "top_retrieval_bm25":   200,
    "top_retrieval_dense":  200,
    "top_anti_persona_sim": 150,
    "title_match_strong":   200,
    "title_mismatch":       150,
    "consulting_only":      100,
    "honeypot_flagged":     100,
    "high_behavioral":      150,
    "low_behavioral":       150,
    "tier1_education":      100,
    "random":              1000,
}


# ------------------------------------------------------------------ #
# Artifact loading
# ------------------------------------------------------------------ #

def load_artifacts() -> dict:
    print("Loading artifacts...")
    embeddings = np.load(ARTIFACTS / "candidate_embeddings.npy").astype(np.float32)
    with open(ARTIFACTS / "candidate_ids.json") as f:
        chunk_ids: list[str] = json.load(f)
    with open(ARTIFACTS / "bm25_index.pkl", "rb") as f:
        bm25_data: dict = pickle.load(f)
    query_vectors = np.load(ARTIFACTS / "jd_query_vectors.npy").astype(np.float32)
    with open(ARTIFACTS / "hypothetical_resumes.json") as f:
        hyp_data: dict = json.load(f)
    return {
        "embeddings": embeddings,
        "chunk_ids": chunk_ids,
        "bm25_data": bm25_data,
        "query_vectors": query_vectors,
        "hyp_data": hyp_data,
    }


def pool_embeddings(embeddings: np.ndarray, chunk_ids: list[str]) -> dict[str, np.ndarray]:
    """Mean-pool per-chunk embeddings into one vector per candidate."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(chunk_ids):
        groups[cid].append(i)
    return {cid: embeddings[idxs].mean(axis=0) for cid, idxs in groups.items()}


def bm25_query_scores(bm25_data: dict, jd_text: str) -> dict[str, float]:
    tokens = re.findall(r"[a-z0-9]+", jd_text.lower())
    scores = bm25_data["bm25"].get_scores(tokens)
    return {cid: float(scores[i]) for i, cid in enumerate(bm25_data["candidate_ids"])}

# ------------------------------------------------------------------ #
# Score functions
# ------------------------------------------------------------------ #

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def dense_scores(
    cand_vecs: dict[str, np.ndarray],
    ideal_vecs: np.ndarray,
) -> dict[str, float]:
    """Max cosine similarity of each candidate to any ideal resume vector."""
    result = {}
    for cid, v in cand_vecs.items():
        result[cid] = max(_cosine(v, q) for q in ideal_vecs)
    return result


def anti_scores(
    cand_vecs: dict[str, np.ndarray],
    anti_vecs: np.ndarray,
) -> dict[str, float]:
    """Max cosine similarity of each candidate to any anti-persona vector."""
    result = {}
    for cid, v in cand_vecs.items():
        result[cid] = max(_cosine(v, q) for q in anti_vecs)
    return result

# ------------------------------------------------------------------ #
# Strata filters
# ------------------------------------------------------------------ #

def _behavioral_score(c: dict) -> float:
    sig = c.get("redrob_signals", {})
    return (sig.get("recruiter_response_rate", 0.0)
            + sig.get("interview_completion_rate", 0.0))


def build_strata_pools(
    candidates: list[dict],
    artifacts: dict,
) -> dict[str, list[str]]:
    from src.consistency_checks import check_consistency
    from src.utils import is_it_services_company

    embeddings = artifacts["embeddings"]
    chunk_ids = artifacts["chunk_ids"]
    bm25_data = artifacts["bm25_data"]
    query_vectors = artifacts["query_vectors"]
    hyp_data = artifacts["hyp_data"]

    resumes = hyp_data["resumes"]
    positive_vecs = query_vectors[
        [i for i, r in enumerate(resumes) if r.get("is_positive", False)]
    ]
    anti_vecs = query_vectors[
        [i for i, r in enumerate(resumes) if not r.get("is_positive", True)]
    ]

    print("Pooling embeddings...")
    cand_vecs = pool_embeddings(embeddings, chunk_ids)

    print("Computing BM25 scores...")
    bm25_sc = bm25_query_scores(bm25_data, hyp_data.get("jd_text", ""))

    print("Computing dense scores...")
    dense_sc = dense_scores(cand_vecs, positive_vecs)

    print("Computing anti-persona scores...")
    anti_sc = anti_scores(cand_vecs, anti_vecs)

    all_ids = [c["candidate_id"] for c in candidates]

    # Pre-sort lists (full set, not deduplicated yet)
    bm25_ranked = sorted(all_ids, key=lambda c: -bm25_sc.get(c, 0.0))
    dense_ranked = sorted(all_ids, key=lambda c: -dense_sc.get(c, 0.0))
    anti_ranked  = sorted(all_ids, key=lambda c: -anti_sc.get(c, 0.0))

    print("Classifying strata...")
    title_match: list[str] = []
    title_mismatch: list[str] = []
    consulting: list[str] = []
    honeypots: list[str] = []
    tier1: list[str] = []

    for c in candidates:
        cid = c["candidate_id"]
        title_lower = c["profile"]["current_title"].lower()
        if any(kw in title_lower for kw in ("engineer", "ml", "ai", "scientist", "research")):
            title_match.append(cid)
        elif dense_sc.get(cid, 0.0) > 0.3:
            title_mismatch.append(cid)
        if all(is_it_services_company(j["company"]) for j in c.get("career_history", [])):
            consulting.append(cid)
        if check_consistency(c)[0]:
            honeypots.append(cid)
        if any(e.get("tier") == "tier_1" for e in c.get("education", [])):
            tier1.append(cid)

    cid_to_cand = {c["candidate_id"]: c for c in candidates}
    behavioral_sorted = sorted(all_ids, key=lambda cid: -_behavioral_score(cid_to_cand[cid]))

    return {
        "top_retrieval_bm25":   bm25_ranked[:500],
        "top_retrieval_dense":  dense_ranked[:500],
        "top_anti_persona_sim": anti_ranked[:300],
        "title_match_strong":   title_match,
        "title_mismatch":       title_mismatch,
        "consulting_only":      consulting,
        "honeypot_flagged":     honeypots,
        "high_behavioral":      behavioral_sorted[:300],
        "low_behavioral":       behavioral_sorted[-300:],
        "tier1_education":      tier1,
        "random":               all_ids,
    }

# ------------------------------------------------------------------ #
# Sampling
# ------------------------------------------------------------------ #

def stratified_sample(
    strata_pools: dict[str, list[str]],
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()

    for stratum, target_n in STRATA_TARGETS.items():
        pool = strata_pools.get(stratum, [])
        eligible = [cid for cid in pool if cid not in seen]
        k = min(target_n, len(eligible))
        if k == 0:
            print(f"  {stratum}: 0 eligible (pool={len(pool)})")
            continue
        chosen = rng.choice(eligible, size=k, replace=False).tolist()
        for cid in chosen:
            selected.append((cid, stratum))
            seen.add(cid)
        print(f"  {stratum}: {k}")

    print(f"Total: {len(selected)}")
    return selected

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    from src.utils import stream_candidates

    DATA.mkdir(exist_ok=True)

    print("Loading all candidates (streaming)...")
    candidates = list(stream_candidates(DATASET / "candidates.jsonl"))
    print(f"Loaded {len(candidates)} candidates")

    artifacts = load_artifacts()

    strata_pools = build_strata_pools(candidates, artifacts)

    rng = np.random.default_rng(42)
    sample = stratified_sample(strata_pools, rng)

    out_path = DATA / "sampled_candidates.json"
    with open(out_path, "w") as f:
        json.dump(
            [{"candidate_id": cid, "stratum": stratum} for cid, stratum in sample],
            f, indent=2,
        )
    print(f"Saved {out_path}  ({len(sample)} candidates)")


if __name__ == "__main__":
    main()
