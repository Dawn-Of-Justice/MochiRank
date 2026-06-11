"""
Stage E — Optional cross-encoder re-rank of top-200.

Uses FlashRank (no torch, ~34MB) if available, otherwise passes through.
Model must be pre-cached during offline setup — zero network calls at rank time.
"""

from __future__ import annotations


def rerank_top_n(
    top_candidates: list,
    jd_text: str,
    n: int = 200,
) -> list:
    """
    Returns [(candidate_id, score), ...] sorted descending by cross-encoder score.
    Falls back to (candidate_id, rank_index) ordering if no reranker available.
    """
    candidates = top_candidates[:n]

    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="artifacts/flashrank")
        passages = []
        for c in candidates:
            text = " ".join(
                [c["profile"].get("summary", ""), c["profile"].get("headline", "")]
                + [j.get("description", "")[:200] for j in c.get("career_history", [])[:3]]
            )[:512]
            passages.append({"id": c["candidate_id"], "text": text})

        request = RerankRequest(query=jd_text[:512], passages=passages)
        results = ranker.rerank(request)
        scored = [(r["id"], float(r["score"])) for r in results]
        return sorted(scored, key=lambda x: -x[1])

    except Exception:
        # No reranker available — preserve XGBoost order
        return [(c["candidate_id"], float(n - i)) for i, c in enumerate(candidates)]
