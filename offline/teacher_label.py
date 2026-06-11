"""
B4 — Claude teacher labeling on the stratified sample.

Technique: semantic rubric anchoring + anchored batching (Malt 2026).
Every batch of 10 real candidates has a known perfect-fit (position 1)
and known non-fit (position last) as calibration anchors.

Requires:
  data/sampled_candidates.json  (from B3 stratified_sampler)
  artifacts/hypothetical_resumes.json (for jd_text)
  dataset/candidates.jsonl
  ANTHROPIC_API_KEY env var

Output:
  data/teacher_labels.csv — columns: candidate_id, score, rationale, stratum

Self-consistency check: double-labels first 100 candidates; logs Pearson r.
If r < 0.85, the rubric needs tightening (switch to Sonnet or tighten anchors).

Usage:
  python -m offline.teacher_label
  python -m offline.teacher_label --no-verify   # skip consistency check
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

ARTIFACTS = Path("artifacts")
DATASET = Path("dataset")
DATA = Path("data")
OUTPUT_PATH = DATA / "teacher_labels.csv"

BATCH_SIZE = 10  # real candidates per batch (+ 2 anchors)

_RUBRIC = """
0.0 — No relevant skills or experience. Completely unable to perform the job.
0.2 — Minor relevance. Some adjacent skills but fundamentally wrong profile.
0.4 — Moderate match. Some relevant skills, significant gaps on core requirements.
0.6 — Good match. Mostly relevant, can perform with some ramp-up. Meets most requirements.
0.8 — Strong match. Highly relevant skills and experience. Ready to perform well.
1.0 — Perfect match. Skills and experience fully aligned. Expert on the topic.
""".strip()

_TEACHER_PROMPT = """You are an objective evaluator for a recruiting platform.

JOB DESCRIPTION:
{jd_text}

SCORING RUBRIC (use ONLY these six values: 0.0  0.2  0.4  0.6  0.8  1.0):
{rubric}

Below are {n} candidate profiles.
- Profile 1 is a CONFIRMED PERFECT FIT — your score for it must be 1.0.
- Profile {n} is a CONFIRMED NON-FIT — your score for it must be 0.0.
- Score profiles 2 through {n_mid} based solely on the rubric above.

For each candidate provide exactly:
  candidate_id : string (copy from profile)
  score        : float (one of 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
  rationale    : one sentence citing specific evidence from their profile

Return a JSON array with {n} objects, one per candidate, in input order. No extra text.

CANDIDATES:
{profiles_json}
"""

# ------------------------------------------------------------------ #
# Anchor profiles (calibration, present in every batch)
# ------------------------------------------------------------------ #

_ANCHOR_PERFECT = {
    "candidate_id": "_ANCHOR_PERFECT",
    "current_title": "Senior ML Engineer — Search & Ranking",
    "years_of_experience": 7,
    "summary": (
        "7 years building production retrieval and ranking systems at product companies. "
        "Shipped hybrid dense+BM25 search serving 10 M queries/day. "
        "Deep expertise in sentence-transformers, FAISS, and LambdaMART. "
        "Led evaluation-framework adoption (NDCG@10 as KPI) and drove 12 % lift via A/B tests."
    ),
    "career": [{
        "title": "Senior ML Engineer", "company": "E-commerce Startup",
        "industry": "E-commerce", "company_size": "201-500",
        "duration_months": 36,
        "description": (
            "Built embedding-based retrieval pipeline + LambdaMART reranker from scratch. "
            "Reduced p99 latency by 40 %, improved NDCG@10 by 12 %. "
            "Owned A/B testing framework and weekly ranking experiments."
        ),
    }],
    "skills_top5": [
        {"name": "sentence-transformers", "proficiency": "expert",   "endorsements": 45, "duration_months": 48},
        {"name": "FAISS",                 "proficiency": "expert",   "endorsements": 38, "duration_months": 42},
        {"name": "XGBoost LambdaMART",    "proficiency": "advanced", "endorsements": 30, "duration_months": 30},
        {"name": "Python",                "proficiency": "expert",   "endorsements": 60, "duration_months": 84},
        {"name": "Elasticsearch",         "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
    ],
    "education": [{"degree": "B.Tech", "field": "Computer Science",
                   "institution": "IIT Delhi", "tier": "tier_1"}],
}

_ANCHOR_NONFIT = {
    "candidate_id": "_ANCHOR_NONFIT",
    "current_title": "Marketing Manager",
    "years_of_experience": 5,
    "summary": (
        "Experienced marketing professional specialising in digital campaigns, SEO, "
        "and content strategy. Passionate about AI trends. "
        "Led team of 5 and managed ₹2 Cr annual ad budget."
    ),
    "career": [{
        "title": "Marketing Manager", "company": "FMCG Brand",
        "industry": "Consumer Goods", "company_size": "1001-5000",
        "duration_months": 36,
        "description": (
            "Managed brand campaigns, social media, and performance marketing. "
            "No technical or ML work."
        ),
    }],
    "skills_top5": [
        {"name": "Digital Marketing", "proficiency": "expert",        "endorsements": 50, "duration_months": 60},
        {"name": "SEO",               "proficiency": "advanced",      "endorsements": 30, "duration_months": 48},
        {"name": "Python",            "proficiency": "beginner",      "endorsements":  2, "duration_months":  3},
        {"name": "Machine Learning",  "proficiency": "beginner",      "endorsements":  1, "duration_months":  1},
        {"name": "AI",                "proficiency": "beginner",      "endorsements":  0, "duration_months":  0},
    ],
    "education": [{"degree": "MBA", "field": "Marketing",
                   "institution": "Private College", "tier": "tier_3"}],
}

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _read_jd_text() -> str:
    hyp_path = ARTIFACTS / "hypothetical_resumes.json"
    if hyp_path.exists():
        with open(hyp_path) as f:
            return json.load(f).get("jd_text", "")
    try:
        import docx
        doc = docx.Document(DATASET / "job_description.docx")
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception:
        return ""


def _candidate_for_teacher(c: dict) -> dict:
    return {
        "candidate_id": c["candidate_id"],
        "current_title": c["profile"]["current_title"],
        "years_of_experience": c["profile"]["years_of_experience"],
        "summary": c["profile"]["summary"][:400],
        "career": [
            {
                "title":          j["title"],
                "company":        j["company"],
                "industry":       j["industry"],
                "company_size":   j["company_size"],
                "duration_months": j["duration_months"],
                "description":    j["description"][:200],
            }
            for j in c.get("career_history", [])[:4]
        ],
        "skills_top5": [
            {
                "name":            s["name"],
                "proficiency":     s["proficiency"],
                "endorsements":    s["endorsements"],
                "duration_months": s.get("duration_months", 0),
            }
            for s in sorted(c.get("skills", []),
                            key=lambda x: x["endorsements"], reverse=True)[:5]
        ],
        "education": [
            {
                "degree": e["degree"],
                "field":  e["field_of_study"],
                "institution": e["institution"],
                "tier":   e.get("tier"),
            }
            for e in c.get("education", [])[:2]
        ],
    }


def _extract_json_array(text: str) -> list:
    text = re.sub(r"```(?:json)?\n?", "", text).strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in response:\n{text[:300]}")
    return json.loads(text[start:end])

# ------------------------------------------------------------------ #
# Labeling
# ------------------------------------------------------------------ #

def _label_batch(client, jd_text: str, batch: list[dict]) -> list[dict]:
    """Wrap batch with anchors, call Claude, return only real-candidate results."""
    profiles = [_ANCHOR_PERFECT] + batch + [_ANCHOR_NONFIT]
    n = len(profiles)
    prompt = _TEACHER_PROMPT.format(
        jd_text=jd_text,
        rubric=_RUBRIC,
        n=n,
        n_mid=n - 1,
        profiles_json=json.dumps(profiles, indent=2),
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    results = _extract_json_array(resp.content[0].text)
    real_ids = {p["candidate_id"] for p in batch}
    return [r for r in results if r.get("candidate_id") in real_ids]

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(verify_consistency: bool = True) -> None:
    import anthropic
    import numpy as np
    from src.utils import stream_candidates

    DATA.mkdir(exist_ok=True)

    jd_text = _read_jd_text()
    if not jd_text:
        raise RuntimeError("Could not load JD text — run generate_hypothetical.py first")

    sample_path = DATA / "sampled_candidates.json"
    if not sample_path.exists():
        raise FileNotFoundError("Run offline/stratified_sampler.py first")

    with open(sample_path) as f:
        sample_meta: list[dict] = json.load(f)
    id_to_stratum = {m["candidate_id"]: m["stratum"] for m in sample_meta}
    sample_ids = set(id_to_stratum.keys())
    print(f"Sample size: {len(sample_ids)}")

    print("Loading candidate profiles from JSONL...")
    candidates: list[dict] = []
    for c in stream_candidates(DATASET / "candidates.jsonl"):
        if c["candidate_id"] in sample_ids:
            candidates.append(c)
        if len(candidates) == len(sample_ids):
            break
    print(f"Loaded {len(candidates)} profiles")

    client = anthropic.Anthropic()
    all_results: list[dict] = []

    print(f"Labeling in batches of {BATCH_SIZE}...")
    for i in range(0, len(candidates), BATCH_SIZE):
        raw_batch = candidates[i : i + BATCH_SIZE]
        batch = [_candidate_for_teacher(c) for c in raw_batch]
        try:
            results = _label_batch(client, jd_text, batch)
            for r in results:
                r["stratum"] = id_to_stratum.get(r["candidate_id"], "unknown")
            all_results.extend(results)
        except Exception as e:
            print(f"  [warn] batch {i // BATCH_SIZE} failed: {e}")
        if (i // BATCH_SIZE + 1) % 20 == 0:
            labeled_so_far = min(i + BATCH_SIZE, len(candidates))
            print(f"  Progress: {labeled_so_far}/{len(candidates)}")
        time.sleep(0.3)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "score", "rationale", "stratum"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved {OUTPUT_PATH}  ({len(all_results)} labels)")

    if not verify_consistency:
        return

    # ---- Self-consistency check ----
    double_pool = candidates[:100]
    if len(double_pool) < 10:
        return

    print(f"\nSelf-consistency check: re-labeling {len(double_pool)} candidates...")
    double_results: list[dict] = []
    for i in range(0, len(double_pool), BATCH_SIZE):
        batch = [_candidate_for_teacher(c) for c in double_pool[i : i + BATCH_SIZE]]
        try:
            double_results.extend(_label_batch(client, jd_text, batch))
        except Exception as e:
            print(f"  [warn] double-label batch failed: {e}")
        time.sleep(0.3)

    primary = {r["candidate_id"]: r["score"] for r in all_results}
    secondary = {r["candidate_id"]: r["score"] for r in double_results}
    common = [cid for cid in primary if cid in secondary]
    if len(common) >= 10:
        x = np.array([primary[cid] for cid in common])
        y = np.array([secondary[cid] for cid in common])
        r = float(np.corrcoef(x, y)[0, 1])
        print(f"Pearson r = {r:.3f}  (need ≥ 0.85)")
        if r < 0.85:
            print("WARNING: Low consistency — tighten rubric or switch to claude-sonnet-4-6")
    else:
        print(f"Not enough overlapping labels for consistency check ({len(common)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args()
    main(verify_consistency=args.verify)
