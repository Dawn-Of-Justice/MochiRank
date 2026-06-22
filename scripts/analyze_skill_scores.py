"""
Analyze skill_assessment_scores vs claimed proficiency across all 100K candidates.
Goal: find the right threshold for "expert claimed but score too low" as a honeypot signal.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import stream_candidates
from src.consistency_checks import check_consistency


def main(candidates_path: str) -> None:
    path = Path(candidates_path)

    # score buckets per proficiency level
    scores_by_prof: dict[str, list[float]] = defaultdict(list)
    # for honeypots vs clean: expert scores
    expert_scores_clean: list[float] = []
    expert_scores_hp: list[float] = []

    # candidates where expert skill has assessment score < threshold
    contradictions: dict[int, list[str]] = defaultdict(list)  # threshold -> [cid]

    total = 0
    honeypot_ids: set = set()

    print("Pass 1: collecting score distributions...")
    for c in stream_candidates(candidates_path):
        total += 1
        is_hp, _, _ = check_consistency(c)
        if is_hp:
            honeypot_ids.add(c["candidate_id"])

        assessment = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
        if not assessment:
            continue

        for skill in c.get("skills", []):
            name = skill.get("name", "")
            prof = skill.get("proficiency", "")
            score = assessment.get(name)
            if score is None:
                continue
            scores_by_prof[prof].append(float(score))
            if prof == "expert":
                if is_hp:
                    expert_scores_hp.append(float(score))
                else:
                    expert_scores_clean.append(float(score))

        if total % 10_000 == 0:
            print(f"  {total:,} scanned", flush=True)

    print(f"\nTotal candidates: {total:,}  |  Honeypots (Stage A): {len(honeypot_ids)}")
    print(f"\n{'='*60}")
    print("Assessment score stats by proficiency level")
    print(f"{'='*60}")
    for prof in ["beginner", "intermediate", "advanced", "expert"]:
        vals = scores_by_prof.get(prof, [])
        if not vals:
            continue
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        p10 = vals_sorted[int(n * 0.10)]
        p25 = vals_sorted[int(n * 0.25)]
        p50 = vals_sorted[int(n * 0.50)]
        p75 = vals_sorted[int(n * 0.75)]
        p90 = vals_sorted[int(n * 0.90)]
        avg = sum(vals_sorted) / n
        print(f"\n  {prof.upper()} (n={n:,})")
        print(f"    avg={avg:.1f}  p10={p10:.1f}  p25={p25:.1f}  "
              f"p50={p50:.1f}  p75={p75:.1f}  p90={p90:.1f}")
        print(f"    min={min(vals_sorted):.1f}  max={max(vals_sorted):.1f}")

    print(f"\n{'='*60}")
    print("Expert skill assessment scores: CLEAN vs HONEYPOT candidates")
    print(f"{'='*60}")
    def _stats(vals: list[float], label: str) -> None:
        if not vals:
            print(f"  {label}: no data")
            return
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        avg = sum(vals_sorted) / n
        p10 = vals_sorted[int(n * 0.10)]
        p25 = vals_sorted[int(n * 0.25)]
        p50 = vals_sorted[int(n * 0.50)]
        print(f"  {label} (n={n}): avg={avg:.1f}  p10={p10:.1f}  p25={p25:.1f}  p50={p50:.1f}  "
              f"min={min(vals_sorted):.1f}  max={max(vals_sorted):.1f}")

    _stats(expert_scores_clean, "Clean candidates - expert scores")
    _stats(expert_scores_hp,    "Honeypots       - expert scores")

    print(f"\n{'='*60}")
    print("Threshold sweep: expert claim + assessment score < T")
    print("(How many new non-honeypot catches vs how many honeypots also have this)")
    print(f"{'='*60}")
    print(f"  {'Threshold':<12} {'New catches (non-HP)':<24} {'Co-occurs w/ HP':<20} {'FP rate'}")
    print(f"  {'-'*70}")

    # Re-scan to count per-candidate contradictions
    cand_min_expert_score: dict[str, float] = {}
    for c in stream_candidates(candidates_path):
        assessment = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
        expert_assessed = [
            float(assessment[s["name"]])
            for s in c.get("skills", [])
            if s.get("proficiency") == "expert" and s.get("name") in assessment
        ]
        if expert_assessed:
            cand_min_expert_score[c["candidate_id"]] = min(expert_assessed)

    for threshold in [10, 15, 20, 25, 30, 35, 40, 50]:
        new_catches = [cid for cid, mn in cand_min_expert_score.items()
                       if mn < threshold and cid not in honeypot_ids]
        hp_cooccur  = [cid for cid, mn in cand_min_expert_score.items()
                       if mn < threshold and cid in honeypot_ids]
        total_assessed_clean = sum(1 for cid in cand_min_expert_score if cid not in honeypot_ids)
        fp_rate = len(new_catches) / total_assessed_clean * 100 if total_assessed_clean else 0
        print(f"  score < {threshold:<4}  new={len(new_catches):<6}  "
              f"hp_cooccur={len(hp_cooccur):<6}  fp={fp_rate:.2f}%")

    print(f"\n{'='*60}")
    print("Candidates with expert claim + assessment score below key thresholds")
    print(f"{'='*60}")
    for threshold in [20, 30]:
        cids = [cid for cid, mn in cand_min_expert_score.items()
                if mn < threshold and cid not in honeypot_ids]
        print(f"\n  score < {threshold}: {len(cids)} non-honeypot candidates")
        for cid in cids[:10]:
            print(f"    {cid}  min_expert_score={cand_min_expert_score[cid]:.1f}")
        if len(cids) > 10:
            print(f"    ... and {len(cids)-10} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="dataset/candidates.jsonl")
    args = parser.parse_args()
    main(args.candidates)
