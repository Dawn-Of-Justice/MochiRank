"""
Simulate Stage F2 on the sample candidates to verify the new scoring logic.
Run against sample first (fast), then optionally full dataset.

Usage:
    python scripts/check_finalists_f2.py
    python scripts/check_finalists_f2.py --candidates dataset/candidates.jsonl
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import stream_candidates, load_candidates_json

REFERENCE_DATE = date(2026, 6, 10)


def finalist_hp_score(c: dict) -> tuple[int, list[str]]:
    """Returns (score, signals). score >= 2 -> remove from top-100."""
    score = 0
    signals = []
    sig = c.get("redrob_signals", {})
    skills = c.get("skills", [])

    signup = sig.get("signup_date", "")
    last_active = sig.get("last_active_date", "")
    if signup and last_active and signup > last_active:
        score += 2
        signals.append(f"signup({signup})>last_active({last_active}) [+2]")

    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 12:
        score += 2
        signals.append(f"expert_skills={expert_count}(>=12) [+2]")
    elif expert_count >= 10:
        score += 1
        signals.append(f"expert_skills={expert_count}(>=10) [+1]")

    return score, signals


def main(candidates_path: str) -> None:
    path = Path(candidates_path)
    print(f"Scanning {path} for any candidate with F2 score >= 1 ...")

    flagged_remove = []
    flagged_watch = []
    total = 0

    loader = (
        load_candidates_json(path)
        if str(path).endswith(".json")
        else stream_candidates(path)
    )
    for c in loader:
        total += 1
        score, signals = finalist_hp_score(c)
        cid = c["candidate_id"]
        if score >= 2:
            flagged_remove.append((cid, score, signals))
        elif score == 1:
            flagged_watch.append((cid, score, signals))

    print(f"\nTotal scanned: {total:,}")
    print(f"Would remove (score >= 2): {len(flagged_remove)}")
    print(f"On-watch (score == 1):     {len(flagged_watch)}")

    print("\n--- WOULD REMOVE (score >= 2) ---")
    for cid, s, sigs in sorted(flagged_remove, key=lambda x: -x[1]):
        print(f"  {cid}  total={s}  {' | '.join(sigs)}")

    print("\n--- ON WATCH (score == 1, not removed) ---")
    for cid, s, sigs in sorted(flagged_watch, key=lambda x: -x[1])[:30]:
        print(f"  {cid}  total={s}  {' | '.join(sigs)}")
    if len(flagged_watch) > 30:
        print(f"  ... and {len(flagged_watch) - 30} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="dataset/sample_candidates.json")
    args = parser.parse_args()
    main(args.candidates)
