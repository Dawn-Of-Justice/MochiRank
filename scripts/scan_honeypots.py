"""
Honeypot scan — streams candidates.jsonl and runs Stage A check_consistency.
Reports total count, trigger breakdown, and all identified IDs.

Usage:
    python scripts/scan_honeypots.py
    python scripts/scan_honeypots.py --candidates dataset/candidates.jsonl
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consistency_checks import check_consistency
from src.utils import stream_candidates


def classify_reason(r: str) -> str:
    if "impossible tenure" in r:
        return "impossible_tenure"
    if "expert skill" in r and "0 months" in r:
        return "expert_zero_months"
    if "assessment score" in r:
        return "expert_low_assessment_score"
    if "yoe=" in r and "graduation" in r:
        return "yoe_impossible_from_grad"
    if "yoe=" in r and "career start" in r:
        return "yoe_impossible_from_career"
    return "other_soft"


def main(candidates_path: str) -> None:
    path = Path(candidates_path)
    if not path.exists():
        print(f"ERROR: {path} not found"); sys.exit(1)

    honeypots: list[tuple[str, list[str]]] = []
    trigger_counter: Counter = Counter()
    total = 0

    print(f"Scanning {path} ...", flush=True)
    for c in stream_candidates(path):
        total += 1
        is_hp, _, reasons = check_consistency(c)
        if is_hp:
            honeypots.append((c["candidate_id"], reasons))
            for r in reasons:
                trigger_counter[classify_reason(r)] += 1
        if total % 10_000 == 0:
            print(f"  {total:,} scanned -- {len(honeypots)} honeypots so far", flush=True)

    print(f"\n{'='*55}")
    print(f"Total scanned        : {total:,}")
    print(f"Honeypots detected   : {len(honeypots)}")
    print(f"Honeypot rate        : {len(honeypots)/total*100:.3f}%")
    print(f"\nTrigger breakdown (reason counts, not unique candidates):")
    for trigger, count in trigger_counter.most_common():
        print(f"  {trigger:<40} {count}")

    print(f"\nAll honeypot IDs:")
    for cid, reasons in honeypots:
        # Show the first HARD reason only
        hard = next((r for r in reasons if "impossible" in r or "expert" in r), reasons[0])
        print(f"  {cid}  ->  {hard[:75]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="dataset/candidates.jsonl")
    args = parser.parse_args()
    main(args.candidates)
