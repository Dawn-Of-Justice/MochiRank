"""
Sanity-check: run consistency engine on all 50 sample candidates.
Prints a summary and flags every flagged profile for eyeballing.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consistency_checks import check_consistency

SAMPLE_PATH = Path(__file__).parent.parent / "dataset" / "sample_candidates.json"


def main():
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        candidates = json.load(f)

    honeypots = []
    soft_violations = []

    for c in candidates:
        cid = c["candidate_id"]
        name = c["profile"].get("anonymized_name", "?")
        is_hp, n_violations, reasons = check_consistency(c)

        if is_hp:
            honeypots.append((cid, name, reasons))
        elif n_violations > 0:
            soft_violations.append((cid, name, n_violations, reasons))

    print(f"\n{'='*60}")
    print(f"Consistency Check — {len(candidates)} sample candidates")
    print(f"{'='*60}")
    print(f"  Honeypots (hard flag): {len(honeypots)}")
    print(f"  Soft violations only:  {len(soft_violations)}")
    print(f"  Clean profiles:        {len(candidates) - len(honeypots) - len(soft_violations)}")

    if honeypots:
        print(f"\n{'-'*60}")
        print("HONEYPOTS:")
        for cid, name, reasons in honeypots:
            print(f"\n  [{cid}] {name}")
            for r in reasons:
                print(f"    FAIL: {r}")

    if soft_violations:
        print(f"\n{'-'*60}")
        print("SOFT VIOLATIONS (not honeypot, but suspicious):")
        for cid, name, n, reasons in soft_violations:
            print(f"\n  [{cid}] {name}  ({n} violation(s))")
            for r in reasons:
                print(f"    WARN: {r}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
