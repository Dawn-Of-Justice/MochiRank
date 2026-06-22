"""
Audit the 100 finalists for any suspicious patterns — soft AND hard.
Since we're only checking 100 candidates (not 100K) we can afford
to be aggressive: flag anything unusual and let a human decide.

Usage:
    python scripts/audit_finalists.py
    python scripts/audit_finalists.py --submission submission.csv --candidates dataset/candidates.jsonl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import FICTIONAL_COMPANIES, REFERENCE_DATE, stream_candidates

_FICTIONAL_NAMES = set(FICTIONAL_COMPANIES.keys()) | {"hooli", "acme corp", "initech"}


def full_audit(c: dict) -> list[tuple[str, str]]:
    """
    Returns list of (severity, reason) for every suspicious signal found.
    severity: 'HARD' (clear honeypot), 'SOFT' (suspicious), 'NOTE' (weak signal)
    """
    findings: list[tuple[str, str]] = []
    profile = c.get("profile", {})
    yoe: float = profile.get("years_of_experience", 0.0)
    career: list[dict] = c.get("career_history", [])
    skills: list[dict] = c.get("skills", [])
    edu: list[dict] = c.get("education", [])
    sig: dict = c.get("redrob_signals", {})

    # ── HARD gates (already in check_consistency) ──────────────────────
    # YOE vs career start
    start_years = [int(j["start_date"][:4]) for j in career
                   if j.get("start_date", "")[:4].isdigit()]
    if start_years:
        max_from_career = REFERENCE_DATE.year - min(start_years) + 2
        if yoe > max_from_career:
            findings.append(("HARD", f"yoe={yoe} > career span max {max_from_career}yr "
                             f"(earliest start {min(start_years)})"))

    # Impossible tenure (existing threshold: >12mo discrepancy)
    for j in career:
        dm = j.get("duration_months", 0)
        sd, ed = j.get("start_date", ""), j.get("end_date", "")
        if dm and sd:
            try:
                s = date.fromisoformat(sd)
                e = date.fromisoformat(ed) if ed else REFERENCE_DATE
                actual = (e.year - s.year) * 12 + (e.month - s.month)
                if dm > actual + 12:
                    findings.append(("HARD", f"impossible tenure at {j.get('company','?')}: "
                                     f"claimed {dm}m, actual span {actual}m"))
                elif dm > actual + 6:
                    findings.append(("SOFT", f"suspicious tenure at {j.get('company','?')}: "
                                     f"claimed {dm}m, actual span {actual}m (6-12mo gap)"))
            except (ValueError, TypeError):
                pass

    # Expert skill with 0 months
    zero_expert = [s for s in skills if s.get("proficiency") == "expert"
                   and s.get("duration_months", 1) == 0]
    if zero_expert:
        findings.append(("HARD", f"{len(zero_expert)} expert skill(s) with 0mo experience: "
                         f"{[s['name'] for s in zero_expert[:3]]}"))

    # ── SOFT checks ────────────────────────────────────────────────────
    # YOE > grad cap (soft but improbable at high excess)
    end_years = [e.get("end_year") for e in edu if e.get("end_year")]
    if end_years:
        latest_grad = max(end_years)
        max_from_grad = REFERENCE_DATE.year - latest_grad
        excess = yoe - (max_from_grad + 1.5)
        if excess > 4:
            findings.append(("SOFT", f"yoe={yoe} exceeds grad cap by {excess:.1f}yr "
                             f"(grad {latest_grad}, max ~{max_from_grad}yr)"))
        elif excess > 0:
            findings.append(("NOTE", f"yoe={yoe} slightly over grad cap "
                             f"(grad {latest_grad}, max ~{max_from_grad}yr, excess {excess:.1f}yr)"))

    # signup_date > last_active_date
    signup = sig.get("signup_date", "")
    last_active = sig.get("last_active_date", "")
    if signup and last_active and signup > last_active:
        findings.append(("SOFT", f"signup_date ({signup}) is after last_active_date ({last_active})"))

    # Overlapping non-concurrent jobs
    closed = [j for j in career if not j.get("is_current")
              and j.get("start_date") and j.get("end_date")]
    overlaps = []
    for i, j1 in enumerate(closed):
        for j2 in closed[i + 1:]:
            if j1["start_date"] < j2["end_date"] and j2["start_date"] < j1["end_date"]:
                overlaps.append(f"{j1.get('company','?')} & {j2.get('company','?')}")
    if overlaps:
        findings.append(("SOFT", f"overlapping employment: {overlaps[:2]}"))

    # Total career months >> YOE*12
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if yoe > 0 and total_career_months > yoe * 12 + 24:
        excess_mo = total_career_months - yoe * 12
        findings.append(("NOTE", f"total career months ({total_career_months}) > "
                         f"YOE*12 ({yoe*12:.0f}) by {excess_mo:.0f}mo"))

    # 10+ expert skills
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 8:
        findings.append(("NOTE" if expert_count < 10 else "SOFT",
                         f"{expert_count} self-reported expert skills"))

    # Fictional company in career (any mention, even with valid dates)
    for j in career:
        co = j.get("company", "").lower()
        for name in _FICTIONAL_NAMES:
            if name in co:
                findings.append(("NOTE", f"fictional company in career: '{j.get('company','?')}'"))
                break

    # Recruiter response rate anomaly (very low for a finalist)
    rrr = sig.get("recruiter_response_rate", 1.0)
    if rrr < 0.10:
        findings.append(("NOTE", f"very low recruiter response rate: {rrr:.0%}"))

    # No recent activity (last_active > 180 days ago)
    if last_active:
        try:
            la = date.fromisoformat(last_active)
            days_inactive = (REFERENCE_DATE - la).days
            if days_inactive > 180:
                findings.append(("NOTE", f"inactive for {days_inactive} days "
                                 f"(last active {last_active})"))
        except (ValueError, TypeError):
            pass

    return findings


def main(submission_path: str, candidates_path: str) -> None:
    # Load finalist IDs from the submission CSV
    finalist_ids: set[str] = set()
    finalist_rank: dict[str, int] = {}
    with open(submission_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["candidate_id"]
            finalist_ids.add(cid)
            finalist_rank[cid] = int(row["rank"])

    print(f"Loaded {len(finalist_ids)} finalists from {submission_path}")

    # Stream candidates.jsonl and collect finalist profiles
    finalist_profiles: dict = {}
    for c in stream_candidates(candidates_path):
        if c["candidate_id"] in finalist_ids:
            finalist_profiles[c["candidate_id"]] = c
        if len(finalist_profiles) == len(finalist_ids):
            break

    print(f"Found {len(finalist_profiles)} profiles in {candidates_path}\n")

    # Audit each finalist
    clean: list = []
    suspicious: list = []
    for cid in sorted(finalist_ids, key=lambda x: finalist_rank[x]):
        c = finalist_profiles.get(cid)
        if c is None:
            print(f"WARNING: {cid} not found in candidates file")
            continue
        findings = full_audit(c)
        rank = finalist_rank[cid]
        if any(sev in ("HARD", "SOFT") for sev, _ in findings):
            suspicious.append((rank, cid, findings))
        elif findings:
            clean.append((rank, cid, findings))  # NOTE-only
        else:
            clean.append((rank, cid, []))

    print(f"{'='*70}")
    print(f"SUSPICIOUS (HARD or SOFT violations): {len(suspicious)} candidates")
    print(f"{'='*70}")
    for rank, cid, findings in suspicious:
        print(f"\n  Rank {rank:>3}  {cid}")
        for sev, reason in findings:
            marker = "!!!" if sev == "HARD" else "  !"
            print(f"    {marker} [{sev}]  {reason}")

    print(f"\n{'='*70}")
    note_only = [(r, c, f) for r, c, f in clean if f]
    print(f"NOTE-only (weak signals, likely fine): {len(note_only)} candidates")
    for rank, cid, findings in note_only[:10]:
        print(f"\n  Rank {rank:>3}  {cid}")
        for sev, reason in findings:
            print(f"         [NOTE]  {reason}")
    if len(note_only) > 10:
        print(f"  ... and {len(note_only)-10} more")

    print(f"\n{'='*70}")
    print(f"Fully clean finalists: {len([x for x in clean if not x[2]])}")
    print(f"\nRECOMMENDATION: manually review {len(suspicious)} SUSPICIOUS candidates "
          f"above before final submission.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--candidates", default="dataset/candidates.jsonl")
    args = parser.parse_args()
    main(args.submission, args.candidates)
