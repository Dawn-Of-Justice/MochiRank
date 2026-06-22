"""
Score every finalist on multiple honeypot signals and show the full picture.
"""
import csv, sys, json
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import stream_candidates, REFERENCE_DATE

SUBMISSION = "C:/Users/salos/Downloads/finalists_all.csv"
CANDIDATES = "dataset/candidates.jsonl"

finalist_rank = {}
with open(SUBMISSION, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        finalist_rank[row["candidate_id"]] = int(row["rank"])

profiles = {}
for c in stream_candidates(CANDIDATES):
    if c["candidate_id"] in finalist_rank:
        profiles[c["candidate_id"]] = c
    if len(profiles) == len(finalist_rank):
        break

print(f"{'Rank':<5} {'Candidate':<16} {'Score':<6} {'Signals fired'}")
print("-" * 100)

rows = []
for cid, rank in sorted(finalist_rank.items(), key=lambda x: x[1]):
    c = profiles.get(cid)
    if not c:
        continue
    sig = c.get("redrob_signals", {})
    skills = c.get("skills", [])

    signals = []
    points = 0

    signup = sig.get("signup_date", "")
    last_active = sig.get("last_active_date", "")
    if signup and last_active and signup > last_active:
        signals.append(f"signup({signup})>last_active({last_active})")
        points += 2  # clear logical impossibility

    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 12:
        signals.append(f"expert_skills={expert_count}(very high)")
        points += 2
    elif expert_count >= 10:
        signals.append(f"expert_skills={expert_count}")
        points += 1

    rrr = sig.get("recruiter_response_rate", 1.0)
    if rrr is not None and float(rrr) < 0.15:
        signals.append(f"recruiter_rr={rrr:.0%}")
        points += 1

    if last_active:
        try:
            days_inactive = (REFERENCE_DATE - date.fromisoformat(last_active)).days
            if days_inactive > 180:
                signals.append(f"inactive={days_inactive}d")
                points += 1
        except (ValueError, TypeError):
            pass

    zero_expert = sum(1 for s in skills
                      if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0)
    if zero_expert:
        signals.append(f"expert_0mo_skills={zero_expert}")
        points += 3

    rows.append((rank, cid, points, signals))

rows.sort(key=lambda x: (-x[2], x[0]))

for rank, cid, points, signals in rows:
    flag = "  <-- REMOVE" if points >= 2 else ""
    sig_str = " | ".join(signals) if signals else "clean"
    print(f"{rank:<5} {cid:<16} {points:<6} {sig_str}{flag}")

print()
to_remove = [(r, c, p, s) for r, c, p, s in rows if p >= 2]
print(f"Candidates to remove (score >= 2): {len(to_remove)}")
for rank, cid, points, signals in to_remove:
    print(f"  Rank {rank:>3}  {cid}  score={points}  {' | '.join(signals)}")
