import sys
sys.path.insert(0, ".")
from src.utils import stream_candidates

targets = {"CAND_0041611", "CAND_0064270", "CAND_0060072", "CAND_0064904",
           "CAND_0092278", "CAND_0015528", "CAND_0061339"}
found = {}
for c in stream_candidates("dataset/candidates.jsonl"):
    if c["candidate_id"] in targets:
        sig = c.get("redrob_signals", {})
        exp = sum(1 for s in c.get("skills", []) if s.get("proficiency") == "expert")
        signup = sig.get("signup_date", "")
        last_active = sig.get("last_active_date", "")
        hp_score = (1 if signup > last_active else 0) + (1 if exp >= 10 else 0)
        found[c["candidate_id"]] = (exp, signup, last_active, hp_score)
    if len(found) == len(targets):
        break

print(f"{'Candidate':<16} {'Expert':>6} {'signup':>12} {'last_active':>12} {'hp_score':>8}  {'ACTION'}")
print("-" * 72)
for cid, (exp, signup, la, hp) in sorted(found.items()):
    action = "REMOVE" if hp >= 2 else "keep"
    print(f"{cid:<16} {exp:>6} {signup:>12} {la:>12} {hp:>8}  {action}")
