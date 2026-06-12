"""
Honeypot and impossible-profile detection (Stage A of rank.py).

Interface contract (frozen — Salo's sampler imports this):
    check_consistency(c: dict) -> tuple[bool, int, list[str]]
        is_honeypot : True  → exclude from top-100 (hard gate)
        violation_count : total logical contradictions found
        reasons : human-readable list for debugging
"""

from src.utils import FICTIONAL_COMPANIES, parse_year, REFERENCE_DATE


def check_consistency(c: dict) -> tuple[bool, int, list[str]]:
    """
    Returns (is_honeypot, violation_count, reasons).

    Honeypot conditions (any single one is sufficient):
      - YOE impossible given graduation date
      - Worked at a fictional company before it was founded
      - Expert skill claimed with 0 months experience

    Soft violations add to violation_count but don't alone trigger honeypot:
      - Overlapping non-concurrent employment dates
      - Total career months >> YOE * 12
      - 10+ self-reported "expert" skills
      - signup_date after last_active_date
    """
    violations: list[str] = []
    honeypot_flags: list[str] = []  # only hard-evidence checks go here

    profile = c.get("profile", {})
    yoe: float = profile.get("years_of_experience", 0.0)
    edu: list[dict] = c.get("education", [])
    career: list[dict] = c.get("career_history", [])
    skills: list[dict] = c.get("skills", [])
    sig: dict = c.get("redrob_signals", {})

    # ------------------------------------------------------------------ #
    # Check 1: YOE impossible given latest graduation date
    # ------------------------------------------------------------------ #
    if edu:
        end_years = [e.get("end_year") for e in edu if e.get("end_year")]
        if end_years:
            latest_grad = max(end_years)
            max_possible_yoe = REFERENCE_DATE.year - latest_grad
            if yoe > max_possible_yoe + 1.5:  # 1.5 yr buffer
                msg = (f"yoe={yoe} claimed but latest graduation {latest_grad} "
                       f"(max possible {max_possible_yoe:.0f}yr)")
                violations.append(msg)  # soft only — noisy for hard-gate

    # ------------------------------------------------------------------ #
    # Check 2: Job duration_months >> actual date span (impossible tenure)
    # e.g., "8 years at a company founded 3 years ago"
    # ------------------------------------------------------------------ #
    for job in career:
        dm = job.get("duration_months", 0)
        sd = job.get("start_date", "")
        ed = job.get("end_date", "")
        if dm and sd:
            try:
                from datetime import date as _date
                s = _date.fromisoformat(sd)
                e = _date.fromisoformat(ed) if ed else REFERENCE_DATE
                actual_months = (e.year - s.year) * 12 + (e.month - s.month)
                if dm > actual_months + 12:  # >1yr discrepancy is impossible
                    msg = (f"impossible tenure: claimed {dm}m at {job.get('company','?')}, "
                           f"actual span {actual_months}m")
                    violations.append(msg)
                    honeypot_flags.append(msg)
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------ #
    # Check 3: Multiple expert skills with 0 months (keyword inflation trap)
    # ------------------------------------------------------------------ #
    zero_expert_count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if zero_expert_count >= 3:
        msg = f"{zero_expert_count} expert skills each with 0 months experience"
        violations.append(msg)
        honeypot_flags.append(msg)

    # ------------------------------------------------------------------ #
    # Check 4: Overlapping non-concurrent employment (soft)
    # ------------------------------------------------------------------ #
    closed_roles = [j for j in career if not j.get("is_current")
                    and j.get("start_date") and j.get("end_date")]
    for i, j1 in enumerate(closed_roles):
        for j2 in closed_roles[i + 1:]:
            # overlap: j1.start < j2.end AND j2.start < j1.end
            if j1["start_date"] < j2["end_date"] and j2["start_date"] < j1["end_date"]:
                violations.append(
                    f"overlapping roles: '{j1.get('title','?')}' at {j1['company']} "
                    f"and '{j2.get('title','?')}' at {j2['company']}"
                )

    # ------------------------------------------------------------------ #
    # Check 5: Total career months >> YOE * 12  (soft)
    # ------------------------------------------------------------------ #
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if yoe > 0 and total_career_months > yoe * 12 + 24:
        violations.append(
            f"total career months ({total_career_months}) >> "
            f"YOE*12 ({yoe * 12:.0f}) by more than 24 months"
        )

    # ------------------------------------------------------------------ #
    # Check 6: 10+ self-reported "expert" skills (keyword inflation, soft)
    # ------------------------------------------------------------------ #
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 10:
        violations.append(f"suspiciously many expert skills: {expert_count}")

    # ------------------------------------------------------------------ #
    # Check 7: signup_date after last_active_date (data integrity, soft)
    # ------------------------------------------------------------------ #
    signup = sig.get("signup_date", "")
    last_active = sig.get("last_active_date", "")
    if signup and last_active and signup > last_active:
        violations.append(f"signup_date ({signup}) is after last_active_date ({last_active})")

    is_honeypot = len(honeypot_flags) > 0
    return is_honeypot, len(violations), violations
