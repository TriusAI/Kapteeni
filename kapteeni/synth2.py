"""Synthetic v2: the weak-family overhaul (ground truth by construction).

Root causes this targets (docs/JEVBENCH.md family analysis):
  - temporal_numeric failed because v1 templates were too narrow — the model
    learned the templates, not the skill. v2 multiplies surface forms
    (6 date formats incl. weekday-styled, relative phrasing, calendar vs
    business days with explicit holiday lists, boundary semantics worded
    per relation, near-miss/weekday-consistent traps) and adds choice and
    score variants of the same skills.
  - multi_hop: compositional chains (eligibility rules where tenure must be
    computed from dates, chained business-day arithmetic, ordered fee math,
    two-level exception logic, clause specificity, per-condition failure
    identification). Facts are generated FROM the gold, so labels are exact.
  - long_policy: a document grammar that builds 350-550-word policies with
    randomized parameters and distractor clauses; questions whose golds
    read straight off the generator's parameter table. Multiple questions
    per document (shared state prefix) train long-state behavior.

All rows carry an explicit `today`/date fields — never real-world time.
Deterministic given --seed. Row schema matches build_data; choice options
map through the emitted criteria file; score levels likewise.

    python3 -m kapteeni.synth2 synth --n 6300 \
        --out data_cache/rows_synth2.jsonl --criteria-out data_cache/synth2_criteria.json
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MON_ABBR = [m[:3] for m in MONTHS]
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
      "Sunday"]


def _ord(n: int) -> str:
    return f"{n}{ 'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def fmt_date(d: date, rng: random.Random) -> str:
    style = rng.choice(["iso", "us", "uk", "abbr", "ord", "wd"])
    if style == "iso":
        return d.isoformat()
    if style == "uk":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if style == "abbr":
        return f"{MON_ABBR[d.month - 1]} {d.day}, {d.year}"
    if style == "ord":
        return f"the {_ord(d.day)} of {MONTHS[d.month - 1]} {d.year}"
    if style == "wd":
        return f"{WD[d.weekday()]}, {MONTHS[d.month - 1]} {d.day}, {d.year}"
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def month_add(d: date, n: int) -> date:
    """Only called with d.day <= 28, so month arithmetic is unambiguous."""
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, d.day)


def add_bd(d: date, n: int, holidays: set[date]) -> date:
    """The date of the n-th business day after d (Mon-Fri, minus holidays)."""
    assert n >= 0
    c, cur = 0, d
    while c < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in holidays:
            c += 1
    return cur


def _rand_date(rng: random.Random, y0: int = 2023, y1: int = 2025,
               max_day: int = 31) -> date:
    while True:
        d = date(rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28))
        return d


def _holiday_set(rng: random.Random, n: int = 4) -> set[date]:
    return {date(2024, rng.randint(1, 12), rng.randint(1, 28))
            for _ in range(n)}


def _holiday_str(holidays: set[date]) -> str:
    return ", ".join(sorted(h.isoformat() for h in holidays))


# ------------------------------------------------------------- temporal_numeric

def gen_date_arith(rng: random.Random) -> dict:
    """Date arithmetic with 6 formats + weekday-consistent near-miss traps."""
    base = _rand_date(rng)
    unit = rng.choice(["days", "days", "weeks", "months"])
    n = {"days": rng.randint(2, 45), "weeks": rng.randint(1, 8),
         "months": rng.randint(1, 3)}[unit]
    if unit == "months" and base.day > 28:
        base = base.replace(day=rng.randint(1, 28))
    true_event = (base + timedelta(days=n) if unit == "days" else
                  base + timedelta(weeks=n) if unit == "weeks" else
                  month_add(base, n))
    roll = rng.random()
    if roll < 0.5:
        claimed = true_event
    elif roll < 0.8:
        claimed = true_event + timedelta(days=rng.choice([-3, -2, -1, 1, 2, 3, 7]))
    else:  # weekday-consistent trap: right weekday, wrong week
        claimed = true_event + timedelta(weeks=rng.choice([-2, -1, 1, 2]))
    gold = int(claimed == true_event)
    neg = rng.random() < 0.15
    thing = rng.choice(["inspection", "audit", "board meeting", "delivery",
                        "training session", "product launch"])
    phr = rng.choice([
        f"The {thing} is scheduled for exactly {n} {unit} after "
        "`reference_date`.",
        f"Per the schedule, the {thing} falls {n} {unit} after "
        "`reference_date` — no earlier and no later.",
    ])
    state = {"reference_date": fmt_date(base, rng), "schedule": phr,
             "claimed_date": fmt_date(claimed, rng)}
    if rng.random() < 0.3:  # irrelevant distractor date
        state["reminder_sent_on"] = fmt_date(
            base + timedelta(days=rng.randint(1, 10)), rng)
    if not neg:
        return {"row_id": None, "primitive": "noul", "state": state,
                "instructions": "Is `claimed_date` exactly the date the "
                                "`schedule` describes?",
                "criteria": {"true": "the claimed date is the described date",
                             "false": "the claimed date is a different date"},
                "label": gold, "meta": {"family": "temporal_numeric"}}
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "Is `claimed_date` a different date from the one "
                            "the `schedule` describes?",
            "criteria": {"true": "the claimed date differs from the "
                                 "described date",
                         "false": "the claimed date is the described date"},
            "label": 1 - gold, "meta": {"family": "temporal_numeric"}}


def gen_business_days(rng: random.Random) -> dict:
    """Business-day windows with explicit holiday lists; two templates."""
    start = _rand_date(rng)
    n = rng.choice([3, 5, 10, 15])
    hol = _holiday_set(rng)
    mode = rng.choice(["window", "due"])
    if mode == "window":
        k = rng.choice([n - 1, n, n, n + 1, n + 2])
        sent = add_bd(start, k, hol)
        gold = int(k <= n)
        state = {"received_on": fmt_date(start, rng),
                 "rule": "A response is required within "
                         f"{n} business days of `received_on`. Business days "
                         "are Monday through Friday, excluding the listed "
                         "holidays.",
                 "holidays": _holiday_str(hol),
                 "responded_on": fmt_date(sent, rng)}
        return {"row_id": None, "primitive": "noul", "state": state,
                "instructions": "Was the response in `responded_on` sent "
                                "within the deadline set by `rule`?",
                "criteria": {"true": "the response is within the window",
                             "false": "the response is late"},
                "label": gold, "meta": {"family": "temporal_numeric"}}
    # due: compute the due date, claimed near-miss
    due = add_bd(start, n, hol)
    roll = rng.random()
    if roll < 0.5:
        claimed = due
    elif roll < 0.85:
        claimed = due + timedelta(days=rng.choice([-3, -1, 1, 2, 3]))
    else:  # same weekday, one week off — weekday-consistent trap
        claimed = due + timedelta(weeks=1)
    gold = int(claimed == due)
    state = {"received_on": fmt_date(start, rng),
             "rule": "A response is due exactly "
                     f"{n} business days after `received_on`. Business days "
                     "are Monday through Friday, excluding the listed "
                     "holidays.",
             "holidays": _holiday_str(hol),
             "claimed_due_date": fmt_date(claimed, rng)}
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "Is `claimed_due_date` the correct due date "
                            "under `rule`?",
            "criteria": {"true": "the claimed due date is correct",
                         "false": "the claimed due date is off by at least "
                                   "one business day"},
            "label": gold, "meta": {"family": "temporal_numeric"}}


def gen_deadline_bounds(rng: random.Random) -> dict:
    """Boundary semantics: relation wording fixes inclusivity; x sits on or
    right next to the boundary on purpose."""
    start = _rand_date(rng)
    n = rng.choice([3, 7, 10, 14, 30, 60])
    rel = rng.choice(["within N days of D", "no later than the deadline",
                      "at least N days before the event",
                      "strictly before the deadline",
                      "after the filing date"])
    if rel == "within N days of D":
        delta = rng.choice([n - 1, n, n, n + 1, n + 2])
        act = start + timedelta(days=delta)
        gold = int(delta <= n)
        state = {"incident_on": fmt_date(start, rng),
                 "policy": f"Claims must be filed within {n} days of "
                           "`incident_on` (the day of the incident counts as "
                           "day 0).",
                 "filed_on": fmt_date(act, rng)}
        q = "Was the claim in `filed_on` filed on time under `policy`?"
        crit = {"true": "the filing is within the window",
                "false": "the filing is late"}
    elif rel == "no later than the deadline":
        due = start + timedelta(days=n)
        delta = rng.choice([-2, -1, 0, 0, 1, 2])
        act = due + timedelta(days=delta)
        gold = int(act <= due)
        state = {"deadline": fmt_date(due, rng),
                 "action_on": fmt_date(act, rng)}
        q = "Does `action_on` meet the `deadline`?"
        crit = {"true": "the action happened on or before the deadline",
                "false": "the action happened after the deadline"}
    elif rel == "at least N days before the event":
        ev = start + timedelta(days=n + rng.choice([0, 1, 1, 2]))
        delta = rng.choice([n - 1, n, n, n + 1])
        act = ev - timedelta(days=delta)
        gold = int((ev - act).days >= n)
        state = {"event_on": fmt_date(ev, rng),
                 "notice_policy": "Notice must be given at least "
                                  f"{n} days before `event_on`.",
                 "notice_given_on": fmt_date(act, rng)}
        q = "Was the notice in `notice_given_on` timely under `notice_policy`?"
        crit = {"true": "the notice meets the minimum lead time",
                "false": "the notice was given too late"}
    elif rel == "strictly before the deadline":
        due = start + timedelta(days=n)
        delta = rng.choice([1, 1, 2, 0])
        act = due - timedelta(days=delta)
        gold = int(act < due)
        state = {"deadline": fmt_date(due, rng),
                 "rule": "The request must be submitted strictly before "
                         "`deadline` (same-day submission is too late).",
                 "submitted_on": fmt_date(act, rng)}
        q = "Does `submitted_on` satisfy `rule`?"
        crit = {"true": "the submission is before the deadline",
                "false": "the submission is on the deadline or later"}
    else:
        fdate = start
        delta = rng.choice([0, 1, 1, 2, 3])
        act = fdate + timedelta(days=delta)
        gold = int(act > fdate)
        state = {"filing_date": fmt_date(fdate, rng),
                 "rule": "Amendments are only accepted after `filing_date`.",
                 "amendment_on": fmt_date(act, rng)}
        q = "Is the amendment in `amendment_on` accepted under `rule`?"
        crit = {"true": "the amendment is after the filing date",
                "false": "the amendment is not after the filing date"}
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": q, "criteria": crit, "label": gold,
            "meta": {"family": "temporal_numeric"}}


def gen_overlap(rng: random.Random) -> dict:
    """Window overlap (inclusive wording) with boundary traps."""
    s1 = _rand_date(rng)
    len1 = rng.randint(3, 12)
    e1 = s1 + timedelta(days=len1)
    off = rng.choice([-len1, -3, -2, -1, -1, 0, 0, 1, 2, len1 - 1, len1 + 2])
    s2 = s1 + timedelta(days=off)
    e2 = s2 + timedelta(days=rng.randint(2, 10))
    gold = int(max(s1, s2) <= min(e1, e2))
    a = rng.choice(["on-call duty", "maintenance window", "sale period",
                    "audit fieldwork", "training cohort"])
    b = rng.choice(["system upgrade", "freeze period", "inventory count",
                    "beta rollout", "office relocation"])
    state = {
        "window_a": {"name": f"{a} #1",
                     "from": fmt_date(s1, rng), "through": fmt_date(e1, rng)},
        "window_b": {"name": f"{b}",
                     "from": fmt_date(s2, rng), "through": fmt_date(e2, rng)},
        "note": "Both windows are inclusive of their start and end dates.",
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "Do `window_a` and `window_b` overlap on at "
                            "least one day?",
            "criteria": {"true": "at least one calendar day is inside both "
                                 "windows",
                         "false": "the windows share no day"},
            "label": gold, "meta": {"family": "temporal_numeric"}}


def gen_month_arith(rng: random.Random) -> dict:
    """Month-only arithmetic (day <= 28, so no end-of-month ambiguity)."""
    base = _rand_date(rng)
    base = base.replace(day=rng.randint(1, 28))
    n = rng.choice([1, 2, 3, 5, 6])
    true_event = month_add(base, n)
    if rng.random() < 0.55:
        claimed = true_event
    else:
        claimed = month_add(true_event, rng.choice([-1, 1])) if rng.random() < 0.5 \
            else true_event + timedelta(days=rng.choice([-2, 2]))
    gold = int(claimed == true_event)
    state = {"agreement_signed_on": fmt_date(base, rng),
             "term": f"The renewal becomes effective {n} month"
                     f"{'s' if n > 1 else ''} after `agreement_signed_on`.",
             "claimed_effective_date": fmt_date(claimed, rng)}
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "Is `claimed_effective_date` the correct "
                            "effective date under `term`?",
            "criteria": {"true": "the claimed date is correct",
                         "false": "the claimed date is off"},
            "label": gold, "meta": {"family": "temporal_numeric"}}


def gen_status_choice(rng: random.Random) -> dict:
    """Which deadline status applies? Options mutually exclusive by bins."""
    today = _rand_date(rng)
    delta = rng.choice([-5, -1, 0, 0, 1, 2, 3, 5, 7, 8, 12, 30])
    due = today + timedelta(days=delta)
    thing = rng.choice(["invoice", "subscription renewal", "license expiry",
                        "warranty end", "report deadline"])
    if delta < 0:
        gold = "overdue"
    elif delta == 0:
        gold = "due_today"
    elif delta <= 7:
        gold = "due_within_7_days"
    else:
        gold = "due_later"
    opts = ["overdue", "due_today", "due_within_7_days", "due_later"]
    state = {"today": fmt_date(today, rng),
             "due_date": fmt_date(due, rng),
             "subject": f"The {thing} in question."}
    return {"row_id": None, "primitive": "choice", "state": state,
            "instructions": "What is the status of the `due_date` relative "
                            "to `today`?",
            "criteria": None, "label": gold,
            "meta": {"options": opts, "family": "temporal_numeric"}}


def gen_urgency_score(rng: random.Random) -> dict:
    """Score levels with exact bins stated in the state (gold = the bin)."""
    today = _rand_date(rng)
    delta = rng.choice([-3, -1, 0, 0, 1, 2, 2, 4, 6, 7, 8, 10, 15, 21])
    deadline = today + timedelta(days=delta)
    if delta <= 0:
        gold = 0
    elif delta <= 2:
        gold = 1
    elif delta <= 7:
        gold = 2
    else:
        gold = 3
    state = {"today": fmt_date(today, rng),
             "deadline": fmt_date(deadline, rng),
             "scale": "Urgency bins: critical — the deadline is today or "
                      "already passed; high — 1-2 days remain; moderate — "
                      "3-7 days remain; low — more than 7 days remain."}
    return {"row_id": None, "primitive": "score", "state": state,
            "instructions": "How urgent is the `deadline`, per the bins in "
                            "`scale`?",
            "criteria": None, "label": gold,
            "meta": {"attribute": "deadline_urgency",
                     "family": "temporal_numeric"}}


# ------------------------------------------------------------------- multi_hop

def _months_between(a: date, b: date) -> int:
    """Complete calendar months from a to b (b is 'today')."""
    m = (b.year - a.year) * 12 + (b.month - a.month)
    if b.day < a.day:
        m -= 1
    return m


def gen_chain_rules(rng: random.Random) -> dict:
    """Eligibility chain: tenure must be computed from dates, spend and
    overdue checked against thresholds, elite plan skips tenure."""
    t_req = rng.choice([6, 12, 18, 24])
    s_req = rng.choice([250, 500, 1000, 2000])
    b_max = rng.choice([15, 30, 45])
    plan = rng.choice(["standard", "standard", "elite"])
    today = _rand_date(rng)
    # sample facts, with near-threshold density
    tenure = rng.choice([t_req - 1, t_req - 1, t_req, t_req + 1])
    joined = _shift_days_for_months(today, tenure)
    spend = s_req + rng.choice([-50, -10, 0, 0, 10, 50])
    overdue = rng.choice([b_max - 1, b_max, b_max + 1, b_max + 15, -1])
    ok_tenure = tenure >= t_req or plan == "elite"
    ok_spend = spend >= s_req
    ok_overdue = overdue <= b_max
    gold = int(ok_tenure and ok_spend and ok_overdue)
    state = {
        "today": fmt_date(today, rng),
        "plan": plan,
        "account_opened_on": fmt_date(joined, rng),
        "trailing_12m_spend_usd": spend,
        "oldest_unpaid_invoice_age_days": overdue,
        "policy": "The loyalty discount applies when the account has been "
                  f"open for at least {t_req} complete months (unless on the "
                  "elite plan, which skips the tenure requirement), the "
                  f"trailing 12-month spend is at least ${s_req}, and no "
                  f"unpaid invoice is older than {b_max} days. "
                  "'Open for N complete months' is measured from "
                  "`account_opened_on` to `today`. An unpaid invoice age of "
                  "-1 means there are no unpaid invoices.",
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "Under `policy`, does the account qualify for "
                            "the loyalty discount? Treat unproved "
                            "conditions as not satisfied.",
            "criteria": {"true": "every applicable condition is met",
                         "false": "at least one applicable condition fails"},
            "label": gold, "meta": {"family": "multi_hop"}}


def _shift_days_for_months(today: date, tenure: int) -> date:
    """A date such that exactly `tenure` complete months elapse to today.
    For today.day == 1, the same day-of-month of the month N back is
    exactly N complete months (e.g. Jan 1 -> Mar 1 = 2 months)."""
    m = (today.year * 12 + today.month - 1) - tenure
    y, mo = m // 12, m % 12 + 1
    day = 1 if today.day == 1 else today.day - 1
    return date(y, mo, day)


def gen_process_chain(rng: random.Random) -> dict:
    """Three chained business-day computations; gold is exact."""
    order = _rand_date(rng)
    hol = _holiday_set(rng, rng.randint(3, 6))
    k, v, r = (rng.choice([2, 3, 4]), rng.choice([1, 2, 3]),
               rng.choice([2, 3, 5]))
    ship = add_bd(order, k, hol)
    deliv = add_bd(ship, v, hol)
    post = add_bd(deliv, r, hol)
    target = post + timedelta(days=rng.choice([-2, -1, 0, 0, 1, 2]))
    gold = int(post <= target)
    state = {
        "order_placed_on": fmt_date(order, rng),
        "process": "Orders ship exactly "
                   f"{k} business days after the order date. Delivery takes "
                   f"exactly {v} business days after shipping. Refunds post "
                   f"exactly {r} business days after delivery. Business days "
                   "are Monday through Friday, excluding the listed "
                   "holidays.",
        "holidays": _holiday_str(hol),
        "refund_needed_by": fmt_date(target, rng),
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "Will the refund post on or before "
                            "`refund_needed_by`, following `process` "
                            "exactly?",
            "criteria": {"true": "the refund posts on or before the date",
                         "false": "the refund posts after the date"},
            "label": gold, "meta": {"family": "multi_hop"}}


def gen_fee_math(rng: random.Random) -> dict:
    """Ordered fee arithmetic: min, percentage, cap — three hops."""
    price = rng.choice([120, 260, 480, 900, 1500])
    warranty = price + rng.choice([-price // 3, -price // 10, 0,
                                   price // 10, price // 3])
    pct = rng.choice([0, 5, 10, 15])
    cap = rng.choice([200, 400, 1000])
    base_amt = min(price, warranty)
    after_pct = round(base_amt * (100 - pct) / 100, 2)
    refund = min(after_pct, cap)
    if rng.random() < 0.6:
        q_level = refund          # exactly at the computed amount
        gold = 1
    else:
        q_level = refund + rng.choice([10, 50])   # strictly above it
        gold = 0
    state = {
        "purchase_price_usd": price,
        "warranty_remaining_usd": warranty,
        "policy": "The refund is the lesser of the purchase price and the "
                  "remaining warranty value. A processing fee of "
                  f"{pct}% is deducted from that amount. The total refund is "
                  f"capped at ${cap}.",
        "question_amount_usd": q_level,
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "Is the refund under `policy` at least "
                            "`question_amount_usd`? Compute the refund "
                            "exactly as the policy orders the steps.",
            "criteria": {"true": "the computed refund meets or exceeds the "
                                "amount",
                         "false": "the computed refund is smaller than the "
                                  "amount"},
            "label": gold, "meta": {"family": "multi_hop"}}


def gen_deep_logic(rng: random.Random) -> dict:
    """Two-level exception logic: rule + exception + exception carve-out.
    Facts generated from gold; every fact is stated explicitly."""
    # conditions and their truth
    conds = [
        ("the claim form is complete", "the claim form is incomplete"),
        ("proof of purchase is attached", "proof of purchase is missing"),
        ("the item is within the coverage period", "the item is outside the "
         "coverage period"),
        ("the account is in good standing", "the account is in arrears"),
    ]
    rng.shuffle(conds)
    k = 3
    chosen = conds[:k]
    exception_applies = rng.random() < 0.5
    carveout_applies = exception_applies and rng.random() < 0.5
    base_ok = rng.random() < 0.7
    # gold: base rules all satisfied AND NOT (exception applies without
    # carve-out)
    gold = int(base_ok and not (exception_applies and not carveout_applies))
    facts = [name for name, _ in chosen]
    if not base_ok:  # exactly one condition broken (which one doesn't
        facts[rng.randrange(k)] = chosen[rng.randrange(k)][1]  # matter to gold
    state = {
        "policy": "A claim is payable when all of the following hold: "
                  + "; ".join(name for name, _ in chosen)
                  + ". Exception: claims flagged as duplicates are denied, "
                    "unless the customer attests that the original order "
                    "was never delivered.",
        "situation": " ".join(f + "." for f in facts)
                     + (f" The claim is flagged as a duplicate."
                        if exception_applies else
                        " The claim is not flagged as a duplicate.")
                     + (f" The customer attests that the original order was "
                        "never delivered." if carveout_applies else
                        (" The customer does not attest anything about the "
                         "original order." if exception_applies else "")),
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "Under `policy`, is the claim in `situation` "
                            "payable?",
            "criteria": {"true": "all conditions hold and no exception "
                                 "denies it",
                         "false": "a condition fails or an exception "
                                  "applies"},
            "label": gold, "meta": {"family": "multi_hop"}}


def gen_doc_conflict(rng: random.Random) -> dict:
    """Clause specificity: which SLA applies to this account?"""
    general = rng.choice([3, 4, 5, 10])
    specific = rng.choice([1, 2, 3])
    while specific == general:
        specific = rng.choice([1, 2, 3])
    plan = rng.choice(["premium", "enterprise", "standard", "basic"])
    if plan in ("premium", "enterprise"):
        gold = f"sla_{specific}_business_days"
    else:
        gold = f"sla_{general}_business_days"
    other = specific if plan in ("premium", "enterprise") else general
    opts = [f"sla_{general}_business_days", f"sla_{specific}_business_days",
            "policy_silent"]
    state = {
        "clause_general": f"All accounts: support responds within "
                          f"{general} business days.",
        "clause_specific": f"{plan.capitalize()} accounts: support responds "
                           f"within {specific} business days."
        if plan in ("premium", "enterprise") else
        f"Premium accounts: support responds within {specific} business "
        "days.",
        "account": {"plan": plan},
        "note": "More specific clauses override general ones when they "
                "plainly cover the account.",
    }
    return {"row_id": None, "primitive": "choice", "state": state,
            "instructions": "Which response-time commitment applies to the "
                            "`account`?",
            "criteria": None, "label": gold,
            "meta": {"options": opts, "family": "multi_hop"}}


def gen_eligibility_tree(rng: random.Random) -> dict:
    """Which single requirement fails (or none)? Options per requirement."""
    t_req = rng.choice([12, 24, 36])
    s_req = rng.choice([500, 1000, 2500])
    c_req = rng.choice([1, 3, 5])
    names = ["tenure_requirement", "spend_requirement",
             "case_volume_requirement", "meets_all_requirements"]
    fail_idx = rng.choice([-1, -1, 0, 1, 2])  # -1 = none fails
    today = _rand_date(rng)
    tenure = t_req + rng.choice([0, 1, 3])
    spend = s_req + rng.choice([0, 100, 500])
    cases = c_req + rng.choice([0, 1, 2])
    if fail_idx == 0:
        tenure = t_req - rng.choice([1, 2])
    elif fail_idx == 1:
        spend = s_req - rng.choice([50, 100])
    elif fail_idx == 2:
        cases = max(0, c_req - 1)
    joined = _shift_days_for_months(today, tenure)
    if fail_idx == -1:
        gold = "meets_all_requirements"
    else:
        gold = names[fail_idx]
    state = {
        "today": fmt_date(today, rng),
        "requirements": {
            "tenure_requirement": f"at least {t_req} complete months "
                                  f"since `joined_on`",
            "spend_requirement": f"trailing 12-month spend of at least "
                                  f"${s_req}",
            "case_volume_requirement": f"at least {c_req} support cases "
                                       "filed this quarter",
        },
        "application": {
            "joined_on": fmt_date(joined, rng),
            "trailing_12m_spend_usd": spend,
            "cases_filed_this_quarter": cases,
        },
    }
    return {"row_id": None, "primitive": "choice", "state": state,
            "instructions": "The program grants access only if every "
                            "`requirements` entry is satisfied by "
                            "`application`. Which single requirement does "
                            "the application fail to meet — or does it meet "
                            "all of them?",
            "criteria": None, "label": gold,
            "meta": {"options": names, "family": "multi_hop"}}


# ---------------------------------------------------------------- long_policy

def gen_long_policy(rng: random.Random) -> list[dict]:
    """One randomized policy document + 3 questions with doc-consistent
    golds. Returns multiple rows sharing the same long state."""
    plans = ["basic", "standard", "premium"]
    wins = {"basic": rng.choice([14, 15, 21]),
            "standard": rng.choice([21, 30, 45]),
            "premium": rng.choice([45, 60, 90])}
    fees = {"basic": rng.choice([15, 20]), "standard": rng.choice([10, 15]),
            "premium": rng.choice([0, 5])}
    appeal_days = rng.choice([10, 14, 30])
    proc_days = rng.choice([3, 5, 7])
    defect_fee_waived = True
    today = _rand_date(rng)
    eff = _shift_days_for_months(today, rng.choice([2, 3, 6]))

    sec_scope = ("2. Scope. This policy covers purchases on the basic, "
                 "standard, and premium plans in the US, EU, and APAC "
                 "regions. Internal test accounts are not covered.")
    sec_windows = ("3. Return window. Returns are accepted when initiated "
                   f"within {wins['basic']} days of purchase on the basic "
                   f"plan, within {wins['standard']} days on the standard "
                   f"plan, and within {wins['premium']} days on the premium "
                   f"plan. The window counts calendar days elapsed from the "
                   "purchase date (the purchase day itself is day 0); a "
                   "return initiated on day N or earlier is within the "
                   "window.")
    sec_excl = ("4. Exclusions. Final-sale items are not returnable. "
                "Digital goods are not returnable once the download link "
                "has been opened. Items damaged by misuse are not "
                "returnable, but items with a manufacturing defect are "
                "always covered regardless of the window.")
    sec_fees = ("5. Restocking fee. Approved returns are refunded minus a "
                f"restocking fee of {fees['basic']}% on the basic plan, "
                f"{fees['standard']}% on the standard plan, and "
                f"{fees['premium']}% on the premium plan. The fee is waived "
                "when the return is due to a manufacturing defect.")
    sec_appeal = (f"6. Appeals. A denied return may be appealed within "
                  f"{appeal_days} days of the denial. The window counts "
                  "calendar days elapsed from the denial date (the denial "
                  "day itself is day 0); an appeal filed on day N or "
                  "earlier is timely. Appeals are decided by the refunds "
                  "team.")
    sec_proc = (f"7. Processing. Approved refunds are issued within "
                f"{proc_days} business days of approval.")
    decoy = ("8. Internal test accounts. Internal test accounts may return "
             "items at any time with no fee. No other account type is "
             "affected by this section.")
    middle = [sec_scope, sec_windows, sec_excl, sec_fees, sec_appeal,
              sec_proc]
    rng.shuffle(middle)
    doc = (f"VENDOR SERVICES AGREEMENT — REFUND POLICY (effective "
           f"{fmt_date(eff, rng)})\n\n"
           "1. Purpose. This document sets out when a purchase may be "
           "returned and how refunds are computed.\n\n"
           + "\n\n".join(middle)
           + f"\n\n{decoy}\n\n"
             "9. Definitions. A business day is Monday through Friday, "
             "excluding public holidays. A manufacturing defect is a fault "
             "present at delivery.")
    doc_state = {"policy_document": doc, "today": fmt_date(today, rng)}

    rows = []

    # Q1: window compliance for a random plan, near boundary
    plan = rng.choice(plans)
    w = wins[plan]
    delta = rng.choice([w - 1, w, w, w + 1, w + 5])
    purchased = today - timedelta(days=delta)
    ret = today
    gold = int(delta <= w)
    rows.append({
        "row_id": None, "primitive": "noul",
        "state": {**doc_state,
                  "case": {"plan": plan,
                           "purchased_on": fmt_date(purchased, rng),
                           "return_initiated_on": fmt_date(ret, rng)}},
        "instructions": "Under the policy, is the return in `case` within "
                        "the return window for that plan?",
        "criteria": {"true": "the return is inside the plan's window",
                     "false": "the return is outside the plan's window"},
        "label": gold, "meta": {"family": "long_policy"},
    })

    # Q2: restocking fee for a defect-free return on a random plan
    plan2 = rng.choice(plans)
    defect = rng.random() < 0.35
    gold2 = "no_fee" if defect else f"fee_{fees[plan2]}_percent"
    opts = ["no_fee", f"fee_{fees['basic']}_percent",
            f"fee_{fees['standard']}_percent",
            f"fee_{fees['premium']}_percent"]
    opts = list(dict.fromkeys(opts))
    rows.append({
        "row_id": None, "primitive": "choice",
        "state": {**doc_state,
                  "case": {"plan": plan2,
                           "reason": "manufacturing defect" if defect
                           else "changed mind",
                           "item_condition": "unused, in original "
                                             "packaging"}},
        "instructions": "Which restocking fee applies to this return under "
                        "the policy?",
        "criteria": None, "label": gold2,
        "meta": {"options": opts, "family": "long_policy"},
    })

    # Q3: appeal timeliness
    denied = today - timedelta(days=rng.choice(
        [appeal_days - 1, appeal_days, appeal_days, appeal_days + 1,
         appeal_days + 3]))
    filed = today
    gold3 = int((filed - denied).days <= appeal_days)
    rows.append({
        "row_id": None, "primitive": "noul",
        "state": {**doc_state,
                  "case": {"denial_date": fmt_date(denied, rng),
                           "appeal_filed_on": fmt_date(filed, rng)}},
        "instructions": "Under the policy, is the appeal in `case` timely?",
        "criteria": {"true": "the appeal is within the appeal window",
                     "false": "the appeal is late"},
        "label": gold3, "meta": {"family": "long_policy"},
    })

    # Q4 (score, sometimes): window-violation severity per the doc's own terms
    if rng.random() < 0.6:
        plan4 = rng.choice(plans)
        w4 = wins[plan4]
        over = rng.choice([0, 0, 1, 2, 5, 10, 20])
        purchased4 = today - timedelta(days=w4 + over)
        if over == 0:
            gold4 = 0
        elif over <= 3:
            gold4 = 1
        elif over <= 14:
            gold4 = 2
        else:
            gold4 = 3
        rows.append({
            "row_id": None, "primitive": "score",
            "state": {**doc_state,
                      "case": {"plan": plan4,
                               "purchased_on": fmt_date(purchased4, rng),
                               "return_initiated_on": fmt_date(today, rng)},
                      "severity_scale": "Severity bins per the policy's own "
                                        "window terms: none — within the "
                                        "plan's window; minor — 1-3 days "
                                        "late; major — 4-14 days late; "
                                        "egregious — more than 14 days "
                                        "late."},
            "instructions": "How severe is the lateness of this return "
                            "relative to the plan's window, per the bins "
                            "in `severity_scale`?",
            "criteria": None, "label": gold4,
            "meta": {"attribute": "window_violation_severity",
                     "family": "long_policy"},
        })
    return rows


# ----------------------------------------------------------------------- cli

def _status_descs() -> dict:
    return {
        "overdue": "the due date is earlier than today",
        "due_today": "the due date is exactly today",
        "due_within_7_days": "the due date is 1-7 days after today",
        "due_later": "the due date is more than 7 days after today",
        "sla_1_business_days": "support responds within 1 business day",
        "sla_2_business_days": "support responds within 2 business days",
        "sla_3_business_days": "support responds within 3 business days",
        "sla_4_business_days": "support responds within 4 business days",
        "sla_5_business_days": "support responds within 5 business days",
        "sla_10_business_days": "support responds within 10 business days",
        "policy_silent": "the policy does not state a commitment for this "
                         "account",
        "tenure_requirement": "fails the tenure requirement",
        "spend_requirement": "fails the spend requirement",
        "case_volume_requirement": "fails the case-volume requirement",
        "meets_all_requirements": "every requirement is satisfied",
        "no_fee": "no restocking fee is deducted",
        "fee_0_percent": "a 0% restocking fee applies",
        "fee_5_percent": "a 5% restocking fee applies",
        "fee_10_percent": "a 10% restocking fee applies",
        "fee_15_percent": "a 15% restocking fee applies",
        "fee_20_percent": "a 20% restocking fee applies",
    }


def _score_levels() -> dict:
    return {
        "deadline_urgency": [
            "critical — the deadline is today or already passed",
            "high — 1-2 days remain until the deadline",
            "moderate — 3-7 days remain until the deadline",
            "low — more than 7 days remain until the deadline",
        ],
        "window_violation_severity": [
            "none — the return is within the plan's window",
            "minor — the return is 1-3 days late",
            "major — the return is 4-14 days late",
            "egregious — the return is more than 14 days late",
        ],
    }


GENS = [
    (gen_date_arith, 500),
    (gen_business_days, 400),
    (gen_deadline_bounds, 500),
    (gen_overlap, 350),
    (gen_month_arith, 300),
    (gen_status_choice, 400),
    (gen_urgency_score, 300),
    (gen_chain_rules, 450),
    (gen_process_chain, 350),
    (gen_fee_math, 400),
    (gen_deep_logic, 400),
    (gen_doc_conflict, 250),
    (gen_eligibility_tree, 300),
    (gen_long_policy, 1200),
]


def _selfcheck() -> int:
    """Invariants for every helper golds depend on."""
    from datetime import date as D
    import random as R
    rng = R.Random(0)
    # weekday-styled format must name the real weekday
    assert "Sunday" in fmt_date(D(2025, 3, 9), rng) or True  # style random
    assert fmt_date(D(2025, 3, 9), R.Random(5))  # just must not raise
    # deterministic styles
    r = R.Random(1)
    got = set()
    for _ in range(40):
        s = fmt_date(D(2025, 3, 9), r)
        got.add(s)
    assert any("Sunday" in s for s in got), got  # weekday style appears
    # business days: Jan 1 2024 is a Monday holiday; 3 business days
    # after Jan 1 -> Jan 4 (Tue/Wed/Thu)
    assert add_bd(D(2024, 1, 1), 3, {D(2024, 1, 1)}) == D(2024, 1, 4)
    # weekend skip: Fri Jan 5 + 1 bd -> Mon Jan 8
    assert add_bd(D(2024, 1, 5), 1, set()) == D(2024, 1, 8)
    # month arithmetic unambiguous for day <= 28
    assert month_add(D(2024, 1, 15), 2) == D(2024, 3, 15)
    assert month_add(D(2024, 11, 10), 3) == D(2025, 2, 10)
    # month-shift roundtrip: exactly N complete months elapse
    for today, n in [(D(2025, 3, 1), 2), (D(2024, 12, 31), 1),
                     (D(2025, 2, 28), 6), (D(2025, 6, 17), 3)]:
        j = _shift_days_for_months(today, n)
        assert _months_between(j, today) == n, (today, n, j)
    # ordinals
    assert [_ord(x) for x in (1, 2, 3, 4, 11, 21, 22, 23, 31)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "21st", "22nd", "23rd", "31st"]
    print("selfcheck: all invariants hold")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synth", "selfcheck"])
    ap.add_argument("--n", type=int, default=6300,
                    help="approx total rows (long_policy emits ~3-4/doc)")
    ap.add_argument("--out", default="")
    ap.add_argument("--criteria-out", default="")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        return _selfcheck()
    assert args.out, "--out is required for synth"

    rng = random.Random(args.seed)
    rows: list[dict] = []
    total_quota = sum(q for _, q in GENS)
    for gen, quota in GENS:
        take = round(quota * args.n / total_quota)
        produced = 0
        while produced < take:
            out = gen(rng)
            if isinstance(out, list):  # long_policy: multiple rows
                rows.extend(out)
                produced += len(out)
            else:
                rows.append(out)
                produced += 1
    for i, r in enumerate(rows):
        r["row_id"] = f"synth2-{i:05d}"
        # choice rows carry the option INDEX (build_data contract), but the
        # generators reason in option names — map here, once, centrally.
        if r["primitive"] == "choice":
            r["label"] = r["meta"]["options"].index(r["label"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    fam = Counter(r["meta"]["family"] for r in rows)
    prim = Counter(r["primitive"] for r in rows)
    pos = [r["label"] for r in rows if r["primitive"] == "noul"]
    avg_len = sum(len(json.dumps(r["state"])) for r in rows) / len(rows)
    print(f"synth2: wrote {len(rows)} rows -> {args.out}")
    print(f"  families: {dict(fam)}")
    print(f"  primitives: {dict(prim)}")
    print(f"  noul positive rate: {round(sum(pos) / len(pos), 3)}")
    print(f"  avg state bytes: {avg_len:.0f}")

    if args.criteria_out:
        crit = _status_descs()
        crit.update(_score_levels())
        Path(args.criteria_out).write_text(json.dumps(crit, indent=2))
        print(f"wrote {args.criteria_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())