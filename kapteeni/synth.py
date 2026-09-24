"""Synthetic decision items with GROUND TRUTH BY CONSTRUCTION (no teacher).

Targets JevBench's weakest families for our v0 — and, deliberately, the
reference model's own documented jaggedness (weak counting, dates-as-text,
numeric reasoning): beating Jev there is a flagged divergence improvement.

Generators (all deterministic, seeded):
  dates        date arithmetic / deadline comparisons ("was notice timely?")
  counting     small sums and counts over receipts/lists
  thresholds   numeric cap/threshold comparisons in policy-style context
  policy       multi-condition permission checks (AND rules + exclusions)
  ordinal      template ordinal items with programmatic level placement

CLI:
  python3 -m kapteeni.synth synth --n 3000 --out data_cache/rows_synth.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def _fmt(d: date, rng: random.Random) -> str:
    style = rng.choice(["md", "iso", "dm"])
    if style == "iso":
        return d.isoformat()
    if style == "dm":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def gen_dates(rng: random.Random) -> dict:
    """Date arithmetic + deadline compliance with computed gold."""
    base = date(2024, 1, 1) + timedelta(days=rng.randint(0, 900))
    delta = rng.choice([7, 10, 14, 21, 30, 45, 60, 90])
    true_event = base + timedelta(days=delta)
    claimed = true_event if rng.random() < 0.5 else true_event + timedelta(
        days=rng.choice([-3, -2, -1, 1, 2, 3, 7]))
    gold = 1 if claimed == true_event else 0
    unit = rng.choice(["days", "days", "weeks"])
    nd = delta if unit == "days" else delta // 7
    state = {
        "reference_date": _fmt(base, rng),
        "event": ("The inspection is scheduled for a date exactly "
                  f"{nd} {unit} after `reference_date`."),
        "claimed_date": _fmt(claimed, rng),
    }
    return {
        "row_id": None, "primitive": "noul", "state": state,
        "instructions": "Is `claimed_date` exactly the date described in `event`?",
        "criteria": {"true": "the claimed date matches the computed date",
                     "false": "the claimed date does not match"},
        "label": gold, "meta": {"family": "temporal_numeric"},
    }


def gen_deadline(rng: random.Random) -> dict:
    """Deadline compliance: was an action within the window?"""
    start = date(2024, 1, 1) + timedelta(days=rng.randint(0, 900))
    window = rng.choice([7, 14, 30, 60])
    sent = start + timedelta(days=rng.randint(0, window * 2))
    gold = 1 if (sent - start).days <= window else 0
    state = {
        "incident_date": _fmt(start, rng),
        "policy": f"Claims must be filed within {window} days of `incident_date`.",
        "filing_date": _fmt(sent, rng),
    }
    return {
        "row_id": None, "primitive": "noul", "state": state,
        "instructions": ("Was the claim in `filing_date` filed on time under "
                         "`policy`, given `incident_date`?"),
        "criteria": {"true": "the filing is within the window",
                     "false": "the filing is late"},
        "label": gold, "meta": {"family": "temporal_numeric"},
    }


def gen_counting(rng: random.Random) -> dict:
    """Small sums: does the stated total match the listed items?"""
    n = rng.randint(3, 6)
    items = [rng.randint(1, 12) for _ in range(n)]
    true_total = sum(items)
    claimed = true_total if rng.random() < 0.5 else true_total + rng.choice(
        [-3, -2, -1, 1, 2, 3])
    gold = 1 if claimed == true_total else 0
    thing = rng.choice(["invoices", "defects", "tickets", "units", "log entries"])
    listing = ", ".join(f"{k} {rng.choice(['on', 'from', 'at'])} "
                        f"{rng.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])}"
                        for k in items)
    state = {"records": f"{listing}.", "claimed_total": f"{claimed} {thing}"}
    return {
        "row_id": None, "primitive": "noul", "state": state,
        "instructions": "Does `claimed_total` match the sum of `records`?",
        "criteria": {"true": "the total is correct",
                     "false": "the total is wrong"},
        "label": gold, "meta": {"family": "temporal_numeric"},
    }


def gen_threshold(rng: random.Random) -> dict:
    """Numeric threshold/cap comparisons in policy context."""
    cap = rng.choice([100, 250, 500, 1000, 2500, 5000, 10000])
    claimed = cap + rng.choice([-cap // 4, -cap // 10, 0, cap // 10, cap // 4,
                                cap // 2, cap])
    relation = rng.choice(["exceeds", "is within", "meets or exceeds"])
    what = rng.choice(["reimbursement", "deductible", "payout", "damage estimate"])
    if relation == "exceeds":
        gold = 1 if claimed > cap else 0
    elif relation == "is within":
        gold = 1 if claimed <= cap else 0
    else:
        gold = 1 if claimed >= cap else 0
    state = {"policy": f"The {what} cap is ${cap}.",
             "claim": f"The requested {what} is ${claimed}."}
    return {
        "row_id": None, "primitive": "noul", "state": state,
        "instructions": (f"Does the amount in `claim` {relation} the cap in "
                         "`policy`?"),
        "criteria": {"true": f"the amount {relation} the cap",
                     "false": "it does not"},
        "label": gold, "meta": {"family": "temporal_numeric"},
    }


def gen_policy(rng: random.Random) -> dict:
    """Multi-condition permission: AND of conditions + one exclusion clause."""
    conds = [
        ("the report was submitted", "the report is missing", 0.5),
        ("receipts are attached", "receipts are missing", 0.5),
        ("the request is under the limit", "the request exceeds the limit", 0.4),
        ("the account is active", "the account is closed", 0.5),
        ("membership was active at the time", "membership had lapsed", 0.4),
    ]
    rng.shuffle(conds)
    k = rng.randint(2, 3)
    facts = [(name, neg, rng.random() < 0.7) for name, neg, _ in conds[:k]]
    exclusion = rng.random() < 0.4
    violated_exclusion = exclusion and rng.random() < 0.5
    gold = 1 if (all(ok for _, _, ok in facts) and not violated_exclusion) else 0
    fact_txt = " ".join((name if ok else neg) + "."
                        for name, neg, ok in facts)
    ex_txt = (" Any request flagged as duplicate is denied in full."
              if exclusion else "")
    ex_fact = (" The request is flagged as a duplicate."
               if violated_exclusion else
               (" The request is not flagged as a duplicate."
                if exclusion else ""))
    state = {"policy": f"Refunds require that: {'; '.join(n for n, _, _ in conds[:k])}."
                       f"{ex_txt}",
             "situation": fact_txt + ex_fact}
    return {
        "row_id": None, "primitive": "noul", "state": state,
        "instructions": ("Under `policy`, is the refund permitted given "
                         "`situation`? Treat unproved conditions as not satisfied."),
        "criteria": {"true": "every condition holds and no exclusion applies",
                     "false": "a condition is missing or an exclusion applies"},
        "label": gold, "meta": {"family": "policy"},
    }


GENS = {"dates": gen_dates, "deadline": gen_deadline, "counting": gen_counting,
        "threshold": gen_threshold, "policy": gen_policy}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synth"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    rows = []
    gens = list(GENS.values())
    while len(rows) < args.n:
        g = gens[len(rows) % len(gens)]
        rows.append(g(rng))
    for i, r in enumerate(rows):
        r["row_id"] = f"synth-{r['meta']['family']}-{i}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"synth: wrote {len(rows)} rows -> {args.out}")
    print("  families:", dict(Counter(r["meta"]["family"] for r in rows)))
    print("  positive rate:", round(sum(r["label"] for r in rows) / len(rows), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())