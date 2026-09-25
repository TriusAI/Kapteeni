"""Score our kapteeni implementation's JevBench public-half run with the OFFICIAL code.

    python3 score_bench.py /tmp/opencode/kapteeni_results.jsonl

Uses jevbench/composite_v13.py (their scoring) on the 231 public items
(easy 48 / standard 72 / hard 111). Caveats vs the official board, stated
plainly in the output:
  - Intelligence: judge tier (146 items) is sealed -> weights renormalize
    over the three public tiers (their intelligence() supports this).
  - Calibration: ECE half only; the board's axis also averages fidelity
    (TVD) against private gold distributions.
  - Speed: measured on OUR hardware (contended Strix Halo), adjusted x2 per
    their self-hosted convention; the board also adds +0.15 s only for their
    own evaluator pods (not our hardware).
  - Cost: hosted list price of the same weights x measured tokens, their
    "est." convention; assumption stated, sensitivity shown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path(__file__).parent / "jevbench"
sys.path.insert(0, str(BENCH))

from jevbench.composite_v13 import calibration as calib_axis  # noqa: E402
from jevbench.composite_v13 import cost as cost_axis_fn  # noqa: E402
from jevbench.composite_v13 import (  # noqa: E402
    TIER_CHANCES,
    chance_corrected_accuracy,
    intelligence,
    jevbench_score,
    speed as speed_axis_fn,
)
from jevbench.metrics import ece_top_label  # noqa: E402
from jevbench.scoring import argmax_label  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

TIERS = {"easy": "datasets/public/easy.jsonl",
         "standard": "datasets/public/original.jsonl",
         "hard": "datasets/public/hard.jsonl"}


def main() -> int:
    root = BENCH
    tasks_by_tier = {}
    for tier, rel in TIERS.items():
        tasks_by_tier[tier] = load_jsonl(str(root / rel))
    id2tier = {t.id: tier for tier, ts in tasks_by_tier.items() for t in ts}
    id2task = {t.id: t for ts in tasks_by_tier.values() for t in ts}

    records = [json.loads(l) for l in open(sys.argv[1])]
    scorable = [r for r in records if r.get("valid")]
    print(f"results: {len(records)} attempted, {len(scorable)} valid, "
          f"{sum(bool(r.get('correct')) for r in scorable)} correct")

    # ---- per-tier accuracy + Intelligence (their code, renormalized weights)
    tiers = {}
    for tier, ts in tasks_by_tier.items():
        ids = {t.id for t in ts if t.expected is not None}
        rs = [r for r in scorable if r["task_id"] in ids]
        acc = sum(bool(r.get("correct")) for r in rs) / len(rs) if rs else None
        tiers[tier] = acc
        cc = chance_corrected_accuracy(acc, TIER_CHANCES[tier]) if acc is not None else None
        print(f"  {tier:<9} n={len(rs):>3}  accuracy={acc:.4f}  "
              f"chance={TIER_CHANCES[tier]:.3f}  chance-corrected={cc:.1f}")

    intel = intelligence(tiers)
    overall = sum(bool(r.get("correct")) for r in scorable) / len(scorable)
    print(f"\npublic accuracy (231 items): {overall:.4f}")
    print(f"Intelligence (3 public tiers, judge sealed -> renormalized): {intel:.2f}")

    # ---- calibration: top-label ECE half only
    pairs = []
    for r in scorable:
        t = id2task[r["task_id"]]
        if r.get("probs"):
            p = r["probs"]
            pairs.append((max(p.values()), argmax_label(p) == str(t.expected)))
    ece = ece_top_label(pairs)["ece"]
    cal = calib_axis(ece)
    print(f"top-label ECE: {ece:.4f} -> Calibration (ECE half only): {cal:.2f}")

    # ---- speed (measured on our hardware; x2 self-hosted adjustment)
    lat = sorted(r["latency_s"] for r in records if r.get("latency_s"))
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    spd_raw = speed_axis_fn(p50, p95, "none")  # no adjustment
    spd = speed_axis_fn(p50, p95, "demo")      # x2 (author hardware convention)
    print(f"\nlatency p50={p50:.2f}s p95={p95:.2f}s (contended Strix Halo)")
    print(f"Speed raw={spd_raw:.1f}  x2-adjusted={spd:.1f} (self-hosted convention)")

    # ---- cost: hosted list price of the same weights x measured tokens
    usage = [r.get("usage", {}) for r in records]
    intok = [u.get("input_tokens") for u in usage if u.get("input_tokens")]
    mean_tok = sum(intok) / len(intok)
    print(f"\nmean input tokens/decision (state+question counted once): {mean_tok:.0f}")
    for price_m in (0.05, 0.14, 0.30):
        per_1k = mean_tok * 1000 * price_m / 1e6
        print(f"  at ${price_m}/M: ${per_1k:.4f}/1k decisions -> Cost {cost_axis_fn(per_1k):.1f}")
    per_1k = mean_tok * 1000 * 0.14 / 1e6
    c = cost_axis_fn(per_1k)

    # ---- composite (their formula), with caveats
    axes = {"intelligence": intel, "calibration": cal, "speed": spd, "cost": c}
    score = jevbench_score(axes)
    print("\naxes:", {k: round(v, 2) for k, v in axes.items()})
    print(f"JevBench-style score (public half, all caveats above): {score:.2f}")
    out = {"axes": axes, "score": score, "tiers": tiers,
           "public_accuracy": overall, "ece": ece, "p50_s": p50, "p95_s": p95,
           "mean_input_tokens": mean_tok,
           "caveats": ["judge tier sealed (weights renormalized)",
                       "calibration = ECE half only (TVD half is private)",
                       "latency measured on contended author hardware, x2-adjusted",
                       "cost = assumption-based estimate"]}
    Path("docs/JEVBENCH.json").write_text(json.dumps(out, indent=2))
    print("\nwrote docs/JEVBENCH.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())