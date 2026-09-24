"""Offline (w, tau, b) landscape on the bench, per primitive.

Two reference deployments are scored alongside the grid:
  - mixed-val fit (refit_kv.py output) — the pre-registered-style config
  - the scan blend (w=0.25, tau=2, b=0) — the Phase-1 public-half-selected one
Grid maxima are PUBLIC-HALF SELECTED and flagged as such; final numbers come
from a real server bench run with the chosen deployment.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "jevbench"))

from jevbench.composite_v13 import (  # noqa: E402
    TIER_CHANCES,
    calibration as calib_fn,
    cost as cost_fn,
    intelligence,
    jevbench_score,
    speed as speed_fn,
)
from jevbench.metrics import ece_top_label  # noqa: E402
from jevbench.scoring import argmax_label, score_task  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

BENCH = Path(__file__).parent / "jevbench"
P50, P95 = 0.45, 2.90          # measured deployed (KV path)
MEAN_TOK, PRICE_M = 597, 0.14  # measured / stated assumption


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def load():
    """Deployed-path ingredients: KV h -> head logits (z) + KV verb scores,
    captured per pass by bench_verb_kv.py (exactly what the server computes)."""
    import torch

    tasks = (load_jsonl(str(BENCH / "datasets/public/easy.jsonl"))
             + load_jsonl(str(BENCH / "datasets/public/original.jsonl"))
             + load_jsonl(str(BENCH / "datasets/public/hard.jsonl")))
    blob = torch.load("data_cache/phase1/bench-verb-kv.pt", weights_only=False)
    ing = blob["by_task"]
    tblob = torch.load("model_cache/kapteeni_v0.pt", weights_only=False)
    T_head = {qt: tblob[qt]["T"] for qt in ("noul", "choice", "score")}
    return tasks, ing, T_head


def blend_probs(t, ing, T_head, params):
    qt = t.question["type"]
    p = params[qt]
    w, tau, b = p["w"], p["tau"], p.get("b", 0.0)
    if qt == "noul":
        z = ing[None]["z"]
        py = _sigmoid(z / T_head["noul"])
        s_h = math.log(max(py, 1e-9) / max(1 - py, 1e-9))
        s = w * s_h + (1 - w) * (ing[None]["sv"] + b) / tau
        pyx = _sigmoid(s)
        return {"yes": pyx, "no": 1 - pyx}
    if qt == "choice":
        keys = list(t.question["criteria"])
        zs = [ing[k]["z"] / T_head["choice"] for k in keys]
        ph = _softmax(zs)
        s = [w * math.log(max(ph[i], 1e-9)) + (1 - w) * (ing[k]["sv"] + b) / tau
             for i, k in enumerate(keys)]
        pp = _softmax(s)
        return {k: pp[i] for i, k in enumerate(keys)}
    keys = list(range(len(t.question["criteria"])))
    zs = [ing[k]["z"] / T_head["score"] for k in keys]
    ph = _softmax(zs)
    s = [w * math.log(max(ph[i], 1e-9)) + (1 - w) * (ing[k]["sv"] + b) / tau
         for i, k in enumerate(keys)]
    pp = _softmax(s)
    return {str(k): pp[i] for i, k in enumerate(keys)}


def full_score(tasks, probs_by_id):
    tiers = {"easy": [], "standard": [], "hard": []}
    pairs = []
    for t in tasks:
        tier = ("easy" if t.id.startswith("easy")
                else "hard" if t.id.startswith("hard") else "standard")
        sc = score_task(probs_by_id[t.id], t)
        tiers[tier].append(sc["correct"] if sc["valid"] else False)
        if sc["valid"] and sc.get("probs"):
            pp = sc["probs"]
            pairs.append((max(pp.values()), argmax_label(pp) == str(t.expected)))
    accs = {k: sum(v) / len(v) for k, v in tiers.items()}
    intel = intelligence(accs)
    ece = ece_top_label(pairs)["ece"]
    spd = speed_fn(P50, P95, "demo")
    c = cost_fn(MEAN_TOK * 1000 * PRICE_M / 1e6)
    axes = {"intelligence": intel, "calibration": calib_fn(ece),
            "speed": spd, "cost": c}
    return accs, intel, ece, jevbench_score(axes), axes


def main() -> int:
    tasks, ing, T_head = load()

    def run(params, label):
        probs_by_id = {
            t.id: blend_probs(t, ing[t.id], T_head, params) for t in tasks
        }
        accs, intel, ece, score, axes = full_score(tasks, probs_by_id)
        print(f"{label:<58} I {intel:>5.1f} ECE {ece:.3f} C {axes['calibration']:.1f} "
              f"-> {score:.1f}")
        return intel, ece, score

    print("=== reference deployments (deployed-path ingredients) ===")
    fit_kv = json.load(open("data_cache/phase1/fit_kv.json"))
    run(fit_kv, "mixed-val fit (w 0.65/0.7/1.0, tau 4/24/0.5, b 4.5)")
    scan = {qt: {"w": 0.25, "tau": 2.0, "b": 0.0}
            for qt in ("noul", "choice", "score")}
    run(scan, "phase-1 scan blend (w=0.25, tau=2, b=0)")
    run({qt: {"w": 1.0, "tau": 2.0, "b": 0.0} for qt in scan}, "head-only")
    run({qt: {"w": 0.0, "tau": 2.0, "b": 0.0} for qt in scan}, "verbalizer-only")

    print("\n=== per-primitive landscape (public-half selected; flagged) ===")
    base = {qt: dict(scan[qt]) for qt in scan}
    for qt in ("noul", "choice", "score"):
        rows = []
        grid_w = [0.0, 0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
        grid_tau = [0.5, 1, 2, 3, 4, 6, 12]
        grid_b = ([0, 0.5, 1, 1.5, 2, 2.5, 3] if qt == "noul" else [0.0])
        for w in grid_w:
            for tau in grid_tau:
                for b in grid_b:
                    params = {k: dict(v) for k, v in base.items()}
                    params[qt] = {"w": w, "tau": tau, "b": b}
                    probs_by_id = {
                        t.id: blend_probs(t, ing[t.id], T_head, params)
                        for t in tasks
                    }
                    _, intel, ece, score, _ = full_score(tasks, probs_by_id)
                    rows.append((score, intel, ece, w, tau, b))
        rows.sort(reverse=True)
        print(f"\n[{qt}] top 8 by composite (others at scan blend):")
        for score, intel, ece, w, tau, b in rows[:8]:
            print(f"  w={w:<5} tau={tau:<5} b={b:<4} -> I {intel:>5.1f} "
                  f"ECE {ece:.3f} score {score:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())