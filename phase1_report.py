"""Phase-1 report: blend sensitivity + full composites for all variants.

Blend weight sensitivity is a diagnostic (the pre-registered choice is the
mixed-val-fitted hybrid; the table shows how the variants trade off). The
composites use the official v1.3 axis math; Speed/Cost are carried over from
the measured run (latency identical for all readout variants).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "jevbench"))

from jevbench.composite_v13 import (  # noqa: E402
    TIER_CHANCES,
    calibration as calib_fn,
    chance_corrected_accuracy,
    cost as cost_fn,
    intelligence,
    jevbench_score,
    speed as speed_fn,
)
from jevbench.metrics import ece_top_label  # noqa: E402
from jevbench.scoring import argmax_label, score_task  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

BENCH = Path(__file__).parent / "jevbench"
P50, P95 = 0.38, 9.48          # measured (contended), carried over
MEAN_TOK = 597                 # measured input tokens/decision
PRICE_M = 0.14                 # stated assumption for the Cost axis


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def load_all():
    import torch

    tasks = load_jsonl_tasks()
    fitted = json.load(open("data_cache/phase1/fit.json"))
    s_verb = torch.load("data_cache/phase1/bench-verb.pt")

    head_recs = {}
    for l in open("docs/bench/kapteeni-v1.4-public-231.jsonl"):
        r = json.loads(l)
        head_recs[r["task_id"]] = r
    return tasks, fitted, s_verb, head_recs


def verb_by_task(passes, s_verb):
    d = defaultdict(dict)
    for p, s in zip(passes, s_verb.tolist()):
        d[p["task_id"]][p["key"]] = s
    return d


def bench_passes():
    from jevbench.adapters.base import build_question
    from kapteeni.serialize import choice_option_pass, noul_pass, score_level_pass

    tasks, out = load_jsonl_tasks(), []
    for t in tasks:
        q = build_question(t)
        if q["type"] == "noul":
            out.append({"task_id": t.id, "type": "noul", "key": None,
                        "text": noul_pass(t.state, q["instructions"], q.get("criteria"))})
        elif q["type"] == "choice":
            for opt, desc in q["criteria"].items():
                out.append({"task_id": t.id, "type": "choice", "key": opt,
                            "text": choice_option_pass(t.state, q["instructions"], opt, desc)})
        else:
            for i, lvl in enumerate(q["criteria"]):
                out.append({"task_id": t.id, "type": "score", "key": i,
                            "text": score_level_pass(t.state, q["instructions"], i,
                                                     len(q["criteria"]), lvl)})
    return tasks, out


def load_jsonl_tasks():
    return (load_jsonl(str(BENCH / "datasets/public/easy.jsonl"))
            + load_jsonl(str(BENCH / "datasets/public/original.jsonl"))
            + load_jsonl(str(BENCH / "datasets/public/hard.jsonl")))


def variant_probs(tasks, verb_scores, head_recs, w, taus):
    """Probability distributions for a given head-verb blend weight w.

    w=1 -> pure head (as recorded), w=0 -> pure verbalizer (taus fit on
    mixed-val). Blending happens in score space: head scores recovered from
    the recorded distribution's log-probs, verb scores as raw yes-no logits.
    """
    out = {}
    for t in tasks:
        qtype = t.question["type"]
        hr = head_recs[t.id]
        hp = hr.get("probs_as_returned") or hr.get("probs")
        sv = verb_scores.get(t.id, {})
        if qtype == "noul":
            s_v = [sv[None] / taus["noul"]]
            py = hp["yes"]
            s_h = [math.log(max(py, 1e-9) / max(1 - py, 1e-9))]
            s = [w * s_h[0] + (1 - w) * s_v[0]]
            p = _sigmoid(s[0])
            out[t.id] = {"yes": p, "no": 1 - p}
        elif qtype == "choice":
            keys = list(hp)
            s_v = [sv[k] / taus["choice"] for k in keys]
            s_h = [math.log(max(hp[k], 1e-9)) for k in keys]
            s = [w * sh + (1 - w) * svv for sh, svv in zip(s_h, s_v)]
            p = _softmax(s)
            out[t.id] = {k: p[i] for i, k in enumerate(keys)}
        else:
            keys = sorted(hp, key=lambda k: int(k))
            s_v = [sv[int(k)] / taus["score"] for k in keys]
            s_h = [math.log(max(hp[k], 1e-9)) for k in keys]
            s = [w * sh + (1 - w) * svv for sh, svv in zip(s_h, s_v)]
            p = _softmax(s)
            out[t.id] = {k: p[i] for i, k in enumerate(keys)}
    return out


def score_variant(tasks, probs_by_id, label):
    tiers = {"easy": [], "standard": [], "hard": []}
    pairs, n_corr = [], 0
    for t in tasks:
        tier = ("easy" if t.id.startswith("easy")
                else "hard" if t.id.startswith("hard") else "standard")
        sc = score_task(probs_by_id[t.id], t)
        tiers[tier].append(sc["correct"] if sc["valid"] else False)
        if sc["valid"]:
            n_corr += bool(sc["correct"])
            p = sc.get("probs", probs_by_id[t.id])
            pairs.append((max(p.values()), argmax_label(p) == str(t.expected)))
    accs = {k: sum(v) / len(v) for k, v in tiers.items()}
    intel = intelligence(accs)
    ece = ece_top_label(pairs)["ece"]
    cal = calib_fn(ece)
    print(f"{label:<28} easy {accs['easy']:.3f} std {accs['standard']:.3f} "
          f"hard {accs['hard']:.3f} | I {intel:>5.1f} | ECE {ece:.3f} -> C {cal:.1f} "
          f"| pub acc {n_corr/len(tasks):.3f}")
    return intel, cal, accs, ece


def main():
    tasks, fitted, s_verb, head_recs = load_all()
    tasks2, passes = bench_passes()
    assert len(tasks2) == len(tasks)
    verb = verb_by_task(passes, s_verb)

    spd = speed_fn(P50, P95, "demo")
    c = cost_fn(MEAN_TOK * 1000 * PRICE_M / 1e6)
    print(f"Speed {spd:.1f} (p50 {P50}s p95 {P95}s, x2-adjusted) | "
          f"Cost {c:.1f} (${MEAN_TOK} tok/dec x ${PRICE_M}/M)\n")

    rows = []
    taus_v = {qt: fitted[qt]["verb_tau"] for qt in ("noul", "choice", "score")}
    for w in [1.0, 0.9, 0.7, 0.5, 0.25, 0.0]:
        probs = variant_probs(tasks, verb, head_recs, w, taus_v)
        label = ("head-only (v0)" if w == 1.0
                 else "verbalizer-only" if w == 0 else f"blend w={w}")
        intel, cal, accs, ece = score_variant(tasks, probs, label)
        axes = {"intelligence": intel, "calibration": cal, "speed": spd, "cost": c}
        sc = jevbench_score(axes)
        rows.append({"label": label, "w": w, "axes": axes, "score": sc,
                     "tiers": accs, "ece": ece})
        print(f"{'':28} -> JevBench-style score {sc:.1f}\n")

    board = json.load(open(BENCH / "results/v1.4/jevbench-v1.4-results.json"))
    ranked = [s for s in board["systems"] if s.get("jevbench_score") is not None]
    for r in rows:
        better = sum(1 for s in ranked if s["jevbench_score"] > r["score"])
        print(f"{r['label']:<28} score {r['score']:.1f}  -> ~#{better+1}/73")

    Path("docs/JEVBENCH-PHASE1.json").write_text(json.dumps(
        {"rows": [{k: v for k, v in r.items()} for r in rows],
         "speed": spd, "cost": c, "fit": fitted}, indent=2))
    print("\nwrote docs/JEVBENCH-PHASE1.json")


if __name__ == "__main__":
    main()