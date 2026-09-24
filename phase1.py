"""Phase 1: verbalizer baseline + head/verbalizer hybrid (honest fitting).

Nothing is fitted on JevBench items. The blend weight and post-temperature
are fitted on a MIXED-DOMAIN validation slice built from our own datasets
(BoolQ/FEVER/MNLI noul val, Banking77 val, HelpSteer2 val) — the bench items
are only ever evaluated.

Stages (checkpointed under data_cache/phase1/):
  val-verb.pt    verbalizer scores for mixed-val passes
  fit.json       fitted (w, tau) per primitive + verbalizer-only tau
  bench-verb.pt  verbalizer scores for the 231 bench passes
  results_hybrid.jsonl / results_verbalizer.jsonl  rebuilt bench records

Usage: python3 phase1.py [--stage all|val|fit|bench|score]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "jevbench"))

from kapteeni.build_data import expand_passes, is_val  # noqa: E402
from kapteeni.metrics import brier  # noqa: E402
from kapteeni.serialize import state_text  # noqa: E402
from kapteeni.verbalizer import load_for_verbalizer, verbalizer_scores  # noqa: E402

BENCH = Path(__file__).parent / "jevbench"
DC = Path("data_cache/phase1")
DC.mkdir(parents=True, exist_ok=True)

NOUL_N = {"boolq": 100, "fever": 100, "mnli": 200}
CHOICE_VAL_ROWS = 40
SCORE_VAL_ROWS = 150

# ---------------------------------------------------------------- mixed val


def build_val_items():
    """Items: {row_id, qtype, state_str, passes: [{key, text, target}], gold}."""
    items = []

    def _add(rows, passes, qtype):
        rid2state = {r["row_id"]: state_text(r["state"]) for r in rows}
        by_row = defaultdict(list)
        for p in passes:
            by_row[p["row_id"]].append(p)
        for rid, ps in by_row.items():
            if qtype == "noul":
                gold = float(ps[0]["gold"])
                items.append({"row_id": rid, "qtype": qtype,
                              "state_str": rid2state[rid],
                              "passes": [{"key": None, "text": p["text"],
                                           "target": gold} for p in ps],
                              "gold": gold})
            elif qtype == "choice":
                gold_name = next(p["key"] for p in ps if p["target"] == 1.0)
                items.append({"row_id": rid, "qtype": qtype,
                              "state_str": rid2state[rid],
                              "passes": [{"key": p["key"], "text": p["text"],
                                           "target": p["target"]} for p in ps],
                              "gold": gold_name})
            else:
                gold_lvl = next(p["key"] for p in ps if p["target"] == 1.0)
                items.append({"row_id": rid, "qtype": qtype,
                              "state_str": rid2state[rid],
                              "passes": [{"key": p["key"], "text": p["text"],
                                           "target": p["target"]} for p in ps],
                              "gold": gold_lvl})

    # noul: BoolQ/FEVER val + MNLI (eval-only, never trained)
    for ds, n in NOUL_N.items():
        rows = [json.loads(l) for l in open(f"data_cache/rows_{ds}.jsonl")]
        soft = {}
        if ds != "mnli":
            for l in open(f"data_cache/soft_{ds}.jsonl"):
                r = json.loads(l)
                soft[r["row_id"]] = {"p": r["p"], "weight": r.get("weight", 1.0)}
        if ds == "mnli":
            # all MNLI rows are held-out for the head (eval-only, never trained)
            vals = rows[:n]
        else:
            vals = sorted([r for r in rows if is_val(r["row_id"])],
                          key=lambda r: r["row_id"])[:n]
        _add(vals, expand_passes(vals, {}, soft), "noul")
    # choice: Banking77 val rows (full 77-option sets)
    rows = [json.loads(l) for l in open("data_cache/rows_banking77.jsonl")]
    crit = json.load(open("data_cache/crit_banking77.json"))
    vals = sorted([r for r in rows if is_val(r["row_id"])],
                  key=lambda r: r["row_id"])[:CHOICE_VAL_ROWS]
    _add(vals, expand_passes(vals, crit, {}), "choice")
    # score: HelpSteer2 val rows (5 level passes each)
    rows = [json.loads(l) for l in open("data_cache/rows_helpsteer2.jsonl")]
    crit = json.load(open("data_cache/crit_helpsteer2.json"))
    vals = sorted([r for r in rows if is_val(r["row_id"])],
                  key=lambda r: r["row_id"])[:SCORE_VAL_ROWS]
    _add(vals, expand_passes(vals, crit, {}), "score")
    return items


def compute_verb(model, tok, texts: list[str], out_path: Path,
                 batch_tokens: int = 16384) -> torch.Tensor:
    if out_path.exists():
        return torch.load(out_path)
    s = verbalizer_scores(model, tok, texts, batch_tokens=batch_tokens)
    torch.save(s, out_path)
    return s


# --------------------------------------------------------------------- head


def _idx_key(r, p):
    if r["qtype"] == "noul":
        return (r["row_id"], "noul", None)
    if r["qtype"] == "choice":
        return (r["row_id"], "choice_opt", p["key"])
    return (r["row_id"], "score_lvl", p["key"])


def head_scores_val2(items):
    """(fixed) raw head logits for val items from emb.pt."""
    blob = torch.load("model_cache/kapteeni_v0.pt", weights_only=False)
    from kapteeni.heads import PassMLP
    heads = {}
    for qt in ("noul", "choice", "score"):
        h = PassMLP(blob[qt]["in_dim"])
        h.load_state_dict(blob[qt]["head"])
        h.eval()
        heads[qt] = h
    emb_blob = torch.load("data_cache/emb.pt", weights_only=False)
    h_all = emb_blob["h"].float()
    rec = emb_blob["records"]
    idx = {}
    for i, r in enumerate(rec):
        idx[(r["row_id"], r["kind"], r["key"])] = i
    out = []
    with torch.no_grad():
        for it in items:
            ixs = [idx[_idx_key(it, p)] for p in it["passes"]]
            z = heads[it["qtype"]](h_all[ixs]).tolist()
            out.append(z)
    return out


# ---------------------------------------------------------------------- fit


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def probs_for(qtype, s, tau):
    if qtype == "noul":
        return _sigmoid(s[0] / tau)
    return _softmax([x / tau for x in s])


def fit(items, z_head, s_verb, taus=(0.05, 0.5, 1, 2, 4, 8)):
    """Grid-fit blend w and post-temperature tau per primitive (Brier)."""
    fit_out = {}
    for qt in ("noul", "choice", "score"):
        idxs = [i for i, it in enumerate(items) if it["qtype"] == qt]
        best = None
        for w in [x / 10 for x in range(11)]:
            for tau in taus:
                bs = []
                for i in idxs:
                    it = items[i]
                    s = [w * zh + (1 - w) * sv
                         for zh, sv in zip(z_head[i], s_verb[i])]
                    if qt == "noul":
                        p = probs_for(qt, s, tau)
                        bs.append((p - it["gold"]) ** 2)
                    else:
                        p = _softmax([x / tau for x in s])
                        keys = [str(pp["key"]) for pp in it["passes"]]
                        gold = str(it["gold"])
                        bs.append(sum((p[k] - int(key == gold)) ** 2
                                      for k, key in zip(range(len(keys)), keys)))
                b = sum(bs) / len(bs)
                if best is None or b < best[2]:
                    best = (w, tau, b)
        # verbalizer-only temperature for the same slice
        best_v = None
        for tau in taus:
            bs = []
            for i in idxs:
                it = items[i]
                sv = s_verb[i]
                if qt == "noul":
                    p = _sigmoid(sv[0] / tau)
                    bs.append((p - it["gold"]) ** 2)
                else:
                    p = _softmax([x / tau for x in sv])
                    keys = [str(pp["key"]) for pp in it["passes"]]
                    gold = str(it["gold"])
                    bs.append(sum((p[k] - int(key == gold)) ** 2
                                  for k, key in zip(range(len(keys)), keys)))
            b = sum(bs) / len(bs)
            if best_v is None or b < best_v[1]:
                best_v = (tau, b)
        fit_out[qt] = {"w": best[0], "tau": best[1], "brier": round(best[2], 4),
                       "verb_tau": best_v[0], "verb_brier": round(best_v[1], 4)}
        print(f"[fit:{qt}] w={best[0]} tau={best[1]} brier={best[2]:.4f} | "
              f"verb-only tau={best_v[0]} brier={best_v[1]:.4f}")
    return fit_out


# --------------------------------------------------------------------- bench


def bench_passes():
    """Rebuild the exact pass texts our server used for each bench task."""
    from jevbench.adapters.base import build_question
    from jevbench.tasks import load_jsonl
    from kapteeni.serialize import choice_option_pass, noul_pass, score_level_pass

    tasks = (load_jsonl(str(BENCH / "datasets/public/easy.jsonl"))
             + load_jsonl(str(BENCH / "datasets/public/original.jsonl"))
             + load_jsonl(str(BENCH / "datasets/public/hard.jsonl")))
    out = []
    for t in tasks:
        q = build_question(t)
        state = t.state
        if q["type"] == "noul":
            out.append({"task_id": t.id, "type": "noul", "key": None,
                        "text": noul_pass(state, q["instructions"], q.get("criteria"))})
        elif q["type"] == "choice":
            for opt, desc in q["criteria"].items():
                out.append({"task_id": t.id, "type": "choice", "key": opt,
                            "text": choice_option_pass(state, q["instructions"], opt, desc)})
        else:
            levels = q["criteria"]
            for i, lvl in enumerate(levels):
                out.append({"task_id": t.id, "type": "score", "key": i,
                            "text": score_level_pass(state, q["instructions"], i,
                                                     len(levels), lvl)})
    return tasks, out


def head_probs_bench():
    """Server-recorded head probabilities per bench item (probs_as_returned)."""
    recs = {}
    for l in open("docs/bench/kapteeni-v1.4-public-231.jsonl"):
        r = json.loads(l)
        recs[r["task_id"]] = r
    return recs


def logit_scores_from_probs(qtype, probs):
    """Recover score-space values from the recorded distribution.

    noul:   s = log(p_yes / p_no)
    choice: s_i = log(p_i)   (up to a constant; cancels in softmax)
    score:  s_i = log(p_i)
    """
    if qtype == "noul":
        py = probs["yes"]
        return [math.log(max(py, 1e-9) / max(1 - py, 1e-9))]
    return [math.log(max(v, 1e-9)) for v in probs.values()]


def rebuild_records(tasks, bench_passes_, s_verb, fitted, head_recs):
    """Build result records for verbalizer-only and hybrid variants."""
    from jevbench.scoring import score_task

    verb_by_task = defaultdict(dict)
    for p, s in zip(bench_passes_, s_verb.tolist()):
        verb_by_task[p["task_id"]][p["key"]] = s

    task_by_id = {t.id: t for t in tasks}
    results = {"verbalizer": [], "hybrid": []}
    for tid, t in task_by_id.items():
        qtype = t.question["type"]
        hr = head_recs[tid]
        head_probs = hr.get("probs_as_returned") or hr.get("probs")
        s_v = verb_by_task[tid]

        if qtype == "noul":
            sv = [s_v[None]]
            s_h = logit_scores_from_probs("noul", head_probs)
            f = fitted["noul"]
            p_verb = _sigmoid(sv[0] / f["verb_tau"])
            s_mix = [f["w"] * s_h[0] + (1 - f["w"]) * sv[0]]
            p_hybr = _sigmoid(s_mix[0] / f["tau"])
            probs_v = {"yes": p_verb, "no": 1 - p_verb}
            probs_h = {"yes": p_hybr, "no": 1 - p_hybr}
        else:
            if qtype == "choice":
                keys = list(head_probs)
                sv = [s_v[k] for k in keys]
            else:  # score: string level indices, numeric order; int pass keys
                keys = sorted(head_probs, key=lambda k: int(k))
                sv = [s_v[int(k)] for k in keys]
            s_h = logit_scores_from_probs(qtype, {k: head_probs[k] for k in keys})
            f = fitted["choice" if qtype == "choice" else "score"]
            p_verb = _softmax([x / f["verb_tau"] for x in sv])
            s_mix = [f["w"] * sh + (1 - f["w"]) * svv for sh, svv in zip(s_h, sv)]
            p_hybr = _softmax([x / f["tau"] for x in s_mix])
            probs_v = {k: p_verb[i] for i, k in enumerate(keys)}
            probs_h = {k: p_hybr[i] for i, k in enumerate(keys)}

        for variant, probs in (("verbalizer", probs_v), ("hybrid", probs_h)):
            sc = score_task(probs, t)
            results[variant].append({
                "task_id": tid, "family": t.family, "split": hr.get("split"),
                "ok": True, "valid": sc["valid"], "correct": sc["correct"],
                "predicted": sc["predicted"], "probs": sc.get("probs", probs),
                "probs_as_returned": probs, "latency_s": hr.get("latency_s"),
                "usage": hr.get("usage", {}),
            })
    return results


# -------------------------------------------------------------------- score


def score_variant(records, tasks):
    """Official math: tier accuracies, intelligence, ECE half, placement."""
    from jevbench.composite_v13 import (TIER_CHANCES, chance_corrected_accuracy,
                                       intelligence)
    from jevbench.metrics import ece_top_label
    from jevbench.scoring import argmax_label

    tiers = {"easy": [], "standard": [], "hard": []}
    tier_of = {}
    for t in tasks:
        tier_of[t.id] = ("easy" if t.id.startswith("easy")
                         else "hard" if t.id.startswith("hard") else "standard")
    pairs = []
    n_corr = 0
    for r in records:
        tiers[tier_of[r["task_id"]]].append(r)
        t = next(x for x in tasks if x.id == r["task_id"])
        if r.get("valid"):
            n_corr += bool(r.get("correct"))
            if r.get("probs"):
                p = r["probs"]
                pairs.append((max(p.values()), argmax_label(p) == str(t.expected)))
    accs = {}
    for tier, rs in tiers.items():
        accs[tier] = sum(bool(r.get("correct")) for r in rs if r.get("valid")) / len(rs)
        print(f"  {tier:<9} acc={accs[tier]:.4f} cc={chance_corrected_accuracy(accs[tier], TIER_CHANCES[tier]):.1f}")
    intel = intelligence(accs)
    ece = ece_top_label(pairs)["ece"]
    print(f"  intelligence={intel:.2f}  top-label ECE={ece:.4f}  "
          f"public acc={n_corr/len(records):.4f}")
    return {"tiers": accs, "intelligence": intel, "ece": ece,
            "public_accuracy": n_corr / len(records)}


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "val", "fit", "bench", "score"])
    args = ap.parse_args()

    fit_path = DC / "fit.json"
    if args.stage in ("all", "val", "fit"):
        model, tok = load_for_verbalizer()
        items = build_val_items()
        print(f"mixed-val items: {len(items)} "
              f"({sum(1 for i in items if i['qtype']=='noul')} noul / "
              f"{sum(1 for i in items if i['qtype']=='choice')} choice / "
              f"{sum(1 for i in items if i['qtype']=='score')} score)")
        texts = [p["text"] for it in items for p in it["passes"]]
        s_val = compute_verb(model, tok, texts, DC / "val-verb.pt")
        per_item = []
        off = 0
        for it in items:
            per_item.append(s_val[off: off + len(it["passes"])].tolist())
            off += len(it["passes"])
        z_val = head_scores_val2(items)
        if args.stage in ("all", "fit"):
            fitted = fit(items, z_val, per_item)
            fit_path.write_text(json.dumps(fitted, indent=2))
            print(f"wrote {fit_path}")
        if args.stage in ("val", "fit"):
            return 0

    # bench + score stages
    fitted = json.load(open(fit_path))
    if args.stage in ("all", "bench"):
        model, tok = load_for_verbalizer()
        tasks, passes = bench_passes()
        print(f"bench passes: {len(passes)} for {len(tasks)} tasks")
        texts = [p["text"] for p in passes]
        s_bench = compute_verb(model, tok, texts, DC / "bench-verb.pt")
    else:
        tasks, passes = bench_passes()
        s_bench = torch.load(DC / "bench-verb.pt")

    head_recs = head_probs_bench()
    results = rebuild_records(tasks, passes, s_bench, fitted, head_recs)
    for variant, recs in results.items():
        out = DC / f"results_{variant}.jsonl"
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n=== {variant} ===")
        score_variant(recs, tasks)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())