"""Refit the blend on the DEPLOYED (shared-prefix KV) path.

The KV path's verbalizer logits drift systematically (~-0.25) from the
full-forward path (bf16 kernel divergence between attention shapes), so the
Phase-1 constants (fit on full-path scores) mis-calibrate the deployed
system (deployed ECE 0.209 vs 0.065 estimated). This recomputes the
verbalizer scores for the same mixed-domain val items via
extract_shared_prefix and refits:

  noul:   p = sigmoid((w * logit(p_head) + (1-w) * (s_verb + b) / tau))
  choice: softmax over blended log-prob scores
  score:  softmax over blended log-prob scores

(w, tau, b) fit by Brier on mixed-val gold — no JevBench data. Writes
data_cache/phase1/fit_kv.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "jevbench"))

import phase1  # noqa: E402
from kapteeni.backbone import extract_shared_prefix, load_backbone  # noqa: E402

DC = Path("data_cache/phase1")
OUT = DC / "fit_kv.json"


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def main() -> int:
    items = phase1.build_val_items()
    print(f"mixed-val items: {len(items)}")

    cache_path = DC / "val-verb-kv.pt"
    if cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        per_item = blob["per_item"]
    else:
        model, tok = load_backbone()
        per_item = []
        for k, it in enumerate(items):
            st = it["state_str"]
            sufs = [p["text"][len(st) + 2:] for p in it["passes"]]
            _, sv = extract_shared_prefix(model, tok, st, sufs)
            per_item.append(sv.tolist())
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(items)} items", flush=True)
        torch.save({"per_item": per_item}, cache_path)
        del model

    z_val = phase1.head_scores_val2(items)
    old = json.load(open(DC / "fit.json"))

    # head probabilities with the bundle temperature (the deployed readout
    # blends at log-prob space: s_h = logit/log of p_head — phase1_report
    # formula, which the server implements)
    blob = torch.load("model_cache/kapteeni_v0.pt", weights_only=False)
    T_head = {qt: blob[qt]["T"] for qt in ("noul", "choice", "score")}

    def head_log_scores(qt, z):
        """(n,) or (k,) raw head logits -> log-prob-space scores."""
        if qt == "noul":
            p = _sigmoid(z[0] / T_head["noul"])
            return [math.log(max(p, 1e-9) / max(1 - p, 1e-9))]
        zs = [x / T_head[qt] for x in z]
        p = _softmax(zs)
        return [math.log(max(x, 1e-9)) for x in p]

    s_head = [head_log_scores(it["qtype"], z_val[i])
              for i, it in enumerate(items)]

    # drift check: full-path vs KV-path verb scores
    flat_new = [v for row in per_item for v in row]
    print(f"KV verb scores: mean {sum(flat_new)/len(flat_new):.3f}")

    fits = {}
    for qt in ("noul", "choice", "score"):
        idxs = [i for i, it in enumerate(items) if it["qtype"] == qt]
        best = None
        grid_w = [x / 20 for x in range(21)]
        grid_tau = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0]
        grid_b = ([round(1.0 + 0.25 * i, 2) for i in range(15)]
                  if qt == "noul" else [0.0])
        for w in grid_w:
            for tau in grid_tau:
                for b in grid_b:
                    bs = []
                    for i in idxs:
                        it = items[i]
                        sh, sv = s_head[i], per_item[i]
                        s = [w * h + (1 - w) * ((x + b) / tau)
                             for h, x in zip(sh, sv)]
                        if qt == "noul":
                            p = _sigmoid(s[0])
                            bs.append((p - it["gold"]) ** 2)
                        else:
                            p = _softmax(s)
                            keys = [str(p_["key"]) for p_ in it["passes"]]
                            gold = str(it["gold"])
                            bs.append(sum((p[j] - int(kk == gold)) ** 2
                                          for j, kk in enumerate(keys)))
                    score = sum(bs) / len(bs)
                    if best is None or score < best[0]:
                        best = (score, w, tau, b)
        score, w, tau, b = best
        fits[qt] = {"w": w, "tau": tau, "b": b,
                   "brier": round(score, 4),
                   "prev": {k: old[qt][k] for k in ("w", "tau")}}
        print(f"[fit-kv:{qt}] w={w} tau={tau} b={b} brier={score:.4f} "
              f"(was w={old[qt]['w']} tau={old[qt]['tau']})")

    OUT.write_text(json.dumps(fits, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())