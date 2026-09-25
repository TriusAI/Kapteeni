"""v1.2.1 finalizer: refit temperatures + blend on the DEPLOYMENT-DIVERSE
val set — the existing mixed-domain val items PLUS the synth2 validation
slice — through the FROZEN seed-3 model artifacts (docs/PREREG-V1.2.1.md).

    python3 -m kapteeni.fit_diverse \
        --lora model_cache/kapteeni_p2_s3/adapter \
        --heads model_cache/kapteeni_p2_s3/heads.pt \
        --out model_cache/kapteeni_v1_2_1.pt \
        --fit-out data_cache/phase1/fit_kv_v1_2_1.json

Identical machinery and grids to p2_finalize; the ONLY change is the fit
set. Diagnostics (per-subset accuracy/ECE at the fitted constants) are
reported so the pre-registered gates can be checked without the bench.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import phase1  # noqa: E402
from kapteeni.build_data import expand_passes, is_val  # noqa: E402
from kapteeni.metrics import ece  # noqa: E402
from kapteeni.p2_finalize import _sigmoid, _softmax, fit_temperature, load_merged  # noqa: E402
from kapteeni.serialize import state_text  # noqa: E402


def build_synth2_val_items() -> list[dict]:
    """synth2 val rows -> items in phase1's shape, tagged source=synth2."""
    rows = [json.loads(l) for l in open("data_cache/rows_synth2.jsonl",
                                        encoding="utf-8")]
    rows = [r for r in rows if is_val(r["row_id"])]
    criteria = json.loads(Path("data_cache/synth2_criteria.json").read_text())
    passes = expand_passes(rows, criteria, {})
    by_row = defaultdict(list)
    for p in passes:
        by_row[p["row_id"]].append(p)
    rid2state = {r["row_id"]: state_text(r["state"]) for r in rows}
    items = []
    for rid, ps in by_row.items():
        qt = ps[0]["qtype"]
        if qt == "noul":
            gold = float(ps[0]["gold"])
            keys = [None]
        elif qt == "choice":
            gold = next(p["key"] for p in ps if p["target"] == 1.0)
            keys = [p["key"] for p in ps]
        else:
            gold = next(p["key"] for p in ps if p["target"] == 1.0)
            keys = [p["key"] for p in ps]
        items.append({"row_id": rid, "qtype": qt,
                      "state_str": rid2state[rid], "gold": gold,
                      "passes": [{"key": p["key"], "text": p["text"]}
                                 for p in ps],
                      "source": "synth2", "keys": keys})
    return items


def _subset_metrics(items, idxs, z_by_item, temps, qt):
    if qt == "noul":
        p = [_sigmoid(z_by_item[i][0] / temps["noul"]) for i in idxs]
        y = [items[i]["gold"] for i in idxs]
        acc = sum((pi >= 0.5) == bool(g) for pi, g in zip(p, y)) / len(y)
        return {"n": len(y), "acc": round(acc, 4), "ece": round(ece(p, y), 4)}
    conf, corr = [], []
    for i in idxs:
        s = [x / temps[qt] for x in z_by_item[i]]
        p = _softmax(s)
        top = max(range(len(s)), key=lambda j: s[j])
        keys = [str(items[i]["passes"][j]["key"])
                for j in range(len(s))] if qt == "choice" else \
               [str(j) for j in range(len(s))]
        gold = str(items[i]["gold"])
        corr.append(1.0 if keys[top] == gold else 0.0)
        conf.append(p[top])
    return {"n": len(corr), "acc": round(sum(corr) / len(corr), 4),
            "ece": round(ece(conf, corr), 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="model_cache/kapteeni_p2_s3/adapter")
    ap.add_argument("--heads", default="model_cache/kapteeni_p2_s3/heads.pt")
    ap.add_argument("--out", default="model_cache/kapteeni_v1_2_1.pt")
    ap.add_argument("--fit-out",
                    default="data_cache/phase1/fit_kv_v1_2_1.json")
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    mixed = phase1.build_val_items()
    for it in mixed:
        it["source"] = "mixed"
    synth2 = build_synth2_val_items()
    items = mixed + synth2
    print(f"diverse val items: {len(items)} "
          f"(mixed {len(mixed)}, synth2 {len(synth2)})", flush=True)

    from kapteeni.backbone import DEFAULT_MODEL, extract_shared_prefix
    from kapteeni.heads import PassMLP

    model, tok = load_merged(args.model or DEFAULT_MODEL, args.lora,
                             args.device)
    heads_sd = torch.load(args.heads, weights_only=True)
    in_dim = model.config.hidden_size
    heads = {}
    for qt in ("noul", "choice", "score"):
        h = PassMLP(in_dim).to(args.device)
        h.load_state_dict(heads_sd[qt])
        h.eval()
        heads[qt] = h

    z_by_item, sv_by_item = [], []
    for k, it in enumerate(items):
        st = it["state_str"]
        sufs = [p["text"][len(st) + 2:] for p in it["passes"]]
        h_m, sv = extract_shared_prefix(model, tok, st, sufs,
                                        device=args.device)
        with torch.no_grad():
            z = heads[it["qtype"]](h_m.to(args.device).float()).cpu().tolist()
        z_by_item.append(z)
        sv_by_item.append(sv.tolist())
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(items)} items", flush=True)

    # 1) temperatures on the DIVERSE set
    temps = {}
    for qt in ("noul", "choice", "score"):
        idxs = [i for i, it in enumerate(items) if it["qtype"] == qt]
        if qt == "noul":
            def readout(T):
                p = [_sigmoid(z_by_item[i][0] / T) for i in idxs]
                y = [items[i]["gold"] for i in idxs]
                return p, y
        else:
            def readout(T):
                conf, corr = [], []
                for i in idxs:
                    s = [x / T for x in z_by_item[i]]
                    p = _softmax(s)
                    keys = [str(pp["key"]) for pp in items[i]["passes"]]
                    gold = str(items[i]["gold"])
                    top = max(range(len(s)), key=lambda j: s[j])
                    corr.append(1.0 if keys[top] == gold else 0.0)
                    conf.append(p[top])
                return conf, corr
        T = fit_temperature(readout)
        p, y = readout(T)
        temps[qt] = T
        print(f"[temp:{qt}] T={T} diverse_ece={ece(p, y):.4f}", flush=True)

    # diagnostics per subset at the fitted temperatures (gates 1 & 2)
    for qt in ("noul", "choice", "score"):
        for src in ("mixed", "synth2"):
            idxs = [i for i, it in enumerate(items)
                    if it["qtype"] == qt and it["source"] == src]
            if idxs:
                m = _subset_metrics(items, idxs, z_by_item, temps, qt)
                print(f"[diag:{qt}:{src}] n={m['n']} acc={m['acc']} "
                      f"ece={m['ece']}", flush=True)

    bundle = {qt: {"head": heads_sd[qt], "T": temps[qt], "in_dim": in_dim,
                   "metrics": {}} for qt in ("noul", "choice", "score")}
    torch.save(bundle, args.out)
    print(f"wrote {args.out}")

    # 2) blend grid on the DIVERSE set — grids unchanged from v1/v1.2
    def head_log_scores(qt, z):
        if qt == "noul":
            p = _sigmoid(z[0] / temps["noul"])
            return [math.log(max(p, 1e-9) / max(1 - p, 1e-9))]
        zs = [x / temps[qt] for x in z]
        p = _softmax(zs)
        return [math.log(max(x, 1e-9)) for x in p]

    s_head = [head_log_scores(it["qtype"], z_by_item[i])
              for i, it in enumerate(items)]
    fits = {}
    for qt in ("noul", "choice", "score"):
        idxs = [i for i, it in enumerate(items) if it["qtype"] == qt]
        best = None
        grid_w = [x / 20 for x in range(21)]
        grid_tau = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
        grid_b = ([round(1.0 + 0.25 * i, 2) for i in range(15)]
                  if qt == "noul" else [0.0])
        for w in grid_w:
            for tau in grid_tau:
                for b in grid_b:
                    bs = []
                    for i in idxs:
                        it = items[i]
                        sh, sv = s_head[i], sv_by_item[i]
                        s = [w * x + (1 - w) * ((v + b) / tau)
                             for x, v in zip(sh, sv)]
                        if qt == "noul":
                            p = _sigmoid(s[0])
                            bs.append((p - it["gold"]) ** 2)
                        else:
                            p = _softmax(s)
                            keys = [str(pp["key"]) for pp in it["passes"]]
                            gold = str(it["gold"])
                            bs.append(sum((p[j] - int(kk == gold)) ** 2
                                          for j, kk in enumerate(keys)))
                    score = sum(bs) / len(bs)
                    if best is None or score < best[0]:
                        best = (score, w, tau, b)
        score, w, tau, b = best
        fits[qt] = {"w": w, "tau": tau, "b": b, "brier": round(score, 4),
                    "source": "v1.2.1 diverse fit (mixed + synth2 val) on "
                              "frozen seed-3 artifacts; no JevBench data"}
        print(f"[fit:{qt}] w={w} tau={tau} b={b} brier={score:.4f}",
              flush=True)
    Path(args.fit_out).write_text(json.dumps(fits, indent=2))
    print(f"wrote {args.fit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())