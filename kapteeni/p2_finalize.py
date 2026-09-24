"""P2 finalizer: merge LoRA, refit temperatures + blend, emit a servable bundle.

    python3 -m kapteeni.p2_finalize --lora model_cache/kapteeni_p2/adapter \
        --heads model_cache/kapteeni_p2/heads.pt --out model_cache/kapteeni_v1.pt

Everything is fitted on the mixed-domain val slices (BoolQ/FEVER/MNLI val,
Banking77 val, HelpSteer2 val) through the MERGED LoRA model — never on
JevBench items. Outputs:
  - kapteeni_v1.pt       bundle {<qt>: {head, T, in_dim}} for serving
  - fit_kv.json          blend (w, tau, b) per primitive for the serving readout
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import phase1  # noqa: E402
from kapteeni.backbone import DEFAULT_MODEL, extract_shared_prefix, load_backbone  # noqa: E402
from kapteeni.heads import PassMLP  # noqa: E402
from kapteeni.metrics import ece  # noqa: E402


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def load_merged(model_name: str, lora_path: str, device: str):
    from peft import PeftModel

    model, tok = load_backbone(model_name, device)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def fit_temperature(probs_of_T, lo=0.05, hi=8.0, iters=40):
    gr = (math.sqrt(5) - 1) / 2
    a, b = math.log(lo), math.log(hi)
    c, d = b - gr * (b - a), a + gr * (b - a)

    def score(x):
        t = math.exp(x)
        p, y = probs_of_T(t)
        return ece(p, y)

    fc, fd = score(c), score(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = score(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = score(d)
    return round(math.exp((a + b) / 2), 4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="model_cache/kapteeni_p2/adapter")
    ap.add_argument("--heads", default="model_cache/kapteeni_p2/heads.pt")
    ap.add_argument("--out", default="model_cache/kapteeni_v1.pt")
    ap.add_argument("--fit-out", default="data_cache/phase1/fit_kv.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    items = phase1.build_val_items()
    print(f"mixed-val items: {len(items)}")
    model, tok = load_merged(args.model, args.lora, args.device)

    heads = {}
    heads_sd = torch.load(args.heads, weights_only=True)
    in_dim = model.config.hidden_size
    for qt in ("noul", "choice", "score"):
        h = PassMLP(in_dim).to(args.device)
        h.load_state_dict(heads_sd[qt])
        h.eval()
        heads[qt] = h

    # extract (h, s_verb) per item through the merged model
    z_by_item, sv_by_item = [], []
    for k, it in enumerate(items):
        st = it["state_str"]
        sufs = [p["text"][len(st) + 2:] for p in it["passes"]]
        h_m, sv = extract_shared_prefix(model, tok, st, sufs, device=args.device)
        with torch.no_grad():
            z = heads[it["qtype"]](h_m.to(args.device).float()).cpu().tolist()
        z_by_item.append(z)
        sv_by_item.append(sv.tolist())
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(items)} items", flush=True)

    # 1) per-head temperature on val (vs gold)
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
        print(f"[temp:{qt}] T={T} ece={ece(p, y):.4f}")

    bundle = {qt: {"head": heads_sd[qt], "T": temps[qt], "in_dim": in_dim,
                   "metrics": {}}
              for qt in ("noul", "choice", "score")}
    torch.save(bundle, args.out)
    print(f"wrote {args.out}")

    # 2) blend (w, tau, b) per primitive on mixed-val (deployed parameterization)
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
                    "source": "P2 fit on mixed-domain val through merged LoRA; "
                              "no JevBench data"}
        print(f"[fit:{qt}] w={w} tau={tau} b={b} brier={score:.4f}")
    Path(args.fit_out).write_text(json.dumps(fits, indent=2))
    print(f"wrote {args.fit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())