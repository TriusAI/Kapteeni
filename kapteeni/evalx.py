"""Post-training evaluation runner (E2 on held-out data).

train.py reports in-split metrics; this runner evaluates the TRAINED model
end-to-end (real serialization -> backbone -> heads) on rows it never saw,
e.g. MNLI for a noul head trained on BoolQ+FEVER.

    python3 -m kapteeni.evalx --bundle model_cache/kapteeni_v0.pt \
        --rows data_cache/rows_mnli.jsonl --out data_cache/mnli_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kapteeni.backbone import DEFAULT_MODEL, extract_h_last, load_backbone
from kapteeni.heads import PassMLP
from kapteeni.metrics import brier, ece
from kapteeni.serialize import noul_pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--rows", required=True, help="noul rows jsonl (eval)")
    ap.add_argument("--soft", default="", help="teacher soft-label jsonl (fidelity report)")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    rows = [r for r in rows if r["primitive"] == "noul"]
    if args.n:
        rows = rows[: args.n]
    soft: dict = {}
    if args.soft:
        for l in open(args.soft, encoding="utf-8"):
            r = json.loads(l)
            soft[r["row_id"]] = r["p"]

    blob = torch.load(args.bundle, weights_only=False)
    head = PassMLP(blob["noul"]["in_dim"]).to(args.device)
    head.load_state_dict(blob["noul"]["head"])
    head.eval()
    T = blob["noul"]["T"]

    model, tok = load_backbone(args.model, args.device)
    texts = [noul_pass(r["state"], r["instructions"], r["criteria"]) for r in rows]
    print(f"evaluating {len(texts)} held-out noul rows ...", flush=True)
    h = extract_h_last(model, tok, texts, device=args.device)

    probs, golds = [], []
    with torch.no_grad():
        z = head(h.to(args.device).float()).cpu()
    for zi, r in zip(z.tolist(), rows):
        probs.append(1 / (1 + pow(2.718281828459045, -zi / T)))
        golds.append(float(r["label"]))

    # reliability table for the report
    bins = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        idx = [i for i in range(len(probs))
               if lo <= probs[i] < hi or (b == 9 and probs[i] >= hi)]
        if not idx:
            continue
        bins.append({
            "p_range": [round(lo, 1), round(hi, 1)],
            "n": len(idx),
            "mean_p": round(sum(probs[i] for i in idx) / len(idx), 3),
            "freq_true": round(sum(golds[i] for i in idx) / len(idx), 3),
        })
    report = {
        "rows": len(probs),
        "dataset": Path(args.rows).stem,
        "ece": round(ece(probs, golds), 4),
        "brier": round(brier(probs, golds), 4),
        "accuracy": round(sum((p >= 0.5) == (g >= 0.5) for p, g in zip(probs, golds)) / len(probs), 4),
        "temperature": T,
        "reliability": bins,
        "model_trained_at": blob.get("meta", {}).get("trained_at"),
    }
    if soft:
        pairs = [(p, soft[r["row_id"]]) for p, r in zip(probs, rows) if r["row_id"] in soft]
        report["fidelity_vs_teacher"] = {
            "n": len(pairs),
            "ece": round(ece([p for p, _ in pairs], [s for _, s in pairs]), 4),
            "brier": round(brier([p for p, _ in pairs], [s for _, s in pairs]), 4),
        }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())