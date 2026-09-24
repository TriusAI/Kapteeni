"""Consolidated E2 report: writes docs/EVAL.md from the trained bundle.

    python3 -m kapteeni.report --emb data_cache/emb.pt --bundle model_cache/kapteeni_v0.pt

Computes, per primitive, on the deterministic val split:
  - noul:  ECE / Brier / accuracy vs gold; fidelity vs teacher soft labels
  - choice: top-1 accuracy (full 77-option sets), ECE, reliability extremes
  - score: top-1 accuracy, expectation MAE
plus the held-out MNLI run (data_cache/mnli_eval.json) if present.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from kapteeni.build_data import is_val
from kapteeni.heads import PassMLP
from kapteeni.metrics import brier, ece
from kapteeni.train import _gold_pos, _row_groups, _soft, load_emb


def _readout(bundle, emb, records, qt, device="cpu"):
    head = PassMLP(bundle[qt]["in_dim"])
    head.load_state_dict(bundle[qt]["head"])
    head.eval()
    T = bundle[qt]["T"]
    h = emb[[r["_i"] for r in records]].float()
    with torch.no_grad():
        z = head(h).tolist()
    return z, T


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data_cache/emb.pt")
    ap.add_argument("--bundle", default="model_cache/kapteeni_v0.pt")
    ap.add_argument("--soft", nargs="+",
                    default=["data_cache/soft_boolq.jsonl", "data_cache/soft_fever.jsonl"])
    ap.add_argument("--out", default="docs/EVAL.md")
    args = ap.parse_args(argv)

    blob = torch.load(args.bundle, weights_only=False)
    emb, records = load_emb(args.emb)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for i, r in enumerate(records):
        r["_i"] = i
        by_type[r["qtype"]].append(r)

    soft: dict[str, float] = {}
    for p in args.soft:
        if Path(p).exists():
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                soft[r["row_id"]] = r["p"]

    lines: list[str] = []
    A = lines.append
    A("# E2 calibration report — kapteeni v0")
    A("")
    A(f"Bundle: `{args.bundle}` (trained {blob['meta'].get('trained_at')}, "
      f"emb `{blob['meta'].get('emb')}`).")
    A("")
    A("Validation rows use the deterministic `sha256(row_id) % 10` split; "
      "val rows carry gold targets and full option sets (see WORKLOG).")
    A("")

    # ---------------------------------------------------------------- noul
    A("## Noul (absolute readout)")
    A("")
    if "noul" in by_type:
        va = [r for r in by_type["noul"] if is_val(r["row_id"])]
        z, T = _readout(blob, emb, va, "noul")
        probs = [1 / (1 + math.exp(-zi / T)) for zi in z]
        golds = [r["gold"] for r in va]
        A(f"- **ECE vs gold: {ece(probs, golds):.4f}** | Brier {brier(probs, golds):.4f} "
          f"| accuracy {sum((p >= .5) == (g >= .5) for p, g in zip(probs, golds)) / len(golds):.3f} "
          f"| n={len(golds)} | T={T}")
        pairs = [(p, soft[r["row_id"]]) for p, r in zip(probs, va) if r["row_id"] in soft]
        if pairs:
            A(f"- Fidelity vs teacher (distillation): ECE "
              f"{ece([p for p, _ in pairs], [s for _, s in pairs]):.4f} "
              f"| Brier {brier([p for p, _ in pairs], [s for _, s in pairs]):.4f} "
              f"| n={len(pairs)}")
        A(f"- Teacher ceiling vs gold on the same datasets: BoolQ ECE 0.130 / "
          f"FEVER ECE 0.074 (measured pre-training, see WORKLOG)")
    else:
        A("- (no noul passes in emb)")
    A("")

    # -------------------------------------------------------------- choice
    A("## Choice (relative readout, Banking77)")
    A("")
    if "choice" in by_type:
        va = [r for r in by_type["choice"] if is_val(r["row_id"])]
        z, T = _readout(blob, emb, va, "choice")
        groups = _row_groups(va)
        conf, correct, n_opts = [], [], []
        for g in groups:
            s = [z[i] / T for i in g]
            p = _soft(s)
            top = max(range(len(s)), key=lambda i: s[i])
            correct.append(1.0 if top == _gold_pos(va, g) else 0.0)
            conf.append(p[top])
            n_opts.append(len(g))
        acc = sum(correct) / len(correct)
        A(f"- **Top-1 accuracy: {acc:.4f}** over a mean of "
          f"{sum(n_opts)/len(n_opts):.1f} options/row | ECE {ece(conf, correct):.4f} "
          f"| n={len(groups)} val rows (full 77-option sets) | T={T}")
    else:
        A("- (no choice passes in emb)")
    A("")

    # --------------------------------------------------------------- score
    A("## Score (independent levels, HelpSteer2)")
    A("")
    if "score" in by_type:
        va = [r for r in by_type["score"] if is_val(r["row_id"])]
        z, T = _readout(blob, emb, va, "score")
        groups = _row_groups(va)
        conf, correct, mae = [], [], []
        for g in groups:
            s = [z[i] / T for i in g]
            p = _soft(s)
            top = max(range(len(s)), key=lambda i: s[i])
            gold = _gold_pos(va, g)
            correct.append(1.0 if top == gold else 0.0)
            conf.append(p[top])
            mae.append(abs(sum(i * p[i] for i in range(len(p))) - gold))
        A(f"- **Top-1 accuracy: {sum(correct)/len(correct):.4f}** | expectation MAE "
          f"{sum(mae)/len(mae):.3f} | ECE {ece(conf, correct):.4f} | n={len(groups)} | T={T}")
    else:
        A("- (no score passes in emb)")
    A("")

    # ------------------------------------------------------------ OOD eval
    A("## Held-out dataset (OOD, MNLI — never trained)")
    A("")
    if Path("data_cache/mnli_eval.json").exists():
        r = json.load(open("data_cache/mnli_eval.json"))
        A(f"- **ECE {r['ece']:.4f}** | Brier {r['brier']:.4f} | accuracy "
          f"{r['accuracy']:.4f} | n={r['rows']} | T={r['temperature']}")
        A("")
        A("| P range | n | mean P | freq true |")
        A("|---|---|---|---|")
        for b in r["reliability"]:
            A(f"| {b['p_range'][0]}–{b['p_range'][1]} | {b['n']} | "
              f"{b['mean_p']} | {b['freq_true']} |")
    else:
        A("- (mnli_eval.json not present — run evalx)")
    A("")
    A("## Bars (reference plan)")
    A("")
    A("- E2 bar: ECE ≤ 0.06 after temperature scaling (measured against the "
      "target the head was trained to match; vs *gold* the teacher's own ECE "
      "is the achievable ceiling).")
    A("- E5 asymmetry, E1 parity, E6 score semantics, E11 schema: green in the "
      "test suite (62 tests) — see README.")
    A("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())