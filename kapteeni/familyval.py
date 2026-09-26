"""Family-val evaluator: per-family accuracy of the SERVED readout (blend)
on the synth2 validation slice, for any bundle/lora combination.

    python3 -m kapteeni.familyval \
        --bundle model_cache/kapteeni_v1_2.pt \
        --lora model_cache/kapteeni_p2_s3/adapter \
        --fit data_cache/phase1/fit_kv_v1_2.json

Runs every synth2 val row (deterministic split) through the deployed
readout path — the exact probabilities a consumer would branch on — and
reports accuracy by family and primitive against gold. Run it once for
v1 (baseline) and once for the candidate; the comparison is the
hypothesis test for v1.2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kapteeni.backbone import DEFAULT_MODEL  # noqa: E402
from kapteeni.build_data import is_val  # noqa: E402
from kapteeni.metrics import ece  # noqa: E402
from kapteeni.model import SystemOneModel  # noqa: E402


def build_questions(row: dict, criteria: dict) -> dict:
    """Row -> contract question in the same shape a consumer would send."""
    prim = row["primitive"]
    if prim == "noul":
        return {"type": "noul", "instructions": row["instructions"],
                "criteria": row["criteria"]}
    if prim == "choice":
        opts = row["meta"]["options"]
        return {"type": "choice", "instructions": row["instructions"],
                "criteria": {o: criteria[o] for o in opts}}
    levels = criteria[row["meta"]["attribute"]]
    return {"type": "score", "instructions": row["instructions"],
            "criteria": levels}


def score_row(answers: dict, row: dict) -> bool:
    """Served answer dict vs gold. answers = model.evaluate's answers[qid]."""
    prim = row["primitive"]
    if prim == "noul":
        return (answers["noul"] >= 0.5) == bool(row["label"])
    if prim == "choice":
        return answers["choice"] == row["meta"]["options"][row["label"]]
    probs = {int(k): v for k, v in answers["probabilities"].items()}
    return max(probs, key=probs.get) == row["label"]


def top_label(answers: dict, row: dict) -> tuple[float, bool]:
    """(confidence of the predicted label, whether it is correct) — the
    inputs for top-label ECE, the direct measure of saturation."""
    prim = row["primitive"]
    if prim == "noul":
        p = answers["noul"]
        pred = p >= 0.5
        return (p if pred else 1.0 - p), pred == bool(row["label"])
    if prim == "choice":
        probs = answers["probabilities"]
        name = max(probs, key=probs.get)
        return probs[name], name == row["meta"]["options"][row["label"]]
    probs = {int(k): v for k, v in answers["probabilities"].items()}
    top = max(probs, key=probs.get)
    return probs[top], top == row["label"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--fit", default="")
    ap.add_argument("--rows", default="data_cache/rows_synth2.jsonl")
    ap.add_argument("--criteria", default="data_cache/synth2_criteria.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    ap.add_argument("--committee", action="store_true",
                    help="ensemble with a second model (--bundle2 etc.)")
    ap.add_argument("--bundle2", default="")
    ap.add_argument("--lora2", default="")
    ap.add_argument("--fit2", default="")
    ap.add_argument("--dist2", default="")
    args = ap.parse_args(argv)

    blend = (json.loads(Path(args.fit).read_text()) if args.fit else None)
    criteria = json.loads(Path(args.criteria).read_text())
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    rows = [r for r in rows if is_val(r["row_id"])]
    print(f"synth2 val rows: {len(rows)}")

    model = SystemOneModel(args.bundle, device=args.device,
                           model_name=args.model or DEFAULT_MODEL,
                           lora=args.lora or None, blend=blend)
    if args.committee:
        from kapteeni.committee import CommitteeModel

        if args.dist2:
            second = SystemOneModel.from_dist(args.dist2,
                                              device=args.device)
        else:
            blend2 = (json.loads(Path(args.fit2).read_text())
                      if args.fit2 else None)
            second = SystemOneModel(args.bundle2, device=args.device,
                                    lora=args.lora2 or None, blend=blend2)
        model = CommitteeModel(model, second)

    stats: dict[tuple, list] = defaultdict(list)
    confs: dict[tuple, list] = defaultdict(list)
    latencies: list[float] = []
    for i, row in enumerate(rows):
        qs = {"q": build_questions(row, criteria)}
        t0 = time.perf_counter()
        answers, _ = model.evaluate(row["state"], qs)
        latencies.append(time.perf_counter() - t0)
        ok = score_row(answers["q"], row)
        conf, ok_tl = top_label(answers["q"], row)
        stats[(row["meta"]["family"], row["primitive"])].append(ok)
        confs[(row["meta"]["family"], row["primitive"])].append(
            (conf, 1.0 if ok_tl else 0.0))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    fams: dict[str, list] = defaultdict(list)
    lines = []
    for (fam, prim), oks in sorted(stats.items()):
        acc = sum(oks) / len(oks)
        cf = confs[(fam, prim)]
        fams[fam].extend(oks)
        lines.append(f"  {fam:16s} {prim:6s} n={len(oks):4d} acc={acc:.4f}"
                     f" ece={ece([c for c, _ in cf], [y for _, y in cf]):.4f}")
    print("\n".join(lines))
    overall = [x for v in stats.values() for x in v]
    fam_out = {}
    for fam, oks in sorted(fams.items()):
        cf = [(c, y) for (f, p), lst in confs.items() if f == fam
              for c, y in lst]
        fam_ece = ece([c for c, _ in cf], [y for _, y in cf])
        fam_out[fam] = {"n": len(oks), "acc": sum(oks) / len(oks),
                        "ece": fam_ece}
        print(f"FAMILY {fam:16s} n={len(oks):4d} acc={sum(oks)/len(oks):.4f}"
              f" ece={fam_ece:.4f}")
    all_conf = [(c, y) for lst in confs.values() for c, y in lst]
    tot_ece = ece([c for c, _ in all_conf], [y for _, y in all_conf])
    print(f"OVERALL                       n={len(overall):4d} "
          f"acc={sum(overall)/len(overall):.4f} ece={tot_ece:.4f}")

    if latencies:
        ls = sorted(latencies)
        p50 = ls[len(ls) // 2]
        p95 = ls[int(len(ls) * 0.95)]
        print(f"latency n={len(ls)} p50={p50:.3f}s p95={p95:.3f}s "
              f"(mean {sum(ls)/len(ls):.3f}s)")

    if args.out:
        payload = {
            "overall_acc": sum(overall) / len(overall),
            "overall_ece": tot_ece,
            "families": fam_out,
            "cells": {f"{f}|{p}": {"n": len(v), "acc": sum(v) / len(v)}
                      for (f, p), v in stats.items()},
            "n_rows": len(overall),
            "latency_s": ({"p50": p50, "p95": p95, "n": len(ls)}
                          if latencies else None),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())