"""v1.1 finalize: per-primitive temperatures + the fitted-ECE gate.

The fifth pre-registered gate (PREREG-KAPTEENI-V11): fitted ECE <= 0.10
on combined held-out val, with per-primitive temperatures fit on that
same combined val (synth3-val + synth3zh-val + text val + synth2zh-val)
— the v1.2.1 lesson (never ship uncalibrated) applied in advance.

Readout: the LM's answer-token logits in the fixed wire answer space
(yes/no for noul; A.. letters for choice/score). One temperature per
primitive, fit by minimizing ECE (kapteeni.p2_finalize.fit_temperature).

  HF_HUB_OFFLINE=1 python3 scripts/v11_finalize.py \
      --adapter model_cache/kapteeni_v11/adapter

Emits model_cache/kapteeni_v11/finalize.json (temps + ECE per primitive
and combined). Run AFTER the gates pass; this measures calibration only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kapteeni.build_data import is_val  # noqa: E402
from kapteeni.metrics import ece  # noqa: E402
from kapteeni.p2_finalize import fit_temperature  # noqa: E402
from kapteeni.train_v import CRITERIA, TEXT_REPLAY  # noqa: E402
from kapteeni.vl_format import LETTERS, synth3_example, text_example, \
    load_rows  # noqa: E402


def collect(model, proc, tok, examples, device) -> list[dict]:
    """Per-example candidate logits from the last real token."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(examples), 8):
            batch = examples[i:i + 8]
            texts, images = [], []
            for ex in batch:
                messages = [{"role": "user", "content": (
                    [{"type": "image", "image": ex["image"]}] if ex["image"]
                    else []) + [{"type": "text", "text": ex["prompt"]}]}]
                texts.append(proc.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False))
                if ex["image"]:
                    images.append(ex["image"])
            inputs = proc(text=texts, images=images or None,
                          return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            for j, ex in enumerate(batch):
                L = int(inputs["attention_mask"][j].sum())
                row = logits[j, L - 1, :]
                if ex["primitive"] == "noul":
                    cand = [tok.tokenizer.encode(t, add_special_tokens=False)[0]
                            for t in ("yes", "no")]
                else:
                    cand = [tok.tokenizer.encode(" " + LETTERS[k],
                                                 add_special_tokens=False)[0]
                            for k in range(ex["n_cands"])]
                out.append({"primitive": ex["primitive"],
                            "source": ex["source"],
                            "logits": row[cand].float().cpu().tolist(),
                            "gold": ex["gold"]})
    return out


def conf_target(item: dict, t: float) -> tuple[float, float]:
    """Top-label confidence after temperature t, and correctness."""
    import math
    z = [x / t for x in item["logits"]]
    m = max(z)
    e = sum(math.exp(x - m) for x in z)
    p = [math.exp(x - m) / e for x in z]
    pred = max(range(len(z)), key=lambda k: p[k])
    return p[pred], float(pred == item["gold"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="model_cache/kapteeni_v11/adapter")
    ap.add_argument("--model", default="model_cache/qwen3.5-4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="model_cache/kapteeni_v11/finalize.json")
    args = ap.parse_args()

    # combined held-out val, exactly the pre-registered set
    val: list = []
    for ds, img_dir in (("synth3", "data_cache/synth3/images"),
                        ("synth3zh", "data_cache/synth3zh/images")):
        for r in load_rows(f"data_cache/{ds}/items.jsonl"):
            if is_val(r["row_id"]):
                ex = synth3_example(r, img_dir)
                ex["source"] = ds
                val.append(ex)
    for name in TEXT_REPLAY:
        crit = json.loads(open(CRITERIA[name]).read()) \
            if name in CRITERIA else {}
        for r in load_rows(f"data_cache/rows_{name}.jsonl"):
            if is_val(r["row_id"]):
                val.append(text_example(r, crit))
    crit2 = json.loads(open("data_cache/synth2_criteria.json").read())
    for r in load_rows("data_cache/rows_synth2.jsonl"):
        if is_val(r["row_id"]):
            val.append(text_example(r, crit2))
    crit2zh = json.loads(open("data_cache/synth2zh_criteria.json").read())
    for r in load_rows("data_cache/rows_synth2zh.jsonl"):
        if is_val(r["row_id"]):
            val.append(text_example(r, crit2zh))
    print(f"combined val: {len(val)} examples", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device)
    model = PeftModel.from_pretrained(model, args.adapter)

    items = collect(model, proc, tok, val, args.device)
    by_prim = defaultdict(list)
    for it in items:
        by_prim[it["primitive"]].append(it)

    temps, eces = {}, {}
    all_pairs = []
    for prim, its in sorted(by_prim.items()):
        def probs_of_T(t, its=its):
            return ([*map(list, zip(*[conf_target(it, t) for it in its]))])
        t = fit_temperature(probs_of_T)
        pairs = [conf_target(it, t) for it in its]
        e = ece([c for c, _ in pairs], [y for _, y in pairs])
        temps[prim] = t
        eces[prim] = round(e, 4)
        all_pairs.extend(pairs)
        print(f"  {prim}: n={len(its)} temp={t} fitted ECE={e:.4f}",
              flush=True)
    combined = ece([c for c, _ in all_pairs], [y for _, y in all_pairs])
    print(f"combined fitted ECE: {combined:.4f} (gate <= 0.10)")

    payload = {
        "combined_val_n": len(val),
        "per_primitive": {p: {"n": len(by_prim[p]), "temperature": temps[p],
                              "fitted_ece": eces[p]}
                          for p in sorted(by_prim)},
        "temperatures": temps,
        "combined_fitted_ece": round(combined, 4),
        "gate": "fitted ECE <= 0.10",
        "gate_pass": bool(combined <= 0.10),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())