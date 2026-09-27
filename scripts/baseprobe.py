"""v1.1 Step 0 base-selection probe (docs/PREREG-KAPTEENI-V11.md).

Frozen, zero training. Measures a candidate backbone on three fixed
slices with the one-pass lettered LM-head readout (the V0/V2 path):

  - synth3-val : all 590 held-out image items (ours)
  - MNLI-noul  : first 150 of data_cache/rows_mnli.jsonl (never trained)
  - OCNLI-noul : first 150 of data_cache/rows_ocnli.jsonl (never trained)

Readout rules (pre-registered, identical across candidates):
  - noul EN (MNLI): argmax over first-tokens of "yes" vs "no"
  - noul ZH (OCNLI): argmax over token groups yes+是 vs no+否
  - choice/score (synth3): argmax over " A", " B", ... (leading space)

Thinking modes are disabled where the chat template supports it
(auto-detected; the used path is recorded in the report).

    python3 scripts/baseprobe.py --model model_cache/qwen3.5-9b \
        --name qwen3.5-9b
    -> data_cache/baseprobe/<name>.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from kapteeni.build_data import is_val  # noqa: E402
from kapteeni.vl_format import (LETTERS, load_rows, synth3_example,  # noqa: E402
                                text_example)

SYNTH3 = ROOT / "data_cache" / "synth3" / "items.jsonl"
IMG_DIR = str(ROOT / "data_cache" / "synth3" / "images")


def first_id(tok, s: str) -> int:
    return tok.tokenizer.encode(s, add_special_tokens=False)[0]


def build_slices():
    def tag(exs, key):
        return [dict(ex, probe_key=key) for ex in exs]

    s3 = [synth3_example(r, IMG_DIR) for r in load_rows(str(SYNTH3))
          if is_val(r["row_id"])]
    mnli = [text_example(r, {})
            for r in load_rows(str(ROOT / "data_cache/rows_mnli.jsonl"))[:150]]
    ocnli = [text_example(r, {})
             for r in load_rows(str(ROOT / "data_cache/rows_ocnli.jsonl"))[:150]]
    return {"synth3": tag(s3, "synth3"), "mnli": tag(mnli, "mnli"),
            "ocnli": tag(ocnli, "ocnli")}


def resolve_think_kwargs(proc, think_kwargs):
    """Try the template with thinking kwargs; fall back to none on
    TypeError (template without the kwarg). Records whether a thinking
    block still lands in the rendered prompt."""
    if not think_kwargs:
        return {}, None, False
    msgs = [{"role": "user",
             "content": [{"type": "text", "text": "probe"}]}]
    try:
        txt = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            **think_kwargs)
        return (dict(think_kwargs), txt[:160],
                "<think>" in txt or "think" in txt.lower()[:300])
    except TypeError:
        return {}, None, False


def render(proc, batch, think_kwargs):
    texts, images = [], []
    for ex in batch:
        content = ([{"type": "image", "image": ex["image"]}] if ex["image"]
                   else []) + [{"type": "text", "text": ex["prompt"]}]
        msgs = [{"role": "user", "content": content}]
        texts.append(proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            **think_kwargs))
        if ex["image"]:
            images.append(ex["image"])
    return texts, images


@torch.no_grad()
def probe(model, proc, tok, examples, device, think_kwargs, bs=8):
    model.eval()
    ids = {"yes": first_id(tok, "yes"), "no": first_id(tok, "no"),
           "shi": first_id(tok, "是"), "fou": first_id(tok, "否")}
    letters = [first_id(tok, " " + L) for L in LETTERS]
    by, fam = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for i in range(0, len(examples), bs):
        batch = examples[i:i + bs]
        texts, images = render(proc, batch, think_kwargs)
        inputs = proc(text=texts, images=images or None,
                      return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        for j, ex in enumerate(batch):
            L = int(inputs["attention_mask"][j].sum())
            row = logits[j, L - 1, :]
            if ex["primitive"] == "noul":
                if ex["probe_key"] == "ocnli":
                    p_yes = row[ids["yes"]] + row[ids["shi"]]
                    p_no = row[ids["no"]] + row[ids["fou"]]
                else:
                    p_yes, p_no = row[ids["yes"]], row[ids["no"]]
                pred = 1 if p_yes > p_no else 0
            else:
                cand = letters[:ex["n_cands"]]
                pred = int(torch.argmax(row[cand]))
            ok = int(pred == ex["gold"])
            by[ex["probe_key"]][0] += ok
            by[ex["probe_key"]][1] += 1
            if ex.get("family"):
                fam[ex["family"]][0] += ok
                fam[ex["family"]][1] += 1
    return {k: round(v[0] / v[1], 4) for k, v in by.items() if v[1]}, \
        {k: round(v[0] / v[1], 4) for k, v in fam.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-run limit per slice")
    ap.add_argument("--think-off", action="store_true", default=True)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device)
    model.eval()

    kw, sample_head, think_in_prompt = resolve_think_kwargs(
        proc, {"enable_thinking": False} if args.think_off else {})
    print(f"think kwargs resolved: {kw} "
          f"(think-marker in prompt: {think_in_prompt})", flush=True)

    slices = build_slices()
    if args.limit:
        slices = {k: v[:args.limit] for k, v in slices.items()}

    report = {"model": args.model, "name": args.name,
              "think_kwargs": {"used": kw, "template_rejected_think_kw":
                               not kw and bool(args.think_off),
                               "think_marker_in_prompt": think_in_prompt,
                               "prompt_head": sample_head},
              "slices": {}}
    t0 = time.time()
    for sname, exs in slices.items():
        acc, fam = probe(model, proc, proc, exs, args.device, kw)
        report["slices"][sname] = acc
        if fam:
            report.setdefault("synth3_families", {})[sname] = fam
        print(f"{sname}: {acc}  ({time.time()-t0:.0f}s)", flush=True)

    vals = [v for s in report["slices"].values() for v in s.values()]
    report["mean"] = round(sum(vals) / len(vals), 4)
    report["n"] = {k: len(v) for k, v in slices.items()}
    out = ROOT / "data_cache" / "baseprobe"
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["slices"], ensure_ascii=False))
    print(f"MEAN = {report['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())