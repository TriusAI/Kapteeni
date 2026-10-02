"""Pack kapteeni-v1.1c for release / Hugging Face.

    HF_HUB_OFFLINE=1 python3 -m kapteeni.pack_v11c

Produces ../kapteeni-v1.1c-dist, mirroring the v1 pack format:
  - the MERGED model (Qwen3.5-4B + LoRA, merge_and_unload) as
    safetensors + tokenizer + processor — no peft dependency at serve
    time; ships the fp32-tower policy implicitly (dtype is bf16 and
    serve_v11c casts the tower at load, exactly as the gated runs did)
  - heads.safetensors (readout heads, no pickles; keys prefixed
    noul./choice./score. like pack.py's format)
  - kapteeni-config.json (served_as, in_dim, head temperatures = the
    pre-registered combined-val fit from final_gates.json)
  - README.md — the v1.1c model card (frontmatter + every gate number)
  - LICENSE (Apache-2.0) + WEIGHTS-LICENSE.md (CC BY-SA 4.0)
  - kapteeni/ — the serving package incl. the demo website, so the
    dist runs without the GitHub repo

Serve the pack:  HF_HUB_OFFLINE=1 python3 -m kapteeni.serve_v11c \
                     --dist ../kapteeni-v1.1c-dist --port 8002
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "model_cache/qwen3.5-4b"
ADAPTER = "model_cache/kapteeni_v11c/adapter"
HEADS = "model_cache/kapteeni_v11c/heads.pt"
GATES = "model_cache/kapteeni_v11c/final_gates.json"
OUT = ROOT.parent / "kapteeni-v1.1c-dist"

CARD = '''---
license: cc-by-sa-4.0
base_model: Qwen/Qwen3.5-4B
tags:
- decision-model
- structured-decisions
- calibration
- probability-estimation
- vision
- image-document-understanding
language:
- en
- zh
library_name: transformers
---

# Kapteeni v1.1c — a multimodal Jev-compatible System One decision model

Send a `state` — optionally carrying an attached **image** — plus typed
questions; get back **calibrated probability distributions** your code can
branch on. No text generation. Kapteeni implements the TypeSafe System One
decision-model interface (`POST /v1/systemone`, the wire format of Jev).

## What it is

One Qwen3.5-4B backbone + LoRA adapter (merged in this snapshot) + tiny
per-primitive readout heads, answering **noul** (absolute probability of
true), **choice** (relative, probabilities sum to exactly 1) and **score**
(independent levels; score = the 0-based expectation) questions over any
state — text-only or carrying `state.image` (base64 PNG/JPEG), in English
and Chinese. The answer space stays `yes/no` + option names in every
language; Chinese enters through state, instructions, and criteria only.

Trained per the v1 phase pipeline ported to the multimodal base
(multi-pass per-option judgment, distillation-weighted targets, group-CE
in-loss, heads on the final hidden state; LoRA r=32/alpha=64 on the seven
language-model projections, lr 1e-4, token budget 8192, ONE epoch —
24.1M real tokens over a mixed image+Chinese+text curriculum), one run,
every pre-registered gate passed on the first reading (2026-10-02):

| gate | reading | bar |
|---|---:|---:|
| synth3-val (n=590, English image decisions) | 0.9661 | > 0.8068 (frozen base) |
| synth3zh-val (n=204, Chinese image decisions; reported) | 0.9412 | — |
| MNLI-noul (n=150, English text, never trained on) | 0.8800 | >= 0.84 |
| OCNLI-noul (n=150, Chinese text, never trained on) | 0.8467 | >= 0.83 |
| synth2zh-val (n=403, Chinese rule skills) | 0.9132 | >= 0.90 |
| synth2-EN-val (n=609, English rule skills) | 0.9048 | >= 0.90 |
| fitted ECE (noul / choice / score) | 0.0241 / 0.0175 / 0.0685 | <= 0.10 |

The rule-skill mastery bars (0.90) had falsified every predecessor of
this arc and were never met by any prior configuration; the multi-pass
judgment structure is what closed them, with MNLI/OCNLI holding
throughout the run.

## Quickstart

```bash
HF_HUB_OFFLINE=1 python3 -m kapteeni.serve_v11c --dist ./ --port 8002
curl localhost:8002/v1/systemone -d '{"state": {"note": "a cafe menu",
  "image": "<base64 png>"}, "model": "kapteeni-v1.1c", "questions": {
  "q1": {"type": "choice", "instructions": "Which drink is the most
          expensive?", "criteria": {"Espresso": null, "Latte": null,
          "Mocha": null}}}}'
# demo website after starting the server: http://localhost:8002/
```

Images are bounded to a 640px longest edge at serve time; quality is
validated at 640x640 (400 vision tokens/pass). Question ids never reach
the model; answers are content-only and deterministic.

## Training data

Synthetic, gold-by-construction, all pixels/text ours (Apache-2.0): 20
image document families (English + fully-Chinese renders), binned
temporal/multi-hop/long-policy rule families (English + Chinese ports),
and English text replay (BoolQ/FEVER/Banking77/CLInc150/GoEmotions/
HelpSteer2-adjacent surfaces reused from the v1.2 union byte-identical).
Zero 8-token overlaps with JevBench or OCNLI gate items (contamination
audit, CJK-aware). Teacher soft labels for noul/replay: k-sample
agreement from a cloud LLM judge (fidelity-documented). Never trained on:
MNLI, OCNLI, synth val slices, imajev-bench, JevBench.

## Citation / provenance

Code + generators: github.com/TriusAI (kapteeni), Apache-2.0. Weights:
CC BY-SA 4.0 (ShareAlike from MultiNLI + FEVER annotation lineage),
see WEIGHTS-LICENSE.md. Backbone: Qwen/Qwen3.5-4B (Apache-2.0).
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--heads", default=HEADS)
    ap.add_argument("--gates", default=GATES)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    print("merging base + adapter ...", flush=True)
    proc = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    proc.save_pretrained(out)
    print("model merged + saved", flush=True)

    # heads -> safetensors (no pickles), keys prefixed per primitive
    from safetensors.torch import save_file
    heads = torch.load(args.heads, weights_only=True)
    tensors = {}
    for qt, sd in heads.items():
        for k, v in sd.items():
            tensors[f"{qt}.{k}"] = v
    save_file(tensors, out / "heads.safetensors")
    print("heads -> heads.safetensors", flush=True)

    # serving config: the pre-registered fitted temperatures
    gates = json.loads(Path(args.gates).read_text())
    (out / "kapteeni-config.json").write_text(json.dumps({
        "served_as": "kapteeni-v1.1c",
        "line": "1.1c (multimodal: images + Chinese + text)",
        "in_dim": 2560,
        "backbone": "Qwen/Qwen3.5-4B",
        "lora": {"r": 32, "alpha": 64, "targets":
                 "language_model q/k/v/o/gate/up/down (tower frozen)"},
        "head_temperatures": gates["fit_temps"],
        "fitted_ece": gates["fitted_ece"],
        "gates": {k: v["acc"] for k, v in gates["gates"].items()},
        "wire_extension": "state.image (base64 PNG/JPEG, bounded to a "
                          "640px longest edge at serve time)",
        "prereg": "docs/PREREG-KAPTEENI-V11C.md",
    }, indent=2))

    (out / "README.md").write_text(CARD)
    shutil.copy(ROOT / "LICENSE", out / "LICENSE")
    shutil.copy(ROOT / "WEIGHTS-LICENSE.md", out / "WEIGHTS-LICENSE.md")
    # the serving package (incl. the demo website), pycache-free
    pkg = ROOT / "kapteeni"
    dst = out / "kapteeni"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(pkg, dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"dist complete -> {out} "
          f"({sum(1 for _ in out.rglob('*'))} files)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())