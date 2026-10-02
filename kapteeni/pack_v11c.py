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

**The flow:** `state` (+ optional `state.image`, one base64 PNG/JPEG) →
every question — and, for choice, every option; for score, every level —
serialized into its **own pass** over the same state (chat-template render;
the image enters through the processor's vision path as ~400 tokens) →
Qwen3.5-4B → each pass's **final-token hidden state** → tiny per-primitive
readout heads (choice = softmax over the row's option passes scored in-loss
during training; score = per-level independent BCE; noul = an absolute
probability — P(A) + P(not-A) can exceed 1 by design, mirroring the
reference) → per-primitive temperature (the only constants fitted anywhere
in the line; held-out validation, never benchmark data) → contract shaping
→ typed answers with exact-1 probability sums. Deterministic end to end;
question ids never reach the model; a question is forwarded on its own, so
adding or removing questions cannot change another question's answer.

## Benchmark: JevBench public half (231 items, official harness, official scorer)

Run of record (2026-10-02), measured through THIS snapshot's packaged
distribution (serve `--dist`/`--hf`) at jevbench upstream `bb05a33`;
231/231 attempted, 0 failures, all schema-valid. Context rows: our shipped
text-only variant kapteeni-v1-meticulous.

| | kapteeni-v1.1c | (context: v1-meticulous) |
|---|---:|---:|
| composite (the benchmark's axes; our hardware assumptions) | **63.72** | 65.71 |
| Intelligence | **68.33** | 60.3 |
| public accuracy (231 items) | **0.766** (177/231) | 0.710 |
| easy / standard / hard | 1.000 / 0.931 / 0.559 | 1.000 / 0.889 / 0.469 |
| top-label ECE → Calibration | 0.0984 → 80.3 | 0.0496 → 90.1 |
| latency p50 / p95 (raw) | 0.26 s / 10.6 s | 0.17 s / 1.17 s |
| mean input tokens/decision | 562 | 597 |

- **n=231 carries a 95% CI of ±5.9pt** — the composite difference vs
  65.71 is inside single-run noise; don't read it as a win or a loss.
- The hard-tier jump (+9.0pt), the accuracy gain, and Intelligence +8.0
  are within-resolution separable, and the family breakdown shows where:
  **multi_hop 0.278 → 0.667**, long_policy 0.316 → 0.421 (the rule skills
  the port targeted), while temporal_numeric (n=15; 0.20 → 0.13) remains
  beyond every model this lineage has trained.
- Latency/cost are raw numbers on our only hardware (AMD Strix Halo
  iGPU, ROCm, contended); plain PyTorch/transformers, no vendor-specific
  code. The board's official Intelligence folds in sealed items (ours
  renormalizes over the three public tiers, as the text variants'
  numbers do); the full section incl. the frozen-v1.5-method note is
  in the repo's docs/JEVBENCH.md.

**When to use which of our models** (numbers first; this repo makes no
leaderboard/placement claims): traffic with **images or Chinese** →
v1.1c, the only variant that handles either; pure-text well-formed
numeric/temporal/multi-step decisions → the measured rule skills favor
v1.1c (and the text variant -intuit over -meticulous); messy,
adversarial, unknown pure-text traffic whose confidence values are
consumed downstream → -meticulous's calibration is the documented safest.

## Pre-registered gates (all passed; one run, one reading)

| gate | reading | bar |
|---|---:|---:|
| synth3-val (n=590, English image decisions) | 0.9661 | > 0.8068 (frozen base) |
| synth3zh-val (n=204, Chinese image decisions; reported) | 0.9412 | — |
| MNLI-noul (n=150, English text, never trained on) | 0.8800 | ≥ 0.84 |
| OCNLI-noul (n=150, Chinese text, never trained on) | 0.8467 | ≥ 0.83 |
| synth2zh-val (n=403, Chinese rule skills) | 0.9132 | ≥ 0.90 |
| synth2-EN-val (n=609, English rule skills) | 0.9048 | ≥ 0.90 |
| fitted ECE (noul / choice / score) | 0.0241 / 0.0175 / 0.0685 | ≤ 0.10 |

## Quickstart

```bash
# serve from this download (merged model — no peft, no training artifacts)
HF_HUB_OFFLINE=1 python3 -m kapteeni.serve_v11c --dist ./ --port 8002
# or directly from this repo id, no download step needed:
python3 -m kapteeni.serve_v11c --hf TriusAI/kapteeni-v1.1c --port 8002
# API: POST http://localhost:8002/v1/systemone — the existing typesafe
# adapter and every Jev harness reach it unchanged.
# Demo website (15 pre-configured cases with hand-checked gold
# annotations, incl. real photographs): http://localhost:8002/
```

Example request:

```json
{"state": {"image": "<base64 png>", "note": "a cafe menu."},
 "model": "kapteeni-v1.1c",
 "questions": {"q1": {"type": "choice",
                      "instructions": "Which drink is the most expensive?",
                      "criteria": {"Espresso": null, "Latte": null,
                                   "Mocha": null}},
               "q2": {"type": "noul",
                      "instructions": "Is any drink above $9.00?",
                      "criteria": {"true": "above", "false": "not above"}}}}
```

**Image handling:** one image per request. Arriving images larger than a
**640px longest edge are downscaled to it** (aspect preserved; no crop,
no paste, no upscale) — small text and fine detail can become unreadable
at the bound, and accuracy may differ from what the full-resolution
image would give, because the model and all its gates were trained and
validated at 640×640; the response carries an advisory `notice` naming
this whenever the bound applies. Precision-critical callers should send
images at ≤ 640px on the long edge, or crop to the region of interest.
Question ids never reach the model; answers are content-only.

## Training data

Synthetic, gold by construction, all pixels/text ours (Apache-2.0): 20
image document families (English + fully-Chinese renders), binned
temporal/multi-hop/long-policy rule families (English + Chinese ports),
and an English text replay unioned with the v1.2-era surfaces. Teacher
soft labels for noul/replay: k-sample agreement from a cloud LLM judge
(fidelity documented). **Contamination:** the 8-token CJK-aware shingle
audit vs JevBench text public items + OCNLI shows zero hits
(scripts/contamination_audit.py); ImageJevBench items were never part of
an item-level audit (items and keys are not distributed), but the image
training data is programmatic template families into which benchmark
content cannot enter by construction. Val slices, MNLI and OCNLI were
never trained on; no gates or constants were adjusted after any
measurement.

## Licenses / provenance

Code + generators: **Apache-2.0** (matching the base model; the serving
package ships inside this snapshot). Trained weights: **CC BY-SA 4.0** —
the ShareAlike term comes from the MultiNLI + FEVER annotation lineage
in the text replay; full provenance and attribution guidance in
WEIGHTS-LICENSE.md. Backbone: Qwen/Qwen3.5-4B (Apache-2.0). Not
affiliated with TypeSafe; "Jev" is their model and trademark.'''


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