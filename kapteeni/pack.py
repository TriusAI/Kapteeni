"""Pack a Kapteeni v1 variant distribution for release / Hugging Face.

    python3 -m kapteeni.pack --served-as kapteeni-v1-meticulous
    python3 -m kapteeni.pack --served-as kapteeni-v1-intuit \
        --lora model_cache/kapteeni_p2_s3/adapter \
        --heads model_cache/kapteeni_p2_s3/heads.pt \
        --bundle model_cache/kapteeni_v1_2_1.pt \
        --fit data_cache/phase1/fit_kv_v1_2_1.json

Produces a self-contained directory (default ../kapteeni-v1-<variant>-dist):
  - merged model (base + LoRA) as safetensors + tokenizer   (--adapter-only
    for a small upload instead: peft adapter + instructions)
  - heads.safetensors (readout heads, no pickles)
  - kapteeni-config.json (variant name, head temperatures, blend constants)
  - README.md  — variant-specific HF model card (frontmatter + numbers +
    variant guidance; no leaderboard claims)
  - LICENSE (Apache-2.0, code) + WEIGHTS-LICENSE.md (CC BY-SA 4.0, weights)
  - kapteeni/  — the serving package, so the dist runs without the GitHub repo

Serve it:  python3 -m kapteeni.serve --dist <dir> --port 8000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kapteeni.p2_finalize import load_merged  # noqa: E402

MODEL_CARD = """---
license: cc-by-sa-4.0
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
- decision-model
- structured-decisions
- calibration
- probability-estimation
- text-classification
language:
- en
library_name: transformers
---

# Kapteeni v1 @@SERVED_AS@@ — a Jev-compatible System One decision model

Send a `state` plus typed questions; get back **calibrated probability
distributions** your code can branch on. No text generation. Kapteeni
implements the [TypeSafe System One](https://docs.typesafe.ai) decision-model
interface (the wire format of Jev, `POST /v1/systemone`).

@@BLURB@@

## Quickstart

```bash
pip install torch transformers safetensors huggingface_hub   # plus the kapteeni package below
python -m kapteeni.serve --dist . --port 8000        # run from this snapshot
curl localhost:8000/v1/systemone -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
      "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry"]}
  }}'
```

Or in-process:

```python
from kapteeni.model import SystemOneModel
m = SystemOneModel.from_dist("<snapshot dir>")
answers, usage = m.evaluate(state, questions)
```

The `kapteeni/` Python package is included in this snapshot; the full repo
(data pipeline, training, tests) lives on GitHub.

## Question types (the Jev contract)

| type | answer | readout |
|---|---|---|
| `noul` | `{type, noul}` — P(yes) in [0,1] | absolute: sigmoid head on the final hidden state; **not** complement-consistent (P(A) + P(not-A) may differ from 1, matching the reference) |
| `choice` | `{type, choice, probabilities, confidence}` | relative: per-option passes through the backbone, shared head, group softmax; probabilities sum to exactly 1 |
| `score` | `{type, score, legend, probabilities, confidence}` | independent levels; score is the probability-weighted expectation and can land between levels |

Serving adds a **verbalizer blend**: the backbone's own next-token yes/no
logits are geometrically blended with the trained head (weights in
`kapteeni-config.json`), which measurably improves out-of-domain robustness
and calibration. `--readout head` or `--readout verb` select pure variants.

## Benchmark (JevBench v1.4 public half, self-reported)

231 public decisions, scored with the benchmark's own code, end-to-end
through the server. Numbers only — no placement claims; see caveats.

@@BENCH@@

Axes common to both variants: Speed 81.0 (p50 0.17 s, p95 1.2 s on an
AMD Strix Halo iGPU, x2 self-hosted adjustment) · Cost 42.4 (597 input
tokens/decision at an assumed $0.14/M hosted price). Composite =
harmonic mean; the Intelligence<50 gate does not apply.

**Caveats, stated plainly:** self-reported public half (judge and sealed
items are private; not an official rank); all serving constants
pre-registered on held-out validation (no benchmark selection);
Intelligence is renormalized without the sealed judge tier; Calibration
shown is the ECE half only. Differences of a few points are within
single-seed pipeline noise (~±2-3 composite points, measured).

## How it was trained

@@TRAINED@@

## Training data provenance

| data | license |
|---|---|
| Qwen3-4B-Instruct-2507 (base) | Apache-2.0 |
| GoEmotions | Apache-2.0 |
| Banking77, HelpSteer2 | CC BY 4.0 |
| BoolQ | CC BY 3.0 |
| MultiNLI | CC BY-SA 4.0 |
| FEVER (underlying) | CC BY-SA 3.0 |
| CLINC150 | research use (no explicit license) |
| synthetic items | Apache-2.0 (this project) |

**Weights: CC BY-SA 4.0** (the ShareAlike terms of MultiNLI/FEVER carry the
recommendation; see `WEIGHTS-LICENSE.md` for the full attribution
obligations). **Code: Apache-2.0** (`LICENSE`).

## Limitations

@@LIMITS@@
- Probabilities are calibrated in aggregate; individual answers are not
  guaranteed correct — branch on confidence where it matters.
- Not affiliated with or endorsed by TypeSafe AI; "Jev" is their model and
  trademark; this is an independent implementation of the documented
  interface, evaluated on the public JevBench items.
"""

_VARIANT_TEXT = {
    "kapteeni-v1-meticulous": {
        "blurb": (
            "**This variant: kapteeni-v1-meticulous** — conservative "
            "confidence. The safe default for unknown or messy traffic.\n\n"
            "Kapteeni v1 ships as two variants of the same architecture. "
            "The other, kapteeni-v1-intuit, decides more accurately on "
            "well-formed numeric, temporal, and multi-step policy traffic "
            "but is less careful about its confidence there."
        ),
        "bench": (
            "| JevBench-style score | Intelligence | top-label ECE | public accuracy |\n"
            "|---:|---:|---:|---:|\n"
            "| **65.71** | 60.3 | 0.0496 (Calibration 90.1) | 0.710 (easy 1.000 / standard 0.889 / hard 0.469) |"
        ),
        "trained": (
            "Qwen3-4B-Instruct-2507 (frozen for the data pipeline, then) "
            "LoRA r=32 on all attention and MLP projections for one epoch "
            "over an 8.7M-token decision mix: BoolQ + FEVER "
            "(teacher-soft-labeled with k=5 sampled agreement), Banking77, "
            "CLINC150, GoEmotions, HelpSteer2, and 3.6k synthetic "
            "temporal/numeric/policy items with ground truth by "
            "construction. Readout heads (2-layer MLPs on the final hidden "
            "state) trained with proper scoring rules only. Per-head "
            "temperature scaling; blend constants fit on held-out "
            "mixed-domain validation. Out-of-domain gate: MNLI excluded "
            "from training, held 0.88 -> 0.893 through all 1,205 steps."
        ),
        "limits": (
            "- English-primary; long-policy and multi-hop reasoning remain "
            "weak (hard-tier accuracy 0.469); temporal/numeric judgment is "
            "unreliable on this variant — prefer -intuit for such traffic."
        ),
    },
    "kapteeni-v1-intuit": {
        "blurb": (
            "**This variant: kapteeni-v1-intuit** — sharper decisions. "
            "Strongest on well-formed numeric, temporal, and multi-step "
            "policy traffic (measured skills-slice accuracy 0.81 vs 0.66 "
            "for -meticulous, with better calibration on that slice).\n\n"
            "Kapteeni v1 ships as two variants of the same architecture. "
            "The other, kapteeni-v1-meticulous, is the conservative-"
            "confidence default for unknown or messy traffic."
        ),
        "bench": (
            "| JevBench-style score | Intelligence | top-label ECE | public accuracy |\n"
            "|---:|---:|---:|---:|\n"
            "| **63.18** | 61.1 | 0.1196 (Calibration 76.1) | 0.714 (easy 1.000 / standard 0.903 / hard 0.468) |"
        ),
        "trained": (
            "Qwen3-4B-Instruct-2507 with LoRA r=32 on all attention and "
            "MLP projections for one epoch over an 11.1M-token union of "
            "the base decision mix (BoolQ/FEVER soft labels, Banking77, "
            "CLINC150, GoEmotions, HelpSteer2, synthetic v1) plus 6.3k "
            "synth2 items: temporal/numeric with wide format and boundary "
            "diversity, multi-hop eligibility/process/fee chains, and "
            "grammar-built long policy documents — all ground truth by "
            "construction. MNLI OOD gate held 0.88 -> 0.907. Serving "
            "constants (temperatures, blend) refit on a deployment-diverse "
            "held-out set (mixed-domain val + the synth2 val slice) per a "
            "pre-registered protocol; saturation-checked end to end."
        ),
        "limits": (
            "- Confidence on messy, adversarial, or out-of-distribution "
            "inputs is less reliable than -meticulous (public-half ECE "
            "0.12 vs 0.05): on unknown traffic, prefer -meticulous."
        ),
    },
}


def _variant_text(served_as: str) -> dict:
    return _VARIANT_TEXT.get(served_as) or _VARIANT_TEXT["kapteeni-v1-meticulous"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="",
                    help="default: ../kapteeni-v1-<variant>-dist")
    ap.add_argument("--served-as", default="kapteeni-v1-meticulous",
                    choices=["kapteeni-v1", "kapteeni-v1-meticulous",
                             "kapteeni-v1-intuit"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--lora", default="model_cache/kapteeni_p2/adapter")
    ap.add_argument("--heads", default="model_cache/kapteeni_p2/heads.pt")
    ap.add_argument("--fit", default="data_cache/phase1/fit_kv.json")
    ap.add_argument("--bundle", default="model_cache/kapteeni_v1.pt",
                    help="head bundle whose temperatures the config carries")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--adapter-only", action="store_true",
                    help="ship the LoRA adapter (~300MB) instead of the "
                         "merged model (~8GB); users merge at load time")
    args = ap.parse_args(argv)

    variant = ("intuit" if "intuit" in args.served_as else "meticulous")
    out = Path(args.out or f"../kapteeni-v1-{variant}-dist")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print("loading merged model ...", flush=True)
    model, tok = load_merged(args.model, args.lora, args.device)
    model.config.use_cache = True

    if args.adapter_only:
        # small upload: ship the adapter; the base model is pulled from HF
        from peft import PeftModel

        pm = PeftModel.from_pretrained(model, args.lora)
        pm.save_pretrained(str(out / "adapter"))
        pm = pm.unload()
    else:
        print("saving merged model + tokenizer ...", flush=True)
        model.save_pretrained(str(out), safe_serialization=True)
        tok.save_pretrained(str(out))

    # heads -> safetensors (qt-prefixed state dicts)
    from safetensors.torch import save_file

    heads_sd = torch.load(args.heads, weights_only=True)
    flat = {}
    for qt, sd in heads_sd.items():
        for k, v in sd.items():
            flat[f"{qt}.{k}"] = v.contiguous().float().cpu()
    save_file(flat, str(out / "heads.safetensors"))
    print(f"heads.safetensors: {len(flat)} tensors")

    fit = json.loads(Path(args.fit).read_text())
    cfg = {
        "served_as": args.served_as,
        "in_dim": model.config.hidden_size,
        "head_temperatures": {},  # filled below
        "blend": fit,
        "readout": "blend",
        "format": "kapteeni-dist-v1",
    }
    # temperatures from the variant's head bundle
    bundle = torch.load(args.bundle, weights_only=False)
    cfg["head_temperatures"] = {qt: bundle[qt]["T"]
                                for qt in ("noul", "choice", "score")}
    (out / "kapteeni-config.json").write_text(json.dumps(cfg, indent=2))

    # model card + licenses + the code package
    vt = _variant_text(args.served_as)
    card = (MODEL_CARD
            .replace("@@SERVED_AS@@", args.served_as)
            .replace("@@BLURB@@", vt["blurb"])
            .replace("@@BENCH@@", vt["bench"])
            .replace("@@TRAINED@@", vt["trained"])
            .replace("@@LIMITS@@", vt["limits"]))
    (out / "README.md").write_text(card, encoding="utf-8")
    root = Path(__file__).parent.parent
    shutil.copy(root / "LICENSE", out / "LICENSE")
    shutil.copy(root / "WEIGHTS-LICENSE.md", out / "WEIGHTS-LICENSE.md")
    pkg = out / "kapteeni"
    shutil.copytree(root / "kapteeni", pkg,
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())