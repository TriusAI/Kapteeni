# Pre-registration: Kapteeni v1.1 — unified base (images + Chinese + new backbone)

Written 2026-09-28, before any v1.1 measurement. v1.1 is a new model
line: one model, one wire contract, images + text + Chinese. It
supersedes the two-variant text-only v1 line (which stays shipped as
kapteeni-v1-meticulous / -intuit) and absorbs the abandoned dedicated-
visual track (V2/V2.1 negative results documented in
PREREG-KAPTEENI-V.md: mixed single-LoRA recipes erode text competence
regardless of replay volume; image skills train robustly).

## Why a new backbone at all

The v1 text backbone was Qwen3-4B-Instruct-2507; the visual track used
its VL sibling. That was inheritance, not a choice. The 2026 landscape
was researched on 2026-09-28 and the Qwen family now spans
Qwen3.5/3.6/3.8 (natively multimodal from 3.5 on). Staying on Qwen by
default means competing inside the family everyone already uses.

## Step 0 — BASE-SELECTION PROBE (fixed now; zero training)

Frozen-backbone probe, one pass per question, same lettered LM-head
readout as the V0 probe (letters for choice/score, yes/no for noul;
thinking modes disabled wherever the chat template supports them;
deterministic, temperature-free).

**Candidates (fixed):**

1. `Qwen/Qwen3.5-9B` — 9B dense VL (2026-03), Apache-2.0, C-Eval 88.2,
   vendor MMMU 78.4 / OCRBench 89.2; hosted fp8 ~$0.04-0.06/M
   → ~$0.024-0.036/1k decisions (Jev-class ✓)
2. `Qwen/Qwen3.5-4B` — same family, budget control; ~$0.018/1k (✓)
3. `zai-org/GLM-4.6V-Flash` — ~10B dense VL (2025-12; verified config:
   `Glm4vForConditionalGeneration`, 40L text + 24-depth ViT), MIT,
   native zh+en (MMBench-CN 85.9); Z.ai API free tier (✓)

**Considered, excluded (documented, with reasons):**
- `Qwen3.8-27B` (2026-08): trainable on 96GB, but $0.45/M input
  (OpenRouter, the only existing price) → $0.27/1k decisions — 3.4×
  over the $0.080 Jev-class cutoff — and it is exactly the monoculture
  base the v1.1 switch is meant to differentiate from.
- `InternVL3.5-8B`: the language backbone is Qwen3 — differentiation
  would be cosmetic.
- `MiniCPM-V 4.5/4.6`: 4.5 archived 2026-02; license lineage requires
  commercial registration; 4.6 is 1B built on Qwen3.5-0.8B.
- `Kimi-VL-A3B`: 2025 generation; 16B resident for serving.
- `Gemma 3/4`: Gemma Terms are not standard-OSS-clean; Chinese quality
  below the Chinese-native candidates.

**Slices (fixed):**
- `synth3-val` — all 590 held-out image items (ours; identical to the
  V-track gate slice).
- `MNLI-noul` — first 150 of `data_cache/rows_mnli.jsonl` (identical to
  the V-track text gate slice; GLUE MNLI validation_matched,
  entailment→true binary noul; never trained on).
- `OCNLI-noul` — first 150 of a NEW `data_cache/rows_ocnli.jsonl`,
  built from CLUEbenchmark/OCNLI `dev.json` with the same conversion as
  MNLI, fully Chinese instruction/criteria templates:
  instructions `前提是否蕴含以下假设？假设：{hypothesis}`,
  criteria true `前提蕴含该假设` / false `前提不蕴含该假设`.
  OCNLI label mapping 0=entailment → true, 1/2 → false (asserted at
  build time against the CLUE reader).
  **License note:** OCNLI is CC BY-NC 2.0 — EVAL-ONLY: never trained
  on, not redistributed, only aggregate accuracies reported; identical
  usage pattern to MNLI. The contamination audit extends to it.

**Bilingual noul readout (fixed):** for the OCNLI slice the model's
binary decision is read as argmax over token groups —
yes-group = P("yes") + P("是"), no-group = P("no") + P("否") — because
a frozen backbone answering a Chinese prompt may answer in either
language. MNLI keeps the plain yes/no readout. No other readout rules
differ between models; all three candidates get the identical harness.

**Selection rule (fixed before measurement):**
- Primary: highest **mean of the three slice accuracies**.
- If candidates fall within 0.03 of the leader's mean: prefer the
  non-Qwen candidate (GLM-4.6V-Flash) for base-family differentiation;
  then prefer lower hosted cost.
- The winner's frozen probe numbers become v1.1's per-gate frozen
  baselines, fixed at probe time, before any training.
- If the winner's frozen synth3-val < 0.70 (the V0 feasibility line),
  the image requirement is re-examined before any training —
  documented, not silently absorbed.

**Cost axis (fixed):** each candidate's hosted list price is recorded
at probe time (~600 input tokens/decision, the v1 measured figure);
Jev-class eligibility requires <= $0.080/1k decisions.

## Step 0 AMENDMENT (2026-09-28, after the GLM smoke, before any full comparison)

The GLM-4.6V-Flash smoke (4 items/slice) exposed a harness-format
assumption: GLM answers the noul prompt with "true"/"false" (echoing
the criteria labels) rather than "yes"/"no" — at the answer position
"false" is top-5 and "yes" sits at rank 147, so the pre-registered
yes/no readout measures noise there (MNLI smoke: 0.5 = chance).
The lettered choice readout is unaffected (GLM's letter argmax matched
gold on the debug item). GLM's template was also verified by rendered-
prompt inspection to honor enable_thinking=False (an empty think pair;
jinja silently ignores unknown kwargs, so "no error" alone would have
proven nothing).

**Instrument repair (applied identically to ALL candidates):**
- noul EN: decision = argmax over probability mass on {yes, true}
  vs {no, false} (sum of exponentials of the readout logits).
- noul ZH: same, over {yes, true, 是} vs {no, false, 否}.
- choice/score: unchanged (argmax over letter logits).

All three candidates are (re-)probed with the amended instrument; the
first-pass Qwen3.5-9B/4B reports (pure yes/no readout) are kept for
the record but superseded. No gate values change; the selection rule
is unchanged; this is a measurement-instrument repair, not a gate
adjustment — a readout that misses a model's answer vocabulary measures
the tokenizer, not the decision.

## Step 0 OUTCOME (2026-09-28) — GLM excluded on stack grounds; the
## tie-break selects Qwen3.5-4B

**Amended-instrument results** (`data_cache/baseprobe/*.json`):

| candidate | synth3-val | MNLI-noul | OCNLI-noul | MEAN | cost /1k |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 0.8203 | 0.8800 | 0.8733 | **0.8579** | ~$0.036 |
| Qwen3.5-4B | 0.8068 | 0.8667 | 0.8600 | **0.8445** | ~$0.018 |
| GLM-4.6V-Flash | — stalled — | — | — | (unmeasured) | ~$0 |

**GLM exclusion (stack, not model):** the GLM probe stalled three
times at the identical stack location — the ViT patch Conv2d
(modeling_glm4v.py:765), grinding single-core for 12+ minutes on
first-touch batches — because its dynamic-resolution tower triggers
per-image-shape MIOpen kernel searches on ROCm (MIOPEN_FIND_MODE=1 did
not help; faulthandler traces recorded in data_cache/probe_glm.log).
Since v1.1 must train and serve on this box, an inference path this
pathological here disqualifies GLM as our base regardless of its
measured quality — which remains UNKNOWN, not low. Revisitable only
under a fresh pre-registration with a different stack (a CUDA host, a
vLLM build, or a standardized single-grid image pipeline, the last of
which would break identical-harness fairness for all candidates).

**Selection (rule applied as pre-registered):** 9B leads by 0.0134
mean — inside the 0.03 tie window (each 150-item slice carries a ~2-3pt
SE; the 9B's per-slice lead is consistent but sub-noise). With no
non-Qwen candidate in the tie, the cost tie-break decides:
**Qwen3.5-4B is the v1.1 backbone** (~$0.018/1k hosted fp8 vs ~$0.036).
Corroborating external evidence recorded after the fact: the top open
system on the JevBench v1.3-era board (SemIf, formerly OpenJev) runs
on Qwen3.5-4B.

**v1.1 frozen baselines (from this probe, fixed):** synth3-val 0.8068,
MNLI-noul 0.8667, OCNLI-noul 0.8600.

## v1.1 proper — recipe pre-registered after the probe, before training

Placeholder until the base is chosen (this section gets the full recipe
then): one wire contract with the `state.image` extension (per the
V-track design), image + text + Chinese traffic, the V2/V2.1 lesson
applied (modality handling chosen at recipe time with text
non-regression as a hard gate), LoRA training on license-clean data
(our synth families incl. Chinese-language variants + existing replay),
per-primitive temperature scaling fit on held-out val from day one.

Gates will be fixed relative to the winner's frozen baselines measured
in Step 0, plus the standing rules: MNLI-noul non-regression,
OCNLI-noul non-regression vs frozen, synth3-val > frozen, calibration
(ECE) gates on held-out val.

## Not allowed

No training or constant selection on any benchmark's items (MNLI,
OCNLI, synth3 val, imajev-bench, JevBench); no gates adjusted after
results; one run per candidate recipe; honest negatives documented.