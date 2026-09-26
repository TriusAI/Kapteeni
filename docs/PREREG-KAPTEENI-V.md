# Pre-registration: Kapteeni-V — the visual decision track

Written 2026-09-27, before any Kapteeni-V measurement. The image track
has **no JevBench exposure at any point** — its gates are entirely
deployment-side (probe/val accuracy, calibration, latency, text
non-regression). The text bench budget stays retired.

## Why this track

The image-Jev incumbents (djev; Visual Jev / PixelJev systems) converge
on one recipe — vision-language backbone, candidate-token readout,
shared image-prefix serving — and share one weakness: **uncalibrated
probabilities** (djev documents its own as experimental; PixelJev:
"accuracy gains do not ensure calibrated target probabilities"). Our
differentiator is the discipline the incumbents skip: temperature
scaling, pre-registered constants fit on deployment-diverse validation
from day one, and per-slice honesty.

## Backbone (fixed now)

**Qwen/Qwen3-VL-4B-Instruct** — the same family and size class as the
text backbone (Qwen3-4B-Instruct-2507), native in transformers 5.15
(`Qwen3VLForConditionalGeneration`), Apache-2.0. Rationale: minimizes
text-competence regression risk (shared tokenizer family, familiar text
behavior), fits the training recipe that worked (LoRA r=32 on the
language projections, vision tower frozen), and fits the box.

## Wire-format extension (fixed now)

`state` gains an image field — `{"image": "<data URI or local path>",
...}` — accepted by the contract as an ordinary state field; the model
layer routes it to the vision tower. Questions/criteria stay textual.
The image occupies the **shared prefix**; question/option/level suffixes
run against it exactly as text suffixes do (the design Visual Jev
measured at 8.9x on multi-question images — and our serving code already
implements it for text).

## Stage gates (fixed before any run)

- **V0 — feasibility probe (frozen backbone, zero training).** The
  synth3 seed renders 27 items (menus, calendars, forms) with gold by
  construction; the frozen model reads them through the lettered
  LM-head readout, one pass per question.
  Gate: **>= 70% accuracy**, deterministic across two runs, every
  answer schema-valid. If the frozen 4B VL cannot read menus and forms,
  the track stops here and the result is documented.
- **V1 — synth3 data at scale.** 20+ rendered template families
  (documents, schedules, UI states, charts), facts-first, ~6-10k items,
  selfcheck invariants + generator determinism. No license risk: every
  pixel is ours (Apache-2.0), extending the synth2 approach.
- **V2 — training.** LoRA on the language projections (vision tower
  frozen) + fresh heads, on image-decision mixes **with text replay**
  from the existing training data.
  Gates: image-val accuracy/ECE strictly better than the frozen
  backbone; **text non-regression** (mixed-val and synth2-val within
  noise of the shipped text variants' numbers; the MNLI text gate still
  >= 0.88) — a visual Kapteeni may not be dumber about text.
- **V3 — constants + serving.** Temps + blend fit on
  **deployment-diverse val from day one** (image val + text mixed val +
  synth2 val — the v1.2.1 lesson applied in advance); shared image-prefix
  KV serving; latency profile recorded.

## Serving identity when shipped

A third variant line: `kapteeni-vX-visual` (name fixed when it ships),
alongside -meticulous and -intuit; same wire contract, model cards with
numbers only, no leaderboard claims anywhere.

## Not allowed

No training or constant selection on any benchmark's items (including
imajev-bench); no gates adjusted after results; text non-regression is
a hard gate, not a preference.