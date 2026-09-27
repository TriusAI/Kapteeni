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

**Amendment (2026-09-27, before any V2 measurement).** The V0 probe
passed at 77.8% with the **one-pass lettered LM-head readout** (no
heads, no training), and Visual Jev's published matched control found
typed heads offer no consistent accuracy advantage over the LM-head
readout on visual tasks. Per-option passes would also repeat each
~400-token image per option (4-5x the training compute for no expected
gain). V2's readout is therefore fixed as: **one pass per question,
options as a letter list, answer-token supervision (yes/no for noul,
letter for choice/score) through the LM head, per-primitive temperature
scaling fit on held-out val.** Typed heads are deferred to a control
experiment only if the primary run underperforms its gates. The text
replay uses the same lettered format so the VL model's text-decision
behavior is measurable in its own readout.
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

## V2 OUTCOME (2026-09-27) — SPLIT: image gate PASSES, text gate FAILS

First training attempt OOM-died at step 60 (fake-unit batch budget
packed ~3x too heavy; silent kernel kill, no checkpoint) — fixed with
real-token batching (peak 17.5G) + 150-step checkpoints/resume, and the
full run completed cleanly: 1,584 steps, ~4.9M real tokens, ~5.3h.

Final gate measurement on the completed adapter
(`data_cache/v2_final_gate.json`):

| gate | required | final | frozen | verdict |
|---|---|---:|---:|---|
| synth3-val (image) | > 0.8119 | **0.8458** | 0.8119 | PASS |
| MNLI noul (text) | >= 0.88 | **0.8333** | 0.90 | **FAIL** |

The image skills are trainable (frozen 0.812 -> 0.846-0.883 across
readings; the data works). The text behavior regressed from the start
and never recovered: in-loop MNLI readings were 0.833 / 0.883 / 0.858 /
0.808 / 0.850 with no trend back toward the frozen model's 0.90 — the
image-heavy mix (~2:1 image:text in real tokens) crowded out text
decisions despite replay. Per the pre-registration this V2 recipe is
NOT shipped and the same recipe is not rerun.

**Pre-registrable next experiment (not run):** token-parity replay —
weight the text mix so text:images is ~1:1 in real tokens (roughly 2x
the text replay volume), same LoRA recipe, same gates. The mechanism
diagnosis (insufficient replay, not a training-skill problem) points
at the mix ratio, not the method.

## V2.1 — token-parity replay (pre-registered 2026-09-27, before running)

The V2 failure mechanism was ratio, not method: text behavior fell at
the FIRST gate and never recovered under a ~2:1 image-heavy token mix.

**Correction (2026-09-27, measured BEFORE any V2.1 run):** real-token
accounting (tokenizer pass) shows the assumptions above were wrong.
Measured: image side = ~2.58M tokens (not 4.3M), text prompts average
~440 tokens (not ~340) — so V2's actual mix was ~0.76:1 image:text,
ALREADY near parity, and the text gate still failed. The "2:1
image-heavy" mechanism story was an estimate error. What survives:
V2's text replay was 6.5k rows (~3.4M tokens) of decision-format-only
items; text competence still eroded 0.90 -> 0.833.

**Amended V2.1 hypothesis and recipe:** the intervention is *much more
text weight* at its strongest cheap setting — text replay = ALL
available non-val rows from the six sources (14,935 rows, ~6.58M
tokens, ~0.39:1 image:text, 2.3x V2's text volume). Everything else
unchanged: image data (all synth3 train rows), LoRA recipe, lr, budget,
gates.

**Decision rule (unchanged):** final gates measured on the completed
adapter — synth3-val > 0.8119 AND MNLI >= 0.88. Pass -> V3. Fail either
-> second documented negative result AND the hypothesis space moves to
structural options (modality-routed adapters, LR, target-module
selection), not further ratio tuning. Same-recipe reruns remain
forbidden.

## V0 OUTCOME (2026-09-27) — PASSED

Frozen Qwen3-VL-4B-Instruct (zero training), lettered LM-head readout,
one pass per question, 27-item probe:

| family | accuracy |
|---|---:|
| menu_board | 11/12 = 0.917 |
| calendar_card | 3/6 = 0.500 |
| form_sheet | 7/9 = 0.778 |
| **overall** | **21/27 = 0.778 >= 0.70 gate** |

Deterministic across two runs (bit-identical probability vectors).
Findings: price reading and threshold comparisons are near-perfect
(noul on menus 6/6, several at conf ~1.0); the calendar family's
"which week" choices fail 0/3 at HIGH confidence (0.81-0.95) — a
layout-counting weakness, precisely the kind of skill that
ground-truth-by-construction training data teaches, and a signal that
the frozen backbone's confidence is NOT calibrated where it is wrong
(the Kapteeni-V differentiator in one sentence).

Report: `data_cache/vprobe/report.json`. The track proceeds to V1.