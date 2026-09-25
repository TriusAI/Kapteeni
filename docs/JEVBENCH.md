# JevBench v1.4 public-half run — kapteeni v0

Run date: 2026-09-24. Harness: official `jevbench` repo (`fstandhartinger/jevbench`,
commit at clone time), `TypeSafeAdapter` pointed at our own server
(`http://127.0.0.1:8000`, trained bundle `model_cache/kapteeni_v0.pt`). 231/231 public
decisions attempted, **0 failures, 231 schema-valid**. Scoring: the repo's own
`composite_v13.py` / `metrics.py`.

> **This is a self-reported public-half result, NOT an official JevBench rank.**
> The judge tier (146 items) is sealed, the calibration axis's gold-distribution
> half is private, and the official v1.4 score requires the 308 sealed items.
> Comparable precedent: the `stuntdouble` reports the benchmark repo links as
> "not an official JevBench result."

## Placement

| | value |
|---|---|
| JevBench-style score (public half) | **17.9** |
| would slot at | **~#52 of 73 ranked systems** |
| public accuracy (231 items) | **0.537** (Jev 1.13.0: 0.866 · 4B rebuilds: 0.78-0.81 · Raw Qwen3-4B logits: 0.697) |

Axes (their formulas):

| Axis | ours | Jev 1.13.0 | notes |
|---|---|---|---|
| Intelligence | **30.9** | 53.1 | 3 public tiers, chance-corrected, weights renormalized (judge sealed) |
| Calibration | **53.6** | 76.3 | ECE half only (0.232 top-label); board also averages private TVD half |
| Speed | **68.5** | 83.3 | p50 0.38 s / p95 9.48 s on **contended** Strix Halo, ×2 self-hosted adjustment |
| Cost | **42.4** | 52.0 | 597 input tokens/decision × assumed hosted Qwen-4B price ($0.14/M → $0.084/1k decisions); at $0.05/M the axis would be 55.8 |

Composite = geometric mean × (I/50)² low-Intelligence gate — the gate is what
caps our score (0.382 multiplier).

## Where it wins and where it fails (per family)

| family | n | acc | read |
|---|---:|---:|---|
| fact | 12 | **1.000** | BoolQ/FEVER noul training transfers perfectly |
| tool_selection | 12 | **1.000** | choice training transfers perfectly |
| extraction | 24 | **0.917** | strong |
| routing_hard | 5 | 0.800 | |
| policy | 12 | 0.667 | |
| intent | 24 | 0.625 | |
| ambiguous / adequacy / adversarial / probability | 6-12 | 0.500 | |
| judge_hard | 17 | 0.471 | |
| long_policy | 19 | 0.316 | 4k-token insurance policies — frozen backbone limits depth |
| trap | 8 | 0.250 | |
| temporal_numeric | 15 | 0.267 | counting/dates — **Jev's own documented jaggedness** |
| multi_hop | 18 | 0.222 | needs reasoning the frozen 4B doesn't do in one pass |
| routing | 12 | 0.250 | 9-category routing rubric, unlike Banking77's intents |
| ordinal | 12 | 0.250 | score-head trained only on HelpSteer2 doesn't transfer |

Paraphrase consistency: 0.722. Ordinal expectation MAE: 0.864.

## Honest reads

1. **The architecture works; the training distribution doesn't cover the task
   space.** Families matching our training data (fact = BoolQ/FEVER,
   tool_selection/extraction = Banking77-style) score 92-100%. Families outside
   it (ordinal rubrics, 9-way routing, multi-hop, long policy) fall to
   22-32%. A frozen backbone + 3.6k-row pilot can't buy reasoning it was never
   shown.
2. **In-domain calibration does not transfer.** Val ECE 0.048 in-domain became
   0.232 top-label ECE here — the heads are overconfident exactly on the
   families they get wrong. The reference's calibration claim survives in our
   in-domain numbers, not on out-of-domain rubrics.
3. **The temporal_numeric weakness reproduces Jev's documented jaggedness** —
   the reference model's own "weak counting, dates-as-text" failures show up in
   our implementation at greater magnitude. Parity of failure class, worse magnitude.
4. **Speed is understated.** The p95 (9.5 s) was measured while the host GPU
   simultaneously trained an unrelated model; hard-tier items are 4k-token
   states × 5 option passes. Uncontended, p95 ≈ 2-4 s → Speed ≈ 74-78 → score
   ≈ 19-20, still ~#50-52.
5. Cost bookkeeping in the run log (`charged=$4.70`) is the harness's nominal
   ledger against a default tariff; the real endpoint is our local GPU — the
   Cost axis above uses their "est." convention (hosted list price of the same
   weights × measured tokens) with the assumption stated.

## What would move the number (P2 and beyond)

- **LoRA on the backbone** (the plan's P2) — the single biggest lever for
  long_policy/multi_hop/trap, where frozen features can't represent the
  reasoning.
- **Wider score/ordinal training** beyond HelpSteer2 (arbitrary rubrics,
  ordinal scales with described levels — synthetic generators).
- **Routing-style choice training** beyond Banking77 (9-way categorical
  routing of task prompts).
- **Distillation on policy-reasoning distributions** — noting the benchmark's
  own warning: the public half may be trained on, but the sealed half evolves;
  generalization, not item-fitting, is the goal.

## Phase 1 (same day): verbalizer hybrid — 17.9 → 41.2

Diagnosis: the board's "Raw Qwen3 4B direct logits" row (I 46.4) outscored our
trained heads (I 30.9) — the frozen backbone's *native* decision readout
beats heads trained on narrow data, exactly on the out-of-distribution
families. We added the native readout (yes/no logits at the `answer:` position
— the same forward pass, ~free) as a second opinion:

| readout | easy | standard | hard | Intelligence | top-label ECE | score | slot |
|---|---:|---:|---:|---:|---:|---:|---:|
| head-only (v0) | 0.979 | 0.472 | 0.387 | 30.9 | 0.232 | 17.9 | ~#52 |
| hybrid w=0.7 *(pre-registered)* | 1.000 | 0.528 | 0.405 | 35.8 | 0.167 | 26.3 | ~#38 |
| hybrid w=0.5 | 1.000 | 0.556 | 0.414 | 37.9 | 0.128 | 30.8 | ~#31 |
| **hybrid w=0.25 (serving default)** | **1.000** | **0.625** | **0.423** | **42.5** | **0.065** | **41.2** | **~#15** |
| verbalizer-only | 0.958 | 0.597 | 0.432 | 40.3 | 0.060 | 36.8 | ~#23 |

Blend: `p ∝ p_head^w · exp((1−w)·s_verb/τ)` per question group; τ fit on
mixed-domain val (BoolQ/FEVER/MNLI/Banking77/HelpSteer2 val slices — no
JevBench data). The improvement is structural, not a lucky point: every step
toward the verbalizer improves OOD Intelligence *and* calibration
monotonically, while the head's contribution keeps the easy tier at 100%.

**Honesty notes:** w=0.25 was selected on the benchmark's public half — the
benchmark permits this ("can be trained on or selected against"), but the
sealed half rotates and the v1.4 gap penalty punishes public-overfitting, so
we flag it. The pre-registered choice (w fitted on mixed-val, 0.7) scores
26.3. A one-scalar selection on 231 items carries low overfitting risk, and
the monotone trend argues the choice is structural. Serving default: blend
(`kapteeni/serve.py --readout blend`, `BLEND_W=0.25`); `--readout head`/`verb`
restore the pure variants.

## P2 (kapteeni-v1): score 65.71, ~#2/73 — with asterisks, stated plainly

LoRA r=32 on all projections + retrained heads, one epoch over the
broadened 8.7M-token mix (BoolQ/FEVER soft labels + 3.6k ground-truth-by-
construction synthetic temporal/numeric/policy items + Banking77 + CLINC150
+ GoEmotions + HelpSteer2). MNLI stayed out of training as the OOD gate:
accuracy held 0.88 -> 0.893 through all 1,205 steps. All serving constants
(head temperatures, blend w/tau/b) refit on mixed-domain val through the
merged model — **zero public-half selection in this chain** (stronger
methodology than v0.1's flagged w=0.25).

| | v0.1 | kapteeni-v1 |
|---|---:|---:|
| easy / standard / hard | 1.000 / 0.681 / 0.405 | **1.000 / 0.889 / 0.469** |
| public accuracy | 0.615 | **0.710** |
| Intelligence | 44.5 | **60.3** |
| top-label ECE -> Calibration | 0.049 -> 90.1 | 0.0496 -> 90.1 |
| score / slot | 48.21 / ~#8 | **65.71 / ~#2 of 73** |

**The asterisks, in order of importance:**

1. **Jev's accuracy on these same public items is still higher** (public
   accuracy 0.866 vs our 0.710). Our Intelligence (60.3) is renormalized
   over the three public tiers because the judge tier (146 items) is
   sealed, while Jev's official 53.1 blends the judge tier in at weight
   0.28. On a like-for-like three-tier basis Jev's Intelligence would be
   higher than ours; our composite edge comes from Calibration (90 vs 76),
   and the Cost/Speed axes. The honest headline: **best open rebuild on the
   board by a wide margin (next: JevK5 62.04), competitive with Jev on the
   measurable public half — not proven ahead of it.**
2. Still a self-reported public-half run (no sealed items, no official
   rank); the sealed set rotates and the gap penalty would apply to a
   sealed measurement we cannot take.
3. Calibration is the ECE half only (the board averages a private
   gold-distribution TVD half); Cost is the stated $0.14/M assumption.

The v0.1 record (48.21) and its raw run remain in `docs/bench/` alongside
the kapteeni-v1 raw run.

## Board as of benchmark v1.4.2 (2026-09-25) — what changed and what didn't

The benchmark updated during our v1.2/v1.2.1 work (releases v1.4.1 and
v1.4.2, clone now at `1bcc55e`). Verified facts:

- **Unchanged:** the 231 public items, the 308 sealed set, the composite
  formula (equal-weight harmonic mean + I<50 gate). We reproduced the
  board's arithmetic exactly (Jev's published axes -> 63.33 vs official
  63.29, rounding), and `score_bench.py`'s v13 helpers compute the same
  composite — all our runs of record remain valid as scored.
- **Changed:** the board grew to 93 systems (89 ranked) and the top of it.
  New verified #1: decider-4b v2 (Mapika) 64.13 — a 4B open system like
  ours — with Jev at #2 (63.29), JevK5 #3 (62.04), Cygnet #4 (61.76).
  classifier.dev (70.82) remains unranked (resold Jev). Six new systems in
  v1.4.1 (none above JevK5) and eleven in v1.4.2.

**Our standing, honestly restated:** kapteeni-v1's self-reported public-half
composite (65.71) is the highest number on the v1.4.2 board as computable
from the public half, but (1) it is self-reported and unverified —
decider-4b v2's #1 passed independent verification including sealed items;
(2) the 1.6-point margin over it is inside our measured noise floor — a
tie, not a lead; (3) every verified system scores 33-39% on the sealed
half (decider: 83.5% public vs 34.7% sealed), which we cannot measure at
all — board Intelligence (49-53) folds that in and ours (60.3, public-only
renormalized) is not comparable; (4) on like-for-like public accuracy we
remain significantly behind Jev (86.6%) and decider-4b (83.5%). The
v1.0.0-era claim "best open rebuild by a wide margin" is retired: the
verified open leaders (decider-4b 64.13, Cygnet 61.76) sit in our tie
band.

**A discipline note the update makes concrete:** the #1-vs-#2 gap on the
official board is 0.84 points — smaller than our measured single-seed
noise floor and smaller than Cost-axis assumption swings. The top of this
board is a tie band by any honest reading.

## Statistical hygiene: how to read every number above

- **Sample sizes are small.** All accuracies are on 231 items (easy 48 /
  standard 72 / hard 111). 95% binomial CIs on public accuracy: v0.1
  0.615 ± 6.3pt, v1 0.710 ± 5.9pt, v1.1 soup 0.697 ± 5.9pt, Jev
  0.866 ± 4.4pt (from the board column). The Jev-vs-us accuracy gap
  survives both CIs; every other ordering claim at this sample size does
  not. Tier-level CIs are wider still (hard: ±9.3pt at v1's 0.469).
- **Neighbor gaps are ties.** The 2–4 composite-point gaps between the
  60–66 band systems are inside run-to-run variance plus axis
  assumptions (the Cost axis alone swings ±7 points across the stated
  price range).
- **A measured noise floor.** v1 vs the soup differ only by training seed
  and val-fit constants and scored 65.71 vs 63.09 — treat ~±2–3 composite
  points as the single-seed pipeline noise floor. Improvements below
  that are not evidence; we pre-register and report deltas against it.
- **ECE at n=231 is biased and high-variance.** Differences below ~0.02
  between single runs are unresolvable; the soup's 0.0496 -> 0.1046
  doubling is a real effect, but 0.05-vs-0.07 comparisons elsewhere in
  the table should not be read as meaningful.
- **Calibration is measured on the benchmark's distribution**, which is
  OOD relative to any deployment: a model whose constants are tuned for
  the bench mix is not thereby calibrated for yours. This failure mode
  is not hypothetical — it is exactly how the soup passed every val gate
  while its bench ECE doubled.
- Benchmark-wide structural limits (self-reported public half, sealed
  judge tier, assumed prices, hardware-confounded Speed) are restated in
  each section above and analyzed alongside a design proposal for a
  deployment-oriented alternative in `docs/DECISIONBENCH-DRAFT.md`.

## v1.2.1 (diverse-constant refit): score 63.18 — directionally fixed, not
## enough; the redesign budget is retired

Pre-registered in docs/PREREG-V1.2.1.md: freeze the seed-3 artifacts,
refit temperatures + blend on the deployment-diverse val set (mixed
590 + synth2 609), grids unchanged. Gates 1 and 2 passed before the run:
synth2-val ECE 0.0427 <= v1's 0.0630 (with every family-accuracy gain
kept: temporal 0.811 / multi-hop 0.917), mixed-val within tolerance; the
saturation itself is gone in the served output (choice 0.951 where v1.2
served 1.000, temperature 0.724 where v1.2's fit gave 0.075).

| | v1 | v1.2 | v1.2.1 |
|---|---:|---:|---:|
| Intelligence | 60.3 | **62.0** | 61.1 |
| top-label ECE -> Calibration | **0.0496** -> 90.1 | 0.2023 -> 59.6 | 0.1196 -> 76.1 |
| **score / slot** | **65.71 / ~#2** | 59.65 / ~#4 | 63.18 / ~#3 |

Gate 3 fails (63.18 < 64.71): **v1 stays the shipped model**, and — as
pre-registered — this is the last change gated on public-half outcomes.

**What the whole v1.x arc establishes:**
1. The family data is real (Intelligence 62.0, hard 0.496, synth2-val
   +19pt/+15pt) — the seed-3 artifacts are kept for any future use that
   prioritizes accuracy over this benchmark's calibration axis.
2. The constants protocol was improved (saturation eliminated, bench
   ECE halved) but v1's artifacts remain better transferable-calibrated
   on the bench distribution; part of the residual is that synthetic
   val slices share their generators' blind spots (v1.2 scores 0.035
   ECE on synth2 val while its bench ECE was 0.2023) — a finding that
   feeds the DecisionBench shift-ladder proposal directly.
3. Below ~66 composite, differences sit inside the measured noise
   floor and axis assumptions; further point-chasing on this
   benchmark is not a productive use of runs.

Raw run: `docs/bench/kapteeni-v1.2.1-record-231.jsonl`; summary:
`data_cache/bench_v1_2_1/summary.json`; artifacts:
`kapteeni_v1_2_1.pt`, `data_cache/phase1/fit_kv_v1_2_1.json`.

## v1.2 (weak-family continuation): score 59.65 — better decisions, broken
## confidence; the flagship gate fails and v1 stays shipped

Pre-registered candidate: seed 3, fresh training on the union of the v1
mix plus the synth2 weak-family data (17,848 rows / 91,554 passes /
11.1M tokens). Every gate passed ahead of the run: final MNLI 0.9067
(best of any seed), mixed-val non-regression (choice/score Brier
improved, noul within the ECE resolution), and the direct hypothesis
test on the 609-row synth2 val slice through the served readout:

| family (n) | v1 | v1.2 |
|---|---:|---:|
| temporal_numeric (286) | 0.601 | **0.794** |
| multi_hop (205) | 0.766 | **0.917** |
| long_policy (118) | 0.610 | 0.644 |
| overall (609) | 0.659 | **0.806** |

The public-half run then split the verdict:

| | v1 | v1.2 |
|---|---:|---:|
| easy / standard / hard | 1.000 / 0.889 / 0.469 | 1.000 / 0.889 / **0.496** |
| public accuracy | 0.710 | **0.723** |
| Intelligence | 60.3 | **62.0** (best measured) |
| top-label ECE -> Calibration | 0.0496 -> 90.1 | **0.2023** -> 59.6 |
| **score / slot** | **65.71 / ~#2** | 59.65 / ~#4 |

The family data DID teach the skills — Intelligence, hard tier, and both
target families improved — but the val-fit choice constants (pure head,
w 1.0, T 0.075) saturate probabilities to 1.0/0.0 and are confidently
wrong off-distribution, quadrupling bench ECE. The flagship gate
(> 1.0 composite drop) fails decisively: **v1 remains the shipped
model.**

**The protocol-level finding.** Two independent runs now show the same
mechanism: the soup (63.09) and v1.2 (59.65) both improved every val
metric while their bench ECE exploded (0.0496 -> 0.1046 / 0.2023).
Conclusion: fitting blend constants on the narrow mixed-domain val
slices is systematically OOD-fragile — val-optimal sharpness does not
transfer. The fix is not tuning against the bench (never); it is
refitting constants on a *deployment-diverse* held-out set (mixed-domain
val + MNLI val + synth2 val — all excluded from training by
construction), pre-registered before any further run.

**Accumulated-exposure note, honestly:** this is our fifth public-half
run, and the last two redesigns were informed by bench outcomes. Each
step was mechanism-driven and one-run-per-candidate, but the aggregate
record is weakly bench-informed; the v1.2.1 protocol fix must be the
last change gated this way, or the "no bench selection" claim erodes.

Artifacts kept: `model_cache/kapteeni_p2_s3/` (adapter + heads),
`kapteeni_v1_2.pt`, `data_cache/phase1/fit_kv_v1_2.json`, family-val
JSONs (`data_cache/familyval_v1*.json`), raw run
`docs/bench/kapteeni-v1.2-record-231.jsonl`.

## v1.1 (three-seed soup): score 63.09 — a negative result, kept honest

Pre-registered (docs/PREREG-V1.1-V1.2.md) before the run: average three
seeds of the exact v1 recipe (seeds 1-2 retrained at budget 8192, both
clearing the MNLI gate: 0.900 / 0.887 vs v1's 0.893), elementwise over
merged weights, refit constants on mixed-val. The mixed-val gates all
PASSED — per-primitive temps-stage ECE improved (noul 0.0574 -> 0.0542,
choice 0.0731 -> 0.0639, score 0.0872 -> 0.0646), so the run was spent.

The public half disagreed, hard:

| | v1 (seed 0) | v1.1 (soup) |
|---|---:|---:|
| easy / standard / hard | 1.000 / 0.889 / 0.469 | 1.000 / 0.875 / 0.451 |
| public accuracy | 0.710 | 0.697 |
| Intelligence | 60.3 | 58.4 |
| top-label ECE -> Calibration | 0.0496 -> 90.1 | 0.1046 -> 79.1 |
| Speed / Cost | 81.1 / 42.4 | 81.0 / 42.3 |
| **score / slot** | **65.71 / ~#2** | **63.09 / ~#3** |

In-domain calibration improved; out-of-domain calibration collapsed
(top-label ECE doubled). The val-fit blend constants for the soup leaned
much harder on the verbalizer (choice w 0.1 -> 0.3, tau 2 -> 8; score
w 0.4 -> 0.7) — optimal on the mixed-domain val slices, fragile off them.
Averaging three runs also plausibly lands the model between loss basins in
a way that shifts the verbalizer's OOD behavior, which the head-side fit
cannot see.

What this result buys us: (a) ensembling is disqualified for this
model+data until the OOD mechanism is understood — v1.2's candidate is the
weak-family continuation alone, per the amendment in the pre-reg; (b) the
val-fit-constants methodology now has a demonstrated failure mode, which
the README's caveats mention; (c) the discipline held — no post-hoc
constant search was run against the bench to "fix" the soup. Raw run:
`docs/bench/kapteeni-v1.1-record-231.jsonl`; summary:
`data_cache/bench_v1_1/summary.json`. Soup weights kept in
`model_cache/kapteeni_soup/` for reproduction.

## Run of record (2026-09-24, post double-softmax fix): score 48.2, ~#8/73

The Phase-1 blend had a serving bug: `_answer()` passed blend PROBABILITIES
into the contract's choice/score shapers, which softmax again — every choice/
score answer was flattened (deployed ECE 0.209; easy-intent-00's 0.4044 is
softmax-of-softmax to the digit). Fixed to pass PRE-softmax scores; verified
in-process, by stage trace, and end-to-end:

| | v0 morning | v0.1 run of record |
|---|---:|---:|
| easy / standard / hard | 0.979 / 0.472 / 0.387 | **1.000 / 0.681 / 0.405** |
| public accuracy | 0.537 | **0.615** |
| Intelligence | 30.9 | **44.5** |
| top-label ECE | 0.232 | **0.049** → Calibration 90.1 |
| p50 / p95 | 0.38s / 9.48s | **0.17s / 1.19s** → Speed 81.1 |
| Cost (assumption) | 42.3 | 42.3 |
| **score** | **17.9** | **48.21** |
| slot | ~#52 | **~#8 of 73** |

Neighbors on the board: metask-jev-4b 47.78, SemIf 47.69, djev 52.23,
reflex 4B 53.99, Winnow-12B 55.58. All previous caveats stand (public half,
self-reported; judge tier sealed; ECE-half calibration; cost = stated
assumption; w=0.25 public-half-selected, flagged).

## Repro

```bash
# server with the trained bundle
python3 -m kapteeni.serve --bundle model_cache/kapteeni_v0.pt --port 8000
# bench (from jevbench/)
TYPESAFE_ENDPOINT=http://127.0.0.1:8000 TYPESAFE_API_KEY=local \
TYPESAFE_PRICE_INPUT_PER_M=0 TYPESAFE_PRICE_OUTPUT_PER_M=0 \
python3 -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --results <out>.jsonl --raw-dir <raw>
python3 ../score_bench.py <out>.jsonl
```