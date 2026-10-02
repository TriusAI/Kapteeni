# Pre-registration: Kapteeni v1.1c — the v1 phase curriculum ported to the
# multimodal base

**Committed before any v1.1c measurement.** One run. Gates frozen below —
identical to v1.1/v1.1b's table, nothing relaxed, nothing added.

Status: the unified arc has two documented negatives (PREREG-KAPTEENI-V11
Outcome: synth2zh 0.7767 < 0.90 with image/NLI gates all passing;
PREREG-KAPTEENI-V11B Outcome: volume falsified — skills climbed to
0.846/0.841, peaked mid-run, then declined under epoch-2 memorization while
MNLI eroded 0.86 -> 0.82). The v1.1b decision rule allows exactly ONE more
pre-reg under a genuinely new mechanism hypothesis, otherwise the
multimodal project parks. User decision (2026-09-29): pursue the
phase-curriculum port — this pre-reg.

## Mechanism hypothesis (new, not volume)

Every unified run so far used **single-pass lettered answer-SFT**: one
prompt per row-question listing all options, supervised on one answer
token. The v1 text flagship used a different structure end to end — the
**phase pipeline**:

- **P1**: per-primitive readout heads (PassMLP) on the frozen backbone's
  final-token hidden state, one pass per (question[, option|level]);
- **P2**: LoRA + heads trained JOINTLY, heads warm-started from P1;
- **multi-pass judgment structure**: each option/level is judged against
  the state in its own pass; choice is a group softmax over the row's
  option passes **in-loss**; score is per-level independent BCE; noul is
  absolute BCE against **teacher-distilled soft targets** (k-sample
  agreement, weight = 1 − std);
- proper scoring rules only — no answer-token convention anywhere.

Evidence this is the right port (all from this repo's own history):

1. v1.2 (the phase pipeline + synth2 passes) reached the best synth2
   skills ever measured (served readout: multi_hop 0.917, temporal
   0.794-0.811, overall ~0.81) **and** the best-ever MNLI gate 0.9067 —
   the only configuration that lifted the rule skills without NLI
   erosion. Every lettered variant (V2, V2.1, v1.1, v1.1b) either eroded
   NLI or plateaued 0.78-0.86 on the skills.
2. The heads readout is answer-vocabulary-free: no letter-convention
   alignment (the 2026-09-28 image-CE spike), no bilingual yes/是
   instrument repair — the decision comes from a learned projection on
   the hidden state, identical machinery for every language and modality.
3. The v1.1b plateau at final-train-loss 0.0002 is a memorization
   signature of hard-target answer SFT; the distillation-weighted soft
   targets and group-CE of the phase pipeline are the documented
   counter-mechanism.

Honest risk, stated before the run: no configuration, including the
pipeline being ported, has ever measured >= 0.90 overall on the full
synth2 val slice (phase-pipeline best ~0.81 through the served blend).
This run tests whether the mechanism transfers to the natively-multimodal
base with the zh data in the mixture. If it does not reach the frozen
bars, the arc parks per the v1.1b rule — no further mechanism on the
table.

## Recipe (frozen)

**Backbone:** `model_cache/qwen3.5-4b` (the v1.1 probe-selected base;
frozen probe baselines carry: synth3-val 0.8068, MNLI-noul 0.8667,
OCNLI-noul 0.8600). Vision tower always frozen.

**Train data** (deterministic splits by `sha256(row_id) % 10`; val rows
and gate rows never trained on):

1. **The v1.2 union verbatim**: `data_cache/passes_p2.jsonl` (81,145
   passes: boolq/fever noul with teacher soft labels + weight;
   bank77/clinc150/goemotions/helpsteer2/synth choice/score gold) +
   `data_cache/passes_synth2.jsonl` (11,498 passes, gold). These files
   are reused byte-identical — the exact texts v1.2 trained on.
2. **synth2zh** — NEW multi-pass expansion of
   `data_cache/rows_synth2zh.jsonl` (same expansion rules as synth2;
   gold targets; `synth2zh_criteria.json` descs/levels).
3. **synth3 + synth3zh image passes** — NEW multi-pass expansion of the
   20 image families: one pass per option/level, the row's render
   attached to EVERY pass of the row (the multi-pass structure is the
   mechanism; option sets are small, 2-7, no subsampling needed).

**Pass text layout:** `kapteeni/serialize.py`'s multi-pass format
(noul_pass / choice_option_pass / score_level_pass), unchanged, wrapped
in the model's chat template (the v1.1 line's serving surface — its
probe baselines and gates are template-based; v1's template-free
convention is a text-backbone choice that does not transfer to a
vision-placeholder model). Image passes attach the image via the
processor's vision placeholder.

**P1 (heads on frozen features):** precompute h_last (final real token,
inner decoder output, bf16, batched, resumable) for all train+val passes;
train the three PassMLP heads with the v0 hyperparameters verbatim
(noul 40 epochs / choice 12 / score 30, lr 1e-3, AdamW wd 1e-4,
group-CE/BCE/weighted-BCE as in `kapteeni/train.py`); fit per-primitive
temperature on the val split.

**P2 (LoRA + heads, joint):** the `kapteeni/train_p2.py` recipe verbatim
— LoRA r=32, alpha=64, dropout 0.05 on the **language model's**
q/k/v/o/gate/up/down projections; lr 1e-4; token budget **8192**; ONE
epoch; warmup 100 steps + cosine; gradient checkpointing; grad clip 1.0;
heads warm-started from P1 and trainable; batches are ROWS (a row's
passes stay together for the group losses), length-sorted, packed to
the real-token budget (image pass cost = text tokens + 399 vision
tokens, the measured Qwen3.5-4B figure). Fresh LoRA (no v1.1-adapter
warm start — entangling the lettered convention with the heads
mechanism would confound the test).

**Monitors (NOT gates), every 300 steps, appended to
`data_cache/v11c_monitors.jsonl`** (the v1.1b log-loss insurance):
MNLI-noul[150], OCNLI-noul[150], synth3-val[:300], synth2zh-val[:200],
synth2-EN-val[:200] — read through the heads readout on the live model.

**Abort rule (carried verbatim from v1.1b's amended rule):** kill the
run if any monitor falls more than **0.10** below the best value that
monitor has shown so far in this run.

**Engineering before the run (allowed; not gate measurement):** h_last
extraction smoke on this backbone; LoRA target-name verification; image
vision-token re-measurement; OOM smoke (~100 steps, peak-memory check);
resume verification; the 8-word contamination audit extended to the
goemotions/synth surfaces (the one mixture addition vs the v1.1-audited
set).

## Gates (final reading on the completed model, full slices, heads
## readout; thresholds IDENTICAL to v1.1b)

| gate | requirement | basis |
|---|---|---|
| synth3-val (n=590) | > 0.8068 | unchanged (beat frozen; the point of the line) |
| MNLI-noul (n=150) | >= 0.84 | unchanged (frozen 0.8667 − 1 binomial SE) |
| OCNLI-noul (n=150) | >= 0.83 | unchanged (frozen 0.8600 − 1 SE) |
| synth2zh-val (n=403) | >= 0.90 | unchanged (the failed v1.1 gate) |
| synth2-EN-val (n=609) | >= 0.90 | unchanged (the failed v1.1b mastery bar) |
| fitted ECE on combined held-out val | <= 0.10 | unchanged |

ECE procedure (fixed, the v1.2.1 lesson applied in advance): per-primitive
temperatures refit on the **combined** held-out val (synth3-val +
synth3zh-val + mixed-domain text val + synth2zh-val + synth2-EN-val),
through the completed model; gate = **every** primitive's fitted
ECE <= 0.10 (metrics.py's 15-bin ECE; noul = sigmoid probs vs gold,
choice/score = top-label confidence vs correctness).

Reported for the record, not gated: synth3zh-val (n=204).

**Decision rule:** pass ALL gates -> fit constants, extend serving with
the `state.image` wire field, ship as **kapteeni-v1.1c** (single
variant). Fail ANY gate -> no ship, honest negative documented with the
monitor trend; **the unified arc parks and v1 remains the shipped line**
(the v1.1b rule — there is no further mechanism hypothesis after this
one).

## Not allowed

No training or constant selection on any benchmark's items (MNLI, OCNLI,
synth3 val, imajev-bench, JevBench); no gates adjusted after results;
one run per candidate recipe; honest negatives documented. Contamination:
the v1.1 audit's zero hits carry over for the shared mixture; the
goemo/synth addition is audited before training; val slices never
trained on.

## ENGINEERING NOTE (2026-09-30, added before any completed run; the
## NaN-casualty attempt never completed and is documented below)

The first P2 launch died at step 20 with loss NaN. Diagnosis (probes
committed as WORKLOG log entries): the **bf16 vision tower produces
non-finite embeds** on borderline image batches under training-mode
perturbation — reproduced twice at the same seeded batch (a
6-level weather-forecast score batch, each image repeated 6x in the
forward, a composition unique to the multi-pass mechanism), localized
to the tower output (everything downstream NaN from layer 0); the
frozen-feature precompute (95,484 passes) is 100% finite, as is every
eval-mode forward and single-pass forward; a fresh model can also be
tipped by batched identical sequences, and fp32-accumulation matmul
knobs do NOT clear it — the base's bf16 tower sits numerically marginal
on this content. Fixes applied, verified on the exact repro (tower
output finite, batch trains, 40 clean steps):

1. **The vision tower runs in fp32** (the multimodal merge already
   casts pixel inputs to the tower's dtype; its fp32 embeds flow back
   into the bf16 stream). The LM stays bf16. Full-fp32 also clears the
   repro but costs ~2x — not taken.
2. **Backstop skip-guard**: any batch whose forward yields non-finite
   h_last is skipped (no optimizer step), logged, and counted; > 20
   skips abort the run. Expected count with the fp32 tower: 0.

These are stack-numerics repairs of the instrument, not recipe changes:
LoRA config, data, losses, lr, budget, gates and decision rule are
exactly as pre-registered above. The NaN casualty never reached a
checkpoint or monitor round and is not a run of record.

## Outcome (2026-10-02, ~09:20) — ALL GATES PASS: ship

One run: 3,398 steps / 24.1M real tokens / ~46h wall / zero skipped
batches (the fp32 tower held for the entire run) / no abort triggers /
peak <= 40G of 96G. Monitor history: data_cache/v11c_monitors.jsonl
(11 rounds; skills flat ~0.70 through step 900, climbed from 1200 on,
one mixture-region dip at 1800 well inside the abort rule, final
round 3300: MNLI 0.88 / OCNLI 0.8533 / synth3 0.96 / synth2zh 0.915 /
synth2-EN 0.90).

Final gates on the completed model (full slices, heads readout, fitted
temperatures 1.2071/0.9911/1.8348 on combined val, per the frozen
procedure):

| gate | result | requirement | verdict |
|---|---:|---|---|
| synth3-val (n=590) | 0.9661 | > 0.8068 | PASS |
| MNLI-noul (n=150) | 0.8800 | >= 0.84 | PASS |
| OCNLI-noul (n=150) | 0.8467 | >= 0.83 | PASS |
| synth2zh-val (n=403) | 0.9132 | >= 0.90 | **PASS** |
| synth2-EN-val (n=609) | 0.9048 | >= 0.90 | **PASS** |
| fitted ECE (noul/choice/score) | 0.0241 / 0.0175 / 0.0685 | <= 0.10 | PASS |
| synth3zh-val (reported, ungated) | 0.9412 (n=204) | — | arc-best |

First configuration to clear the 0.90 mastery bars — v1.1 0.7767,
v1.1b 0.8462/0.8407 with MNLI eroded 0.86 -> 0.82, best-ever text
line (v1.2 through the served blend) ~0.81. The full-slice readings
sat AT OR ABOVE the [:200] monitor subslices (the lettered
instrument's gap did not reappear). The mechanism verdict: the
multi-pass judgment structure + distillation-weighted targets +
heads readout delivered the rule skills AND the NLI gates AND the
image families simultaneously — the "skills vs. erosion" trade that
every single-pass answer-SFT run hit (V2, V2.1, v1.1, v1.1b) is
absent from this run's entire trajectory.

Per the pre-registered decision rule: fit constants (done — the
fitted temperatures above), extend serving with the `state.image`
wire field, ship as **kapteeni-v1.1c**.