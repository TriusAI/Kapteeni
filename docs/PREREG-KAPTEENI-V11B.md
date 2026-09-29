# Pre-registration: Kapteeni v1.1b (synth2-heavy continuation)

**Committed before any measurement.** One run. Gates frozen below.
Status: the v1.1 unified recipe is a documented negative (see its
Outcome section: synth2zh gate 0.7767 < 0.90; the diagnosis shows
synth2-EN equally weak (0.775) and the weakness replicated across the
V2 backbone/LoRA/LR (0.7373) — the single-epoch unified replay cannot
re-master the synth2 rule skills, while zh transfer itself is healthy).

## Hypothesis

Volume: the synth2 rule skills (temporal_numeric, multi_hop,
long_policy) need more than one epoch on the multimodal base. The v1
pipeline reached mastery through synth2-heavy phases; if N additional
synth2-only epochs on top of the v1.1 adapter close the gap, the
unified approach survives with a volume-corrected curriculum. If they
do not, the volume hypothesis is dead and the unified arc's verdict is
"phase-style curriculum or nothing" — documented either way.

## Recipe (the b-run)

- **Warm start**: the saved v1.1 adapter (3,327 steps; gates 1-3
  already passing on it — re-measured at the end, never assumed).
- **2 additional epochs**, each containing ONLY:
  - synth2-EN non-val (5,691 rows) + synth2-zh non-val (3,799 rows),
    shuffled together — 9,490 rows per epoch (verified counts; the
    v1.1 build_examples 'text' source 14,935 = five-source replay
    9,244 + synth2-EN 5,691).
  - No image rows, no NLI/classification replay, no new data, no
    distillation changes. Rationale: v1.1's monitors stayed perfectly
    stable through a full mixed epoch (zero erosion), so pure-synth2
    epochs are the maximal-volume test of the hypothesis; the monitor
    rule below catches erosion if wrong.
- **LoRA, LR, optimizer: unchanged** from v1.1 (r=32 attention-only,
  lr 5e-5, AdamW, bf16, right-padding, answer tokens yes/no + letters
  in all languages). Gradient checkpointing ON (the 22G stable
  config); ckpt + resume every 150 steps; the process runs detached;
  every monitor round appended to data_cache/v11b_monitors.jsonl (log
  loss insurance after the first attempt lost its monitors).
- In-loop monitors every 300 steps (diagnostics, NOT gates): MNLI[150],
  OCNLI[150], synth3-val[:300], synth2zh-val[:200], synth2-EN-val[:200].
- **Abort rule (pre-registered; AMENDED before the run started — the
  original >0.03-between-rounds threshold fired on the 100-row smoke
  at step 18 from pure sampling noise (SE of a difference of two n=200
  readings is ~0.038; a 0.03 drop is <1 sigma), so it was
  miscalibrated, not evidence-driven): kill the run if any monitor
  falls more than 0.10 below the best value that monitor has shown
  so far in this run — noise-immune, catches only catastrophic
  erosion (the V-track's worst observed erosion moved ~7 points).**

## Gates (final reading on the completed adapter; full slices; same
thresholds as v1.1 — not relaxed, one added)

| gate | requirement | basis |
|---|---|---|
| synth3-val (n=590) | > 0.8068 | unchanged (the v1.1 bar) |
| MNLI-noul (n=150) | >= 0.84 | unchanged |
| OCNLI-noul (n=150) | >= 0.83 | unchanged |
| synth2zh-val (n=403) | >= 0.90 | unchanged (the failed v1.1 gate) |
| synth2-EN-val (n=609) | >= 0.90 | NEW — mastery bar, both languages |
| fitted ECE (combined val) | <= 0.10 | unchanged |

## Decision rule

Pass ALL gates -> fit per-primitive temperatures on combined val
(the v1.1 finalize step), extend serving with the `state.image` wire
field, ship as **kapteeni-v1.1b** (single variant). Fail ANY gate ->
no ship, negative documented with the monitor trend; the unified arc
gets ONE more pre-reg only under a genuinely new mechanism hypothesis
(not "more volume"), otherwise the multimodal project parks and v1
remains the shipped line.

## Not allowed

No training or constant selection on any benchmark's items (MNLI,
OCNLI, synth3 val, imajev-bench, JevBench); no gates adjusted after
results; one run per candidate recipe; honest negatives documented.
Contamination: datasets unchanged from v1.1 (the extended audit's zero
hits carry over); val slices never trained on.

## Outcome (2026-09-29, ~22:35) — NEGATIVE: no ship

One run (1,802 steps = 2 synth2-only epochs on the v1.1 adapter;
monitor history data_cache/v11b_monitors.jsonl). Final gates (full
slices): synth3-val 0.8729 PASS; OCNLI 0.8533 PASS; **MNLI 0.82 FAIL**
(gate 0.84; eroded from the warm start's 0.86 under pure-synth2
pressure); **synth2zh 0.8462 FAIL**; **synth2-EN 0.8407 FAIL**;
fitted ECE moot.

The volume hypothesis is FALSIFIED as the fix: +2 epochs lifted both
skill gates substantially (+7.0 ZH, +6.6 EN — real, monitor-confirmed
climbs that peaked mid-run at 0.895 / 0.845 around step 1200) and then
DECLINED as epoch-2 memorization took over (final train loss 0.0002;
the late-round monitors reversed: synth2zh 0.895 -> 0.86, EN 0.845 ->
0.80). The plateau sits ~15 points below the mastery bar, and the same
pressure that lifts the skills erodes the NLI gates (MNLI 0.86 ->
0.82; OCNLI 0.8867 -> 0.8533). Cross-check: the four unified-track
runs to date (v1.2, V2, v1.1, v1.1b) top out at 0.82-0.85 on synth2
mastery under every replay/epoch variation tried; the v1 phase
pipeline (multi-pass, distillation-heavy) is the only configuration
that has reached mastery on these families.

Per the decision rule: the "more volume" mechanism is spent; the
remaining genuinely-different mechanism on the table is porting the
v1 phase curriculum (synth2-heavy phases with distillation) to the
multimodal base — a build decision, not a recipe tweak.