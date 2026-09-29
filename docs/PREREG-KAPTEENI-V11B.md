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
- **Abort rule (pre-registered)**: if any monitor drops by more than
  0.03 absolute between consecutive rounds, kill the run — that is an
  erosion signal the recipe cannot absorb; document and stop.

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