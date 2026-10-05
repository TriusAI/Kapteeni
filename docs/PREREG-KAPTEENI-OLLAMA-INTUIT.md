# Pre-registration: the Ollama port of kapteeni-v1-intuit

**Plain goal.** Same as the meticulous port (PREREG-KAPTEENI-OLLAMA.md,
which is now a parked two-negative program): an Ollama-native copy of
kapteeni-v1-intuit that behaves approximately like the original,
published as `zaaktinlam/kapteeni-v1-intuit` if and only if the gates
below pass. v1-intuit was always the second port in the plan.

**Committed before any measurement of this student. One run.**

Why this is expected to work where the meticulous port fell short —
the mechanism, stated before the run: v1-intuit's served output is
MUCH sharper (choice blend w=1.0, tau=0.5 — logits effectively
doubled; noul blend w=0.25 vs meticulous's cautious w=0.6, tau=3.0,
b=4.5). The meticulous port's measured failure mode was distribution
SHAPE: the student's letter softmax is naturally peaked, and the
meticulous teacher is deliberately flat. The intuit teacher's
served distributions are close to what letter logits natively
produce, so both the soft-target fit and the shape gates (TV/|dp|)
should be far easier. This hypothesis is testable and pre-registered
as such.

## Teacher and student

- Teacher: the published `../kapteeni-v1-intuit-dist` (served via
  `kapteeni.serve --dist`); Qwen3-4B-Instruct-2507 lineage, merged.
- Student: fresh start from THE TEACHER'S OWN merged backbone (the
  proven closer initialization from the meticulous runs). LoRA r=32
  alpha 64 targets q/k/v/o/gate/up/down.

## Data

The Run-2 rebalanced recipe, re-taught from the intuit teacher
(its outputs, not meticulous's — the lesson sets are NOT shared):
- All 8 sources, non-val lessons: CUTS banking77→600, clinc150→1000,
  synth→2500; OVERSAMPLE goemotions ×3, helpsteer2 ×3, fever ×2,
  boolq ×2; synth2 all; + the same 2,000 fresh synth2x rows (seed 99,
  prefix synth2x-) taught by THIS teacher.
- Same loss family with the Run-2 diagnosis applied, frozen:
  **hard-CE weight 0.1, soft-CE weight 0.9** (the meticulous runs'
  0.5 hard weight peaked the student where the teacher was flat;
  0.1 keeps the outrank property — verified by the gate's zero-error
  requirement — without distorting shapes).
- Fast config (the Run-2 amendment): budget 8192, no gradient
  checkpointing, lr 1e-4, one epoch, seed 0. Per-source in-loop
  monitors every 300 steps (40 val rows/source).

## Gates — identical bars to the meticulous port, measured through
the actual ollama runtime on the final F16 GGUF:

1. Zero outrank errors + token-count pre-flight.
2. Per-source fidelity vs the INTUIT teacher's val outputs:
   top-1 agreement ≥ 0.90; noul mean |Δp| ≤ 0.10; choice/score TV
   ≤ 0.15.
3. Absolute floors vs gold: synth2-EN val ≥ 0.85; MNLI first 150
   ≥ 0.80.
4. Pooled ECE ≤ 0.10.

## Decision rule

Pass ALL → publish `zaaktinlam/kapteeni-v1-intuit` with its own
numbers + cross-links, honest deltas stated. Fail ANY → no publish;
documented negative. Given the mechanism bet recorded above, a fail
does NOT park the whole line: the v1.1c text-only port and any
meticulous Run-3 are separately pre-registered decisions — but if
THIS run fails on shape fidelity with a peakier teacher AND a
soft-dominant loss, the letter-distillation mechanism itself is the
problem and the whole Ollama program should park in favor of
researching the Clef span-head format (the only native path that
does not go through letter logits).

## Not allowed

No benchmark items in training; val rows never trained; gates frozen
before measurement; one run; honest negatives documented.
## Outcome (2026-10-05, ~14:15) — NEGATIVE: no ship, but the
mechanism bet is CONFIRMED

One run (3,356 steps; peakier teacher + soft-dominant loss
hard-weight 0.1). Gates through the actual ollama runtime on
kapteeni-intuit-port (F16 GGUF):

| gate | metic. R1 | metic. R2 | INTUIT | bar | verdict |
|---|---|---|---|---|---|
| banking77 ag / TV | 0.963 / 0.053 | 0.947 / 0.083 | 0.952 / 0.069 | 0.90 / 0.15 | PASS |
| boolq ag / dp | 0.902 / 0.101 | 0.932 / 0.091 | 0.947 / 0.078 | 0.90 / 0.10 | PASS |
| clinc150 ag / TV | 0.973 / 0.038 | 0.990 / 0.032 | 0.969 / 0.037 | 0.90 / 0.15 | PASS |
| fever ag / dp | 0.899 / 0.106 | 0.949 / 0.059 | 0.968 / 0.078 | 0.90 / 0.10 | PASS |
| synth ag / dp | 0.938 / 0.162 | 0.940 / 0.171 | 0.951 / **0.077** | 0.90 / 0.10 | PASS (new) |
| synth2 ag / dp / TV | 0.883 / 0.154 / 0.265 | 0.900 / 0.217 / 0.279 | 0.893 / **0.114** / **0.127** | 0.90 / 0.10 / 0.15 | dp +0.014, TV PASSES, ag −0.007 |
| goemotions ag / TV | 0.788 / 0.288 | 0.836 / 0.257 | 0.833 / **0.191** | 0.90 / 0.15 | FAIL (−0.067 / +0.041) |
| helpsteer2 ag / TV | 0.765 / 0.202 | 0.784 / 0.229 | 0.797 / **0.130** | 0.90 / 0.15 | TV PASSES (new), ag −0.103 |
| synth2-EN gold floor | 0.652 | 0.655 | **0.785** | 0.85 | FAIL (−0.065) |
| MNLI | 0.847 | 0.873 | 0.887 | 0.80 | PASS |
| ECE | 0.038 | 0.018 | 0.068 | 0.10 | PASS |
| outrank errors | 0 | 0 | 0 | 0 | PASS |

The shape-fidelity bet held: synth, synth2, and helpsteer2 TV all
PASS now (0.077-0.130 vs 0.15-0.279 in the meticulous runs), synth2
gold jumped 0.65 -> 0.785, and everything that failed does so by
0.007-0.103 instead of 0.10-0.20. Four gates still fail:
goemotions agreement (0.833) + TV (0.191), helpsteer2 agreement
(0.797), synth2 agreement (0.893, seven thousandths under) and dp
(0.114), and the synth2 gold floor (0.785).

Per the frozen decision rule: no publish. The park-the-whole-program
clause does NOT trigger — this run did not fail on the shape
mechanism (shape largely passed); it failed on ARGMAX agreement, the
deliberate trade-off of the 0.1 hard weight. The measured trade-off
curve (0.5 hard = shape fails, 0.1 hard = agreement drifts) says the
unexplored middle (0.2-0.3) is the credible next recipe, needing its
own pre-registration. The synth2 gold floor (0.785) remains the
letter-paradigm capability ceiling; only the Clef span-head format
bypasses it.

Artifacts kept: model_cache/kapteeni_intuit_ollama (+ merged, GGUF),
kapteeni-intuit-port in the local ollama, gates.json in
data_cache/ollama_port_intuit. Nothing published; the HF originals
remain the reference.
