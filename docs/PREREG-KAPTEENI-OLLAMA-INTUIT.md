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