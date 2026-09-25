# Pre-registration: v1.2.1 — deployment-diverse constant refit

Written before any run. This is the LAST experiment gated on public-half
bench outcomes; whatever happens, the bench-informed redesign budget is
retired after it (see the accumulated-exposure note in JEVBENCH.md).

## Background (mechanism, not bench-tuning)

Two independent pre-registered runs failed the same way: the soup
(bench ECE 0.0496 -> 0.1046) and v1.2 (-> 0.2023) both improved every
mixed-val metric while bench calibration collapsed. Mechanism: serving
constants (head temperatures T, blend w/tau/b) fitted on the narrow
mixed-domain val slices select sharpness/verbalizer mixes that do not
transfer out of domain. Both v1.2 model artifacts (adapter + heads) are
the best measured on Intelligence (62.0) and on the family skills
(temporal 0.79 / multi-hop 0.92 on synth2 val).

## The experiment

- **Model artifacts are FROZEN**: seed 3's adapter + heads, exactly as
  evaluated in v1.2. No retraining of anything.
- **Only the constants change**: temperatures and blend (w, tau, b) are
  refit on the union of
  (a) the existing mixed-domain val items (590: BoolQ/FEVER/MNLI val,
      Banking77 val, HelpSteer2 val) and
  (b) the synth2 validation slice (609 held-out rows, never trained on,
      added via the same expand_passes machinery),
  all extracted through the same merged seed-3 model. **The fit grids are
  unchanged from v1/v1.2** (no new search dimensions).
- Output: `kapteeni_v1_2_1.pt` + `data_cache/phase1/fit_kv_v1_2_1.json`.
  v1's and v1.2's constant files remain untouched.

## Gates — fixed now, before any measurement

1. **Saturation gate (measured off-bench):** top-label ECE on the synth2
   val slice (familyval.py) for v1.2.1 must be <= v1's on the same slice.
   This is the directly measurable symptom of the failure mode; v1.2's
   value is expected to fail it.
2. **Mixed-val non-regression:** per-primitive mixed-val ECE (the old fit
   set, reported separately by the fit script) within +0.02 of v1's
   (noul 0.0574 / choice 0.0731 / score 0.0872).
3. **Flagship gate (the one bench run):** public-half composite
   >= 64.71 (= v1's 65.71 minus the 1.0 tolerance) -> v1.2.1 becomes the
   shipped model and README headline. Below that -> v1 stays shipped and
   the result is reported as the final negative result. Either way: one
   run, no re-runs, no post-hoc constant edits, and no further
   bench-gated design changes in this repo.

## What is NOT allowed

- No new grids, no grid restriction chosen by looking at v1.2's bench
  answers, no per-family or per-tier constant tuning, no blend variants
  evaluated on bench items. The diverse fit set is the only change.

## OUTCOME (2026-09-26, after the single pre-registered run)

Gates 1 and 2 passed (synth2-val ECE 0.0427 <= v1's 0.0630 with all
family-accuracy gains kept; mixed-val within tolerance; saturation
eliminated: served choice 0.951 vs v1.2's 1.000, choice T 0.724 vs
0.075). Gate 3 failed: composite 63.18 < 64.71 (Intelligence 61.1,
bench ECE 0.1196 — halved from v1.2's 0.2023 but above v1's 0.0496).
**v1 stays shipped.** The bench-informed redesign budget is retired as
pre-committed. Secondary finding: synthetic val slices are not OOD
proxies — v1.2's constants score 0.035 ECE on synth2 val while its bench
ECE was 0.2023.