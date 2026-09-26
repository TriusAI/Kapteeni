# Pre-registration: kapteeni-v1-committee (output-ensembling the two variants)

Written before any committee measurement. The bench-informed redesign
budget is retired (docs/PREREG-V1.2.1.md): this experiment is evaluated
**entirely on deployment-side measurements** — the synth2 validation
slice and measured latency — with no benchmark runs at any point.

## Mechanism

The model soup failed by averaging WEIGHTS across loss basins (65.71 ->
63.09). Output-averaging is the theoretically safe form of ensembling:
each model stays in its own basin; per-decision probabilities are
averaged and every downstream field (argmax, expectation, confidence,
rounding) is re-derived with the contract's own formulas
(`kapteeni/committee.py`). Serving cost is honestly reported: input
tokens are the decision's (billed once), output tokens count BOTH
models' internal passes; true serving cost is ~2x GPU time.

Committee members (fixed):
- **kapteeni-v1-meticulous**: `kapteeni_v1.pt` + `kapteeni_p2/adapter`
  + `data_cache/phase1/fit_kv.json`
- **kapteeni-v1-intuit**: `kapteeni_v1_2_1.pt` + `kapteeni_p2_s3/adapter`
  + `data_cache/phase1/fit_kv_v1_2_1.json`

## Hypothesis

Averaging captures most of intuit's decision accuracy while keeping
meticulous's calibration — i.e., the committee's accuracy/ECE profile
dominates or matches the better of the two members on most of the slice.

## Measurements and gates (fixed in advance)

On the 609-row synth2 validation slice, through the merged readout
(familyval.py, --committee), on an idle GPU:

| gate | criterion | reference values |
|---|---|---|
| G1: accuracy | committee >= **0.75** | meticulous 0.659, intuit 0.814 |
| G2: calibration | committee top-label ECE <= **0.055** | meticulous 0.063, intuit 0.043 |
| G3: latency | measured p50 <= **0.45 s** (~0.9 s adjusted; Jev-class with margin) | members ~0.17 s measured |

Per-family numbers are reported for temporal_numeric / multi_hop /
long_policy alongside.

## Decision rule

- All gates pass: ship **kapteeni-v1-committee** as a third documented
  serving mode with its numbers in the README variant table.
- Any gate fails: document as a negative result in the WORKLOG; keep the
  code (the merge is pure and tested) but do not promote the mode.

No re-runs, no gate adjustments after the fact, no benchmark exposure.

## OUTCOME (2026-09-26, negative result)

| gate | criterion | measured | verdict |
|---|---|---|---|
| G1 accuracy | >= 0.75 | **0.801** | pass (97% of intuit's gain) |
| G2 top-label ECE | <= 0.055 | **0.0795** | **fail** (members: 0.063 / 0.043) |
| G3 latency | p50 <= 0.45 s | **0.336 s** | pass |

Per-family: temporal 0.776 (ECE 0.081), multi_hop 0.917 (ECE **0.137**),
long_policy 0.661 (ECE 0.092). Mechanism: output-averaging dilutes
intuit's correct confidence with meticulous's hedging — the committee
makes intuit's decisions at meticulous's confidence, i.e. systematic
UNDERconfidence, which top-label ECE punishes more than either member's
own error. The mode is NOT promoted; the merge code stays (pure, tested).
Third failed combining mechanism after weight-space (soup) and
constants-space (v1.2/v1.2.1): meticulous's calibration does not average.
The surviving idea from this line is ROUTING (pick one member per
request) rather than merging — a new experiment would need its own
pre-registration.