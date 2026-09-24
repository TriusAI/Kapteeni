# E2 calibration report — kapteeni v0

Bundle: `model_cache/kapteeni_v0.pt` (trained 2026-09-24 12:01, emb `data_cache/emb.pt`).

Validation rows use the deterministic `sha256(row_id) % 10` split; val rows carry gold targets and full option sets (see WORKLOG).

## Noul (absolute readout)

- **ECE vs gold: 0.1081** | Brier 0.1187 | accuracy 0.834 | n=470 | T=1.0249
- Fidelity vs teacher (distillation): ECE 0.0479 | Brier 0.0595 | n=422
- Teacher ceiling vs gold on the same datasets: BoolQ ECE 0.130 / FEVER ECE 0.074 (measured pre-training, see WORKLOG)

## Choice (relative readout, Banking77)

- **Top-1 accuracy: 0.9259** over a mean of 77.0 options/row | ECE 0.0370 | n=189 val rows (full 77-option sets) | T=1.1365

## Score (independent levels, HelpSteer2)

- **Top-1 accuracy: 0.5621** | expectation MAE 0.893 | ECE 0.0480 | n=153 | T=2.4053

## Held-out dataset (OOD, MNLI — never trained)

- **ECE 0.0495** | Brier 0.0829 | accuracy 0.8800 | n=500 | T=1.0249

| P range | n | mean P | freq true |
|---|---|---|---|
| 0.0–0.1 | 259 | 0.022 | 0.019 |
| 0.1–0.2 | 35 | 0.141 | 0.257 |
| 0.2–0.3 | 15 | 0.247 | 0.467 |
| 0.3–0.4 | 8 | 0.342 | 0.5 |
| 0.4–0.5 | 9 | 0.457 | 0.556 |
| 0.5–0.6 | 7 | 0.557 | 0.286 |
| 0.6–0.7 | 5 | 0.641 | 0.0 |
| 0.7–0.8 | 20 | 0.753 | 0.6 |
| 0.8–0.9 | 38 | 0.871 | 0.868 |
| 0.9–1.0 | 104 | 0.944 | 0.933 |

## Bars (reference plan)

- E2 bar: ECE ≤ 0.06 after temperature scaling (measured against the target the head was trained to match; vs *gold* the teacher's own ECE is the achievable ceiling).

## Trained-model interface suites (run against the real bundle)

- E1 (batched == single, adding questions never changes others, ids never reach
  the model), E5 (noul absolute / choice relative / non-complement noul),
  E6 (score expectation, legend echo): **all green** — 11 tests, after the
  content-keyed embedding cache made inference bit-exact across call shapes
  (bf16 kernel jitter otherwise perturbs the 4th decimal).
- E11 wire-format suite (mock server): green, 62/62 total in mock mode.

## Live demo (docs' own example requests, served model)

| Request | Answer |
|---|---|
| "Help! My payouts have been failing for 3 days." — urgency (noul) | **0.94** |
| same state — department (choice) | **technical 0.99**, confidence 0.96 |
| "API returning 500s, orders blocked" — is_urgent | **0.95** |
| same — frustration (score, 3 levels) | **1.10** (0.24/0.43/0.33), confidence **0.02** — honest uncertainty on a genuinely ambiguous case |
| State-page refund workflow (dot-paths) | refund_requested **0.99**, policy_supports **0.98** |
| held-out fact check (FEVER-style) | **0.96** |
- E5 asymmetry, E1 parity, E6 score semantics, E11 schema: green in the test suite (62 tests) — see README.
