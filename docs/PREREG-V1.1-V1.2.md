# Pre-registration: v1.1 (3-seed soup) and v1.2 (weak-family data)

Written **before** either public-half benchmark run. The repo's credibility
rests on this discipline: constants are fit on mixed-domain validation,
bench items are never tuned against, and any result — positive or negative
— is reported with its caveats.

## Discovered recipe fact (erratum)

`train_p2.py`'s argparse default is `--token-budget 12288`, but the shipped
v1 run used **8192** (its log: 1205 steps, 8.7M tokens, 7.2k tok/step = 88%
pack fill; 12288 yields 765 batches, 7168 yields 1381 — only 8192 gives
1205). The docstring now records this. Soup seeds must run at 8192 or they
are not seeds of v1.

## v1.1 — three-seed uniform soup

**Hypothesis.** Averaging independently-seeded runs of the exact v1 recipe
improves accuracy and calibration at zero serving cost.

**What varies.** Only the seed: 0 = the shipped v1 adapter
(`model_cache/kapteeni_p2/adapter`), 1 and 2 = fresh runs
(`kapteeni_p2_s1`, `kapteeni_p2_s2`) with the same data
(`passes_p2.jsonl`), same warm start (`kapteeni_v0.pt`), same budget
(8192), lr (1e-4), epochs (1). Seeds auto-chain via
`scripts/chain_seed2.sh`.

**Soup method.** Elementwise mean of the *merged* (base + scale·B@A)
weights — basis-independent; raw-adapter averaging would not be
(mean of B@A ≠ mean(B)@mean(A)). `kapteeni/soup.py`. Heads averaged the
same way (all three warm-start from the same P1 bundle).

**Constants.** Temperatures + blend refit through the soup on the SAME
mixed-domain val slices with the SAME grids as v1 (`p2_finalize
--merged-dir model_cache/kapteeni_soup --heads .../heads_soup.pt --out
model_cache/kapteeni_v1_1.pt --fit-out data_cache/phase1/fit_kv_v1_1.json`).
No new hyperparameter search.

**Gates — decided now, before any run:**
- Each seed's final MNLI OOD gate ≥ 0.88 (from its training log).
- Soup heads: if mixed-val head-readout accuracy per primitive drops more
  than 1 point below seed 0's, the averaged heads are abandoned and heads
  are retrained on soup features (v0-style) instead; the doc is updated
  with whichever happened.
- If mixed-val (temps stage) accuracy/ECE of the soup is *worse* than
  seed 0 by more than noise (>0.5pt acc or >20% ECE), the hypothesis is
  reported as failed and **no** public-half run is spent on it.

**Run.** Exactly one 231-item public-half run of the soup
(`kapteeni-v1.1`), fresh ledger, cap $1000, same harness as v0/v0.1/v1.
No re-runs.

**Reporting.** Whatever happens: a v1.1 row in docs/JEVBENCH.md and the
README benchmark section, with the deltas and gates that passed/failed.
Distribution artifacts ship only if the soup wins.

## v1.2 — weak-family continuation

**Hypothesis.** The two worst public-half families (temporal_numeric,
multi_hop; long_policy weak on hard tier) fail for *data* reasons — v1's
synthetic templates were too narrow, so the model learned templates, not
skill. Ground-truth-by-construction data with wide surface diversity and
trap density fixes them without touching the base model.

**New data (`kapteeni/synth2.py`, deterministic, seeded):**
- 6,300 rows / 11,498 passes / ~2.56M tokens; 609 rows land in the val
  split by the repo-wide deterministic rule.
- Families: temporal_numeric 2,839 (6 date formats incl. weekday-styled,
  relative phrasing, business days with explicit holiday lists, boundary
  semantics per relation wording, near-miss/weekday-consistent traps,
  choice + score variants), multi_hop 2,220 (eligibility chains with
  tenure computed from dates, 3-hop business-day process chains, ordered
  fee arithmetic, two-level exception logic, clause specificity,
  per-condition failure identification), long_policy 1,241 (document
  grammar: randomized parameters, shuffled sections, distractor clauses,
  3-4 questions per doc whose gold reads off the generator's table).
- Gold correctness is guarded by `synth2 selfcheck` (helper invariants —
  it caught a real `today.day == 1` labeling bug during development) and
  by tests (determinism, schema, criteria resolution, bin coverage).
- Choice rows store the option index per the build_data contract;
  criteria file: `data_cache/synth2_criteria.json`.

**Training.** One fresh seed (seed 3) trained from the base with the v0
warm start on the **union** `passes_p2.jsonl + passes_synth2.jsonl`
(1 epoch, budget 8192, lr 1e-4). From-scratch beats continue-training the
soup: no double-descent on replayed data, and it keeps v1.1 interpretable.

**Candidate selection — on val only, never the bench:**
- Measure mixed-val (temps/blend fit stage) AND the synth2 val slice
  (family accuracy for temporal_numeric / multi_hop / long_policy through
  the served readout).
- Candidates: seed 3 alone, and the 4-seed soup {0,1,2,3}. Pick the best
  mixed-val + family-val; only the winner gets the public-half run.

**Amendment (2026-09-25, after v1.1's completed run, before any v1.2 run):**
v1.1 showed the soup's val-fitted constants are OOD-fragile (mixed-val ECE
improved everywhere while bench ECE doubled; 63.09 vs v1's 65.71). The
4-seed-soup candidate is therefore dropped. v1.2's candidate is **seed 3
alone**, judged on the same gates; the comparison stays against v1 (the
shipped model), and one public-half run is spent only if the gates pass.

**Gates.** MNLI OOD ≥ 0.88 during seed 3; English flagship: if the
public-half composite drops more than 1.0 below v1's, that is reported
honestly and v1 stays the headline.

**Run + reporting.** Same as v1.1: one run, one row in the docs, caveats
included, no re-runs.

## Later tracks (explicitly out of scope here)

Chinese support and adaptive re-asking are separate pre-registrations;
nothing in this document touches them.