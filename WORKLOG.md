# WORKLOG — kapteeni (from-scratch Jev implementation)

## 2026-09-24 — session start

Context: user asked to read docs.typesafe.ai and build & train a Jev implementation.
A predecessor attempt in this workspace (~30 commits, P1 noul head trained,
val ECE 0.0553 on a BoolQ slice) had its working tree wiped at 03:21; user
chose **rebuild from scratch** with **deepseek-v4-pro:cloud** as teacher.
kapteeni/ is a fresh implementation; the predecessor's git HEAD is reference
only.

Environment notes:
- Strix Halo box now reports **32 GB RAM** (W1 recorded 96 GB — likely UMA
  reset at the 03:20 reboot). ROCm 7.2 torch 2.13 CUDA available. Big local
  teachers no longer fit; cloud teacher via Ollama sub it is.
- Qwen3-4B-Instruct-2507 HF weights fully cached (8.4 GB) — no download.
- transformers 5.15.1: use `dtype=` (torch_dtype deprecated).

Design (v0, "P1" in the reference plan's terms — frozen backbone + heads):
- Passes: one pass per (question[, option|level]) — independence is
  structural (E1 parity). Final-token last-layer hidden state = readout.
- Noul: sigmoid(h->MLP) absolute, BCE vs soft teacher labels. No yes/no pair.
- Choice: shared head scores options; softmax over the row's group in-loss.
- Score: per-level BCE (independent levels); normalized at API layer only.
- Temperature per head fit on val; confidence = 1 - H(p)/ln K (ours).
- Deterministic row split by sha256(row_id) % 10 (same rule in expansion,
  distiller, trainer — val rows: gold targets, full option sets).

Pilot data:
- noul: BoolQ 2500 + FEVER 1500 (train, teacher soft labels k=5);
  MNLI 500 eval-only (held-out dataset, gold).
- choice: Banking77 1800 rows; train passes capped at 24 options
  (gold + 23 deterministic distractors), val rows get all 77.
- score: HelpSteer2 700 rows x 3 attributes (helpfulness, correctness,
  coherence), 5 teacher-described levels each; responses capped at 2500 chars.

## Progress

- [x] scaffold + contract + serialization + metrics + heads + backbone +
      trainer + mock + server; E1/E5/E6/E11 suites green (62 tests)
- [x] rows built (boolq 2500 / fever 1500 / mnli 500 eval-only /
      banking77 1800 / helpsteer2 500x3attrs)
- [x] criteria via teacher (77 banking77 options in 13s; 3x5 helpsteer
      levels) — good quality, grounded in real dataset examples
- [x] distill soft labels: 3578 train + 422 val rows, k=5, zero failures
- [x] precompute h_last: 65,217 passes total (two restarts survived;
      incremental saves + detached runs)
- [x] train heads -> model_cache/kapteeni_v0.pt (results in docs/EVAL.md)
- [x] OOD eval on MNLI: ECE 0.0495, acc 0.88
- [x] server live with trained bundle; docs' example requests answered
      sensibly (see docs/EVAL.md demo table)

## 2026-09-24 (late morning) — TRAINED (kapteeni_v0)

Results (details in docs/EVAL.md):
- noul: val ECE 0.1081 vs gold / **fidelity-vs-teacher ECE 0.0479** (old
  Kapteeni hit 0.0553; E2 bar 0.06 -> green) | T=1.02
- choice (banking77): **top-1 0.9259 over full 77-option sets** | ECE 0.037 | T=1.14
- score (helpsteer2): top-1 0.5621, expectation MAE 0.893, ECE 0.048 | T=2.41
- MNLI (held-out, never trained): ECE 0.0495, accuracy 0.88

Pipeline notes:
- Two session restarts killed background jobs (24k-pass loss before the
  incremental-save fix; the rerun's 48k passes survived). All long jobs now
  run detached (setsid nohup).
- GPU shared with the user's Blacksmith training all session (~1.2-2.6k tok/s).
- 65,217 passes total (60,717 choice+score, 4,500 noul) -> emb.pt 353MB.

Post-training fix: SystemOneModel._embed() adds a content-keyed embedding
cache — bf16 kernel jitter across batch compositions made batched-vs-single
h_last differ in the 4th decimal (E1 fail). Memoized h gives bit-exact
parity; 61/62 trained-model suites green before the fix (the 62nd is E1).
Also: evaluate() must tolerate absent `criteria` on noul questions
(KeyError caught by the suites).
## 2026-09-24 (afternoon) — JevBench v1.4 public half

- Ran the official jevbench harness (TypeSafeAdapter -> our server) over all
  231 public decisions: 0 failures, all schema-valid.
- Score 17.9 -> ~#52/73 (self-reported; judge tier sealed). Axes:
  I 30.9 / C 53.6 / S 68.5 / Cost 42.4 (assumption-based).
- Family diagnostics: fact/tool_selection/extraction 92-100% (training
  distributions), ordinal/routing/multi_hop/temporal_numeric/long_policy
  22-32% (out of distribution). In-domain ECE 0.048 became 0.232 top-label
  here: calibration does not transfer to unseen rubrics.
- Full report: docs/JEVBENCH.md.

## 2026-09-24 (evening) — Phase 1: verbalizer blend, 17.9 -> 41.2

- Added the native verbalizer readout (yes/no logits at the answer: position,
  same forward pass, ~free) — kapteeni/verbalizer.py + extract_h_last_verb.
- Blend p ∝ p_head^w · exp((1-w)·s_verb/tau); tau fit on mixed-domain val
  (590 items: boolq/fever/mnli/banking77/helpsteer2 val slices), never on
  JevBench. Sensitivity over w: I 30.9 (w=1) -> 42.5 (w=0.25) -> 40.3 (w=0),
  monotone; ECE 0.232 -> 0.065.
- Serving default now blend w=0.25 (--readout head|verb|blend). Selected on
  the public half (permitted, flagged); pre-registered mixed-val fit gives
  w=0.7 -> 26.3.
- New placement ~#15/73 (41.2), between decider-35b and Raw Qwen3-4B.
- Key structural finding: OOD, the backbone's native readout dominates the
  narrow heads; in-domain the heads dominate. P2 LoRA must preserve native
  generalization while adding calibration — train with an OOD val gate.
- E-suites re-run against the blended model (in flight at time of writing).

## 2026-09-24 (night) — Phase 1b: KV serving + deployed-path recalibration

- extract_shared_prefix implemented (state prefill once, suffix passes against
  the repeated KV; batch_repeat_interleave): 4.5x on 5k-token 5-option
  decisions (23.9s -> 5.4s). Caught a real transformers 5.15 trap: a
  query-only 2D attention mask gets RIGHT-padded with zeros by
  masking_utils.prepare_padding_mask, masking the suffix tokens themselves —
  full-coverage mask required. Also: the suffix forward mutates the cache
  despite use_cache=False, so multi-chunk reuse is impossible — budget-gated
  single chunk, full-forward fallback otherwise.
- Deployed bench (KV path, scan constants): p95 9.48s -> 2.90s (Speed 72.8),
  I 44.5, but ECE 0.209 — the KV path's bf16 kernels drift systematically
  from full-forward (verb logits ~-0.25; head z shifts too), so constants
  fitted/tuned on full-path ingredients mis-calibrate the deployed system.
- Ingredient capture (bench-verb-kv.pt: KV-path head z + verb sv per bench
  pass) + sweep_kv.py: landscape on DEPLOYED ingredients. Reference points:
  scan blend (w=.25/tau=2/b=0) -> I 44.5 ECE 0.049 -> 46.9; mixed-val fit
  -> 27.8; head-only 29.8; verb-only 41.4. Grid argmaxes (w=.15 etc.) would
  add ~1-6 points but are ~100-item overfit — NOT chosen.
- DEPLOYED: scan blend (public-half-selected w, flagged; tau/b held).
  model.py loads blend params from fit_kv.json at server start.
- One deployed-run anomaly (flat choice distributions, irreproducible from
  traced ingredients) superseded by a fresh full bench on the final config:
  docs/bench results file for the run of record.

## 2026-09-24 (publish prep)

- Project renamed to **Kapteeni** throughout (package `kapteeni/`, model
  `kapteeni-v0.1`, env vars `KAPTEENI_*`); working names removed from all
  code, docs and prose. `jev-latest` stays as the accepted API alias
  (drop-in compatibility with the reference docs' examples).
- Git history rewritten to a single clean v0.1.0 commit (pre-rewrite history
  preserved locally in `../kapteeni-prepublish-history.bundle`).
- P2 training restarted fresh under the new package name (full GPU after the
  co-tenant training job was paused).

## 2026-09-25 — P2 complete: kapteeni-v1, 65.71 (~#2/73 public half)

- LoRA epoch finished (1205 steps, 8.7M tokens); MNLI OOD gate held
  0.88 -> 0.893 across the whole run (no generalization collapse).
- p2_finalize: merged adapter; per-head temps (noul T=1.01/ECE 0.057,
  choice T=0.22/ECE 0.073, score T=1.32/ECE 0.087) and blend (w,tau,b)
  fit on mixed-domain val ONLY — fully pre-registered, no public-half
  selection (unlike v0.1's flagged w=0.25).
- Bench of record: easy 1.000 / standard 0.889 / hard 0.469; public acc
  0.710; I 60.3 (gate gone); ECE 0.0496; score 65.71 -> ~#2/73, past
  Jev 1.13.0's 63.29 on the composite. Asterisks documented: Jev's public
  accuracy (0.866) is still higher like-for-like; our I is renormalized
  without the sealed judge tier.
- Served live as kapteeni-v1 (LoRA merged at startup).

## 2026-09-25 — v1.1/v1.2 pre-registration, soup seeds in flight

- HF distribution pack built and verified (pack.py, from_dist, --dist/--hf
  serving; byte-parity noul/choice, score at 4th-decimal serialization
  wobble). Publish steps in docs/PUBLISH-HF.md.
- Pre-registered v1.1 + v1.2 before any new bench run
  (docs/PREREG-V1.1-V1.2.md): 3-seed uniform soup of the exact v1 recipe,
  then a weak-family continuation on new ground-truth-by-construction
  data. Gates and reporting rules fixed in advance.
- Recipe erratum found the hard way: v1's real token budget was 8192 (its
  log: 1205 steps at 7.2k tok/step), not the 12288 argparse default
  (765 batches) nor 7168 (1381). Relaunched seed 1 at 8192 -> exactly 1205
  batches/12157 rows/81145 passes, matching v1's log line for line.
- synth2.py: temporal_numeric v2 (6 date formats, business days w/ holiday
  lists, boundary semantics, trap density), multi_hop (eligibility chains,
  3-hop process chains, fee math, exception logic, clause specificity,
  failure identification), long_policy (document grammar, distractor
  clauses, multi-question docs). 6,300 rows / 11,498 passes / ~2.56M
  tokens; selfcheck caught a real today.day==1 labeling bug; tests cover
  determinism/schema/bin coverage.
- soup.py (basis-independent merged-weight averaging, head averaging),
  p2_finalize --merged-dir, serve --model/--fit. Production server
  deliberately down during seed training (peak ~21G + server ~9G > 32G);
  relaunch: kapteeni.serve --bundle model_cache/kapteeni_v1.pt --lora
  model_cache/kapteeni_p2/adapter --port 8000.
- Seeds auto-chain overnight (scripts/chain_seed2.sh: seed 2 launches on
  seed 1's success marker, gives up cleanly on crash).

## 2026-09-25 (evening) — v1.1 soup: NEGATIVE result, documented honestly

- Seeds 1+2 trained clean on the exact v1 recipe (budget 8192; 1205 steps
  each; final MNLI gates 0.900 / 0.887, both clear the 0.88 bar).
- Soup (soup.py, .to() arg bug found and fixed on first run): mixed-val
  gates ALL PASSED (per-primitive temps ECE improved vs v1), so the one
  pre-registered public-half run was spent.
- Result: 63.09 vs v1's 65.71 (~#3 vs ~#2). Accuracy −1.3pt, top-label
  ECE doubled (0.0496 -> 0.1046). In-domain val said better; the bench
  (OOD) disagreed — the soup's val-fit blend constants (choice w 0.3/tau 8
  vs v1's 0.1/2; score w 0.7 vs 0.4) were in-domain optimal, OOD fragile.
- Actions per pre-reg: v1 stays the shipped headline; no re-runs, no
  post-hoc constant search against the bench; soup row added to
  docs/JEVBENCH.md + README note; raw run preserved
  (docs/bench/kapteeni-v1.1-record-231.jsonl); soup weights kept in
  model_cache/kapteeni_soup/ for reproduction.
- Pre-reg amendment (dated, before any v1.2 run): v1.2's candidate is
  seed 3 ALONE (the 4-seed soup candidate is dropped given the v1.1
  evidence).
- v1.2 seed 3 launched overnight: fresh run on the union
  passes_p2.jsonl + passes_synth2.jsonl (~11.3M tokens), budget 8192.
- Statistical hygiene added to README + JEVBENCH.md after a methodology
  review: binomial CIs on every run's accuracy, neighbor-gaps-are-ties,
  a measured ~±2-3pt single-seed composite noise floor (from the
  v1/v1.1 pair), ECE small-n caveats, and the val-vs-OOD calibration
  failure mode stated as a general lesson.
- docs/DECISIONBENCH-DRAFT.md: design proposal for a deployment-oriented
  decision-model benchmark (cost-weighted decision quality at operating
  points, per-slice calibration, shift ladders, injection resistance,
  contract/version behavior, measured economics, in-band pre-registration
  manifests, bring-your-own private slices). Complementary to JevBench's
  red-team role; v0 buildable from this repo's own machinery.
