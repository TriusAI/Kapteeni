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

## 2026-09-26 — v1.2: Intelligence 62.0 (best measured) but flagship gate
## fails on calibration; v1 stays shipped; protocol-level finding

- Seed 3 (v1.2 candidate) trained on the passes_p2 + passes_synth2 union:
  1507 steps / 11.1M tokens; final MNLI gate 0.9067, best of all seeds.
- familyval.py built (new evaluator: per-family accuracy through the served
  blend on the synth2 val slice; 90 tests green). Hypothesis CONFIRMED on
  val: temporal_numeric 0.601->0.794, multi_hop 0.766->0.917, overall
  0.659->0.806; mixed-val non-regression held (choice/score Brier better).
- Public-half run: Intelligence 62.0, hard 0.496, accuracy 0.723 — all
  best-ever — but val-fit choice constants (w 1.0, T 0.075) saturate to
  1.0/0.0 and bench ECE quadrupled (0.0496 -> 0.2023). Composite 59.65 vs
  v1's 65.71: flagship gate fails by 5+ points. v1 stays the shipped
  headline; production server restored on v1.
- Protocol-level finding (two independent failures: soup ECE x2, v1.2 ECE
  x4): fitting blend constants on the narrow mixed-domain val is
  systematically OOD-fragile. Proposed v1.2.1: same seed-3 model
  artifacts, constants refit on a pre-registered deployment-diverse val
  set (mixed + MNLI + synth2 val), one run, same gates. Also recorded the
  accumulated-exposure caveat (fifth run; redesigns must stop being
  bench-gated after this).
- Artifacts kept: kapteeni_p2_s3, kapteeni_v1_2.pt, fit_kv_v1_2.json,
  familyval JSONs, raw run docs/bench/kapteeni-v1.2-record-231.jsonl.

## 2026-09-26 (later) — v1.2.1 diverse refit: gates 1-2 pass, gate 3 fails;
## redesign budget retired, v1 stays shipped

- fit_diverse.py built (temps+blend refit on mixed 590 + synth2 609 val,
  grids unchanged, artifacts frozen) per docs/PREREG-V1.2.1.md; 93 tests
  green (familyval ECE extension + fit_diverse).
- Gate 1: v1.2.1 synth2-val ECE 0.0427 <= v1's 0.0630, family accuracies
  kept (temporal 0.811/multi-hop 0.917). Gate 2: mixed-val within +0.02.
  Saturation eliminated at the source (choice T 0.075 -> 0.724; served
  0.951 where v1.2 served 1.000).
- Gate 3 (the run): 63.18 < 64.71 — Intelligence 61.1, bench ECE 0.1196
  (halved from v1.2 but above v1's 0.0496). v1 stays the shipped model;
  production restored on it. Per the pre-registration, no further
  bench-gated design changes.
- Arc conclusions (all documented): family data is real but its artifacts
  are less OOD-calibrated than v1's on the bench mix; synthetic val
  slices share generator blind spots (v1.2: 0.035 ECE synth2 val vs 0.2023
  bench) — feeds the DecisionBench shift-ladder design; sub-66 composite
  differences sit inside the noise floor.
- Artifacts kept: kapteeni_v1_2_1.pt, fit_kv_v1_2_1.json, familyval
  v1/v1.2/v1.2.1 JSONs, raw run docs/bench/kapteeni-v1.2.1-record-231.jsonl.

## 2026-09-26 (later) — benchmark updated to v1.4.2 mid-work; docs restated

- jevbench clone pulled to v1.4.2 (1bcc55e) via the local proxy (direct
  github access was down; port 10081). v1.4.1 adds 6 systems, v1.4.2 adds
  11 (93 total, 89 ranked); items, sealed set and composite formula all
  unchanged — our runs of record remain valid as scored (reproduced the
  board arithmetic: Jev's axes -> 63.33 vs official 63.29).
- New verified #1: decider-4b v2 (Mapika) 64.13 — a 4B open system;
  classifier.dev (70.82) unranked. (Standing claims about the board were
  removed from the docs later the same day, per the entry below; the
  benchmark numbers of our own runs stay in docs/JEVBENCH.md.)

## 2026-09-26 (final) — two shipped variants; leaderboard claims removed

- User decision: (1) remove ALL leaderboard/placement claims from the
  public docs — benchmark numbers stay, ranks and cross-system claims go;
  score_bench.py no longer prints slot/neighbors; pack.py's model card is
  numbers-only. (2) ship BOTH v1 (as kapteeni-v1-meticulous, conservative
  confidence, the default) and v1.2.1 (as kapteeni-v1-intuit, sharper
  decisions on well-formed numeric/temporal/multi-step traffic) — the
  "good at different things" split is documented with per-variant numbers
  and use-when guidance in the README's side-by-side table.
- serve.py: --served-as (variant names accepted in requests alongside
  jev-latest and the legacy kapteeni-v1; responses report the real variant
  name; /v1/models describes the loaded variant). pack.py: --served-as /
  --bundle, per-variant model cards. PUBLISH-HF.md rewritten for the two
  packs. 94 tests green.
- Both distribution packs rebuilt: ../kapteeni-v1-meticulous-dist and
  ../kapteeni-v1-intuit-dist (the old single dist removed).

## 2026-09-26 — Jev-class eligibility math + verified list price

- The benchmark's Jev-class rule (cost <= $0.080/1k decisions, adjusted
  median <= 1.30s): verified the hosted list price for the Qwen3-4B class
  (Novita qwen3-4b-fp8 $0.03/M; Alibaba's nearest official tier qwen-turbo
  $0.05/M). Our 597 tokens/decision -> $0.018-0.030/1k, 2.7-4.5x under the
  cutoff; the old $0.14/M assumption ($0.0835/1k) was the only thing that
  ever had us over the line, and it was pessimistic. README Cost row and
  caveats updated with the verified prices.
- Committee (both variants averaged at serving): adjusted p50 doubles
  0.34s -> 0.68s, still 1.9x inside the 1.30s cutoff; cost axis unchanged
  under the benchmark's per-decision billing convention (internal passes
  invisible, as for every multi-pass system incl. Jev's per-option
  passes). True serving cost is 2x GPU time — stated plainly in the docs.
  Eligibility note added as descriptive fact (no placement claims,
  per policy).

## 2026-09-26 (evening) — committee (output-ensembling): negative result on
## calibration; mode not promoted; servers restored

- Pre-registered (docs/PREREG-COMMITTEE.md, deployment-side gates only, no
  bench): output-average meticulous + intuit (committee.py: pure merge with
  the contract's own shaping formulas; serve.py --committee; familyval
  --committee + per-item latency recording; 104 tests green).
- Measured (609-row synth2 val, idle GPU): accuracy 0.801 (G1 pass, 97% of
  intuit's gain), latency p50 0.336s (G3 pass, Jev-class with margin), but
  top-label ECE 0.0795 (G2 FAIL — worse than both members' 0.063/0.043).
- Mechanism: averaging dilutes intuit's correct confidence with
  meticulous's hedging -> systematic underconfidence (multi_hop: acc 0.917
  kept, ECE 0.137). Third failed combining mechanism (weight-space soup,
  constants-space v1.2/v1.2.1, output-space committee): meticulous's
  calibration does not average. Mode not promoted; merge code kept.
  Surviving idea: per-request ROUTING between members (own pre-reg needed).
- Servers restored: meticulous :8000, intuit :8001 (from their dists).

## 2026-09-26 (night) — bench request drafted; posting pending the code push

- Read the benchmark's bench-request issues (#79 decider-4b v2, #100
  deck-4B) as format templates; drafted ours in docs/BENCH-REQUEST.md
  (pinned HF revisions 8c1abcf/6466d70 — verified current incl. the user's
  refreshed cards; exact per-variant measured tables; identity checks;
  run commands; limits; full disclosures incl. the zero-hit 8-word
  contamination audit — scripts/contamination_audit.py).
- Code pinned at tag bench-request-v1 (avoids self-reference); the tag +
  2 commits await the user's SSH push (the fine-grained gh PAT cannot
  write the TriusAI repo; the gpg-agent socket has no key loaded for
  non-interactive use).
- Issue body staged at /tmp/opencode/bench-request-body.md; gh CLI is
  authenticated (bctnry) and the API is reachable via proxy 10081.

## 2026-09-26 (final) — bench request POSTED: fstandhartinger/jevbench#102

- https://github.com/fstandhartinger/jevbench/issues/102 — open, under
  bctnry, body verified byte-identical to docs/BENCH-REQUEST.md (7,376
  chars; all pinned revisions and numbers present).
- Posting required a classic PAT (fine-grained PATs cannot create issues
  in third-party repos regardless of permission toggles — documented the
  finding when it failed twice).
- The request pins: HF TriusAI/kapteeni-v1-meticulous @ 8c1abcf and
  TriusAI/kapteeni-v1-intuit @ 6466d70 (verified current, refreshed cards
  included), code at tag bench-request-v1 (7d365aa). Sealed-half
  evaluation now possible via their offline-container process if
  accepted.

## 2026-09-27 (V2 complete) — split verdict: image gate passes, text gate fails

- First launch OOM-died at step 60 (fake-unit batch packing ~3x too
  heavy — 52.8G peak on 32G until the kernel killed it silently). Fixed
  with real-token batching (tokenizer-measured prompt lengths + 400
  vision tokens/image, 4096 real-token budget), 150-step checkpoints
  with adapter+optimizer+step resume. Full run completed: 1,584 steps /
  ~4.9M real tokens / ~5.3h, peak 17.8G.
- Final gates on the completed adapter: synth3-val 0.8458 > frozen
  0.8119 (PASS — the image skills are trainable); MNLI 0.8333 < 0.88
  (FAIL — frozen 0.90; text behavior regressed from the first gate and
  never recovered; the ~2:1 image-heavy token mix crowded out text
  despite replay).
- Per the pre-reg: not shipped, same recipe not rerun, gates not
  adjusted. The mechanism (insufficient replay ratio, not method
  failure) motivates a pre-registrable follow-up: token-parity replay
  (~2x text volume), same recipe and gates.

## 2026-09-28 — V2.1 (all-text replay): second negative result; mixed-recipe space closed

- Completed cleanly: 2,743 steps / ~10.2M real tokens / ~7h / peak 17.9G
  / no OOM (the real-token batching + checkpoint/resume carried over).
- Final gates: synth3-val 0.8932 (PASS, +8.1pt over frozen 0.8119,
  beyond the 590-item CI); MNLI 0.8267 (FAIL vs >= 0.88, frozen 0.90).
- Series conclusion: text erosion (~0.90 -> ~0.83) is INVARIANT to a
  2.3x change in text replay volume — structural to the mixed recipe
  (answer-SFT + full-projection LoRA), not a ratio problem. The image
  side trains robustly in both settings.
- Per the amended pre-reg's fail-path: next is the structural option,
  modality-routed serving — text through the frozen backbone (MNLI
  0.90 by construction), images through the V2.1 adapter (0.893) —
  no further training required. To be pre-registered as V3-ROUTED
  before serving work claims it.

## 2026-09-28 (early) — v1.1 track opened; base-selection probe pre-registered

- Direction set by the user: abandon the dedicated-image-model line;
  v1.1 will be a NEW unified model (images + Chinese + a researched
  backbone). The Qwen3 choice for v1/V was inherited, not chosen.
- Researched the 2026 small-VLM landscape. Exclusions documented with
  reasons (Qwen3.8-27B: $0.27/1k decisions, 3.4x over the Jev-class
  cutoff + the monoculture pick; InternVL3.5: Qwen3 backbone inside;
  MiniCPM: registration-clause lineage / 1B current; Kimi-VL: 2025 gen;
  Gemma: terms + weaker zh).
- Pre-registered Step 0 (docs/PREREG-KAPTEENI-V11.md, committed BEFORE
  any measurement): frozen probe over Qwen3.5-9B / Qwen3.5-4B /
  GLM-4.6V-Flash on synth3-val (590) + MNLI-noul (150) + new
  OCNLI-noul (150, CC BY-NC eval-only, Chinese templates, bilingual
  yes+是 / no+否 group readout). Selection rule: highest mean; within
  0.03 -> prefer the non-Qwen candidate, then lower hosted cost.
- scripts/build_ocnli.py: rows_ocnli.jsonl built (500 rows; dev labels
  are explicit strings, so the entailment->true mapping is
  unambiguous). data_cache/ stays untracked -> no NC redistribution.
- scripts/baseprobe.py: the multi-model probe harness (chat-template
  render with thinking-kwarg auto-resolution, one-pass lettered
  readout, per-family synth3 breakdown). Slices verified to render.
- Downloads launched (9B -> GLM -> 4B chained, proxy + Xet disabled).

## 2026-09-28 — base-selection probe complete: Qwen3.5-4B selected for v1.1

- Amended-instrument probes: Qwen3.5-9B mean 0.8579 (synth3 0.8203 /
  MNLI 0.88 / OCNLI 0.8733); Qwen3.5-4B mean 0.8445 (0.8068 / 0.8667 /
  0.86). Gap 0.0134 < the 0.03 tie window; cost tie-break (no non-Qwen
  candidate remained) selects the 4B (~$0.018/1k fp8). GLM excluded on
  stack grounds: its ViT patch-conv grinds per-image-shape MIOpen
  kernel searches on our ROCm box (3 stalls at modeling_glm4v.py:765;
  MIOPEN_FIND_MODE=1 didn't help) — a training-box disqualifier, not a
  model-quality verdict (quality unmeasured; revisitable under a fresh
  pre-reg on a different stack).
- v1.1 frozen gate baselines fixed from the probe: synth3-val 0.8068,
  MNLI-noul 0.8667, OCNLI-noul 0.8600.
- External corroboration noted post hoc: top open JevBench system
  (SemIf, ex-OpenJev) runs on Qwen3.5-4B.

## 2026-09-28 — v1.1 recipe pre-registered (Qwen3.5-4B, images + Chinese)

- Backbone Qwen3.5-4B; frozen baselines from the probe. Wire answer
  space fixed: yes/no + letters in every language; Chinese enters via
  state/instructions/criteria only.
- Data: synth3 (5,412) + NEW synth3zh (~2,000 Chinese questions over
  the same image families) + full EN text replay (~14,935) + NEW
  synth2zh (~4,000 Chinese ports of the three skill families).
- Recipe applies the V2/V2.1 mechanism diagnosis from two sides:
  attention-only LoRA targets (narrower footprint) + lr 5e-5 (half of
  V2), plus the stated new hypothesis: the natively-multimodal base may
  not erode text the way the bolted-on Qwen3-VL did.
- Gates fixed: synth3-val > 0.8068; MNLI >= 0.84; OCNLI >= 0.83
  (each = frozen − 1 binomial SE); synth2zh-val >= 0.90; fitted ECE
  <= 0.10 on combined held-out val. Fail -> no ship, next step is a
  new pre-reg (lower LR / narrower / modality-routed).

## 2026-09-28 — v1.1 data + trainer built; pre-run engineering done

- synth2zh (4,202 rows): the three synth2 skill families, Chinese
  surfaces, gold logic imported from synth2 (shared invariants). State
  keys stay English (wire identifiers, OCNLI convention); Chinese date
  formats incl. weekday-styled; Chinese policy docs; Chinese criteria
  descs. is_val split 403.
- synth3zh (2,000 rows / 643 images): all 20 image families fully
  Chinese — Noto Sans/Serif CJK renders (Style(zh=True); DejaVu default
  unchanged so English synth3 regenerates byte-identical), Chinese
  questions/criteria, zh-/synth3zh- id prefixes. QA: determinism
  selfcheck, border-clip scan over all 643 images (0 hits), per-char
  glyph check (0 missing), visual inspection of 10 layouts. Fixed en
  route: seating row labels + bar-chart axis labels were left-clipped;
  paperclip emoji -> plain 附件.pdf chip (Noto CJK has no emoji).
- Contamination audit extended (CJK-aware tokens: alnum words + single
  CJK chars, 8-token shingles — identical behavior on English): JevBench
  231 items AND OCNLI 500 gate rows vs the full v1.1 mixture (28,804
  surfaces, val excluded) — zero items with any overlap, both gates.
- train_v11: mixture verified vs the pre-reg exactly (synth3 5,412 +
  synth3zh 1,796 + EN replay 14,935 + synth2zh 3,799 = 25,942; val
  590 + 204; MNLI/OCNLI slices n=150). LoRA r=32 attention-only
  (6.29M trainable, 0.14%), lr 5e-5, budget 4096, image tokens
  re-measured for Qwen3.5-4B: 399 per 640x640 render (the 400 constant
  carries). --gates-only runs the five final gates post-run.
- Pre-run smokes: peak memory 22.8G (96G box — ample); resume verified
  in practice (picked up an aborted run's step-30 ckpt, adapter+opt+
  step); monitor path exercised twice. Two defects caught and fixed:
  synth2zh monitor source tag (printed None) and run_gate's default
  limit=120 silently truncating monitors below registered sizes.
- Full-run shape: 3,327 batches, ~11.0M real tokens. Monitors are trend
  instruments; the fresh-LoRA numbers (MNLI~0.41, OCNLI~0.29) just mean
  the answer convention isn't learned in the first dozen steps.
- Gates-only plumbing verified on the throwaway 51-step smoke adapter
  (all five slices render; MNLI already 0.8133 / synth3 0.7881 from
  400 examples — trajectory, not a gate reading). Overnight v1.1 run
  launched (3,327 batches, ~11.0M real tokens, monitors every 300
  steps, ckpt+resume every 150).
- 08:34 first overnight attempt: reached step 600 (ckpt 09:32), then
  died silently after 09:32 — no reboot, no journal/OOM trace; the
  opencode background-shell record itself was lost (no output file,
  no completion notification), so the harness likely killed the child
  process group. Cause unknown, diagnostics from steps 300/600 lost
  with the log (monitors are trend instruments only — no gate data
  lost). Relaunched 20:05 DETACHED (setsid+nohup -> /tmp/opencode/
  v11_run.log, pid 443869): resumed cleanly at step 600, loss 0.49 at
  step 620, peak 15.6G. ETA ~01:00-01:30.
- 21:00 spike investigation (ckpt-1050 probe, no training): image
  decisions intact (synth3-val[:120]=0.8917 — above frozen 0.8119);
  train==eval CE rules out mode artifacts. Mass probe found the
  mechanism: image CHOICE rows put 54% on the no-space letter id 32
  (fresh-model line-start lettering) vs the supervised ' A' id 357 at
  0.1% -> CE ~6.9, exactly the spike; text choice rows are 99.6% on
  the with-space letter (converged during the text-first curriculum).
  NOT divergence: the LoRA is aligning image rows to the wire answer
  convention; argmax acc was never broken. Separate issue: image
  batches ran ~3x slower than text (GDN chunked-scan recompute +
  vision tower recompute under checkpointing). Fix: pad_to=512 in
  encode_batch/run_gate/finalize — one sequence shape for every
  sub-512 batch (pad tokens masked; training math identical). Relaunched
  from ckpt-1050 at 22:07: image region ~11s/step. ETA ~03:00.
- 23:30 no-ckpt memory findings: WITH pad_to=512 the fp32 logits tower
  (bf16 copy + float() + grad ≈ 9.5G per 12x512 batch) plus
  un-checkpointed activations OOM'd (run3, ~83G). Natural widths +
  no-ckpt (run4, pid 477902, from ckpt-1200): peak stable 79.8G over
  the image region so far, ~15s/step vs 24s checkpointed (recompute
  eliminated). Watcher armed for auto-relaunch if the unified-memory
  pool shrinks (desktop shares the 96G). ETA ~04:30.
- 01:19 run4 (no-ckpt) OOM'd at ~step 1690: the unified-memory pool
  went to 0 free (desktop shares the 96G; expandable_segments
  mapping failed). ckpt-1650 survived; monitor 1500 still clean
  (MNLI 0.86 / OCNLI 0.8867 / synth3-val 0.87 / synth2zh 0.84).
- 04:48 run5: back to the PROVEN config (gradient checkpointing on,
  natural widths, 21.5G peak — OOM-proof against desktop
  fluctuation), resumed at 1650. The no-ckpt experiments bought
  ~540 steps and cost two dead runs; the memory-hungry variant is
  shelved for daytime use only. Remaining ~1,677 steps, ETA ~09:00.
## 2026-09-29 evening — v1.1b: NEGATIVE (no ship); unified-arc verdict

- b-run: 1,802 steps (2 synth2-only epochs on the v1.1 adapter), monitors
  in data_cache/v11b_monitors.jsonl (the log-loss insurance paid off —
  clean 6-round history). Gates: synth3-val 0.8729 and OCNLI 0.8533
  PASS; MNLI 0.82 FAIL (eroded from 0.86 under synth2-only pressure);
  synth2zh 0.8462 / synth2-EN 0.8407 FAIL (mastery bar 0.90).
- The trend is the result: skills climbed (+7 ZH / +6.6 EN), peaked
  mid-run (0.895 / 0.845 at step ~1200), then DECLINED as epoch-2
  memorization capped (final loss 0.0002) — plateau ~15 points short
  of mastery while NLI gates eroded. Note the monitors also caught an
  abort-rule near-miss worth having survived: MNLI's step-900 dip was
  1 sigma noise; the amended catastrophic-only rule behaved correctly
  throughout.
- Volume mechanism spent (4 unified runs now top out 0.82-0.85 on
  synth2 mastery). Remaining mechanism: the v1 phase curriculum on
  the multimodal base (a build) — or park. User decision pending.
- GPU free; text servers still down pending the decision.

## 2026-09-30 — v1.1c opened: the v1 phase curriculum port (user decision)

- User decision on the parked multimodal arc: pursue the remaining
  documented mechanism — port the v1 phase pipeline (multi-pass per-
  option judgment, distillation-weighted soft targets, group-CE/level
  BCE in-loss, tiny per-primitive heads on h_last, P1 heads -> P2
  LoRA+heads joint) to Qwen3.5-4B. Modality-routed serving was offered
  and declined in favor of the unified-mechanism test.
- Pre-registered BEFORE any measurement: docs/PREREG-KAPTEENI-V11C.md.
  Gates identical to v1.1b's table (synth3 > 0.8068, MNLI >= 0.84,
  OCNLI >= 0.83, synth2zh >= 0.90, synth2-EN >= 0.90, fitted ECE
  <= 0.10). Data: the v1.2 union REUSED BYTE-IDENTICAL
  (passes_p2.jsonl + passes_synth2.jsonl, incl. the goemo/synth rows
  the lettered line never used) + NEW multi-pass expansions of
  synth2zh and the synth3/synth3zh image families (image attached to
  every pass of a row — the mechanism's cost: 399 vision tokens/pass).
  v1.2's exact recipe: LoRA r=32/α=64 all 7 language-model projections
  (tower frozen), lr 1e-4, budget 8192, 1 epoch, warmup+cosine; v0's
  exact head training (40/12/30 epochs, lr 1e-3) in P1. Honest risk
  stated up front: nothing has ever measured >= 0.90 on the full
  synth2 val, including the pipeline being ported.
- Build (committed): kapteeni/v11c.py (pass builders, chat-template VL
  encode, h_last extraction, heads readout), v11c_precompute.py
  (frozen-feature precompute, incremental shards + resume),
  train_v11c.py (--phase1/--p2/--gates-only), scripts/chain_v11c.sh,
  10 new tests (full suite 163 green).
- Engineering smokes (pre-registered engineering, not gate
  measurement): h_last through Qwen3_5Model with images (batch 8
  optimal: text 11.6 pass/s, image 1.5 — bigger batches slower);
  LoRA targets verified (128 language-model projections, 42.5M params,
  tower excluded); image-grad probe peak 16.0G at budget 8192;
  limit-120 P2 smoke exercised train loop + monitor round through the
  heads readout on real slices + ckpt; resume verified live — and
  caught a REAL BUG: v1.1b's resume-skip condition (base + bi < step)
  is wrong under the v1.2 shuffled batch order (bi is a cost-sorted
  index, not the training counter) — fixed with an explicit iteration
  counter and re-verified.
- Contamination audit extended to the v1.1c mixture (adds goemotions
  + synth surfaces; 35,804 training rows vs JevBench 231 + OCNLI 500):
  zero 8-token overlaps, both gates. docs/contamination_audit.json.
- Full-run shape: P1 precompute 95,484 passes (75,829 train incl.
  19,374 image + 19,655 val; ~6h at measured rates) -> P1 heads
  (minutes) -> P2 ~21M real tokens (~10-14h + ~1.5h of monitor
  rounds; ~1,900 batches). Chain launched detached
  (scripts/chain_v11c.sh -> /tmp/opencode/v11c_*.log), ETA for the
  final gates roughly 18-22h from launch. Monitors every 300 steps to
  data_cache/v11c_monitors.jsonl; abort rule + 150-step ckpt/resume
  armed. The gates run ONCE on the completed model.
