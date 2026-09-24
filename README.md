# Kapteeni — a Jev-compatible System One decision model

**Status: v1 (P2 LoRA) trained, evaluated & served** — `docs/EVAL.md` for
calibration, `docs/JEVBENCH.md` for the JevBench v1.4 public-half placement:
kapteeni-v1 scores 65.71, ~#2 of 73 systems (best open rebuild; asterisks
stated in that document).

Jev (docs.typesafe.ai) is a *System One* decision model: you send a `state` plus
typed questions and get back **calibrated probability distributions your code
can branch on** — no text generation. Kapteeni implements that behavior
faithfully end-to-end on local hardware:

```
state + questions ──▶ serialize ──▶ Qwen3-4B (LoRA-adapted) ──▶ h_last per pass
                                      (one pass per question/option/level)
                 ──▶ readout heads + verbalizer blend ──▶ temperature ──▶ contract shaping
                 ──▶ {answers: typed + probabilities + confidence, usage}
```

## Benchmark: JevBench v1.4 public half

Kapteeni v1 is evaluated on the public half of JevBench (the benchmark for
Jev-class decision models; 231 items: easy 48 / standard 72 / hard 111) using
the benchmark's own official scoring code, through the live server, end to end.

### Headline

| | kapteeni-v1 | Jev 1.13.0 (TypeSafe) | JevK5 | Hopper |
|---|---:|---:|---:|---:|
| **JevBench-style score** | **65.71** | 63.29 | 62.04 | 59.43 |
| public half placement | **~#2 of 73 ranked systems** | #3 | #4 | #5 |

Only classifier.dev (70.82) sits above — and its fast tier *is* Jev's model
resold. On the composite as computable from the public half, kapteeni-v1 is
the **best open rebuild on the board by a wide margin**, and competitive with
Jev itself. (Asterisks below.)

### The four axes (official formulas)

| axis | value | how it is measured |
|---|---:|---|
| **Intelligence** | **60.3 / 100** | chance-corrected tier accuracy, weights easy .14 / standard .28 / hard .30 (renormalized — the judge tier is sealed) |
| **Calibration** | **90.1 / 100** | from top-label ECE **0.0496** across all 231 answers (ECE half only; the board also averages a private gold-distribution half) |
| **Speed** | **81.1 / 100** | measured server latency p50 **0.17 s**, p95 **1.17 s** on an AMD Strix Halo iGPU, x2 self-hosted adjustment applied |
| **Cost** | **42.4 / 100** | 597 input tokens/decision x assumed $0.14/M hosted list price = $0.083/1k decisions (at $0.05/M the axis would be 55.8) |

Composite = equal-weight harmonic mean of the four axes (x the
`(Intelligence/50)^2` gate for I<50 — kapteeni-v1 clears it at 60.3).

### Per-tier accuracy (231 items)

| tier | n | accuracy |
|---|---:|---:|
| easy | 48 | **1.000** |
| standard | 72 | **0.889** |
| hard | 111 | **0.469** |
| pooled | 231 | **0.710** |

### Per-family accuracy (public items)

| family | n | acc | | family | n | acc |
|---|---:|---:|---|---|---:|---:|
| fact | 12 | 1.000 | | routing | 12 | 0.750 |
| tool_selection | 12 | 1.000 | | ambiguous | 7 | 0.429 |
| ordinal | 12 | 1.000 | | probability | 10 | 0.400 |
| extraction | 24 | 0.917 | | long_policy | 19 | 0.316 |
| intent | 24 | 0.958 | | multi_hop | 18 | 0.278 |
| policy | 12 | 0.917 | | temporal_numeric | 15 | 0.200 |
| adequacy | 12 | 0.917 | | tradeoff | 6 | 0.167 |
| adversarial | 6 | 1.000 | | judge_hard | 17 | 0.706 |
| trap | 8 | 0.875 | | routing_hard | 5 | 1.000 |

Biggest gains vs v0.1 (pre-LoRA): ordinal 0.25 → **1.00**, routing 0.25 →
0.75, trap 0.25 → 0.88, policy 0.67 → 0.92, intent 0.63 → 0.96,
judge_hard 0.47 → 0.71. One honest negative: the synthetic temporal/numeric
training did **not** transfer to the bench's temporal_numeric family
(0.27 → 0.20) — template diversity was too narrow; that family and
multi_hop remain the frontier. Ordinal expectation MAE: 0.482 levels;
paraphrase consistency: 0.861.

### Methodology & caveats (read this)

- **Self-reported public-half run** (the benchmark's 146 judge-tier items and
  308 sealed items are private). Not an official rank. Raw per-item outputs:
  `docs/bench/kapteeni-v1-record-231.jsonl`; v0.1's run is preserved alongside.
- **All serving constants for v1 are pre-registered**: head temperatures and
  blend weights fitted on our own mixed-domain validation set (BoolQ/FEVER/
  MNLI/Banking77/HelpSteer2 val slices) through the final merged model — no
  benchmark selection anywhere in the v1 chain.
- **Jev's accuracy on the same public items is still higher** (0.866 vs our
  0.710): our Intelligence is renormalized over three public tiers because the
  judge tier is sealed, while Jev's official figure blends it at weight 0.28.
  Like-for-like, Jev's Intelligence would exceed ours; our composite edge comes
  from Calibration (90 vs 76) and the Cost/Speed axes. Honest headline: best
  open rebuild, competitive with Jev on what is measurable — not proven ahead.
- Calibration is the ECE half only (the board also averages private
  gold-distribution fidelity); Cost rests on the stated $/M assumption;
  the sealed set rotates and a sealed measurement could differ.

Full details: `docs/JEVBENCH.md` (includes the v0 → v0.1 → v1 progression and
every fix along the way); calibration report: `docs/EVAL.md`.

## What is replicated

| Aspect | Reference behavior | kapteeni v0 |
|---|---|---|
| API | `POST /v1/systemone`, `GET /v1/models` | same wire format; docs' examples run unchanged (E11) |
| Noul | absolute P(true) ∈ [0,1]; no confidence field | sigmoid head, BCE on teacher soft labels — **absolute** (E5: P(A)+P(¬A) ≠ 1) |
| Choice | relative: probabilities sum to exactly 1 | per-option scores, softmax over the group, exact-1 decimal sum |
| Score | independent levels; score = expectation (can be fractional) | per-level BCE head, normalized at the API layer; legend echoes levels |
| Confidence | derived from distribution shape | 1 − H(p)/ln K, floored (**documented divergence**: the reference's formula is not public) |
| Calibration | "P=0.2 events fire ≈20% of the time" | temperature scaling per head; ECE/Brier measured on val + held-out datasets (E2) |
| Independence | adding/removing questions never changes others | structural: passes are content-only, question ids never serialized (E1) |
| Determinism | reference shows run-to-run std ≈ 0.01 | **fully deterministic** (documented improvement; ensemble noise is a future flag) |

## Layout

- `kapteeni/contract.py` — the wire format as executable code (validation, answer
  shaping, confidence, exact-1 probability sums)
- `kapteeni/serialize.py` — pass serialization (state JSON text + `[noul]/
  [choice]/[score]` suffixes; ids never serialized)
- `kapteeni/backbone.py` — frozen Qwen3-4B-Instruct-2507 (bf16/ROCm), batched
  `h_last` precompute with resume
- `kapteeni/heads.py`, `kapteeni/train.py` — per-primitive readout heads, proper-scoring
  losses (BCE/CE, soft targets), per-head temperature fit
- `kapteeni/build_data.py`, `kapteeni/criteria.py`, `kapteeni/distill.py` — datasets → rows →
  passes; teacher-written rubrics; k-sample soft labels
- `kapteeni/model.py`, `kapteeni/serve.py` — runtime model + stdlib HTTP server
- `tests/` — E1 (parity), E5 (asymmetry), E6 (score semantics), E11 (schema,
  end-to-end over HTTP); the suites run against the mock by default and against
  the trained model via `KAPTEENI_TEST_BUNDLE=`

## Run

```bash
cd kapteeni
python3 -m pytest                          # 62+ tests, mock model, no GPU needed

# serve the released model (from Hugging Face, or a local distribution pack)
python3 -m kapteeni.serve --hf <user>/kapteeni-v1 --port 8000
python3 -m kapteeni.serve --dist ./kapteeni-v1-dist --port 8000

# or from local training artifacts
python3 -m kapteeni.serve --bundle model_cache/kapteeni_v1.pt \
    --lora model_cache/kapteeni_p2/adapter --port 8000

curl localhost:8000/v1/systemone -d '{"state":"...","model":"jev-latest","questions":{...}}'
```

Prebuilt distribution packs (merged model, heads, serving constants, model
card — built with `python3 -m kapteeni.pack`) run without any local training
artifacts. Publishing one to Hugging Face: `docs/PUBLISH-HF.md`.

## Retrain from scratch

```bash
# full pipeline (see WORKLOG for the exact commands used)
python3 -m kapteeni.build_data ...              # datasets -> rows
python3 -m kapteeni.criteria ...                # teacher rubrics
python3 -m kapteeni.distill ...                 # teacher soft labels
python3 -m kapteeni.build_data passes ...       # rows -> passes
python3 -m kapteeni.backbone --passes ... --out data_cache/emb.pt
python3 -m kapteeni.train --emb data_cache/emb.pt --out model_cache/kapteeni_v0.pt
python3 -m kapteeni.train_p2 --passes ... --out model_cache/kapteeni_p2   # LoRA + heads
python3 -m kapteeni.p2_finalize                 # temps + blend, emits servable bundle
python3 -m kapteeni.pack --out ../kapteeni-v1-dist
```

## Design decisions worth knowing

1. **Noul is trained absolute** — a single sigmoid against soft targets, never
   a yes/no softmax pair. A complement-consistent Noul would be exactly the
   bug the reference's numbers (0.72 + 0.47 = 1.19) rule out; there is a test
   asserting non-identity.
2. **Choice softmax happens over the row's option group in-loss** so gradients
   shape relative separation, and at the API layer the distribution is rounded
   to 4 decimals with the argmax repaired so the decimal sum is exactly 1
   (matching the reference's published examples).
3. **Score levels are judged independently** (per-level BCE, "does the state
   match this level?"), normalized only at the API layer — this reproduces the
   reference's documented "numbers-only levels fail" asymmetry by construction.
4. **Soft labels from teacher k-sample agreement** (mean of k judge calls,
   weight = 1 − std). Ambiguous rows are kept and down-weighted: the
   calibration band is learned *from* disagreement.
5. **Deterministic row splits** by `sha256(row_id) % 10` — the same rule in
   expansion, distillation, and training, so val rows never leak into train
   passes, and the distiller never labels val rows.
6. **No chat template** — passes are plain text; behavior is template-independent
   and fully deterministic.

## Known divergences (documented, not bugs)

- Confidence formula is ours (theirs is unpublished).
- Our model is deterministic across repeats (reference: std ≈ 0.01).
- Usage accounting mirrors their semantics (input = state + questions,
  output = serialized answers) but token counts are our tokenizer's.
- v0 trains heads on a frozen backbone (the reference plan's "P1"); LoRA and
  ensemble noise are the next steps.

## Prior work

An earlier private attempt in this workspace established the verified dataset
facts (HF mirrors, label semantics) and environment lessons (offline HF mode,
proxy bypass, fence-tolerant JSON parsing) that this codebase builds on. The
code here was written fresh.

## License

- **Code, docs, generators:** Apache-2.0 (`LICENSE`) — matching the
  Apache-2.0 base model (Qwen3-4B-Instruct-2507).
- **Trained weights from this pipeline:** CC BY-SA 4.0, with the full
  training-data provenance and attribution guidance in
  `WEIGHTS-LICENSE.md` (the ShareAlike term comes from MultiNLI and the
  underlying FEVER annotations).