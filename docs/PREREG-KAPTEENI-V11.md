# Pre-registration: Kapteeni v1.1 — unified base (images + Chinese + new backbone)

Written 2026-09-28, before any v1.1 measurement. v1.1 is a new model
line: one model, one wire contract, images + text + Chinese. It
supersedes the two-variant text-only v1 line (which stays shipped as
kapteeni-v1-meticulous / -intuit) and absorbs the abandoned dedicated-
visual track (V2/V2.1 negative results documented in
PREREG-KAPTEENI-V.md: mixed single-LoRA recipes erode text competence
regardless of replay volume; image skills train robustly).

## Why a new backbone at all

The v1 text backbone was Qwen3-4B-Instruct-2507; the visual track used
its VL sibling. That was inheritance, not a choice. The 2026 landscape
was researched on 2026-09-28 and the Qwen family now spans
Qwen3.5/3.6/3.8 (natively multimodal from 3.5 on). Staying on Qwen by
default means competing inside the family everyone already uses.

## Step 0 — BASE-SELECTION PROBE (fixed now; zero training)

Frozen-backbone probe, one pass per question, same lettered LM-head
readout as the V0 probe (letters for choice/score, yes/no for noul;
thinking modes disabled wherever the chat template supports them;
deterministic, temperature-free).

**Candidates (fixed):**

1. `Qwen/Qwen3.5-9B` — 9B dense VL (2026-03), Apache-2.0, C-Eval 88.2,
   vendor MMMU 78.4 / OCRBench 89.2; hosted fp8 ~$0.04-0.06/M
   → ~$0.024-0.036/1k decisions (Jev-class ✓)
2. `Qwen/Qwen3.5-4B` — same family, budget control; ~$0.018/1k (✓)
3. `zai-org/GLM-4.6V-Flash` — ~10B dense VL (2025-12; verified config:
   `Glm4vForConditionalGeneration`, 40L text + 24-depth ViT), MIT,
   native zh+en (MMBench-CN 85.9); Z.ai API free tier (✓)

**Considered, excluded (documented, with reasons):**
- `Qwen3.8-27B` (2026-08): trainable on 96GB, but $0.45/M input
  (OpenRouter, the only existing price) → $0.27/1k decisions — 3.4×
  over the $0.080 Jev-class cutoff — and it is exactly the monoculture
  base the v1.1 switch is meant to differentiate from.
- `InternVL3.5-8B`: the language backbone is Qwen3 — differentiation
  would be cosmetic.
- `MiniCPM-V 4.5/4.6`: 4.5 archived 2026-02; license lineage requires
  commercial registration; 4.6 is 1B built on Qwen3.5-0.8B.
- `Kimi-VL-A3B`: 2025 generation; 16B resident for serving.
- `Gemma 3/4`: Gemma Terms are not standard-OSS-clean; Chinese quality
  below the Chinese-native candidates.

**Slices (fixed):**
- `synth3-val` — all 590 held-out image items (ours; identical to the
  V-track gate slice).
- `MNLI-noul` — first 150 of `data_cache/rows_mnli.jsonl` (identical to
  the V-track text gate slice; GLUE MNLI validation_matched,
  entailment→true binary noul; never trained on).
- `OCNLI-noul` — first 150 of a NEW `data_cache/rows_ocnli.jsonl`,
  built from CLUEbenchmark/OCNLI `dev.json` with the same conversion as
  MNLI, fully Chinese instruction/criteria templates:
  instructions `前提是否蕴含以下假设？假设：{hypothesis}`,
  criteria true `前提蕴含该假设` / false `前提不蕴含该假设`.
  OCNLI label mapping 0=entailment → true, 1/2 → false (asserted at
  build time against the CLUE reader).
  **License note:** OCNLI is CC BY-NC 2.0 — EVAL-ONLY: never trained
  on, not redistributed, only aggregate accuracies reported; identical
  usage pattern to MNLI. The contamination audit extends to it.

**Bilingual noul readout (fixed):** for the OCNLI slice the model's
binary decision is read as argmax over token groups —
yes-group = P("yes") + P("是"), no-group = P("no") + P("否") — because
a frozen backbone answering a Chinese prompt may answer in either
language. MNLI keeps the plain yes/no readout. No other readout rules
differ between models; all three candidates get the identical harness.

**Selection rule (fixed before measurement):**
- Primary: highest **mean of the three slice accuracies**.
- If candidates fall within 0.03 of the leader's mean: prefer the
  non-Qwen candidate (GLM-4.6V-Flash) for base-family differentiation;
  then prefer lower hosted cost.
- The winner's frozen probe numbers become v1.1's per-gate frozen
  baselines, fixed at probe time, before any training.
- If the winner's frozen synth3-val < 0.70 (the V0 feasibility line),
  the image requirement is re-examined before any training —
  documented, not silently absorbed.

**Cost axis (fixed):** each candidate's hosted list price is recorded
at probe time (~600 input tokens/decision, the v1 measured figure);
Jev-class eligibility requires <= $0.080/1k decisions.

## Step 0 AMENDMENT (2026-09-28, after the GLM smoke, before any full comparison)

The GLM-4.6V-Flash smoke (4 items/slice) exposed a harness-format
assumption: GLM answers the noul prompt with "true"/"false" (echoing
the criteria labels) rather than "yes"/"no" — at the answer position
"false" is top-5 and "yes" sits at rank 147, so the pre-registered
yes/no readout measures noise there (MNLI smoke: 0.5 = chance).
The lettered choice readout is unaffected (GLM's letter argmax matched
gold on the debug item). GLM's template was also verified by rendered-
prompt inspection to honor enable_thinking=False (an empty think pair;
jinja silently ignores unknown kwargs, so "no error" alone would have
proven nothing).

**Instrument repair (applied identically to ALL candidates):**
- noul EN: decision = argmax over probability mass on {yes, true}
  vs {no, false} (sum of exponentials of the readout logits).
- noul ZH: same, over {yes, true, 是} vs {no, false, 否}.
- choice/score: unchanged (argmax over letter logits).

All three candidates are (re-)probed with the amended instrument; the
first-pass Qwen3.5-9B/4B reports (pure yes/no readout) are kept for
the record but superseded. No gate values change; the selection rule
is unchanged; this is a measurement-instrument repair, not a gate
adjustment — a readout that misses a model's answer vocabulary measures
the tokenizer, not the decision.

## Step 0 OUTCOME (2026-09-28) — GLM excluded on stack grounds; the
## tie-break selects Qwen3.5-4B

**Amended-instrument results** (`data_cache/baseprobe/*.json`):

| candidate | synth3-val | MNLI-noul | OCNLI-noul | MEAN | cost /1k |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 0.8203 | 0.8800 | 0.8733 | **0.8579** | ~$0.036 |
| Qwen3.5-4B | 0.8068 | 0.8667 | 0.8600 | **0.8445** | ~$0.018 |
| GLM-4.6V-Flash | — stalled — | — | — | (unmeasured) | ~$0 |

**GLM exclusion (stack, not model):** the GLM probe stalled three
times at the identical stack location — the ViT patch Conv2d
(modeling_glm4v.py:765), grinding single-core for 12+ minutes on
first-touch batches — because its dynamic-resolution tower triggers
per-image-shape MIOpen kernel searches on ROCm (MIOPEN_FIND_MODE=1 did
not help; faulthandler traces recorded in data_cache/probe_glm.log).
Since v1.1 must train and serve on this box, an inference path this
pathological here disqualifies GLM as our base regardless of its
measured quality — which remains UNKNOWN, not low. Revisitable only
under a fresh pre-registration with a different stack (a CUDA host, a
vLLM build, or a standardized single-grid image pipeline, the last of
which would break identical-harness fairness for all candidates).

**Selection (rule applied as pre-registered):** 9B leads by 0.0134
mean — inside the 0.03 tie window (each 150-item slice carries a ~2-3pt
SE; the 9B's per-slice lead is consistent but sub-noise). With no
non-Qwen candidate in the tie, the cost tie-break decides:
**Qwen3.5-4B is the v1.1 backbone** (~$0.018/1k hosted fp8 vs ~$0.036).
Corroborating external evidence recorded after the fact: the top open
system on the JevBench v1.3-era board (SemIf, formerly OpenJev) runs
on Qwen3.5-4B.

**v1.1 frozen baselines (from this probe, fixed):** synth3-val 0.8068,
MNLI-noul 0.8667, OCNLI-noul 0.8600.

## v1.1 RECIPE (pre-registered 2026-09-28, after the probe, before any v1.1 training)

**Backbone:** `Qwen/Qwen3.5-4B` (probe-selected above). Frozen
baselines fixed by the probe: synth3-val **0.8068**, MNLI-noul
**0.8667**, OCNLI-noul **0.8600**.

**Wire answer space (fixed):** {yes, no} for noul and letter tokens for
choice/score in EVERY language. Chinese enters through state text,
instructions, and criteria only — never through the answer tokens.
Serving readouts stay unambiguous across languages; contract code is
unchanged.

**Training data** (all deterministic, gold by construction, ours or
license-clean, zero benchmark items; MNLI/OCNLI/val slices never
trained on):
1. **synth3** — the existing 5,412 image rows (20 document families,
   English questions, unchanged from the V-track).
2. **synth3zh** — NEW Chinese-language question/state/criteria
   templates over the same 20 image families (fresh deterministic
   renders; same generators, Chinese text; answers stay yes/no +
   letters). ~2,000 rows, is_val split as always.
3. **English text replay at full non-val volume** — the same six
   sources as V2.1 (~14,935 rows).
4. **synth2zh** — NEW Chinese ports of the three skill families
   (temporal_numeric with Chinese date formats / 工作日 semantics,
   multi_hop with Chinese rule chains, long_policy with Chinese clause
   grammar; facts generated from gold exactly as synth2 does).
   ~4,000 rows, is_val split.
5. Contamination: the 8-word audit extends to OCNLI before training;
   the new zh generators cannot touch benchmark content by
   construction (pure template families).

**Recipe (the V2/V2.1 mechanism diagnosis applied from two sides):**
- LoRA r=32, alpha=64, dropout 0.05 on **attention projections only**
  (q/k/v/o of the language model) — narrower update footprint than
  V2's all-projection config.
- lr **5e-5** (half of V2's 1e-4).
- Vision tower frozen; 1 epoch; real-token budget 4096; grad
  checkpointing; checkpoints + resume every 150 steps.
- Answer-token SFT in the lettered format for all three data kinds
  (images, EN text, ZH text).
- New-backbone hypothesis, stated honestly: Qwen3.5 is natively
  multimodal (early-fusion pretraining), unlike Qwen3-VL-4B's bolted-on
  tower — the V2/V2.1 text erosion may have been an artifact of
  adapting that architecture. This recipe tests that.

**Gates (fixed; final reading on the completed adapter, full slices):**

| gate | requirement | basis |
|---|---|---|
| synth3-val (n=590) | > 0.8068 | strictly beat frozen (the point of v1.1) |
| MNLI-noul (n=150) | >= 0.84 | frozen 0.8667 − 1 binomial SE (non-regression, noise-tolerant) |
| OCNLI-noul (n=150) | >= 0.83 | frozen 0.8600 − 1 SE |
| synth2zh val | >= 0.90 | in-distribution; must be mastered |
| fitted ECE on combined held-out val | <= 0.10 | per-primitive temperatures fit on combined val (synth3-val + synth3zh-val + text val + synth2zh-val), the v1.2.1 lesson applied in advance |

In-loop monitors (not gates): MNLI + OCNLI + synth3-val[:300] +
synth2zh-val[:200] every 300 steps, for erosion diagnosis.

**Engineering before the run (allowed; not gate measurement):** OOM
smoke (~100 steps with peak-memory verification; the per-image vision
token count re-measured from the Qwen3.5 processor, since the 400/image
constant was Qwen3-VL-specific), resume mechanism verified.

**Decision rule:** pass ALL gates -> fit constants, extend serving with
the `state.image` wire field, ship as **kapteeni-v1.1** (single
variant; meticulous-style conservative calibration is the identity).
Fail ANY gate -> no ship, negative result documented with the in-loop
trend, same recipe not rerun; the next step is a NEW pre-registration
guided by which gate failed (candidate directions: lower LR, still-
narrower targets, or modality-routed serving — text/Chinese through the
frozen backbone, images through the adapter).

## Not allowed

No training or constant selection on any benchmark's items (MNLI,
OCNLI, synth3 val, imajev-bench, JevBench); no gates adjusted after
results; one run per candidate recipe; honest negatives documented.
## Outcome (2026-09-29) — NEGATIVE: no ship

One run. Final gates on the completed adapter (full slices):

| gate | result | requirement | verdict |
|---|---|---|---|
| synth3-val (n=590) | 0.8881 | > 0.8068 | PASS |
| MNLI-noul (n=150) | 0.86 | >= 0.84 | PASS |
| OCNLI-noul (n=150) | 0.8867 | >= 0.83 | PASS |
| synth2zh-val (n=403) | 0.7767 | >= 0.90 | **FAIL** |
| fitted ECE | not fit | <= 0.10 | moot (gate already failed) |

In-loop monitor trend (every 300 steps): MNLI/OCNLI/synth3-val
remarkably stable from step 900 to 3327 (0.86 / 0.8867 / 0.87);
synth2zh-val flat at 0.84 on its [:200] slice (generation-ordered,
temporal-heavy) — the full 403-row slice reads 0.7767, so the deficit
sits in the later families.

Diagnosis (data_cache/v11_synth2_diagnosis.json): the same weakness
appears on the synth2-EN val (n=609): v1.1 0.775, V2 adapter 0.7373
(the FAILED quota-replay recipe; WORSE than v1.1) — replicated across
backbone (Qwen3.5 vs Qwen3-VL), LoRA footprint (attention-only vs
all-projection), and LR (5e-5 vs 1e-4). long_policy/score = 0.32 vs
0.25 chance on BOTH adapters (a shared blind spot). Conclusion: the
single-epoch unified replay cannot reconstitute the synthetic rule
skills the original multi-phase curriculum taught (v1's synth2-heavy
phases); capacity and LR are ruled out as discriminators. The zh
transfer itself is HEALTHY: synth2zh (0.7767) >= synth2-EN (0.775)
on the same skill families, and OCNLI (0.8867) beats MNLI (0.86).

Also recorded (engineering): two mid-run restarts (harness killed the
first overnight process; a no-checkpoint variant OOM'd when the
unified-memory pool shrank); fixed-width padding hypothesis
disproven; the loss spike at the image-region transition is convention
alignment, not divergence (image decisions stayed intact throughout,
synth3-val 0.87 -> final 0.8881).
