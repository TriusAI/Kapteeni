# Kapteeni on Ollama — investigation, conclusion, and port plan

**Status (2026-10-03):** investigation complete. Conclusion: the packed
v1.1c artifact cannot be published on Ollama as a decision model — its
calibrated readout lives outside the weights, where Ollama's decision
runner cannot execute it. The viable path is a **behavior-distilled,
Ollama-native variant** trained to Ollama's own System One prompt and
candidate-token convention. Plan below; nothing has been trained or
pushed yet.

---

## 1. Why we looked

Prompt was `https://ollama.com/library/tev1` — Together AI's decision
model, published on Ollama. It is Kapteeni's exact model class: a
**Qwen3.5-4B fine-tune** (same backbone as v1.1c) answering Jev-style
typed questions. Since **Ollama 0.35 (2026-09-29)**, Ollama ships
native decision-model support, which changes what "publish on Ollama"
means for this project. An earlier first pass — made before knowing
about 0.35 — concluded decision models were inexpressible on Ollama
(GGUF runs next-token generation only). That conclusion is obsolete;
this document replaces it.

## 2. Findings: what Ollama supports as of 0.35 / 0.35.1

All checked 2026-10-03:

- **`/v1/systemone` endpoint** — the Jev API wire format: `state`
  (string or JSON) + 1–64 named `choice` / `noul` / `score` questions →
  typed answers with per-candidate probabilities and an entropy-based
  `confidence` (1 − H(p)/ln N). This is the wire format `serve_v11c`
  already speaks.
- **Published decision models:** `nimble` (Bespoke, Qwen3.5-9B),
  `tev1` / `tev1:0.8b` (Together, Qwen3.5-4B / 0.8B, 4.4 GB / 812 MB
  tags — Q8_0-class sizes). **Vision decision models exist:** `clef` /
  `clef-flash` take base64 `images` alongside `state` (require 0.35.1).
- **Third-party publishing is a first-class path:** the Modelfile grew
  `CAPABILITY decision` ("declare only for weights trained for System
  One's decision prompts") and `REQUIRES <version>`; publish flow is
  `ollama create` → `ollama cp <user>/<model>` → `ollama push` (account
  + public key registered at ollama.com/settings/keys).
- **Runtime mechanics** (from the API reference + model cards): the
  runner builds the decision prompt itself ("Ollama builds Tev1's
  prompt for you"); weights must be GGUF on a scoring-capable runner
  (cloud and MLX/Safetensors models are rejected for systemone —
  though `ollama create` can import a safetensors directory and build
  the GGUF itself). Scoring reads **candidate answer-token logits
  inside the model**: "Nimble reads the prompt once per question and
  scores the answer tokens directly… no reasoning step"; usage counts
  "tokens generated internally for scoring, including prefix
  preparation and retries", and shared state is re-counted "when the
  model scores questions separately" — i.e. per-question passes over a
  shared state prefix. Limits: 64 KiB request body without images,
  32 MiB with; `noul` returned as P(true) normalized over the
  {false, true} candidates.
- **Qwen3.5 hybrid support in llama.cpp is real but fresh:** gated
  deltanet linear-attention + full-attention layers execute correctly
  in the runtime (unsloth-converted GGUFs in the wild; Tev1/Nimble/Clef
  are all Qwen3.5-based). The official `convert_hf_to_gguf.py` had a
  live conversion bug in the `_reorder_v_heads` path (llama.cpp issue
  #27019) — re-check status before converting; unsloth's converter is
  the known-good fallback.

## 3. The gap: why the packed v1.1c is not the publishable artifact

Ollama's decision runner scores **answer-token logits inside the LM
head**. Kapteeni v1.1c's calibrated readout lives **outside the
weights**:

- each pass's final-token hidden state → `heads.safetensors` (31 MB,
  `noul./choice./score.`-prefixed readout heads over the 2560-dim
  state) — no execution slot exists for these in the GGUF runner;
- fitted per-primitive temperatures (noul 1.2071 / choice 0.9911 /
  score 1.8348 — the only constants fitted anywhere in the line) and
  contract shaping, applied in Python at serve time;
- the multi-pass serialization itself (per-question — and for choice,
  per-option; for score, per-level — passes over the same state) is
  rendered by `serve_v11c` through the chat template, while Ollama
  renders the decision prompt per its own System One convention, which
  the weights must have been trained for.

Publishing the current merged backbone with `CAPABILITY decision`
would emit letter guesses from a model never trained to emit letters;
every calibration claim in the card would be false for that artifact.
**Rejected.** Semantic deltas to carry into the port's card rather
than fight: images arrive via the `images` array (Clef-style), not
`state.image`; the 640px-bound advisory `notice` has no channel
through Ollama's response schema; Ollama's `noul` normalization and
entropy-based `confidence` replace our contract shaping (our exact-1
sums are compatible).

## 4. Verification: what is actually in the v1.1c pack

Checked in code (`kapteeni/pack_v11c.py`) and empirically on disk
(2026-10-03), comparing `../kapteeni-v1.1c-dist/model.safetensors`
against the base checkpoint `model_cache/qwen3.5-4b`:

- `pack_v11c.py` loads base + adapter, calls `merge_and_unload()`,
  saves with `safe_serialization=True` — **the LoRA is merged into the
  backbone at pack time; no adapter ships.**
- Empirical: 723 tensors, **zero** `lora`-named keys. LoRA-targeted
  tensors (`mlp.gate/up/down_proj`) are byte-different from base;
  frozen tensors (`embed_tokens`, `visual.blocks.0.attn.qkv`) are
  byte-identical. Matches the declared recipe (`language_model
  q/k/v/o/gate/up/down; tower frozen`) in `kapteeni-config.json`.

Implications: the checkpoint is self-contained and directly
GGUF-convertible (no merge step needed for any future pack), and the
merged backbone alone is precisely the artifact that must **not** be
published under the Kapteeni name on Ollama (§3).

## 5. The plan: a behavior-distilled Ollama-native variant

Working name **kapteeni-ollama** (final naming is an open decision,
§6). Same backbone, same training-data discipline, readout moved
inside the model as candidate-token logits.

### Phase 0 — pin the prompt contract (no GPU; ~1 day)

- Sources of truth: the `ollama/ollama` runner source (MIT) for the
  decision/scoring path; `togethercomputer/tev1` and
  `bespokelabsai/nimble` training repos; local ground truth via
  `OLLAMA_DEBUG` on a 0.35.1 install with `ollama pull tev1`, issuing
  `/v1/systemone` requests and capturing the rendered prompts and
  scoring internals.
- Deliverable: `docs/OLLAMA-PROMPT-SPEC.md` — the exact prompt
  skeleton per primitive, the candidate-token convention (letters vs
  keys), the image path, captured fixtures.
- **Design risk to resolve here:** whether the runner scores a
  question's candidates in a single pass (one letter distribution) or
  can do per-candidate passes. v1.1c's choice readout is a softmax over
  per-option passes scored in-loss; if the runner is single-pass per
  question, the port's choice mechanism changes structurally — that
  must be measured (Phase 2), not assumed away.

### Phase 1 — training (distillation)

- Same backbone + LoRA recipe as v1.1c; readout = answer-candidate
  token logits — the `verbalizer.py` "Design-B" paradigm already in
  the repo as a baseline class.
- **Teacher = the packaged v1.1c pipeline itself:** train the
  candidate-token distribution to match v1.1c's calibrated,
  post-temperature outputs (the `distill.py` / `committee.py`
  infrastructure is reusable). Hard labels from the existing synth
  generators remain the data backbone; add the candidate-target
  conversion per the Phase 0 spec.
- Keep the line's disciplines: question ids never rendered; per-question
  independence; no benchmark data anywhere near training.
- Vision: train with images through the pinned Ollama image path — or
  ship text-only first and add vision after (open decision, §6).

### Phase 2 — re-gating (pre-registered; the v1.1c gates do NOT transfer)

- New prereg doc (`docs/PREREG-KAPTEENI-OLLAMA.md`), same gate set
  (synth3 / synth3zh / MNLI / OCNLI / synth2zh / synth2-EN / fitted
  ECE), measured **on the Ollama artifact through the actual Ollama
  runtime** — a Python simulation of the runner is not a gate
  environment.
- Calibration is the line's differentiator: ECE first-class; also
  re-run the JevBench public half through Ollama's `/v1/systemone`
  (the official harness already speaks this wire format — that is the
  point of the port).
- **Quantization is part of the artifact now:** evaluate Q8_0 and any
  smaller quant separately — quantization moves logits, therefore
  probabilities, therefore calibration. Ship only quants the numbers
  defend.

### Phase 3 — packaging + publishing

- Pack exactly like `pack_v11c.py` (`merge_and_unload`; verified
  pattern §4).
- Convert to GGUF (`convert_hf_to_gguf.py` on latest master, re-check
  issue #27019; unsloth converter as fallback) — or `ollama create`
  directly from the safetensors directory.
- Modelfile: `FROM` the GGUF; `CAPABILITY decision`; `REQUIRES
  0.35.0` (text-only launch) or `0.35.1` (vision); `LICENSE` carrying
  the CC BY-SA 4.0 weights terms + WEIGHTS-LICENSE.md attribution.
- Smoke-test `/v1/systemone` locally (the TypeSafe SDK pointed at
  `localhost:11434` works as a client).
- Publish: ollama.com account + public key → `ollama cp
  <user>/kapteeni…` → `ollama push`.

### Phase 4 — cards and cross-linking

- Ollama card: disclose the artifact as a behavior-distilled port of
  `TriusAI/kapteeni-v1.1c` with **its own** Phase 2 numbers;
  no-leaderboard-claims policy as for the HF cards; document what
  differs by construction (no 640px-bound notice channel, `images`
  array input, Ollama's noul normalization and confidence).
- HF v1.1c (pinned rev `9cbc9e32ef547b8376eae75b80719d1a16a6021b`)
  remains the reference artifact; both cards cross-link. The port does
  not replace the heads pipeline for gate or claims purposes.

## 6. Open decisions

- Name on ollama.com (namespace + `kapteeni` vs `kapteeni-ollama` vs
  version-suffixed).
- Text-first vs vision-at-launch.
- Which quants ship.
- If the calibration gates fail at acceptable margins, the honest
  fallback is an explicitly **experimental** label (cf. tev1's), not a
  silent downgrade of the line's claims.

## 7. Non-goals

- No attempt to run the heads pipeline on Ollama (impossible by runtime
  construction, §3).
- No proxying Ollama requests to `serve_v11c` (Ollama has no
  remote-model support for systemone; the Python server remains the
  faithful artifact and stays published on HF).

## Sources (accessed 2026-10-03)

- https://ollama.com/library/tev1 · /library/nimble — model cards,
  runtime notes, benchmark tables
- https://ollama.com/blog/ollama-now-supports-jev-style-decision-models
  — 0.35 announcement
- https://docs.ollama.com/api/systemone — OpenAPI: request/response
  schemas, limits, usage semantics
- https://docs.ollama.com/capabilities/decision — decision guide incl.
  Clef vision models (0.35.1)
- https://docs.ollama.com/modelfile — `CAPABILITY`, `REQUIRES`
- https://docs.ollama.com/import — create/cp/push publishing flow
- https://www.together.ai/blog/how-to-train-your-own-jev — Tev1
  training recipe (single-letter answers, temperature 0, max_tokens 8)
- https://github.com/ggml-org/llama.cpp/issues/27019 — qwen3_5
  conversion-pipeline bug status
- Local: `kapteeni/pack_v11c.py`, `kapteeni-v1.1c-dist/` contents,
  safetensors header/hash inspection vs `model_cache/qwen3.5-4b`