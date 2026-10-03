# Ollama System One prompt + scoring spec (pinned for the Kapteeni port)

Phase 0 deliverable of PLAN_OLLAMA.md. Sources of truth, in order:

1. **The runner source**, tag `v0.35.1` (`decision/systemone.go`,
   `decision/clef.go`, `decision/types.go`, `llm/score.go`,
   `llm/llama_server_score.go`, `server/routes.go:892`) — the code
   below cites it; this is the normative reference.
2. The published API reference (docs.ollama.com/api/systemone, 0.35.1).
3. Live fixtures against `tev1` (§6) — **capture pending** the local
   0.35.1 upgrade; this section will be filled with the rendered
   prompts and scoring traces, and any discrepancy with §2–§4 gets
   resolved there.

## 1. Wire contract (what requests mean)

Request: `{model, state, images?, questions: {name: question},
keep_alive?}`; 1–64 questions; state = nonempty string, or
object/array serialized to text; each question is one of:

- `noul`: `{type, instructions, criteria?: {"false": desc, "true": desc}}`
  (defaults No / Yes)
- `choice`: `{type, instructions, criteria: {key: description|null}}`
  — 2–26 options; a null description uses the key itself
- `score`: `{type, instructions, criteria: [descriptions]}` — 2–26,
  ordered lowest (index 0) to highest

Semantics that matter to the port: answers are **not** passed to later
questions (per-question independence is the runner's contract too);
requests ≤64 KiB text-only / ≤32 MiB with images; input is never
truncated (too long = error).

Response: `noul` → `P(true)` as a number; `choice` → argmax key +
per-key probabilities + confidence; `score` → **probability-weighted
average of zero-based level indices** (an expectation, not an argmax
level) + legend + per-level probabilities + confidence;
`confidence = 1 − H(p)/ln N` everywhere (concentration, explicitly
"not calibrated correctness"). Probabilities are a plain softmax over
candidate logits, summing to 1.

## 2. The letter encoding (the port's target; `decision.type` unset)

`server/routes.go` passes the model-metadata string `decision.type` to
the compiler; **unset/empty selects the candidate-scoring format shared
by Nimble and Tev**. This encoding rejects `images`.

### 2.1 Prompt construction (decision/systemone.go)

- `context` = the state: a string passes through verbatim; an object
  or array is **json.Compact**'d (key order preserved, no whitespace).
- `schema` = every question compiled to
  `{name, description, choices: [{code, value, description}...]}` where
  `code` is the letter `"A"`, `"B"`, … assigned **in option order**:
  - noul: **A = false** ("No"/custom), **B = true** ("Yes"/custom)
  - choice: A, B, C… in the criteria keys' insertion order
  - score: A = level 0, B = level 1, …
  - `value` is the wire value (noul: JSON true/false; choice: the key;
    score: the index as a decimal string); `description` is the
    criterion text (null descriptions already resolved to keys).
- Each question becomes its own single user message:

      <json.Marshal({"context": ..., "schema": [...]})>
      \n\nRequested field: <json.Marshal(question name)>

  i.e. the **same full schema payload for every question**, differing
  only in the quoted `Requested field` name. `instructions` follow the
  same content rule as state (string verbatim; object/array compacted).
- The user message is rendered through the **model's own chat
  template** (`Compiled.Render` calls the model's render function —
  the Modelfile/GGUF template). The port must therefore train against
  the exact template it will ship with.

### 2.2 Scoring (llm/llama_server_score.go) — the design risk, resolved

**One pass per question; one next-token distribution per question.**
There are no per-option passes: each question's row is scored once and
every candidate is read from that single distribution.

- Each candidate string must append **exactly one ordinary token** to
  the tokenized prompt (asserted by re-tokenizing prompt+candidate and
  the candidate alone); duplicate candidate tokens are rejected. So
  candidates are single tokens — the letters A–Z in the model's
  tokenizer, in a fresh-prompt continuation position.
- The runner requests one token with the candidates **logit-biased by
  +100** (retry at +1000), `top_k = len(candidates)`, temperature 1,
  and reads post-sampling probabilities; candidate logits are
  recovered as `ln p`. A shared bias cancels in the softmax, so the
  answer probabilities are **the softmax over the candidates' original
  next-token logits at T = 1**.
- **There is no temperature anywhere in the runner.** Any calibration
  must live in the weights themselves. Our fitted serving
  temperatures have no equivalent here.
- Hard constraint: the biased candidates must outrank **every other
  token** in the top-k distribution, else the request fails with
  "scoring candidates were outranked by other tokens". A well-trained
  port puts its answer mass on the letter tokens, above all other
  vocabulary, for every decision prompt.
- Answer mapping (systemone.go `Answer`): softmax over per-question
  logits; `noul` = probability of the true-letter; `choice` = argmax
  letter → option key; `score` = Σ j·p_j over level letters.

### 2.3 Usage mechanics

Shared-prefix **priming**: for ≥2 questions the runner evaluates the
shared prefix plus 4 tokens (checkpoint offset) so llama-server saves
a rollback checkpoint exactly at the end of the shared tokens — built
for **recurrent/hybrid layers like Qwen 3.5**, which cannot trim cache
like plain attention models. `output_tokens` = 1 discarded primer token
+ 1 per question; `input_tokens` sums every row's full length (shared
context is recounted per question). Requires two token positions of
headroom (prompt + 1 for scoring, +1 truncation guard).

## 3. The Clef encoding (`decision.type=clef`) — research note

A second, model-specific encoding exists: a raw ChatML prompt
(`STATE:` / `SCHEMA FIELDS:` / `FIELD n` blocks with `OPTION n:` spans)
fed to llama-server's `/embedding` with **`score_fields`** — the
runner maps per-option **token spans** to hidden-state readouts and
returns per-field logits computed by a **decision head carried in the
GGUF**. It accepts `images` at a fixed segment boundary (this is how
Clef/Clef Flash do vision decisions; joint per-field decisions; the
JSON canonicalization is Python `json.dumps(ensure_ascii=False,
sort_keys=True, separators=(",",":"))`-exact).

Implications, stated plainly:

1. **Heads-style readouts are executable on Ollama** — for models
   carrying the llama.cpp decision-head format. If that format is
   publicly trainable, a future native port of v1.1c (span heads over
   option texts, images included) becomes possible without the letter
   paradigm at all. Follow-up: pin the llama.cpp-side decision-head
   GGUF format (the pinned llama-server's `/embedding` +
   `score_fields` implementation).
2. **The letter encoding cannot ship v1.1c's image capability**
   (it rejects `images` outright). Therefore: the v1.1c Ollama port is
   **text-only in this encoding** — a constraint of the runtime, not a
   training choice. Images on Ollama today go through Clef-class
   models only.

## 4. What Phase 1 trains, concretely

- Prompt = the §2.1 user message rendered through the port's own chat
  template (the one the Modelfile will declare — pin it in the
  pre-reg). One training example per (row, question).
- Target = the **letter token** of the correct candidate at the next
  assistant position; with teacher distillation, the target is the
  teacher's output distribution **mapped onto the letters**:
  teacher choice-probabilities / noul P(true) / score level-probs →
  [P(A)…P(N)] (renormalized), so the port's letter softmax reproduces
  the teacher's calibrated distribution — the teacher's fitted
  temperatures fold in via its post-temperature outputs.
- The letters must dominate the next-token distribution (outrank
  check) — letter SFT produces this naturally; verify per-family.
- Independence contract maps 1:1: our multi-question rows exercise
  the exact per-question prompt shape the runner builds (full schema,
  one requested field).
- Evaluation must read **through the runner** (Phase 2): probabilities
  are T=1 softmaxes; the port's ECE is measured on exactly what
  /v1/systemone returns.

## 5. Risks carried forward (unchanged from the plan, now sharper)

- The letter paradigm's measured ceiling in this repo is ~0.85 on
  synth2 mastery (v1.1/v1.1b); the heads readout reached 0.91–0.90.
  Distillation from the calibrated teachers is the strongest available
  recipe for this paradigm, but the 0.90 bars are at-risk, honestly.
- Calibration lives in weights: no serving-side temperature exists to
  repair quantization damage — each quant is its own artifact needing
  its own gate numbers (Phase 2 evaluates quants separately).
- Qwen3.5 (v1.1c's backbone) conversion is the fresh path
  (llama.cpp #27019 — recheck; unsloth converter fallback). The v1
  text pair's Qwen3-4B converts cleanly; hence v1-meticulous first.

## 6. Live fixtures — PENDING

To capture after the local upgrade to 0.35.1: `ollama pull tev1`,
`ollama show tev1 --modelfile` (the exact TEMPLATE), then
`OLLAMA_DEBUG=1` requests with noul/choice/score/multi-question
payloads to confirm: the rendered prompts (template + payload
equality with §2.1), the letter candidates, primer/usage counts, and
the outrank-check behavior on an untrained-letter model (expected
400). This section records the captured fixtures verbatim.