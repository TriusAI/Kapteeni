# Bench request draft — post as a GitHub issue to fstandhartinger/jevbench

> Copy everything below the line. Title:
> **[bench request]: kapteeni-v1-meticulous and kapteeni-v1-intuit (Qwen3-4B + LoRA + calibrated heads, two variants, TypeSafe wire format, self-hosted)**

---

**Weights:**
[TriusAI/kapteeni-v1-meticulous](https://huggingface.co/TriusAI/kapteeni-v1-meticulous) at
[`8c1abcf`](https://huggingface.co/TriusAI/kapteeni-v1-meticulous/tree/8c1abcf8279837b855a7e099898cd98c51353b89)
and
[TriusAI/kapteeni-v1-intuit](https://huggingface.co/TriusAI/kapteeni-v1-intuit) at
[`6466d70`](https://huggingface.co/TriusAI/kapteeni-v1-intuit/tree/6466d703989805d049cd5a247cf66249d8a05d98).
Each is a self-contained snapshot: merged bf16 model, tokenizer, readout heads
(safetensors), serving constants, the `kapteeni/` package, and a model card.

**Code:** https://github.com/TriusAI/Kapteeni at
[`3adc8bb`](https://github.com/TriusAI/Kapteeni/tree/3adc8bbf16) (Apache-2.0).
The repository carries the full pipeline (data, training, serving, 104 tests),
every raw public-half run record, the pre-registrations, and the negative
results.

Please add **kapteeni-v1-meticulous** and **kapteeni-v1-intuit** as their own
rows (or one row for `-meticulous` with `-intuit` noted, at your discretion).
They are two variants of one architecture — same base, same wire format,
different training data and serving constants — and are genuinely different
deployments, not a rerun. `-meticulous` is the default variant.

## What it is

A decision model that does not generate text. Every question — and, for
choice, every option; for score, every level — is its own pass through the
backbone (Qwen3-4B-Instruct-2507, 4.0B, Apache-2.0; frozen for the data
pipeline, then adapted with LoRA r=32 on all attention and MLP projections).
A 2-layer MLP readout head on the final hidden state and the backbone's own
yes/no next-token logits are blended geometrically; per-head temperatures and
blend weights are fit on held-out validation slices (never on JevBench data).
Serving shares one state prefill across all question/option/level passes.
Deterministic: the same build and input return the same probability
dictionary (asserted by the repo's test suite; verified across restarts and
serving paths). Trained weights are **CC BY-SA 4.0** (dataset provenance and
attribution obligations are in `WEIGHTS-LICENSE.md` in the repo); code
Apache-2.0. Not affiliated with TypeSafe; "Jev" is their model and trademark.

The two variants differ in what they are careful about: `-meticulous` was
trained on the base decision mix and is the conservative-confidence default;
`-intuit` adds 6.3k programmatic temporal/numeric, multi-hop and long-policy
items (ground truth by construction) and refits constants on a
deployment-diverse validation set — it decides more accurately on
well-formed structured traffic and less carefully on messy traffic. Our
request includes both because we document that trade-off per variant rather
than picking for every user.

## What we measured

With JevBench's CLI and the official `typesafe` adapter, one request at a
time, loopback, no other GPU load. `-meticulous` was measured at harness
revision `2fa63fa` (v1.4.0), `-intuit` at `1bcc55e` (v1.4.2); we verified the
public item sets are identical across those releases, so the splits are
directly comparable.

| | -meticulous | -intuit |
|---|---:|---:|
| correct | 164/231 | 165/231 |
| easy / standard / hard | 48/48 · 64/72 · 52/111 | 48/48 · 65/72 · 52/111 |
| answered and valid | 231/231 (failures 0) | 231/231 (failures 0) |
| ECE (10 bins, top-label) | 0.0496 | 0.1196 |
| latency p50 / p95 | 0.17 s / 1.17 s | 0.17 s / 1.19 s |
| mean input tokens | 597 | 597 |

Latency is raw, measured on an AMD Strix Halo iGPU (ROCm) — our only
available hardware; the stack is plain PyTorch/transformers with no
vendor-specific code, and CPU inference works (slowly), but we have not been
able to verify an NVIDIA pod ourselves.

**Identity check:** a correct setup reproduces 164/231 with that tier split
(48 · 64 · 52) and model name `kapteeni-v1-meticulous`; and 165/231
(48 · 65 · 52) with `kapteeni-v1-intuit`. Requests addressed to `jev-latest`
are accepted (the reference's alias behavior) and the response reports the
served variant's real name.

## To run it

The snapshots speak your wire format (`POST /v1/systemone`), so the existing
`typesafe` adapter reaches them with no new code:

```sh
huggingface-cli download TriusAI/kapteeni-v1-meticulous \
    --revision 8c1abcf8279837b855a7e099898cd98c51353b89 --local-dir k-m
cd k-m
pip install torch transformers safetensors   # plus the bundled package above
python -m kapteeni.serve --dist . --port 8000
# prints: listening on http://127.0.0.1:8000 (served as kapteeni-v1-meticulous)

python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8000 --model jev-latest \
  --key-env "" --results out.jsonl --raw-dir raw
```

Same for `-intuit` with `TriusAI/kapteeni-v1-intuit` at revision
`6466d70…` (server reports `kapteeni-v1-intuit`). Both servers run without
auth by default; `KAPTEENI_API_KEY` optionally adds a bearer check.

## Limits

English only. Context limit is the base model's; states beyond it are
rejected, not truncated. Option-count independent (one pass per option; no
cap). Readout constants are per-variant and shipped in each snapshot.

## Disclosures

- **No JevBench item text appears in any training file.** A normalized
  eight-word-sequence audit of every training sequence (18,457 rows; pass
  texts embed their states) against all 231 public items found **zero
  hits** — `scripts/contamination_audit.py`, result in
  `docs/contamination_audit.json` in the repo; rerunnable.
- **Development was benchmark-informed in three disclosed ways.** (1) The
  synthetic family data behind `-intuit` was designed *after* reading our own
  measured per-family public-half results and the benchmark's published
  family taxonomy — it targets the three families where we were weakest, by
  name. The generators are programmatic (`kapteeni/synth2.py`, seeded); no
  item text was used (zero-hit audit above). (2) An early v0.1-era blend
  scalar (w=0.25) was selected on the public half; that model is not part of
  this request, and both submitted variants' constants were fit on held-out
  validation slices with zero JevBench data. (3) We have run the public half
  six times across development; the three candidate evaluations after v1
  were each pre-registered with a one-run-per-candidate rule and their gates
  used public-half outcomes (the flagship gate). The full record — including
  the three negative results that failed those gates — is in
  `docs/JEVBENCH.md` and the `docs/PREREG-*.md` files.
- **Self-hosted, no provider tariff.** Under the benchmark's est.
  convention the cost is 597 input tokens/decision at the base model's
  public list price: $0.03/M (Novita, qwen3-4b-fp8) → $0.018/1k decisions,
  or $0.05/M (Alibaba's nearest official tier, qwen-turbo) → $0.030/1k.
- **Weights: CC BY-SA 4.0** (MultiNLI/FEVER ShareAlike terms carry the
  recommendation; full attribution obligations in `WEIGHTS-LICENSE.md`).
  Code: Apache-2.0. The base checkpoint (Qwen3-4B-Instruct-2507) was not
  trained by us; the LoRA adapters, readout heads, and constants were.

We will run whatever sealed-set process you prescribe for self-hosted
submissions (offline container, evaluator-owned pod, etc.) and can provide
the two snapshots pinned to the revisions above for it.