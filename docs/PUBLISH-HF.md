# Publishing Kapteeni v1 variants to Hugging Face

**Status (2026-09-26): both repos exist under TriusAI**
(`TriusAI/kapteeni-v1-meticulous`, `TriusAI/kapteeni-v1-intuit`) and the
full distributions are uploaded. The remaining workflow is incremental:

**Card-only updates** (numbers/caveats in the model cards, no weights
change): regenerate locally without a GPU —

```bash
python3 -m kapteeni.pack --cards-only --served-as kapteeni-v1-meticulous
python3 -m kapteeni.pack --cards-only --served-as kapteeni-v1-intuit
```

— then push just the file (seconds; the HF repo stores files
independently, so weights are untouched):

```bash
cd ../kapteeni-v1-meticulous-dist
huggingface-cli upload TriusAI/kapteeni-v1-meticulous README.md README.md \
    --repo-type model --commit-message "Card: <what changed>"
# likewise for ../kapteeni-v1-intuit-dist -> TriusAI/kapteeni-v1-intuit
```

The same `--cards-only` run also refreshes the bundled `kapteeni/`
package inside each dist; push it the same way if desired
(`huggingface-cli upload <repo> kapteeni kapteeni --repo-type model`).
From the build machine, prefix `HTTPS_PROXY=http://127.0.0.1:10081`
(the direct route to huggingface.co is blocked here; the proxy works).

**Full repacks** are needed only when weights/heads/constants change:
`python3 -m kapteeni.pack --served-as <variant>` (see below for the
intuit flags) — ~12 min each, then upload.

---

Kapteeni v1 ships as **two variants** (same architecture and wire format,
different training data and serving constants):

- **kapteeni-v1-meticulous** — conservative confidence; the default for
  unknown or messy traffic. Built from the original v1 artifacts.
- **kapteeni-v1-intuit** — sharper decisions on well-formed numeric /
  temporal / multi-step policy traffic. Built from the seed-3 (weak-family)
  artifacts with deployment-diverse-refit constants.

The build machine cannot reach huggingface.co (DNS is poisoned on its
network), so the upload runs from any machine that can. The distributions
are fully self-contained — no HF token needed to USE them, only to publish.

## 0. One-time setup (on a network that can reach HF)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # token with write access
```

## 1. Create the model repos

```bash
huggingface-cli repo create kapteeni-v1-meticulous --type model -y
huggingface-cli repo create kapteeni-v1-intuit --type model -y
```

## 2. Build the packs (on the machine with the training artifacts)

```bash
# meticulous (defaults)
python3 -m kapteeni.pack --served-as kapteeni-v1-meticulous

# intuit
python3 -m kapteeni.pack --served-as kapteeni-v1-intuit \
    --lora model_cache/kapteeni_p2_s3/adapter \
    --heads model_cache/kapteeni_p2_s3/heads.pt \
    --bundle model_cache/kapteeni_v1_2_1.pt \
    --fit data_cache/phase1/fit_kv_v1_2_1.json
```

Each pack (default `../kapteeni-v1-<variant>-dist`, ~7.6 GB) contains the
merged bf16 model + tokenizer, heads.safetensors, kapteeni-config.json
(variant name, head temperatures, blend constants), a variant-specific
model card (numbers only — no leaderboard claims), LICENSE +
WEIGHTS-LICENSE.md, and the kapteeni/ serving package. `--adapter-only`
builds a ~300MB pack instead (users pull the base model and merge at
load).

## 3. Upload

```bash
cd /path/to/kapteeni-v1-meticulous-dist
huggingface-cli upload TriusAI/kapteeni-v1-meticulous . . \
    --repo-type model --commit-message "Kapteeni v1 meticulous"
# likewise for the intuit pack
```

Or with Python (resumable, faster with hf_transfer):

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for variant in ("kapteeni-v1-meticulous", "kapteeni-v1-intuit"):
    api.create_repo(repo_id=f"TriusAI/{variant}", exist_ok=True)
    api.upload_large_folder(repo_id=f"TriusAI/{variant}",
                            repo_type="model",
                            folder_path=f"kapteeni-v1-{variant.split('-')[-1]}-dist")
EOF
```

## 4. How users run them after download

```bash
python -m kapteeni.serve --hf TriusAI/kapteeni-v1-meticulous --port 8000
python -m kapteeni.serve --hf TriusAI/kapteeni-v1-intuit --port 8000
```

Requirements: `torch`, `transformers`, `safetensors`, `huggingface_hub`.
The `kapteeni/` package ships inside each pack and also lives in the
GitHub repo (data pipeline, training code, tests). Requests may address
either variant by name, by the legacy `kapteeni-v1`, or by the
wire-compat alias `jev-latest`; responses always report the served
variant's real name.

## Checklist before pushing

- [ ] Model card frontmatter: `license: cc-by-sa-4.0` (weights; code is
      Apache-2.0 — both files included), `base_model:
      Qwen/Qwen3-4B-Instruct-2507`
- [ ] The cards' benchmark sections carry numbers + caveats only (the
      repo's no-leaderboard-claims policy; verified in pack.py's template)
- [ ] Add the GitHub repo URL to each card once public
- [ ] Optionally open a JevBench discussion/PR on the benchmark repo if
      you want the public-half runs listed (their submission process will
      re-run the sealed half — the one axis we cannot measure)
## 5. kapteeni-v1.1c (the multimodal line)

The multimodal pack is built by the mirrored packer:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python3 -m kapteeni.pack_v11c
# -> ../kapteeni-v1.1c-dist (8.6G: MERGED model + heads.safetensors +
#    kapteeni-config.json + card + licenses + the kapteeni/ serving
#    package incl. the demo website)
```

Validate the pack BEFORE upload (merged numerics must reproduce the
gated artifact-path behavior; LoRA merge is exact-linear, so the only
drift is bf16 rounding in the 2nd-3rd decimal — no decision flips):

```bash
HF_HUB_OFFLINE=1 python3 -m kapteeni.serve_v11c \
    --dist ../kapteeni-v1.1c-dist --port 8021
# then run the 15-case demo verification against :8021
# expected: same decisions as the artifact path; probabilities within
# the 3rd decimal; 20/23 vs gold
```

Upload target: `TriusAI/kapteeni-v1.1c`

```bash
huggingface-cli upload TriusAI/kapteeni-v1.1c ../kapteeni-v1.1c-dist . \
    --repo-type model
# then pin the revision in docs/BENCH-REQUEST.md / README as usual
```

Users run it exactly as the text variants:

```bash
HF_HUB_OFFLINE=1 python3 -m kapteeni.serve_v11c --hf TriusAI/kapteeni-v1.1c
# → API at /v1/systemone (state.image carries base64 images),
#   demo website at /
```

Card checklist for 1.1c: `base_model: Qwen/Qwen3.5-4B` (Apache-2.0
backbone), weights CC BY-SA 4.0, the six-gate table verbatim (images
0.9661 / Chinese images 0.9412 / MNLI 0.88 / OCNLI 0.8467 / synth2zh
0.9132 / synth2-EN 0.9048 / ECE 0.0241-0.0685), the state.image wire
extension documented, and the no-leaderboard-claims policy as for v1.

## 6. UPLOAD DONE (2026-10-02) — kapteeni-v1.1c is live

- https://huggingface.co/TriusAI/kapteeni-v1.1c — public, verified via
  the HF API: 69 files (the full pack incl. kapteeni/ + demo), card
  tags resolved (license: cc-by-sa-4.0, base Qwen/Qwen3.5-4B, en, zh,
  image-text-to-text).
- **Pinned revision: `17affb8670dcf466550a755bfd78a19b3dce615d`** —
  this is the hash bench requests / README citations must carry for
  the 1.1c line (the text variants' pinned revisions are 8c1abcf /
  6466d70).
- Card verified live (nextcard): the six-gate table + the
  state.image extension + demo instructions render as written.
