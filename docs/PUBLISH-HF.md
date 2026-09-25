# Publishing Kapteeni v1 variants to Hugging Face

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
huggingface-cli upload <your-user>/kapteeni-v1-meticulous . . \
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
    api.create_repo(repo_id=f"<your-user>/{variant}", exist_ok=True)
    api.upload_large_folder(repo_id=f"<your-user>/{variant}",
                            repo_type="model",
                            folder_path=f"kapteeni-v1-{variant.split('-')[-1]}-dist")
EOF
```

## 4. How users run them after download

```bash
python -m kapteeni.serve --hf <your-user>/kapteeni-v1-meticulous --port 8000
python -m kapteeni.serve --hf <your-user>/kapteeni-v1-intuit --port 8000
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