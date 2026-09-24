# Publishing Kapteeni v1 to Hugging Face

The build machine cannot reach huggingface.co (DNS is poisoned on its
network), so the upload runs from any machine that can. The distribution is
fully self-contained — no HF-token needed to USE it, only to publish.

## 0. One-time setup (on a network that can reach HF)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # token with write access
```

## 1. Create the model repo

```bash
huggingface-cli repo create kapteeni-v1 --type model -y
```

## 2. Upload the distribution

The pack is at `kapteeni-v1-dist/` (7.6 GB: merged bf16 model, tokenizer,
heads.safetensors, kapteeni-config.json, model card README with frontmatter,
LICENSE + WEIGHTS-LICENSE.md, and the kapteeni/ serving package).

```bash
cd /path/to/kapteeni-v1-dist
huggingface-cli upload <your-user>/kapteeni-v1 . . \
    --repo-type model --commit-message "Kapteeni v1.0.0"
```

Or with Python (resumable, faster with hf_transfer):

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="<your-user>/kapteeni-v1", exist_ok=True)
api.upload_large_folder(repo_id="<your-user>/kapteeni-v1",
                         repo_type="model",
                         folder_path="kapteeni-v1-dist")
EOF
```

## 3. Rebuild the pack if needed

From the repo root (the machine with the training artifacts):

```bash
python3 -m kapteeni.pack --out ../kapteeni-v1-dist          # full (7.6GB)
python3 -m kapteeni.pack --adapter-only --out ../kapteeni-v1-lite  # ~300MB
```

The `--adapter-only` pack ships the LoRA adapter + heads + config; users
pull the base model from `Qwen/Qwen3-4B-Instruct-2507` themselves and merge
at load time (`SystemOneModel(bundle, lora=..., model_name="Qwen/...")`).
Prefer the full pack unless bandwidth is constrained — it just works with
`from_pretrained` and the HF Hub caching.

## 4. How users run it after download

```bash
# from the Hub (no local artifacts needed):
python -m kapteeni.serve --hf <your-user>/kapteeni-v1 --port 8000

# or from a local snapshot:
python -m kapteeni.serve --dist ./kapteeni-v1-dist --port 8000
```

Requirements: `torch`, `transformers`, `safetensors`, `huggingface_hub`.
The `kapteeni/` package ships inside the distribution and also lives in the
GitHub repo (data pipeline, training code, tests).

## Checklist before pushing

- [ ] Model card frontmatter: `license: cc-by-sa-4.0` (weights; code is
      Apache-2.0 — both files included), `base_model:
      Qwen/Qwen3-4B-Instruct-2507`
- [ ] The card's benchmark table cites the self-reported public-half run and
      its caveats (already written into the card by `pack.py`)
- [ ] Consider adding the GitHub repo URL to the card ("Full code & data
      pipeline: ...") once the repo is public
- [ ] Optionally open a JevBench discussion/PR on the benchmark repo with
      the self-reported result if you want it listed (their submission
      process will re-run the sealed half)