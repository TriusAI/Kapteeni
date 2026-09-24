#!/bin/zsh
# Post-CS-precompute pipeline: noul precompute -> merge -> train -> eval.
# Run after data_cache/emb.pt has the choice+score records.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1

echo "=== 1/5: precompute noul passes (appends to emb.pt) ==="
python3 -u -m kapteeni.backbone --passes data_cache/passes_noul.jsonl \
    --out data_cache/emb.pt --batch-tokens 16384

echo "=== 2/5: train heads ==="
python3 -u -m kapteeni.train --emb data_cache/emb.pt --out model_cache/kapteeni_v0.pt

echo "=== 3/5: OOD eval on held-out MNLI ==="
python3 -u -m kapteeni.evalx --bundle model_cache/kapteeni_v0.pt \
    --rows data_cache/rows_mnli.jsonl --out data_cache/mnli_eval.json

echo "=== 4/5: consolidated E2 report ==="
python3 -u -m kapteeni.report --emb data_cache/emb.pt --bundle model_cache/kapteeni_v0.pt

echo "=== 5/5: full test suite against the trained model ==="
KAPTEENI_TEST_BUNDLE=model_cache/kapteeni_v0.pt python3 -m pytest -q

echo "ALL DONE"