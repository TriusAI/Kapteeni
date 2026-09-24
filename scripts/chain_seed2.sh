#!/bin/zsh
# Launches P2 seed 2 the moment seed 1 finishes successfully (single-GPU box:
# seeds must run sequentially). Gives up if seed 1 dies or takes > 11h.
set -u
cd "$(dirname "$0")/.."
deadline=$(( $(date +%s) + 11 * 3600 ))
while true; do
  if grep -q "^done: " data_cache/train_s1.log 2>/dev/null; then
    echo "$(date) seed1 done -> launching seed2"
    HF_HUB_OFFLINE=1 python3 -u -m kapteeni.train_p2 \
      --passes data_cache/passes_p2.jsonl \
      --bundle model_cache/kapteeni_v0.pt \
      --out model_cache/kapteeni_p2_s2 \
      --token-budget 8192 --lr 1e-4 --epochs 1 --seed 2 \
      > data_cache/train_s2.log 2>&1
    echo "$(date) seed2 exit=$?"
    exit 0
  fi
  if ! pgrep -f "kapteeni\.train_p2" > /dev/null; then
    sleep 60
    if ! grep -q "^done: " data_cache/train_s1.log 2>/dev/null; then
      echo "$(date) seed1 died without a success marker; NOT launching seed2"
      exit 1
    fi
  fi
  if (( $(date +%s) > deadline )); then
    echo "$(date) timed out waiting for seed1"
    exit 1
  fi
  sleep 180
done