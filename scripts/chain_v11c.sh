#!/bin/zsh
# v1.1c pipeline chain (single-GPU box, stages sequential): the P1
# precompute -> P1 heads -> the P2 run. Each stage starts only on the
# previous stage's success marker; any death stops the chain cleanly
# (every stage is itself resumable). Launch DETACHED:
#
#   setsid nohup zsh scripts/chain_v11c.sh > /tmp/opencode/v11c_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

echo "$(date) stage 1: P1 precompute"
HF_HUB_OFFLINE=1 python3 -u -m kapteeni.v11c_precompute \
  > /tmp/opencode/v11c_precompute.log 2>&1
if [[ $? != 0 || ! -f data_cache/v11c_emb.pt ]]; then
  echo "$(date) precompute failed (see /tmp/opencode/v11c_precompute.log)"; exit 1
fi

echo "$(date) stage 2: P1 heads"
HF_HUB_OFFLINE=1 python3 -u -m kapteeni.train_v11c --phase1 \
  > /tmp/opencode/v11c_p1.log 2>&1
if [[ $? != 0 || ! -f model_cache/kapteeni_v11c_p1.pt ]]; then
  echo "$(date) P1 failed (see /tmp/opencode/v11c_p1.log)"; exit 1
fi

echo "$(date) stage 3: P2 (the run)"
HF_HUB_OFFLINE=1 python3 -u -m kapteeni.train_v11c --p2 \
  > /tmp/opencode/v11c_p2.log 2>&1
echo "$(date) P2 exit=$?"
exit 0