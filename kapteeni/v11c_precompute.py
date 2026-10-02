"""v1.1c P1 precompute: h_last for every P1 training + val pass, on the
frozen Qwen3.5-4B (bf16, batched, incremental shard saves + resume).

    HF_HUB_OFFLINE=1 setsid nohup python3 -m kapteeni.v11c_precompute \
        > /tmp/opencode/v11c_precompute.log 2>&1 &

Output: data_cache/v11c_emb.pt {"h": fp16 (N, hidden), "records": [...]}
in the v0 emb format (P1 consumes it via the same contract).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kapteeni.build_data import data_dir
from kapteeni.v11c import (extract_h, p1_train_passes, p1_val_passes)

MODEL = "model_cache/qwen3.5-4b"
OUT = f"{data_dir()}/v11c_emb.pt"
CKPT = Path(f"{data_dir()}/v11c_precompute")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=4000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.base)
    tok = proc
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16).to(args.device)
    model.eval()
    inner = model.model  # multimodal inner decoder (frozen, no LoRA here)

    train = p1_train_passes()
    val = p1_val_passes()
    records = train + val
    print(f"P1 passes: {len(train)} train + {len(val)} val = {len(records)}",
          flush=True)

    CKPT.mkdir(parents=True, exist_ok=True)
    state_path = CKPT / "state.json"
    state = {"done": 0}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"resuming at pass {state['done']}", flush=True)

    h_parts, done = [], state["done"]
    shard_paths = sorted(CKPT.glob("h_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    for p in shard_paths:
        h_parts.append(torch.load(p, weights_only=True))
    n_have = sum(t.shape[0] for t in h_parts)
    if n_have != done:
        raise RuntimeError(f"shards hold {n_have} rows but state says {done}")
    shard_idx = len(shard_paths)

    t0 = time.time()
    todo = records[done:]
    for i in range(0, len(todo), args.save_every):
        chunk = todo[i: i + args.save_every]
        h = extract_h(inner, proc, tok, chunk, args.device, args.batch)
        h_parts.append(h.to(torch.float16))
        shard_idx += 1
        torch.save(h.to(torch.float16), CKPT / f"h_{shard_idx}.pt")
        state["done"] += len(chunk)
        state_path.write_text(json.dumps(state))
        dt = time.time() - t0
        print(f"  {state['done']}/{len(records)} passes "
              f"({state['done']/dt:.1f}/s, peak "
              f"{torch.cuda.max_memory_allocated()/2**30:.1f}G)", flush=True)

    h = torch.cat(h_parts)
    for r_i, r in enumerate(records):
        r["_i"] = r_i
    torch.save({"h": h, "records": records}, args.out)
    print(f"done: {tuple(h.shape)} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())