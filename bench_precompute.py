"""Benchmark precompute throughput on real passes (GPU idle while distill runs)."""
import json
import time

import torch

from kapteeni.backbone import load_backbone, extract_h_last
from kapteeni.build_data import expand_passes

rows_b = [json.loads(l) for l in open("data_cache/rows_banking77.jsonl")][:400]
crit_b = json.load(open("data_cache/crit_banking77.json"))
passes_b = expand_passes(rows_b, crit_b, {}, opts_per_row=24)
rows_h = [json.loads(l) for l in open("data_cache/rows_helpsteer2.jsonl")][:60]
crit_h = json.load(open("data_cache/crit_helpsteer2.json"))
passes_h = expand_passes(rows_h, crit_h, {})

model, tok = load_backbone()
for name, passes, bt in (("banking77 (short passes)", passes_b[:1920], 16384),
                         ("helpsteer2 (long passes)", passes_h[:600], 16384)):
    texts = [p["text"] for p in passes]
    ntok = sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])
    t0 = time.time()
    h = extract_h_last(model, tok, texts, batch_tokens=bt)
    dt = time.time() - t0
    print(f"{name}: {len(texts)} passes, {ntok} tokens -> {dt:.1f}s "
          f"= {ntok/dt:.0f} tok/s, {len(texts)/dt:.1f} pass/s  h={tuple(h.shape)}")
print("peak GPU mem:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")