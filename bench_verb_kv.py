"""Compute KV-path verbalizer scores for the 231 bench items (once).

Artifact: data_cache/phase1/bench-verb-kv.pt — per-task {key: s_verb} aligned
with jevbench's build_question, exactly as the server's passes would produce.
Used for offline (w, tau, b) sweeps; the final number always comes from a
real server bench run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "jevbench"))

import phase1  # noqa: E402
from kapteeni.backbone import extract_shared_prefix, load_backbone  # noqa: E402

OUT = Path("data_cache/phase1/bench-verb-kv.pt")


def main() -> int:
    if OUT.exists():
        blob = torch.load(OUT, weights_only=False)
        print(f"cached: {len(blob['by_task'])} tasks")
        return 0
    tasks, passes = phase1.bench_passes()
    from collections import defaultdict
    from kapteeni.serialize import state_text

    by_task = defaultdict(dict)
    model, tok = load_backbone()
    # head from the bundle (same as the server) for z on the deployed path
    from kapteeni.heads import PassMLP

    blob = torch.load("model_cache/kapteeni_v0.pt", weights_only=False)
    heads = {}
    for qt in ("noul", "choice", "score"):
        h = PassMLP(blob[qt]["in_dim"])
        h.load_state_dict(blob[qt]["head"])
        h.eval()
        heads[qt] = h
    # group passes by task; each task's passes share its state
    groups = defaultdict(list)
    for p in passes:
        groups[p["task_id"]].append(p)
    for k, (tid, ps) in enumerate(sorted(groups.items())):
        t = next(x for x in tasks if x.id == tid)
        st = state_text(t.state)
        sufs = [p["text"][len(st) + 2:] for p in ps]
        h_kv, sv = extract_shared_prefix(model, tok, st, sufs)
        qt = t.question["type"]
        with torch.no_grad():
            z = heads[qt](h_kv.float()).tolist()
        for p, s, zz in zip(ps, sv.tolist(), z):
            by_task[tid][p["key"]] = {"sv": s, "z": zz}
        if (k + 1) % 40 == 0:
            print(f"  {k+1}/{len(groups)} tasks", flush=True)
    torch.save({"by_task": dict(by_task)}, OUT)
    torch.save({"by_task": dict(by_task)}, OUT)
    print(f"wrote {OUT} ({len(by_task)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())