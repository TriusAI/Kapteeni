"""Trace: server in-process vs offline ingredients for one choice task."""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "jevbench")

from kapteeni.model import SystemOneModel  # noqa: E402
from sweep_kv import load  # noqa: E402

tasks, ing, T_head = load()
t = next(x for x in tasks if x.question["type"] == "choice"
         and x.id == "easy-intent-00")

m = SystemOneModel("model_cache/kapteeni_v0.pt")
q = {"type": "choice", "instructions": t.question["instructions"],
     "criteria": t.question["criteria"]}
passes = m._build_passes(t.state, {"decision": q})
h, sv_server = m._embed_texts(
    __import__("kapteeni.serialize", fromlist=["state_text"]).state_text(t.state),
    [p[3] for p in passes],
    [p[3][len(__import__("kapteeni.serialize", fromlist=["state_text"]).state_text(t.state)) + 2:]
     for p in passes],
)
with m.model.device:
    pass
import torch  # noqa: E402

zs_server = m.heads["choice"](h.to(m.device).float()).cpu().tolist()
print(f"{'option':<28} {'server z':>9} {'offline z':>9} | {'server sv':>9} {'offline sv':>9}")
for i, p in enumerate(passes):
    opt = p[2]
    print(f"{opt:<28} {zs_server[i]:>9.3f} {ing[t.id][opt]['z']:>9.3f} | "
          f"{sv_server[i]:>9.3f} {ing[t.id][opt]['sv']:>9.3f}")
print("T_head choice:", m.T["choice"])