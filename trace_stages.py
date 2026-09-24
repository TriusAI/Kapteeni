"""Instrument evaluate()'s actual stages for easy-intent-00."""
import json
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "jevbench")

from jevbench.adapters.base import build_question  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402
from kapteeni.model import SystemOneModel, _probs  # noqa: E402
from kapteeni.serialize import state_text  # noqa: E402

tasks = load_jsonl("jevbench/datasets/public/easy.jsonl")
t = next(x for x in tasks if x.id == "easy-intent-00")
q = build_question(t)

m = SystemOneModel("model_cache/kapteeni_v0.pt")
state = t.state
questions = {"decision": q}
passes = m._build_passes(state, questions)
state_str = state_text(state)
texts = [p[3] for p in passes]
suffixes = [p[3][len(state_str) + 2:] for p in passes]
h, s_verb = m._embed_texts(state_str, texts, suffixes)

head_of = {"noul": "noul", "choice_opt": "choice", "score_lvl": "score"}
z_flat = []
verb_pos = {qt: [] for qt in head_of}
for qt in ("noul", "choice_opt", "score_lvl"):
    idx = [i for i, p in enumerate(passes) if p[1] == qt]
    if not idx:
        continue
    hs = h[idx].float()
    import torch  # noqa: E402

    with torch.no_grad():
        z = m.heads[head_of[qt]](hs.to(m.device)).cpu().tolist()
    for i, val in zip(idx, z):
        z_flat.append((i, val / m.T[head_of[qt]]))
        verb_pos[qt].append((i, s_verb[i]))

print("pass order:")
for i, p in enumerate(passes):
    print(f"  {i}: key={p[2]:<18} z/T={z_flat[i][1]:>8.3f}  sv={s_verb[i]:>7.3f}")

scores_by_qid = {}
for i, v in z_flat:
    qid, kind, key, _ = passes[i]
    scores_by_qid.setdefault(qid, {"kind": kind, "scores": {}})["scores"][key] = v

info = scores_by_qid["decision"]
p_head = _probs(info["kind"], info["scores"])
print("\np_head:", {k: round(v, 6) for k, v in p_head.items()})

vmap = {}
for (i, sv) in verb_pos["choice_opt"]:
    pi = passes[i]
    if pi[0] == "decision":
        vmap[pi[2]] = sv
print("vmap:  ", {k: round(v, 3) for k, v in vmap.items()})

w, tau, b = m.blend_w["choice"], m.verb_tau["choice"], m.verb_b["choice"]
keys = list(p_head)
s = [w * math.log(max(p_head[k], 1e-9)) + (1 - w) * (vmap[k] + b) / tau
     for k in keys]
print("\nblend terms (key: w*log(p) + (1-w)*sv/tau):")
for k, si in zip(keys, s):
    print(f"  {k:<18} w*log(p)={w * math.log(max(p_head[k], 1e-9)):>8.3f}  "
          f"(1-w)*sv/tau={(1 - w) * (vmap[k] + b) / tau:>7.3f}  s={si:>7.3f}")