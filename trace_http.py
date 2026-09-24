"""Compare in-process evaluate() vs the live HTTP server for one task."""
import json
import sys
import urllib.request

sys.path.insert(0, ".")
sys.path.insert(0, "jevbench")

from jevbench.adapters.base import build_question  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

tasks = load_jsonl("jevbench/datasets/public/easy.jsonl")
t = next(x for x in tasks if x.id == "easy-intent-00")
q = build_question(t)
body = {"state": t.state, "model": "jev-latest", "questions": {"decision": q}}

# 1) in-process
from kapteeni.model import SystemOneModel  # noqa: E402

m = SystemOneModel("model_cache/kapteeni_v0.pt")
answers, usage = m.evaluate(t.state, {"decision": q})
print("in-process :", json.dumps(answers["decision"]["probabilities"]))
print("  params:", m.blend_w, m.verb_tau, m.verb_b)

# 2) live server
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/systemone",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as r:
    resp = json.loads(r.read())
print("live server:", json.dumps(resp["answers"]["decision"]["probabilities"]))