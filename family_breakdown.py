"""Per-family breakdown for the README benchmark section (kapteeni-v1 run)."""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "jevbench")

from jevbench.summarize import summarize
from jevbench.tasks import load_jsonl

tasks = (load_jsonl("jevbench/datasets/public/easy.jsonl")
         + load_jsonl("jevbench/datasets/public/original.jsonl")
         + load_jsonl("jevbench/datasets/public/hard.jsonl"))
records = [json.loads(l) for l in open("docs/bench/kapteeni-v1-record-231.jsonl")]
s = summarize(tasks, records)
print(f"{'family':<18} {'n':>3} {'acc':>6} {'ece':>6}")
for fam, m in sorted(s["per_family"].items()):
    ece = m["ece"]["ece"] if isinstance(m["ece"], dict) else (m["ece"] or 0)
    print(f"{fam:<18} {m['n_scorable']:>3} {m['accuracy']:>6.3f} {ece:>6.3f}")
print("paraphrase consistency:", s["paraphrase_consistency"]["agreement"])
print("ordinal MAE:", round(s["ordinal_mae"], 3))