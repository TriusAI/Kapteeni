"""Contamination audit: normalized 8-word-sequence overlap between every
training sequence (pass texts embed their states) and the 231 public
JevBench items.

    python3 scripts/contamination_audit.py

Emits docs/contamination_audit.json (for the bench request's disclosures)
and prints a summary. Zero-hit result is the expectation: the synthetic
generators are programmatic and the dataset rows come from public corpora
that predate the benchmark.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "jevbench"
N = 8  # shingle width, matches the benchmark's request conventions


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + N]) for i in range(len(words) - N + 1)}


def load_bench() -> dict[str, set]:
    """task_id -> shingles of the full item text (state + question +
    options/levels)."""
    out = {}
    import sys
    sys.path.insert(0, str(BENCH))
    from jevbench.tasks import load_jsonl
    for rel in ("datasets/public/easy.jsonl", "datasets/public/original.jsonl",
                "datasets/public/hard.jsonl"):
        for t in load_jsonl(str(BENCH / rel)):
            parts = [t.state if isinstance(t.state, str) else json.dumps(t.state)]
            q = t.question if isinstance(t.question, str) else json.dumps(t.question)
            parts.append(q)
            if getattr(t, "options", None):
                parts.extend(str(o) for o in t.options)
            if getattr(t, "levels", None):
                parts.extend(str(x) for x in t.levels)
            out[t.id] = shingles(" ".join(parts))
    return out


def load_training() -> dict[str, set]:
    """row_id -> shingles of every training pass text (text embeds state)."""
    out = {}
    for p in ("data_cache/passes_p2.jsonl", "data_cache/passes_synth2.jsonl"):
        for line in open(ROOT / p, encoding="utf-8"):
            r = json.loads(line)
            rid = r["row_id"]
            s = out.setdefault(rid, set())
            s |= shingles(r["text"])
    return out


def main() -> int:
    bench = load_bench()
    train = load_training()
    print(f"bench items: {len(bench)}; training rows: {len(train)}")

    hits: dict[str, dict] = {}
    for tid, tsh in bench.items():
        found = [(rid, len(tsh & rsh)) for rid, rsh in train.items()
                 if tsh & rsh]
        if found:
            hits[tid] = {"training_rows": {rid: n for rid, n in found},
                         "task_total_shingles": len(tsh)}
    total_shingle_hits = sum(
        n for v in hits.values() for n in v["training_rows"].values())

    payload = {
        "shingle_width_words": N,
        "normalization": "lowercase alnum word sequences",
        "bench_items": len(bench),
        "training_rows": len(train),
        "items_with_hits": len(hits),
        "total_shingle_hits": total_shingle_hits,
        "hits": hits,
    }
    (ROOT / "docs" / "contamination_audit.json").write_text(
        json.dumps(payload, indent=2))
    print(f"items with any 8-word-sequence overlap: {len(hits)} "
          f"({total_shingle_hits} shingle hits total)")
    if hits:
        for tid, v in list(hits.items())[:10]:
            top = Counter(v["training_rows"]).most_common(3)
            print(f"  {tid}: hits in {top}")
    else:
        print("zero hits — no training sequence shares an 8-word run with "
              "any public item")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())