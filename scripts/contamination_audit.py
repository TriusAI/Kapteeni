"""Contamination audit for the v1.1 mixture: normalized 8-token-sequence
overlap between (a) every training surface and the 231 public JevBench
items, and (b) every training surface and the 500 OCNLI gate rows.

Tokenization is CJK-aware: tokens are maximal alnum runs (as before —
identical behavior on English text) plus single CJK characters, so an
8-token shingle is an 8-word run in English and an 8-character run in
Chinese. The zh training surfaces are programmatic template families,
so zero hits is the expectation; any hit would be inspected and
disclosed, never silently dropped.

    python3 scripts/contamination_audit.py

Emits docs/contamination_audit.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "jevbench"
N = 8  # shingle width in tokens, matches the benchmark's request conventions

_TOK = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]")


def tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def shingles(text: str) -> set[tuple[str, ...]]:
    toks = tokens(text)
    return {tuple(toks[i:i + N]) for i in range(len(toks) - N + 1)}


def _row_text(r: dict) -> str:
    """The full text surface a training example renders (state embeds
    into the prompt; options/levels/criteria complete the answer space)."""
    parts = [r.get("state_text") or json.dumps(
        r.get("state"), ensure_ascii=False)]
    parts.append(r.get("instructions") or "")
    m = r.get("meta") or {}
    parts.extend(str(o) for o in (m.get("options") or r.get("options") or []))
    parts.extend(str(x) for x in (m.get("levels") or r.get("levels") or []))
    c = r.get("criteria")
    if isinstance(c, dict):
        parts.extend(str(v) for v in c.values())
    elif isinstance(c, list):
        parts.extend(str(v) for v in c)
    return " ".join(p for p in parts if p)


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
    """row_id -> shingles of the v1.1 training surface.

    Sources per PREREG-KAPTEENI-V11: the six English replay sources
    (boolq, fever, banking77, clinc150, helpsteer2, synth2) at full
    non-val volume, synth2zh, and the two image datasets (synth3,
    synth3zh). Val rows are excluded — they are held out, not trained.
    """
    out: dict[str, set] = {}
    for name in ("boolq", "fever", "banking77", "clinc150", "helpsteer2",
                 "synth2", "synth2zh"):
        for line in open(ROOT / f"data_cache/rows_{name}.jsonl",
                         encoding="utf-8"):
            r = json.loads(line)
            out[r["row_id"]] = shingles(_row_text(r))
    for ds in ("synth3", "synth3zh"):
        for line in open(ROOT / f"data_cache/{ds}/items.jsonl",
                         encoding="utf-8"):
            r = json.loads(line)
            out[r["row_id"]] = shingles(_row_text(r))
    return out


def load_ocnli() -> dict[str, set]:
    """The OCNLI gate slice (eval-only, CC BY-NC, never trained on)."""
    out = {}
    for line in open(ROOT / "data_cache/rows_ocnli.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out[r["row_id"]] = shingles(_row_text(r))
    return out


def _audit(name: str, gates: dict[str, set], train: dict[str, set],
           n_gate: int) -> dict:
    hits: dict[str, dict] = {}
    for tid, tsh in gates.items():
        found = [(rid, len(tsh & rsh)) for rid, rsh in train.items()
                 if tsh & rsh]
        if found:
            hits[tid] = {"training_rows": {rid: n for rid, n in found},
                         "gate_total_shingles": len(tsh)}
    total = sum(n for v in hits.values() for n in v["training_rows"].values())
    print(f"[{name}] gate items: {n_gate}; training rows: {len(train)}; "
          f"items with any 8-token overlap: {len(hits)} ({total} hits)")
    if hits:
        for tid, v in list(hits.items())[:10]:
            top = Counter(v["training_rows"]).most_common(3)
            print(f"  {tid}: hits in {top}")
    else:
        print(f"[{name}] zero hits — no training surface shares an "
              f"8-token run with any gate item")
    return {"gate_items": n_gate, "training_rows": len(train),
            "items_with_hits": len(hits), "total_shingle_hits": total,
            "hits": hits}


def main() -> int:
    train = load_training()
    bench = load_bench()
    ocnli = load_ocnli()
    res_bench = _audit("jevbench", bench, train, len(bench))
    res_ocnli = _audit("ocnli", ocnli, train, len(ocnli))
    payload = {
        "shingle_width_tokens": N,
        "normalization": "lowercase alnum words + single CJK chars",
        "training_sources": [
            "rows_boolq", "rows_fever", "rows_banking77", "rows_clinc150",
            "rows_helpsteer2", "rows_synth2", "rows_synth2zh",
            "synth3/items", "synth3zh/items"],
        "val_rows_excluded": True,
        "jevbench": res_bench,
        "ocnli": res_ocnli,
    }
    (ROOT / "docs" / "contamination_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    print("wrote docs/contamination_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())