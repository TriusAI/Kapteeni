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
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kapteeni.build_data import data_dir  # noqa: E402

DATA = Path(data_dir())  # KAPTEENI_DATA_DIR override (forge runs)
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
    """row_id -> shingles of the training surface.

    Sources per PREREG-KAPTEENI-V11 (the v1.1 mixture: the six English
    replay sources at full non-val volume, synth2zh, and the two image
    datasets) PLUS the v1.1c additions per PREREG-KAPTEENI-V11C: the
    goemotions and synth rows that enter through the reused v1.2 text
    union (passes_p2). Val rows are excluded — held out, not trained.
    """
    out: dict[str, set] = {}
    for name in ("boolq", "fever", "banking77", "clinc150", "helpsteer2",
                 "synth2", "synth2zh", "goemotions", "synth"):
        for line in open(DATA / f"rows_{name}.jsonl",
                         encoding="utf-8"):
            r = json.loads(line)
            out[r["row_id"]] = shingles(_row_text(r))
    for ds in ("synth3", "synth3zh"):
        for line in open(DATA / f"{ds}/items.jsonl",
                         encoding="utf-8"):
            r = json.loads(line)
            out[r["row_id"]] = shingles(_row_text(r))
    return out


def load_ocnli() -> dict[str, set]:
    """The OCNLI gate slice (eval-only, CC BY-NC, never trained on)."""
    out = {}
    for line in open(DATA / "rows_ocnli.jsonl", encoding="utf-8"):
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


def load_feedback(paths: list[str]) -> dict[str, set]:
    """User-feedback store files as EXTRA training-side surfaces: any
    surface that might one day be trained on must be audited like the
    generated rows (feedback corrections are future training data).
    Records map to their state+question text via _feedback_text."""
    import glob as _glob
    out: dict[str, set] = {}
    for pattern in paths:
        files = sorted(_glob.glob(pattern))
        if not files:
            raise SystemExit(f"--feedback pattern matched nothing: {pattern}")
        for fp in files:
            for line in open(fp, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                out[f"feedback:{r.get('id', '?')}"] = shingles(
                    _feedback_text(r))
    return out


def _feedback_text(r: dict) -> str:
    parts = []
    st = r.get("state")
    if st:
        parts.append(st if isinstance(st, str) else json.dumps(
            st, ensure_ascii=False))
    q = r.get("question") or {}
    if isinstance(q, dict):
        parts.append(str(q.get("instructions", "")))
    else:
        parts.append(str(q))
    return " ".join(p for p in parts if p)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="contamination audit: training surfaces vs JevBench "
                    "public items + OCNLI gate rows")
    ap.add_argument("--out", default=str(ROOT / "docs" /
                                         "contamination_audit.json"),
                    help="report path (default: docs/contamination_audit"
                    ".json — the shipped record)")
    ap.add_argument("--feedback", action="append", default=[],
                    help="glob of feedback-store JSONL files to include "
                    "as training-side surfaces (the forge passes the "
                    "run's feedback store)")
    args = ap.parse_args(argv)
    train = load_training()
    train.update(load_feedback(args.feedback))
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
        "feedback_surfaces": sorted({k.rsplit(":", 1)[0]
                                     for k in train
                                     if k.startswith("feedback:")}),
        "jevbench": res_bench,
        "ocnli": res_ocnli,
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())