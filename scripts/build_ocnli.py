"""Build data_cache/rows_ocnli.jsonl — the OCNLI Chinese NLI gate rows.

Mirrors rows_mnli.jsonl's schema exactly (see build_data.build_mnli),
with Chinese instruction/criteria templates as pre-registered in
docs/PREREG-KAPTEENI-V11.md:

  instructions: 前提是否蕴含以下假设？假设：{hypothesis}
  criteria true/false: 前提蕴含该假设 / 前提不蕴含该假设

Source: CLUEbenchmark/OCNLI dev.json (data_cache/ocnli_dev.json,
CC BY-NC 2.0 — EVAL-ONLY: never trained on, not redistributed).
The dev file labels are explicit strings (entailment / neutral /
contradiction / '-'), which makes the pre-registered entailment->true
mapping unambiguous; '-' rows are skipped. First 500 valid rows in
file order (deterministic), gate slice = first 150 (as with MNLI).

    python3 scripts/build_ocnli.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kapteeni.build_data import data_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(data_dir())
SRC = ROOT / DATA / "ocnli_dev.json"
OUT = ROOT / DATA / "rows_ocnli.jsonl"

N_ROWS = 500


def main() -> int:
    src = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    rows, skipped = [], Counter()
    for i, r in enumerate(src):
        lab = r["label"]
        if lab == "-":
            skipped["unlabeled"] += 1
            continue
        if len(rows) >= N_ROWS:
            break
        rows.append({
            "row_id": f"ocnli-{i}",
            "primitive": "noul",
            "state": {"premise": r["sentence1"]},
            "instructions":
                f"前提是否蕴含以下假设？假设：{r['sentence2']}",
            "criteria": {"true": "前提蕴含该假设",
                         "false": "前提不蕴含该假设"},
            "label": 1 if lab == "entailment" else 0,
            "meta": {"level": r.get("level"), "genre": r.get("genre")},
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labs = Counter(r["label"] for r in rows)
    print(f"wrote {len(rows)} rows -> {OUT}")
    print(f"label balance: true={labs[1]} false={labs[0]} "
          f"({labs[1]/len(rows):.1%} entailment)")
    print(f"levels: {Counter(r['meta']['level'] for r in rows)}")
    print(f"skipped in source: {dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())