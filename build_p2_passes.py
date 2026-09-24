"""Build the curated P2 training pass set (train rows only).

Mix (one epoch ~= 13-14M tokens, sized for an overnight run on the shared
Strix Halo):
  noul    BoolQ 2236 (soft) + FEVER 1342 (soft) + synth ~3600 (gold)  ~7.2k
  choice  banking77 1200x24 + clinc150 1200x16 + goemotions 1200x16    ~67k
  score   helpsteer2 ~1350 train rows x 5 levels                       ~6.8k

Val rows are excluded (deterministic is_val split) — they are the honest
in-domain gate; MNLI stays fully out (the OOD gate).
"""

from __future__ import annotations

import json
from pathlib import Path

from kapteeni.build_data import expand_passes, is_val

SPECS = [
    # (rows file, criteria file or None, opts cap, max train rows)
    ("data_cache/rows_boolq.jsonl", None, 1, 0),
    ("data_cache/rows_fever.jsonl", None, 1, 0),
    ("data_cache/rows_synth.jsonl", None, 1, 0),
    ("data_cache/rows_banking77.jsonl", "data_cache/crit_banking77.json", 24, 1200),
    ("data_cache/rows_clinc150.jsonl", "data_cache/crit_clinc150.json", 16, 1200),
    ("data_cache/rows_goemotions.jsonl", "data_cache/crit_goemotions.json", 16, 1200),
    ("data_cache/rows_helpsteer2.jsonl", "data_cache/crit_helpsteer2.json", 5, 0),
]


def main() -> int:
    out = Path("data_cache/passes_p2.jsonl")
    total = 0
    with open(out, "w", encoding="utf-8") as f:
        for rows_path, crit_path, cap, max_rows in SPECS:
            rows = [json.loads(l) for l in open(rows_path, encoding="utf-8")]
            train = sorted([r for r in rows if not is_val(r["row_id"])],
                           key=lambda r: r["row_id"])
            if max_rows:
                train = train[:max_rows]
            soft = {}
            stem = rows_path.split("rows_")[1].split(".")[0]
            soft_path = Path(f"data_cache/soft_{stem}.jsonl")
            if soft_path.exists():
                for l in open(soft_path, encoding="utf-8"):
                    r = json.loads(l)
                    soft[r["row_id"]] = {"p": r["p"], "weight": r.get("weight", 1.0)}
            crit = json.load(open(crit_path)) if crit_path else {}
            passes = expand_passes(train, crit, soft, opts_per_row=cap)
            for p in passes:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            total += len(passes)
            print(f"{stem:<12} rows={len(train):>5} passes={len(passes):>6}")
    print(f"total passes: {total} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())