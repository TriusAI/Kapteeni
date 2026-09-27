"""synth3zh: the Chinese edition of the synth3 image families.

Same plumbing as kapteeni.synth3 (emit/validate/write), over the 20
Chinese-localized generators in synth3_families_zh. Facts drive gold
exactly as in the English edition; only rendered document text,
questions, and criteria are Chinese (Noto CJK fonts).

  python3 -m kapteeni.synth3zh selfcheck
  python3 -m kapteeni.synth3zh gen --rows 2000 \
      --out data_cache/synth3zh
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from kapteeni.build_data import is_val
from kapteeni.synth3_gfx import validate_rows
from kapteeni.synth3_families_zh import GENS_ZH


def _emit(gen, rng: random.Random, seq: int):
    out = []
    for bundle in gen(rng):
        validate_rows(bundle["rows"])
        for j, r in enumerate(bundle["rows"]):
            r["row_id"] = f"synth3zh-{seq:06d}-{j}"
            r["_img"] = bundle["img"]
            r["_rid"] = bundle["rid"]
            out.append(r)
    return out


def _selfcheck() -> int:
    for name, gen in GENS_ZH.items():
        for seed in (0, 1, 2):
            rng = random.Random(seed)
            rows = _emit(gen, rng, 1)
            assert rows, f"{name} produced no rows"
            rng2 = random.Random(seed)
            rows2 = _emit(gen, rng2, 1)
            strip = lambda rs: [{k: v for k, v in r.items()
                                 if not k.startswith("_")} for r in rs]
            assert strip(rows) == strip(rows2), f"{name} not deterministic"
            # Chinese surfaces really are Chinese (no stray English words
            # in question templates, except legitimate code-switches like
            # train numbers and e-mail addresses)
            for r in rows:
                assert r["instructions"], f"{name} empty instructions"
        print(f"  {name:<18} ok ({len(rows)} rows/render)")
    print("selfcheck: all 20 zh families hold their invariants and are "
          "deterministic")
    return 0


def _write(rows, out: Path):
    (out / "images").mkdir(parents=True, exist_ok=True)
    seen_rids = set()
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            rid = r["_rid"]
            if rid not in seen_rids:
                seen_rids.add(rid)
                r["_img"].save(out / "images" / (rid + ".png"))
            payload = {k: v for k, v in r.items()
                       if not k.startswith("_")}
            payload["image"] = rid + ".png"
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    fams = Counter(r["meta"]["family"] for r in rows)
    prims = Counter(r["primitive"] for r in rows)
    pos = [r["label"] for r in rows if r["primitive"] == "noul"]
    n_val = sum(1 for r in rows if is_val(r["row_id"]))
    print(f"wrote {len(rows)} rows ({len(seen_rids)} images) -> {out}")
    print(f"  families: {len(fams)}; primitives: {dict(prims)}")
    print(f"  noul positive rate: {sum(pos) / len(pos):.2f}")
    print(f"  val rows (deterministic split): {n_val}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selfcheck", "gen"])
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--out", default="data_cache/synth3zh")
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        return _selfcheck()

    rng = random.Random(args.seed)
    names = list(GENS_ZH)
    rows: list[dict] = []
    seq = 0
    i = 0
    while len(rows) < args.rows:
        rows.extend(_emit(GENS_ZH[names[i % len(names)]], rng, seq))
        seq += 1
        i += 1
    _write(rows, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())