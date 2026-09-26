"""synth3: image decision items with GROUND TRUTH BY CONSTRUCTION.

The synth2 philosophy, pixel edition. Facts are sampled before rendering;
the same facts drive the image and every question, so gold is exact.

20 families across two modules:
  synth3_families.py   documents (menu, price tags, invoices, forms,
                       schedules, attendance, sales tables, hours signs,
                       tickets, spec sheets)
  synth3_families_v.py visual/UI (calendars incl. the week-counting rows
                       the V0 probe exposed, bar charts, seating maps,
                       weather, notification badges, kanban, number grids,
                       analog clocks, email headers, shape scenes)

Every render randomizes palette, fonts, and margins (synth2's
anti-narrow-template lesson). All pixels are ours (Apache-2.0); no
external images, no benchmark items.

CLI:
  python3 -m kapteeni.synth3 selfcheck
  python3 -m kapteeni.synth3 gen --rows 6000 --out data_cache/synth3
  python3 -m kapteeni.synth3 probe                  # 27-item V0 probe
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from kapteeni.build_data import is_val
from kapteeni.synth3_gfx import validate_rows
from kapteeni.synth3_families import (
    gen_attendance, gen_form, gen_invoice, gen_key_value, gen_menu,
    gen_price_tag, gen_schedule, gen_sign, gen_table_chart, gen_ticket,
)
from kapteeni.synth3_families_v import (
    gen_bar_chart, gen_calendar, gen_clock, gen_email_card, gen_grid_pattern,
    gen_kanban, gen_notification, gen_seating, gen_shape_scene,
    gen_weather,
)

GENS = {
    "menu_board": gen_menu, "price_tag": gen_price_tag,
    "invoice_receipt": gen_invoice, "form_sheet": gen_form,
    "schedule_table": gen_schedule, "attendance_sheet": gen_attendance,
    "table_chart": gen_table_chart, "sign_poster": gen_sign,
    "ticket_stub": gen_ticket, "key_value_panel": gen_key_value,
    "bar_chart": gen_bar_chart, "calendar_card": gen_calendar,
    "seating_chart": gen_seating, "weather_panel": gen_weather,
    "notification_card": gen_notification, "kanban_board": gen_kanban,
    "grid_pattern": gen_grid_pattern, "clock_face": gen_clock,
    "email_card": gen_email_card, "shape_scene": gen_shape_scene,
}


def _emit(gen, rng: random.Random, seq: int):
    """One render -> rows with globally unique row_ids."""
    out = []
    for bundle in gen(rng):
        validate_rows(bundle["rows"])
        for j, r in enumerate(bundle["rows"]):
            r["row_id"] = f"synth3-{seq:06d}-{j}"
            r["_img"] = bundle["img"]
            r["_rid"] = bundle["rid"]
            out.append(r)
    return out


def _selfcheck() -> int:
    for name, gen in GENS.items():
        for seed in (0, 1, 2):
            rng = random.Random(seed)
            rows = _emit(gen, rng, 1)
            assert rows, f"{name} produced no rows"
            # determinism: same seed -> identical rows and labels
            rng2 = random.Random(seed)
            rows2 = _emit(gen, rng2, 1)
            strip = lambda rs: [{k: v for k, v in r.items()
                                 if not k.startswith("_")} for r in rs]
            assert strip(rows) == strip(rows2), f"{name} not deterministic"
        print(f"  {name:<18} ok ({len(rows)} rows/render)")
    print("selfcheck: all 20 synth3 families hold their invariants and "
          "are deterministic")
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
    ap.add_argument("cmd", choices=["selfcheck", "gen", "probe"])
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        return _selfcheck()

    if args.cmd == "probe":  # V0 compatibility: fixed 9 renders
        args.out = args.out or "data_cache/vprobe"
    else:
        args.out = args.out or "data_cache/synth3"

    rng = random.Random(args.seed)
    names = list(GENS)
    probe_fams = ["menu_board", "calendar_card", "form_sheet"]
    rows: list[dict] = []
    seq = 0
    if args.cmd == "probe":
        for fam in probe_fams:
            for _ in range(3):
                rows.extend(_emit(GENS[fam], rng, seq))
                seq += 1
    else:
        i = 0
        while len(rows) < args.rows:
            rows.extend(_emit(GENS[names[i % len(names)]], rng, seq))
            seq += 1
            i += 1
    _write(rows, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())