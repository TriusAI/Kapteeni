"""synth3 seed: facts-first IMAGE decision items — the pixel edition of the
synth2 philosophy. Facts are generated BEFORE rendering; the same facts
drive both the rendered image and the questions, so gold is exact by
construction.

Three families for the V0 feasibility probe (frozen-backbone readout):
  menu_board     rendered café menu          -> cheapest/most-expensive
                                              (choice), price comparison (noul)
  calendar_card  rendered month grid + event  -> before/after day (noul),
                                              which week (choice)
  form_sheet     rendered form w/ checkboxes  -> selected? (noul),
                                              which option (choice)

CLI:
  python3 -m kapteeni.synth3 probe --out data_cache/vprobe

The V1 expansion (hundreds of template families) grows from this seed;
the probe only needs a handful of items per family.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    FONT = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    FONT_BIG = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
    FONT_SM = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
except OSError:  # pragma: no cover - fallback if the font is missing
    FONT = FONT_BIG = FONT_SM = ImageFont.load_default()

W, H = 640, 640
BG, INK, ACCENT = "#ffffff", "#111111", "#7a1f1f"


def _canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 46), title, font=FONT_BIG, fill=ACCENT, anchor="ma")
    return img, d


# ------------------------------------------------------------------ families

def gen_menu(rng: random.Random) -> list[dict]:
    """Café menu: 4 items with distinct prices -> 2 choice + 2 noul items."""
    names = rng.sample(["Espresso", "Latte", "Cappuccino", "Flat White",
                        "Mocha", "Americano", "Macchiato", "Cortado"], 4)
    prices = rng.sample(range(250, 800, 25), 4)  # cents, distinct
    items = list(zip(names, prices))
    rng.shuffle(items)
    img, d = _canvas("CAFE MENU")
    y = 120
    for name, cents in items:
        price = f"${cents // 100}.{cents % 100:02d}"
        d.text((60, y), name, font=FONT, fill=INK)
        tw = d.textlength(price, font=FONT)
        d.text((W - 60 - tw, y), price, font=FONT, fill=INK)
        x0, x1 = 60 + d.textlength(name, font=FONT) + 12, W - 75 - tw
        d.line((x0, y + 30, x1, y + 30), fill="#999999", width=2)
        y += 64
    rid = f"menu-{rng.randint(0, 10**6):06d}"
    exp_i = max(range(4), key=lambda i: items[i][1])
    chp_i = min(range(4), key=lambda i: items[i][1])
    rows = []
    # choice: most expensive / cheapest (gold by construction)
    for q, gold_i in (("Which drink is the most expensive?", exp_i),
                      ("Which drink is the cheapest?", chp_i)):
        rows.append({
            "row_id": f"synth3-{rid}-c", "primitive": "choice",
            "image": f"{rid}.png", "state_text": "A cafe menu board.",
            "instructions": q,
            "criteria": None,
            "options": [n for n, _ in items], "label": gold_i,
            "meta": {"family": "menu_board"},
        })
    # noul: threshold comparisons with near-miss traps
    for _ in range(2):
        name, cents = items[rng.randrange(4)]
        threshold = cents + rng.choice([-25, 25, -50, 50])
        rows.append({
            "row_id": f"synth3-{rid}-n", "primitive": "noul",
            "image": f"{rid}.png", "state_text": "A cafe menu board.",
            "instructions": (f"Does the {name} cost more than "
                             f"${threshold // 100}.{threshold % 100:02d}?"),
            "criteria": {"true": "the price is above the threshold",
                         "false": "the price is not above it"},
            "label": int(cents > threshold),
            "meta": {"family": "menu_board"},
        })
    return [{"img": img, "rid": rid, "rows": rows}]


def gen_calendar(rng: random.Random) -> list[dict]:
    """Month grid with one highlighted event -> day comparisons, weeks."""
    month_names = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
                   "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER",
                   "DECEMBER"]
    m = rng.randint(0, 11)
    event_day = rng.choice([7, 13, 14, 15, 16, 22, 28])
    img, d = _canvas(f"{month_names[m]}")
    y0 = 116
    for col, label in enumerate(("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")):
        d.text((76 + col * 72, y0), label, font=FONT_SM, fill=ACCENT)
    for day in range(1, 31):
        row, col = (day - 1) // 7, (day - 1) % 7
        x, y = 76 + col * 72, y0 + 52 + row * 62
        if day == event_day:
            d.rectangle((x - 12, y - 8, x + 44, y + 44), fill=ACCENT)
            d.text((x, y), str(day), font=FONT, fill="#ffffff")
        else:
            d.text((x, y), str(day), font=FONT, fill=INK)
    d.text((76, y0 + 52 + 5 * 62), "TEAM LUNCH (highlighted)",
          font=FONT_SM, fill=ACCENT)
    rid = f"cal-{m:02d}-{event_day:02d}"
    return [{"img": img, "rid": rid, "rows": [
        {"row_id": f"synth3-{rid}-n", "primitive": "noul",
         "image": f"{rid}.png",
         "state_text": "A calendar page with one highlighted event.",
         "instructions": "Is the highlighted team lunch scheduled after "
                         "the 15th of the month?",
         "criteria": {"true": "the event day is after the 15th",
                      "false": "the event day is on or before the 15th"},
         "label": int(event_day > 15),
         "meta": {"family": "calendar_card"}},
        {"row_id": f"synth3-{rid}-c", "primitive": "choice",
         "image": f"{rid}.png",
         "state_text": "A calendar page with one highlighted event.",
         "instructions": "In which week of the month is the highlighted "
                         "team lunch?",
         "criteria": None,
         "options": ["Week 1", "Week 2", "Week 3", "Week 4"],
         "label": min(3, (event_day - 1) // 7),
         "meta": {"family": "calendar_card"}},
    ]}]


def gen_form(rng: random.Random) -> list[dict]:
    """Delivery form with checkboxes; exactly one shipping method chosen."""
    express = rng.random() < 0.5
    insurance = rng.random() < 0.5
    gift = rng.random() < 0.5
    img, d = _canvas("DELIVERY FORM")
    y = 130
    for label, checked in (("Express shipping (+$5.00)", express),
                           ("Shipment insurance", insurance),
                           ("Gift wrapping", gift)):
        d.rectangle((70, y, 110, y + 40), outline=INK, width=3)
        if checked:
            d.line((76, y + 20, 88, y + 34), fill=INK, width=5)
            d.line((88, y + 34, 106, y + 6), fill=INK, width=5)
        d.text((130, y + 4), label, font=FONT_SM, fill=INK)
        y += 70
    d.text((70, y + 30), "Recipient: A. Customer", font=FONT_SM, fill=INK)
    rid = f"form-{int(express)}{int(insurance)}{int(gift)}"
    return [{"img": img, "rid": rid, "rows": [
        {"row_id": f"synth3-{rid}-n", "primitive": "noul",
         "image": f"{rid}.png",
         "state_text": "A delivery form with checkboxes.",
         "instructions": "Is express shipping selected on the form?",
         "criteria": {"true": "the express shipping box is checked",
                      "false": "it is not checked"},
         "label": int(express),
         "meta": {"family": "form_sheet"}},
        {"row_id": f"synth3-{rid}-c", "primitive": "choice",
         "image": f"{rid}.png",
         "state_text": "A delivery form with checkboxes.",
         "instructions": "Which shipping method is selected?",
         "criteria": None,
         "options": ["Standard shipping", "Express shipping"],
         "label": int(express),
         "meta": {"family": "form_sheet"}},
        {"row_id": f"synth3-{rid}-s", "primitive": "score",
         "image": f"{rid}.png",
         "state_text": "A delivery form with checkboxes.",
         "instructions": "How many optional extras are selected on the "
                         "form?",
         "criteria": None,
         "levels": ["None", "One", "Two", "All three"],
         "label": int(express) + int(insurance) + int(gift),
         "meta": {"family": "form_sheet"}},
    ]}]


GENS = [gen_menu, gen_calendar, gen_form]


def _selfcheck() -> int:
    rng = random.Random(0)
    for gen in GENS:
        for bundle in gen(rng):
            for r in bundle["rows"]:
                if r["primitive"] == "choice":
                    assert 0 <= r["label"] < len(r["options"])
                if r["primitive"] == "noul":
                    assert r["label"] in (0, 1)
                if r["primitive"] == "score":
                    assert 0 <= r["label"] < len(r["levels"])
    # menu golds really match the facts
    rng = random.Random(3)
    b = gen_menu(rng)[0]
    choice = next(r for r in b["rows"] if r["primitive"] == "choice")
    assert "expensive" in choice["instructions"] or "cheapest" in \
        choice["instructions"]
    print("selfcheck: all synth3 seed invariants hold")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "selfcheck"])
    ap.add_argument("--out", default="data_cache/vprobe")
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        return _selfcheck()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for gen in GENS:
        for _ in range(args.per_family):
            bundle = gen(rng)
            for b in bundle:
                b["img"].save(out / "images" / (b["rid"] + ".png"))
                items.extend(b["rows"])
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    fams = Counter(r["meta"]["family"] for r in items)
    prims = Counter(r["primitive"] for r in items)
    pos = [r["label"] for r in items if r["primitive"] == "noul"]
    print(f"synth3 probe: {len(items)} items, {len(fams)} families "
          f"({dict(fams)}), primitives {dict(prims)}, "
          f"noul positive rate {sum(pos) / len(pos):.2f}")
    print(f"-> {out}/items.jsonl + {out}/images/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())