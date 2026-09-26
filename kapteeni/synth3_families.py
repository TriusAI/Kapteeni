"""synth3 families, part 1: document-style items (text-in-image reading).

Every generator returns [{"img": PIL.Image, "rid": str, "rows": [...]}].
Facts are sampled FIRST; the same facts drive the render and every gold.
Trap discipline: near-miss thresholds, distinct argmaxes, counting rows
with exact counts, distractor lines that must be ignored.
"""

from __future__ import annotations

import random

from kapteeni.synth3_gfx import (SIZE, Style, canvas, checkbox, choice_row,
                                 leaders, noul_row, score_row)


def _rid(prefix, rng):
    return f"{prefix}-{rng.randint(0, 999999):06d}"


def _fmt_usd(cents: int) -> str:
    return f"${cents // 100}.{cents % 100:02d}"


def _fmt_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# 1 ---------------------------------------------------------------- menu
def gen_menu(rng: random.Random) -> list[dict]:
    names = rng.sample(["Espresso", "Latte", "Cappuccino", "Flat White",
                        "Mocha", "Americano", "Macchiato", "Cortado",
                        "Chai Latte", "Hot Chocolate", "Green Tea",
                        "Iced Coffee"], rng.randint(5, 7))
    cents = rng.sample(range(220, 900, 10), len(names))
    items = list(zip(names, cents))
    rng.shuffle(items)
    st = Style(rng)
    img, d = canvas(st, rng.choice(["CAFE MENU", "DRINKS", "MENU"]))
    y = 118
    for name, c in items:
        d.text((st.margin, y), name, font=st.body, fill=st.ink)
        price = _fmt_usd(c)
        pw = d.textlength(price, font=st.body)
        x0 = st.margin + d.textlength(name, font=st.body) + 14
        leaders(d, x0, SIZE - st.margin - pw - 8, y, st)
        d.text((SIZE - st.margin - pw, y), price, font=st.body, fill=st.ink)
        y += 58
    if rng.random() < 0.4:  # distractor footer, never affects gold
        d.text((st.margin, y + 26), "All prices include tax.",
               font=st.small, fill=st.dim())
    rid = _rid("menu", rng)
    mx = max(items, key=lambda t: t[1]); mn = min(items, key=lambda t: t[1])
    rows = [choice_row(f"synth3-{rid}-c", f"{rid}.png", "A cafe menu.",
                       "Which drink is the most expensive?",
                       [n for n, _ in items], items.index(mx), "menu_board"),
            choice_row(f"synth3-{rid}-c2", f"{rid}.png", "A cafe menu.",
                       "Which drink is the cheapest?",
                       [n for n, _ in items], items.index(mn), "menu_board")]
    for name, c in rng.sample(items, 2):
        t = c + rng.choice([-10, 10, -30, 30, 5])
        rows.append(noul_row(f"synth3-{rid}-n", f"{rid}.png", "A cafe menu.",
                             f"Does the {name} cost more than {_fmt_usd(t)}?",
                             "the price is above the threshold",
                             "the price is not above it",
                             int(c > t), "menu_board"))
    return [{"img": img, "rid": rid, "rows": rows}]


# 2 ------------------------------------------------------------ price tags
def gen_price_tag(rng: random.Random) -> list[dict]:
    names = rng.sample(["Milk", "Bread", "Cheese", "Apples", "Juice",
                        "Eggs", "Rice", "Pasta", "Oil", "Salt"], 4)
    cents = rng.sample(range(150, 1200, 25), 4)
    pairs = list(zip(names, cents))
    st = Style(rng)
    img, d = canvas(st, "SHELF PRICES")
    x = st.margin
    for i, (name, c) in enumerate(pairs):
        bx = x + i * ((SIZE - 2 * st.margin) // 4)
        d.rectangle((bx, 150, bx + 120, 420), outline=st.ink, width=4)
        d.text((bx + 60, 240), name, font=st.body, fill=st.ink, anchor="mm")
        d.text((bx + 60, 330), _fmt_usd(c), font=st.head, fill=st.accent,
               anchor="mm")
    rid = _rid("tag", rng)
    mn = min(pairs, key=lambda t: t[1])
    (a, ca), (b, cb) = rng.sample(pairs, 2)
    return [{"img": img, "rid": rid, "rows": [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A shelf with priced "
                   "items.", "Which item is the cheapest?",
                   [n for n, _ in pairs], pairs.index(mn), "price_tag"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A shelf with priced "
                 "items.", f"Does the {a} cost less than the {b}?",
                 f"the {a} is cheaper", f"the {a} is not cheaper",
                 int(ca < cb), "price_tag")]}]


# 3 --------------------------------------------------------------- invoice
def gen_invoice(rng: random.Random) -> list[dict]:
    goods = rng.sample(["Widget", "Gadget", "Cable", "Adapter", "Case",
                        "Stand", "Bulb", "Filter"], rng.randint(3, 4))
    # distinct line amounts so the argmax gold is unique
    while True:
        qty = [rng.randint(1, 4) for _ in goods]
        unit = rng.sample(range(300, 2500, 50), len(goods))
        amts = [q * u for q, u in zip(qty, unit)]
        if len(set(amts)) == len(amts):
            break
    lines = [(g, q, u, q * u) for (g, q, u) in zip(goods, qty, unit)]
    subtotal = sum(l[3] for l in lines)
    has_delivery = rng.random() < 0.6
    delivery = rng.choice([300, 500, 700]) if has_delivery else 0
    total = subtotal + delivery
    st = Style(rng)
    img, d = canvas(st, "INVOICE")
    y = 118
    for g, q, u, amt in lines:
        d.text((st.margin, y), f"{q} x {g} @ {_fmt_usd(u)}", font=st.body,
               fill=st.ink)
        s = _fmt_usd(amt)
        d.text((SIZE - st.margin - d.textlength(s, font=st.body), y), s,
               font=st.body, fill=st.ink)
        y += 52
    if has_delivery:
        d.text((st.margin, y), "Delivery", font=st.body, fill=st.ink)
        s = _fmt_usd(delivery)
        d.text((SIZE - st.margin - d.textlength(s, font=st.body), y), s,
               font=st.body, fill=st.ink)
        y += 52
    y += 16
    d.text((st.margin, y), "TOTAL", font=st.head, fill=st.accent)
    s = _fmt_usd(total)
    d.text((SIZE - st.margin - d.textlength(s, font=st.head), y), s,
           font=st.head, fill=st.accent)
    rid = _rid("inv", rng)
    big = max(lines, key=lambda l: l[3])
    t = total + rng.choice([-100, 100, -200, 200])
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "An invoice.",
                   "Which line item has the largest amount?",
                   [f"{g} x{q}" for g, q, _, _ in lines], lines.index(big),
                   "invoice_receipt"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "An invoice.",
                 f"Is the total more than {_fmt_usd(t)}?",
                 "the total exceeds that amount",
                 "the total is at or below it", int(total > t),
                 "invoice_receipt"),
        noul_row(f"synth3-{rid}-n2", f"{rid}.png", "An invoice.",
                 "Does the invoice include a delivery charge?",
                 "a delivery line is present", "no delivery line is shown",
                 int(has_delivery), "invoice_receipt"),
        score_row(f"synth3-{rid}-s", f"{rid}.png", "An invoice.",
                  "How many distinct items are on the invoice (excluding "
                  "delivery and totals)?",
                  ["One", "Two", "Three", "Four"], len(lines) - 1,
                  "invoice_receipt"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 4 ------------------------------------------------------------------ form
def gen_form(rng: random.Random) -> list[dict]:
    extras = ["Express shipping", "Shipment insurance", "Gift wrapping",
              "Priority handling"]
    k = rng.randint(0, 3)
    chosen = set(rng.sample(range(4), k))
    express = 0 in chosen
    st = Style(rng)
    img, d = canvas(st, "DELIVERY FORM")
    y = 128
    for i, label in enumerate(extras):
        checkbox(d, st.margin, y, st, i in chosen)
        d.text((st.margin + 70, y + 4), label, font=st.body, fill=st.ink)
        y += 66
    d.text((st.margin, y + 24), "Recipient: A. Customer", font=st.small,
           fill=st.dim())
    rid = _rid("form", rng)
    rows = [
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A delivery form.",
                 "Is express shipping selected?",
                 "the box is checked", "the box is not checked",
                 int(express), "form_sheet"),
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A delivery form.",
                  "Which shipping method is selected?",
                  ["Standard shipping", "Express shipping"], int(express),
                  "form_sheet"),
        score_row(f"synth3-{rid}-s", f"{rid}.png", "A delivery form.",
                  "How many optional extras are selected?",
                  ["None", "One", "Two", "Three"], k, "form_sheet"),
    ]
    if k:  # a named extra
        i = rng.choice(sorted(chosen))
        rows.append(noul_row(f"synth3-{rid}-n2", f"{rid}.png",
                             "A delivery form.",
                             f"Is '{extras[i].lower()}' selected on the form?",
                             "the box is checked", "the box is not checked",
                             1, "form_sheet"))
    else:
        rows.append(noul_row(f"synth3-{rid}-n2", f"{rid}.png",
                             "A delivery form.",
                             f"Is '{rng.choice(extras).lower()}' selected "
                             "on the form?", "the box is checked",
                             "the box is not checked", 0, "form_sheet"))
    return [{"img": img, "rid": rid, "rows": rows}]


# 5 -------------------------------------------------------------- schedule
def gen_schedule(rng: random.Random) -> list[dict]:
    trains = [f"T{rng.randint(10, 99)}" for _ in range(4)]
    while len(set(trains)) < 4:
        trains = [f"T{rng.randint(10, 99)}" for _ in range(4)]
    times = sorted(rng.sample(range(6 * 60, 12 * 60, 15), 4))
    stops = ["Central", "Riverside", "Northgate", "Old Town", "Airport",
             "Harbor"]
    stop_sets = []
    for _ in trains:
        n = rng.randint(2, 3)
        stop_sets.append(set(rng.sample(stops, n)))
        if not stop_sets[-1]:
            stop_sets[-1] = {stops[0]}
    st = Style(rng)
    img, d = canvas(st, "DEPARTURES")
    y = 116
    d.text((st.margin, y), "Train", font=st.small, fill=st.dim())
    d.text((st.margin + 130, y), "Departs", font=st.small, fill=st.dim())
    d.text((st.margin + 260, y), "Stops", font=st.small, fill=st.dim())
    y += 44
    for tr, tm, ss in zip(trains, times, stop_sets):
        d.text((st.margin, y), tr, font=st.body, fill=st.ink)
        d.text((st.margin + 130, y), _fmt_hhmm(tm), font=st.body,
               fill=st.ink)
        d.text((st.margin + 260, y), " - ".join(sorted(ss)), font=st.small,
               fill=st.ink)
        y += 74
    rid = _rid("sched", rng)
    i, j = rng.sample(range(4), 2)
    ask_stop = rng.choice(sorted(set.intersection(set(stop_sets[i]),
                                                 stop_sets[j])
                                 or {sorted(stop_sets[i])[0]}))
    thr = times[j] + rng.choice([-15, 15, -30, 30])
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A departure board.",
                   "Which train departs first?", trains, 0, "schedule_table"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A departure board.",
                 f"Does train {trains[i]} stop at {ask_stop}?",
                 f"the stop is listed for {trains[i]}",
                 f"that stop is not listed", int(ask_stop in stop_sets[i]),
                 "schedule_table"),
        noul_row(f"synth3-{rid}-n2", f"{rid}.png", "A departure board.",
                 f"Does train {trains[j]} depart before {_fmt_hhmm(thr)}?",
                 "it departs before that time",
                 "it departs at or after it", int(times[j] < thr),
                 "schedule_table"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 6 ------------------------------------------------------------ attendance
def gen_attendance(rng: random.Random) -> list[dict]:
    people = rng.sample(["Ana", "Ben", "Cleo", "Dara", "Eli", "Faye"], 4)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    # unique attendance argmax BEFORE any rendering
    while True:
        mark = [[rng.random() < 0.6 for _ in days] for _ in people]
        counts = [sum(r) for r in mark]
        if counts.count(max(counts)) == 1 and max(counts) > 0:
            break
    st = Style(rng)
    img, d = canvas(st, "ATTENDANCE")
    x0, y0 = st.margin, 130
    for j, day in enumerate(days):
        d.text((x0 + 110 + j * 100, y0), day, font=st.small, fill=st.dim())
    for i, p in enumerate(people):
        y = y0 + 56 + i * 84
        d.text((x0, y), p, font=st.body, fill=st.ink)
        for j in range(5):
            cx = x0 + 116 + j * 100
            d.rectangle((cx, y - 8, cx + 46, y + 40), outline=st.ink,
                        width=2)
            if mark[i][j]:
                d.line((cx + 6, y + 16, cx + 18, y + 34), fill=st.accent,
                       width=5)
                d.line((cx + 18, y + 34, cx + 40, y + 0), fill=st.accent,
                       width=5)
    rid = _rid("att", rng)
    top = counts.index(max(counts))
    pi, di = rng.randrange(4), rng.randrange(5)
    rows = [
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "An attendance sheet.",
                 f"Did {people[pi]} attend on {days[di]}?",
                 "the day is marked", "the day is not marked",
                 int(mark[pi][di]), "attendance_sheet"),
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "An attendance sheet.",
                   "Who attended the most days?", people, top,
                   "attendance_sheet"),
        score_row(f"synth3-{rid}-s", f"{rid}.png", "An attendance sheet.",
                  f"How many days did {people[pi]} attend?",
                  ["None", "One", "Two", "Three", "Four", "Five"],
                  counts[pi], "attendance_sheet"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 7 ------------------------------------------------------------ sales table
def gen_table_chart(rng: random.Random) -> list[dict]:
    prods = rng.sample(["Alpha", "Beta", "Gamma", "Delta"], 3)
    qs = ["Q1", "Q2", "Q3"]
    # distinct Q2 values so the argmax gold is unique
    while True:
        vals = {p: rng.sample(range(20, 200, 5), 3) for p in prods}
        q2 = [vals[p][1] for p in prods]
        if len(set(q2)) == len(q2):
            break
    st = Style(rng)
    img, d = canvas(st, "SALES BY QUARTER (units)")
    y = 130
    d.text((st.margin + 150, y), "  ".join(qs), font=st.body, fill=st.dim())
    y += 54
    for p in prods:
        d.text((st.margin, y), p, font=st.body, fill=st.ink)
        for j, v in enumerate(vals[p]):
            d.text((st.margin + 160 + j * 130, y), str(v), font=st.body,
                   fill=st.ink)
        y += 74
    rid = _rid("tab", rng)
    a, b = rng.sample(prods, 2)
    q2_top = max(prods, key=lambda p: vals[p][1])
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A sales table.",
                   "Which product had the highest Q2 sales?", prods,
                   prods.index(q2_top), "table_chart"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A sales table.",
                 f"Did {a} sell more units than {b} in Q3?",
                 f"{a} outsold {b} that quarter", f"{b} sold at least as "
                 "many", int(vals[a][2] > vals[b][2]), "table_chart"),
        noul_row(f"synth3-{rid}-n2", f"{rid}.png", "A sales table.",
                 f"Did {a}'s sales increase from Q1 to Q2?",
                 "the number went up", "the number went down or stayed",
                 int(vals[a][1] > vals[a][0]), "table_chart"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 8 ----------------------------------------------------------------- hours
def gen_sign(rng: random.Random) -> list[dict]:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # distinct open-hour durations so the argmax gold is unique
    durs = rng.sample(range(360, 780, 30), 7)
    opens = {}
    k = 0
    for day in days:
        if day == "Sun" and rng.random() < 0.5:
            opens[day] = None  # closed
        else:
            o = rng.randint(7, 10) * 60
            opens[day] = (o, o + durs[k])
            k += 1
    open_days = [x for x in days if opens[x] is not None]
    st = Style(rng)
    img, d = canvas(st, "OPENING HOURS")
    y = 124
    for day in days:
        d.text((st.margin, y), day, font=st.body, fill=st.ink)
        v = opens[day]
        txt = "CLOSED" if v is None else f"{_fmt_hhmm(v[0])} - {_fmt_hhmm(v[1])}"
        d.text((st.margin + 150, y), txt, font=st.body, fill=st.ink)
        y += 64
    rid = _rid("sign", rng)
    open_days = [x for x in days if opens[x] is not None]
    longest = max(open_days, key=lambda x: opens[x][1] - opens[x][0])
    ask = rng.choice(days)
    v = opens[ask]
    rows = [
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "An opening-hours sign.",
                 f"Is the shop open on {ask}?", "hours are listed for "
                 "that day", "the sign says CLOSED", int(v is not None),
                 "sign_poster"),
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "An opening-hours sign.",
                   "Which day has the longest opening hours?",
                   open_days, open_days.index(longest), "sign_poster"),
    ]
    if v is not None:
        t = rng.choice([v[0] - 30, v[0], v[0] + 30])
        rows.append(noul_row(
            f"synth3-{rid}-n2", f"{rid}.png", "An opening-hours sign.",
            f"Does the shop open at or before {_fmt_hhmm(t)} on {ask}?",
            "opening time is at or before that", "it opens later",
            int(v[0] <= t), "sign_poster"))
    return [{"img": img, "rid": rid, "rows": rows}]


# 9 ---------------------------------------------------------------- ticket
def gen_ticket(rng: random.Random) -> list[dict]:
    gate = rng.choice(["A", "B", "C", "D"])
    row_n = rng.randint(3, 25)
    seat = rng.randint(1, 40)
    dom = rng.randint(2, 27)
    hour = rng.randint(13, 22)
    minute = rng.choice([0, 15, 30, 45])
    st = Style(rng)
    img, d = canvas(st, "ADMIT ONE")
    fields = [("EVENT", "Spring Concert"), ("DATE", f"June {dom}, 2026"),
              ("TIME", _fmt_hhmm(hour * 60 + minute)),
              ("GATE", gate), ("ROW", str(row_n)), ("SEAT", str(seat))]
    y = 128
    for label, val in fields:
        d.text((st.margin, y), label, font=st.small, fill=st.dim())
        d.text((st.margin + 160, y), val, font=st.body, fill=st.ink)
        y += 74
    rid = _rid("tix", rng)
    thr = rng.choice([row_n - 1, row_n, row_n + 1])
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "An event ticket.",
                   "Which gate does the ticket say?", ["A", "B", "C", "D"],
                   "ABCD".index(gate), "ticket_stub"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "An event ticket.",
                 f"Is the seat in row {thr} or higher?",
                 f"the row number is >= {thr}", "the row number is lower",
                 int(row_n >= thr), "ticket_stub"),
        noul_row(f"synth3-{rid}-n2", f"{rid}.png", "An event ticket.",
                 "Is the event in the first half of the month?",
                 "the day is between the 1st and 15th",
                 "the day is the 16th or later", int(dom <= 15),
                 "ticket_stub"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 10 ----------------------------------------------------------- key/value
def gen_key_value(rng: random.Random) -> list[dict]:
    weight = rng.randint(5, 95)  # tenths of kg
    colors = rng.sample(["Red", "Blue", "Green", "Black", "White"], 3)
    ordered = list(colors)
    rng.shuffle(ordered)
    dims = (rng.randint(20, 40), rng.randint(15, 35), rng.randint(15, 30))
    st = Style(rng)
    img, d = canvas(st, "PRODUCT SPECS")
    fields = [("Weight", f"{weight / 10:.1f} kg"),
              ("Dimensions", f"{dims[0]} x {dims[1]} x {dims[2]} cm"),
              ("Colors", ", ".join(ordered))]
    y = 130
    for label, val in fields:
        d.text((st.margin, y), label, font=st.small, fill=st.dim())
        d.text((st.margin + 200, y), val, font=st.body, fill=st.ink)
        y += 84
    rid = _rid("spec", rng)
    t = weight + rng.choice([-10, 10, -5, 5])
    rows = [
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A product spec sheet.",
                 f"Is the weight more than {t / 10:.1f} kg?",
                 "the listed weight is above that",
                 "the listed weight is at or below it",
                 int(weight > t), "key_value_panel"),
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A product spec sheet.",
                   "Which color is listed first?",
                   ["Red", "Blue", "Green", "Black", "White"],
                   ["Red", "Blue", "Green", "Black", "White"].index(
                       ordered[0]), "key_value_panel"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]