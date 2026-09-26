"""synth3 families, part 2: visual, spatial, and UI-state items.

Same contract as part 1. gen_calendar is the expanded seed family —
its "which week" rows are the exact skill the V0 probe found the frozen
backbone gets wrong at high confidence (0/3), so this family carries
extra week-counting volume.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from kapteeni.synth3_gfx import (SIZE, Style, canvas, checkbox, choice_row,
                                 noul_row, score_row, star)


def _rid(prefix, rng):
    return f"{prefix}-{rng.randint(0, 999999):06d}"


MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
          "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
WEEKDAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]


# 11 -------------------------------------------------------------- calendar
def gen_calendar(rng: random.Random) -> list[dict]:
    """Sequential 7-column month grid (days 1-31, rows = weeks). Two
    events; questions include the week-counting skill V0 exposed."""
    m = rng.randint(0, 11)
    days = rng.choice([28, 30, 31])
    e1 = rng.randint(1, days)
    e2 = e1
    while abs(e2 - e1) < 4:
        e2 = rng.randint(1, days)
    st = Style(rng, allow_dark=False)  # calendars stay light for grid legibility
    img, d = canvas(st, MONTHS[m])
    y0 = 116
    for col, wd in enumerate(WEEKDAYS):
        d.text((st.margin + 14 + col * 78, y0), wd, font=st.small,
               fill=st.accent)
    n_rows = (days - 1) // 7 + 1
    for day in range(1, days + 1):
        row, col = (day - 1) // 7, (day - 1) % 7
        x, y = st.margin + 14 + col * 78, y0 + 54 + row * 60
        if day == e1:
            d.rectangle((x - 10, y - 6, x + 52, y + 46), fill=st.accent)
            d.text((x, y), str(day), font=st.body, fill="#ffffff")
        elif day == e2:
            d.rectangle((x - 10, y - 6, x + 52, y + 46), outline=st.ink,
                       width=3)
            d.text((x, y), str(day), font=st.body, fill=st.ink)
        else:
            d.text((x, y), str(day), font=st.body, fill=st.ink)
    d.text((st.margin, y0 + 54 + n_rows * 60 + 24),
           f"solid = team lunch (day {e1}); box = review (day {e2})",
           font=st.small, fill=st.dim())
    rid = f"cal-{m:02d}-{days}-{e1:02d}-{e2:02d}"
    week_of = lambda day: (day - 1) // 7  # 0-indexed week row
    rows = [
        noul_row(f"synth3-{rid}-n", f"{rid}.png",
                 "A calendar page with two marked events.",
                 "Is the solid team lunch scheduled after the 15th of "
                 "the month?", "the event day is after the 15th",
                 "the event day is on or before the 15th", int(e1 > 15),
                 "calendar_card"),
        choice_row(f"synth3-{rid}-c", f"{rid}.png",
                   "A calendar page with two marked events.",
                   "In which week of the month is the solid team lunch?",
                   ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"],
                   week_of(e1), "calendar_card"),
        noul_row(f"synth3-{rid}-n2", f"{rid}.png",
                 "A calendar page with two marked events.",
                 "Does the solid team lunch happen before the boxed "
                 "review?", "the solid event is on an earlier day",
                 "the solid event is on the same or a later day",
                 int(e1 < e2), "calendar_card"),
        noul_row(f"synth3-{rid}-n3", f"{rid}.png",
                 "A calendar page with two marked events.",
                 "Is the boxed review in the same week as the solid "
                 "team lunch?", "both fall in the same week row",
                 "they are in different week rows",
                 int(week_of(e1) == week_of(e2)), "calendar_card"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 12 ------------------------------------------------------------- bar chart
def gen_bar_chart(rng: random.Random) -> list[dict]:
    cats = rng.sample(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], 4)
    while True:
        vals = rng.sample(range(10, 100, 5), 4)
        if len(set(vals)) == 4:
            break
    st = Style(rng, allow_dark=False)
    img, d = canvas(st, "QUARTERLY UNITS")
    x0, y0, y1 = st.margin + 40, 460, 150
    for gv in range(0, 101, 25):  # gridlines
        y = y0 - (y0 - y1) * gv / 100
        d.line((x0 - 14, y, SIZE - st.margin, y), fill=st.dim(), width=1)
        d.text((st.margin - 8, y), str(gv), font=st.small, fill=st.dim(),
               anchor="rm")
    for i, (cat, v) in enumerate(zip(cats, vals)):
        bx = x0 + 24 + i * ((SIZE - x0 - 60) // 4)
        bw = 66
        top = y0 - (y0 - y1) * v / 100
        d.rectangle((bx, top, bx + bw, y0), fill=st.accent)
        d.text((bx + bw // 2, y0 + 14), cat, font=st.small, fill=st.ink,
               anchor="ma")
        d.text((bx + bw // 2, top - 6), str(v), font=st.small, fill=st.ink,
               anchor="ms")
    rid = _rid("bar", rng)
    a, b = rng.sample(range(4), 2)
    over = sum(1 for v in vals if v > 50)
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A bar chart.",
                   "Which bar is the tallest?", cats, vals.index(max(vals)),
                   "bar_chart"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A bar chart.",
                 f"Does {cats[a]} have a higher value than {cats[b]}?",
                 f"{cats[a]}'s bar is taller", f"{cats[b]}'s bar is at "
                 "least as tall", int(vals[a] > vals[b]), "bar_chart"),
        score_row(f"synth3-{rid}-s", f"{rid}.png", "A bar chart.",
                  "How many bars have a value above 50?",
                  ["None", "One", "Two", "Three", "Four"], over,
                  "bar_chart"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 13 --------------------------------------------------------------- seating
def gen_seating(rng: random.Random) -> list[dict]:
    names = ["A. Chen", "B. Diaz", "C. Erin", "D. Fox", "E. Gray"]
    grid = [[rng.random() < 0.5 for _ in range(5)] for _ in range(4)]
    while True:
        row_counts = [sum(r) for r in grid]
        if row_counts and row_counts.count(max(row_counts)) == 1 \
                and sum(row_counts) < 20:
            break
        grid = [[rng.random() < 0.5 for _ in range(5)] for _ in range(4)]
    st = Style(rng)
    img, d = canvas(st, "DESK MAP")
    x0, y0 = st.margin + 60, 140
    for r in range(4):
        for c in range(5):
            x, y = x0 + c * 100, y0 + r * 104
            d.rectangle((x, y, x + 80, y + 78), outline=st.ink, width=3)
            if grid[r][c]:
                d.text((x + 40, y + 39), names[(r * 5 + c) % 5],
                       font=st.small, fill=st.accent, anchor="mm")
            else:
                d.text((x + 40, y + 39), "empty", font=st.small,
                       fill=st.dim(), anchor="mm")
    for r in range(4):
        d.text((st.margin - 8, y0 + r * 104 + 39), f"R{r + 1}",
               font=st.small, fill=st.ink, anchor="rm")
    rid = _rid("seat", rng)
    empty = sum(1 for r in grid for v in r if not v)
    pi, pj = rng.randrange(4), rng.randrange(5)
    top_row = row_counts.index(max(row_counts))
    rows = [
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "An office desk map.",
                 f"Is the desk in row R{pi + 1}, position {pj + 1}, "
                 "occupied?", "a name is shown in that desk",
                 "the desk is marked empty", int(grid[pi][pj]),
                 "seating_chart"),
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "An office desk map.",
                   "Which row has the most occupied desks?",
                   ["R1", "R2", "R3", "R4"], top_row, "seating_chart"),
        score_row(f"synth3-{rid}-s", f"{rid}.png", "An office desk map.",
                  "How many desks are empty?",
                  ["None", "1-5", "6-10", "11-15", "16-20"],
                  min(4, (empty + 4) // 5), "seating_chart"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 14 --------------------------------------------------------------- weather
def gen_weather(rng: random.Random) -> list[dict]:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    while True:
        temps = rng.sample(range(-2, 28), 5)
        if temps.count(max(temps)) == 1 and temps.count(min(temps)) == 1:
            break
    icons = ["sun", "cloud", "rain", "snow"]
    wx = [rng.choice(icons) for _ in days]
    st = Style(rng)
    img, d = canvas(st, "5-DAY FORECAST")
    for j, (day, t, w) in enumerate(zip(days, temps, wx)):
        x = st.margin + 40 + j * 112
        d.text((x + 40, 130), day, font=st.body, fill=st.ink, anchor="ma")
        d.text((x + 40, 220), w.upper(), font=st.small, fill=st.dim(),
               anchor="ma")
        d.text((x + 40, 330), f"{t}\u00b0", font=st.head, fill=st.accent,
               anchor="ma")
    rid = _rid("wx", rng)
    a, b = rng.sample(range(5), 2)
    cold = sum(1 for t in temps if t < 10)
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A weather forecast.",
                   "Which day is the warmest?", days, temps.index(max(temps)),
                   "weather_panel"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A weather forecast.",
                 f"Is {days[a]} warmer than {days[b]}?",
                 f"{days[a]}'s temperature is higher",
                 f"{days[b]} is at least as warm",
                 int(temps[a] > temps[b]), "weather_panel"),
        score_row(f"synth3-{rid}-s", f"{rid}.png", "A weather forecast.",
                  "How many days are colder than 10 degrees?",
                  ["None", "One", "Two", "Three", "Four", "Five"], cold,
                  "weather_panel"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 15 ---------------------------------------------------------- notifications
def gen_notification(rng: random.Random) -> list[dict]:
    apps = rng.sample(["Mail", "Chat", "News", "Photos", "Bank", "Music",
                       "Store", "Fitness"], 5)
    counts = rng.sample(range(1, 20), 5)
    st = Style(rng, allow_dark=True)
    img, d = canvas(st, "NOTIFICATIONS")
    y = 124
    for app, c in zip(apps, counts):
        d.text((st.margin, y), app, font=st.body, fill=st.ink)
        d.ellipse((SIZE - st.margin - 56, y - 4, SIZE - st.margin + 6,
                   y + 58), fill=st.accent)
        d.text((SIZE - st.margin - 25, y + 24), str(c), font=st.body,
               fill="#ffffff", anchor="mm")
        y += 84
    rid = _rid("notif", rng)
    top = counts.index(max(counts))
    a = rng.randrange(5)
    many = sum(1 for c in counts if c >= 10)
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png",
                   "A list of apps with notification badges.",
                   "Which app has the most notifications?", apps, top,
                   "notification_card"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png",
                 "A list of apps with notification badges.",
                 f"Does {apps[a]} have more than 9 unread notifications?",
                 "its badge shows 10 or more", "its badge shows 9 or fewer",
                 int(counts[a] >= 10), "notification_card"),
        score_row(f"synth3-{rid}-s", f"{rid}.png",
                  "A list of apps with notification badges.",
                  "How many apps have 10 or more unread notifications?",
                  ["None", "One", "Two", "Three", "Four", "Five"], many,
                  "notification_card"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 16 ----------------------------------------------------------------- kanban
def gen_kanban(rng: random.Random) -> list[dict]:
    cols = ["TO DO", "DOING", "DONE"]
    tasks = ["auth flow", "dark mode", "fix crash", "docs", "billing",
             "search", "export", "mobile"]
    rng.shuffle(tasks)
    board = {c: [] for c in cols}
    for t in tasks:
        board[rng.choice(cols)].append(t)
    while any(len(v) == 0 for v in board.values()):  # every column nonempty
        board = {c: [] for c in cols}
        for t in tasks:
            board[rng.choice(cols)].append(t)
    st = Style(rng)
    img, d = canvas(st, "SPRINT BOARD")
    colw = (SIZE - 2 * st.margin) // 3
    for ci, col in enumerate(cols):
        x = st.margin + ci * colw
        d.rectangle((x, 120, x + colw - 16, 560), outline=st.ink,
                    width=3)
        d.text((x + colw // 2 - 8, 134), col, font=st.small, fill=st.accent,
               anchor="ma")
        for ti, t in enumerate(board[col]):
            d.rectangle((x + 12, 180 + ti * 78, x + colw - 28,
                        240 + ti * 78), fill=st.dim(), width=0)
            d.text((x + colw // 2 - 8, 194 + ti * 78 + 12), t,
                   font=st.small, fill=st.ink, anchor="mm")
    rid = _rid("kanban", rng)
    ask = rng.choice(tasks)
    n_todo = len(board["TO DO"])
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A task board.",
                   f"Which column contains '{ask}'?", cols,
                   next(i for i, c in enumerate(cols) if ask in board[c]),
                   "kanban_board"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A task board.",
                 f"Is the task 'billing' in the DOING column?",
                 "it is in DOING", "it is in another column",
                 int("billing" in board["DOING"]), "kanban_board"),
        score_row(f"synth3-{rid}-s", f"{rid}.png", "A task board.",
                  "How many cards are in the TO DO column?",
                  ["None", "One", "Two", "Three", "Four or more"],
                  min(4, n_todo), "kanban_board"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 17 ------------------------------------------------------------ grid cells
def gen_grid_pattern(rng: random.Random) -> list[dict]:
    nums = [rng.randint(10, 99) for _ in range(16)]
    while len(set(nums)) != 16:
        nums = [rng.randint(10, 99) for _ in range(16)]
    colored = set(rng.sample(range(16), rng.randint(2, 6)))
    st = Style(rng, allow_dark=False)
    img, d = canvas(st, "CODE GRID")
    x0, y0 = st.margin + 80, 150
    for i, n in enumerate(nums):
        r, c = divmod(i, 4)
        x, y = x0 + c * 112, y0 + r * 96
        if i in colored:
            d.rectangle((x, y, x + 96, y + 80), fill=st.accent)
            d.text((x + 48, y + 40), str(n), font=st.body, fill="#ffffff",
                   anchor="mm")
        else:
            d.rectangle((x, y, x + 96, y + 80), outline=st.ink, width=2)
            d.text((x + 48, y + 40), str(n), font=st.body, fill=st.ink,
                   anchor="mm")
    rid = _rid("grid", rng)
    corner = nums[3]  # top-right
    r2c3 = nums[6]  # row 2, col 3 (0-indexed)
    opts = [corner] + rng.sample([x for x in nums if x != corner], 3)
    rng.shuffle(opts)
    rows = [choice_row(f"synth3-{rid}-c", f"{rid}.png",
                       "A 4x4 grid of numbered cells.",
                       "What number is in the top-right corner cell?",
                       [str(x) for x in opts], opts.index(corner),
                       "grid_pattern"),
            noul_row(f"synth3-{rid}-n", f"{rid}.png", "A 4x4 grid of "
                     "numbered cells.",
                     "Is the number in row 2, column 3 (counting from the "
                     "top left, one-indexed) greater than 55?",
                     "that number exceeds 55", "it is 55 or less",
                     int(r2c3 > 55), "grid_pattern"),
            score_row(f"synth3-{rid}-s", f"{rid}.png", "A 4x4 grid of "
                      "numbered cells.", "How many cells are highlighted "
                      "in the accent color?",
                      ["None", "Two", "Three or four", "Five or six"],
                      0 if not colored else (1 if len(colored) == 2 else
                                             (2 if len(colored) <= 4 else 3)),
                      "grid_pattern")]
    return [{"img": img, "rid": rid, "rows": rows}]


# 18 ------------------------------------------------------------ analog clock
def gen_clock(rng: random.Random) -> list[dict]:
    hour = rng.randint(1, 12)
    minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    st = Style(rng, allow_dark=False)
    img, d = canvas(st, None)
    cx, cy, R = SIZE // 2, SIZE // 2 + 20, 230
    d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=st.ink, width=8)
    for i in range(12):
        ang = i * 30 - 90
        import math
        x1 = cx + (R - 30) * math.cos(math.radians(ang))
        y1 = cy + (R - 30) * math.sin(math.radians(ang))
        x2 = cx + (R - 12) * math.cos(math.radians(ang))
        y2 = cy + (R - 12) * math.sin(math.radians(ang))
        d.line((x1, y1, x2, y2), fill=st.ink, width=6)
    ang_h = (hour % 12 + minute / 60) * 30 - 90
    ang_m = minute * 6 - 90
    import math
    d.line((cx, cy, cx + (R - 90) * math.cos(math.radians(ang_h)),
            cy + (R - 90) * math.sin(math.radians(ang_h))),
           fill=st.ink, width=14)
    d.line((cx, cy, cx + (R - 40) * math.cos(math.radians(ang_m)),
            cy + (R - 40) * math.sin(math.radians(ang_m))),
           fill=st.accent, width=9)
    d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=st.ink)
    total = hour * 60 + minute
    rid = f"clock-{hour:02d}{minute:02d}"
    thr_h, thr_m = rng.choice([(hour, (minute + 30) % 60),
                               (hour % 12 + 1, minute), (hour, minute)])
    thr_total = thr_h * 60 + thr_m if thr_h else 720
    if abs(thr_total - total) < 3:  # avoid degenerate equality
        thr_total += 30
    rows = [
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "An analog clock face.",
                 "Is the time shown before noon?",
                 "the hour hand is in the AM half (before 12:00, "
                 "12:00 counts as noon)", "the time is noon or after",
                 int(hour < 12), "clock_face"),
        noul_row(f"synth3-{rid}-n2", f"{rid}.png", "An analog clock face.",
                 "Is the minute hand past the 6 (i.e., more than 30 "
                 "minutes)?", "the minute hand is past the 6",
                 "it is at or before the 6", int(minute > 30),
                 "clock_face"),
        noul_row(f"synth3-{rid}-n3", f"{rid}.png", "An analog clock face.",
                 "Is the time before "
                 f"{((thr_total // 60 - 1) % 12) + 1}:{thr_total % 60:02d}?",
                 "the shown time is earlier", "the shown time is at or "
                 "after it", int(total < thr_total), "clock_face"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 19 ------------------------------------------------------------ email card
def gen_email_card(rng: random.Random) -> list[dict]:
    senders = ["billing@acme-corp.com", "no-reply@shop-now.example",
               "team@projectflow.io", "alerts@bankmail.example"]
    sender = rng.choice(senders)
    has_attachment = rng.random() < 0.5
    folder = rng.choice(["Inbox", "Work", "Newsletter"])
    subj = rng.choice(["Your invoice for August", "Weekly digest",
                       "Re: contract", "Payment received", "New login"])
    st = Style(rng)
    img, d = canvas(st, "MAIL")
    y = 130
    d.text((st.margin, y), "From:", font=st.small, fill=st.dim())
    d.text((st.margin + 110, y), sender, font=st.body, fill=st.ink)
    y += 62
    d.text((st.margin, y), "Subject:", font=st.small, fill=st.dim())
    d.text((st.margin + 110, y), subj, font=st.body, fill=st.ink)
    y += 62
    d.text((st.margin, y), "Folder:", font=st.small, fill=st.dim())
    d.text((st.margin + 110, y), folder, font=st.body, fill=st.ink)
    y += 80
    if has_attachment:
        d.rectangle((st.margin, y, st.margin + 220, y + 64),
                   outline=st.ink, width=3)
        d.text((st.margin + 16, y + 18), "📎 attachment.pdf", font=st.body,
               fill=st.ink)
    rid = _rid("mail", rng)
    domain = sender.split("@")[1]
    ask_folder = rng.choice(["Inbox", "Work", "Newsletter"])
    rows = [
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "An email header.",
                 "Does the email have an attachment?", "an attachment "
                 "badge is shown", "no attachment is shown",
                 int(has_attachment), "email_card"),
        noul_row(f"synth3-{rid}-n2", f"{rid}.png", "An email header.",
                 f"Is the email filed in the {ask_folder} folder?",
                 f"the folder label matches", "the folder label differs",
                 int(ask_folder == folder), "email_card"),
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "An email header.",
                   f"Which domain is the sender's address from?",
                   sorted({s.split('@')[1] for s in senders}),
                   sorted({s.split('@')[1] for s in senders}).index(domain),
                   "email_card"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 20 ------------------------------------------------------------- shapes
def gen_shape_scene(rng: random.Random) -> list[dict]:
    import math
    kinds = ["triangle", "circle", "square"]
    colors = ["red", "blue", "green", "amber"]
    counts = {}
    while True:
        counts = {k: rng.randint(0, 5) for k in kinds}
        vals = list(counts.values())
        if sum(vals) >= 4 and vals.count(max(vals)) == 1:
            break
    st = Style(rng, allow_dark=False)
    img, d = canvas(st, "SCENE")
    pen = {"red": "#c0392b", "blue": "#2471a3", "green": "#1e8449",
           "amber": "#b9770e"}
    placed = []
    for kind, n in counts.items():
        for _ in range(n):
            col = rng.choice(colors)
            for _try in range(60):
                x, y = rng.randint(80, 560), rng.randint(120, 560)
                if all((x - px) ** 2 + (y - py) ** 2 > 78 ** 2
                       for px, py in placed):
                    break
            placed.append((x, y))
            if kind == "circle":
                d.ellipse((x - 26, y - 26, x + 26, y + 26),
                          outline=pen[col], width=6)
            elif kind == "square":
                d.rectangle((x - 24, y - 24, x + 24, y + 24),
                           outline=pen[col], width=6)
            else:
                p = [(x, y - 30), (x + 28, y + 22), (x - 28, y + 22)]
                d.polygon(p, outline=pen[col], width=6)
    rid = _rid("shape", rng)
    top_kind = max(kinds, key=lambda k: counts[k])
    a, b = rng.sample(kinds, 2)
    rows = [
        choice_row(f"synth3-{rid}-c", f"{rid}.png", "A scene of shapes.",
                   "Which kind of shape appears most often?",
                   kinds, kinds.index(top_kind), "shape_scene"),
        noul_row(f"synth3-{rid}-n", f"{rid}.png", "A scene of shapes.",
                 f"Are there more {a}s than {b}s?",
                 f"more {a}s are shown", f"there are at least as many "
                 f"{b}s", int(counts[a] > counts[b]), "shape_scene"),
        score_row(f"synth3-{rid}-s", f"{rid}.png", "A scene of shapes.",
                  f"How many {a}s are shown?",
                  ["None", "One or two", "Three or four", "Five"],
                  min(3, counts[a]), "shape_scene"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]