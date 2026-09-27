"""synth3 families, Chinese edition: the same 20 image families with
Chinese-rendered documents and Chinese questions.

Same contract as synth3_families(.v): facts are sampled before
rendering and drive every gold. Fact sampling and gold computation are
copied verbatim from the English generators; only content pools,
rendered text, questions, and criteria are localized. Fonts come from
Noto Sans/Serif CJK via Style(zh=True). Row ids use `synth3zh-` and
image rids use `zh-` prefixes, so nothing collides with the English
dataset directories.
"""

from __future__ import annotations

import math
import random

from kapteeni.synth3_gfx import (SIZE, Style, canvas, checkbox, choice_row,
                                 noul_row, score_row)


def _rid(prefix, rng):
    return f"zh-{prefix}-{rng.randint(0, 999999):06d}"


def _fmt_cny(cents: int) -> str:
    return f"¥{cents // 100}.{cents % 100:02d}"


def _fmt_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


MONTHS_ZH = ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月",
             "九月", "十月", "十一月", "十二月"]
WEEKDAYS_ZH = ["一", "二", "三", "四", "五", "六", "日"]
DAYS_ZH = ["周一", "周二", "周三", "周四", "周五"]


# 1 ---------------------------------------------------------------- menu
def gen_menu_zh(rng: random.Random) -> list[dict]:
    names = rng.sample(["浓缩咖啡", "拿铁", "卡布奇诺", "澳白", "摩卡",
                        "美式", "玛奇雅朵", "可塔多", "印度拉茶", "热巧克力",
                        "绿茶", "冰咖啡"], rng.randint(5, 7))
    cents = rng.sample(range(220, 900, 10), len(names))
    items = list(zip(names, cents))
    rng.shuffle(items)
    st = Style(rng, zh=True)
    img, d = canvas(st, rng.choice(["咖啡菜单", "饮品单", "菜单"]))
    y = 118
    for name, c in items:
        d.text((st.margin, y), name, font=st.body, fill=st.ink)
        price = _fmt_cny(c)
        pw = d.textlength(price, font=st.body)
        x0 = st.margin + d.textlength(name, font=st.body) + 14
        d.line((x0, y + 28, SIZE - st.margin - pw - 8, y + 28),
               fill=st.dim(), width=2)
        d.text((SIZE - st.margin - pw, y), price, font=st.body, fill=st.ink)
        y += 58
    if rng.random() < 0.4:
        d.text((st.margin, y + 26), "所有价格均已含税。", font=st.small,
               fill=st.dim())
    rid = _rid("menu", rng)
    mx = max(items, key=lambda t: t[1])
    mn = min(items, key=lambda t: t[1])
    rows = [choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张咖啡菜单。",
                       "哪种饮品最贵？",
                       [n for n, _ in items], items.index(mx), "menu_board"),
            choice_row(f"synth3zh-{rid}-c2", f"{rid}.png", "一张咖啡菜单。",
                       "哪种饮品最便宜？",
                       [n for n, _ in items], items.index(mn), "menu_board")]
    for name, c in rng.sample(items, 2):
        rows.append(noul_row(
            f"synth3zh-{rid}-n{c}", f"{rid}.png", "一张咖啡菜单。",
            f"{name}的价格超过 ¥5.00 吗？", "该饮品价格高于 ¥5.00",
            "该饮品价格不超过 ¥5.00", int(c > 500), "menu_board"))
    return [{"img": img, "rid": rid, "rows": rows}]


# 2 ------------------------------------------------------------ price tags
def gen_price_tag_zh(rng: random.Random) -> list[dict]:
    names = rng.sample(["牛奶", "面包", "奶酪", "苹果", "果汁", "鸡蛋",
                        "大米", "意面", "食用油", "食盐"], 4)
    cents = rng.sample(range(150, 1200, 25), 4)
    pairs = list(zip(names, cents))
    st = Style(rng, zh=True)
    img, d = canvas(st, "货架价签")
    x = st.margin
    for i, (name, c) in enumerate(pairs):
        bx = x + i * ((SIZE - 2 * st.margin) // 4)
        d.rectangle((bx, 150, bx + 128, 420), outline=st.ink, width=4)
        d.text((bx + 64, 240), name, font=st.body, fill=st.ink, anchor="mm")
        d.text((bx + 64, 330), _fmt_cny(c), font=st.head, fill=st.accent,
               anchor="mm")
    rid = _rid("tag", rng)
    mn = min(pairs, key=lambda t: t[1])
    (a, ca), (b, cb) = rng.sample(pairs, 2)
    return [{"img": img, "rid": rid, "rows": [
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "货架上的商品价签。",
                   "哪件商品最便宜？",
                   [n for n, _ in pairs], pairs.index(mn), "price_tag"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "货架上的商品价签。",
                 f"{a}比{b}便宜吗？", f"{a}更便宜", f"{a}并不更便宜",
                 int(ca < cb), "price_tag")]}]


# 3 --------------------------------------------------------------- invoice
def gen_invoice_zh(rng: random.Random) -> list[dict]:
    goods = rng.sample(["零部件", "小工具", "线缆", "转接头", "保护壳",
                        "支架", "灯泡", "滤芯"], rng.randint(3, 4))
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
    st = Style(rng, zh=True)
    img, d = canvas(st, "发票")
    y = 118
    for g, q, u, amt in lines:
        d.text((st.margin, y), f"{q} x {g} @ {_fmt_cny(u)}", font=st.body,
               fill=st.ink)
        s = _fmt_cny(amt)
        d.text((SIZE - st.margin - d.textlength(s, font=st.body), y), s,
               font=st.body, fill=st.ink)
        y += 52
    if has_delivery:
        d.text((st.margin, y), "运费", font=st.body, fill=st.ink)
        s = _fmt_cny(delivery)
        d.text((SIZE - st.margin - d.textlength(s, font=st.body), y), s,
               font=st.body, fill=st.ink)
        y += 52
    y += 16
    d.text((st.margin, y), "合计", font=st.head, fill=st.accent)
    s = _fmt_cny(total)
    d.text((SIZE - st.margin - d.textlength(s, font=st.head), y), s,
           font=st.head, fill=st.accent)
    rid = _rid("inv", rng)
    big = max(lines, key=lambda l: l[3])
    t = total + rng.choice([-100, 100, -200, 200])
    rows = [
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张发票。",
                   "哪一行的金额最大？",
                   [f"{g} x{q}" for g, q, _, _ in lines], lines.index(big),
                   "invoice_receipt"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张发票。",
                 f"总额超过 {_fmt_cny(t)} 吗？", "总额超过该金额",
                 "总额不超过该金额", int(total > t), "invoice_receipt"),
        noul_row(f"synth3zh-{rid}-n2", f"{rid}.png", "一张发票。",
                 "这张发票包含运费吗？", "列出了运费一行", "没有运费一行",
                 int(has_delivery), "invoice_receipt"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png", "一张发票。",
                  "发票上有几种不同的商品（不含运费与合计行）？",
                  ["一种", "两种", "三种", "四种"], len(lines) - 1,
                  "invoice_receipt"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 4 ------------------------------------------------------------------ form
def gen_form_zh(rng: random.Random) -> list[dict]:
    extras = ["加急配送", "运输保险", "礼品包装", "优先处理"]
    k = rng.randint(0, 3)
    chosen = set(rng.sample(range(4), k))
    express = 0 in chosen
    st = Style(rng, zh=True)
    img, d = canvas(st, "快递单")
    y = 128
    for i, label in enumerate(extras):
        checkbox(d, st.margin, y, st, i in chosen)
        d.text((st.margin + 70, y + 4), label, font=st.body, fill=st.ink)
        y += 66
    d.text((st.margin, y + 24), "收件人：王明", font=st.small, fill=st.dim())
    rid = _rid("form", rng)
    rows = [
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张快递单。",
                 "勾选了加急配送吗？", "该项已勾选", "该项未勾选",
                 int(express), "form_sheet"),
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张快递单。",
                  "选择了哪种配送方式？", ["普通配送", "加急配送"],
                  int(express), "form_sheet"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png", "一张快递单。",
                  "勾选了几项附加服务？", ["零项", "一项", "两项", "三项"],
                  k, "form_sheet"),
    ]
    if k:
        i = rng.choice(sorted(chosen))
        rows.append(noul_row(f"synth3zh-{rid}-n2", f"{rid}.png",
                             "一张快递单。",
                             f"单子上勾选了“{extras[i]}”吗？", "该项已勾选",
                             "该项未勾选", 1, "form_sheet"))
    else:
        rows.append(noul_row(f"synth3zh-{rid}-n2", f"{rid}.png",
                             "一张快递单。",
                             f"单子上勾选了“{rng.choice(extras)}”吗？",
                             "该项已勾选", "该项未勾选", 0, "form_sheet"))
    return [{"img": img, "rid": rid, "rows": rows}]


# 5 -------------------------------------------------------------- schedule
def gen_schedule_zh(rng: random.Random) -> list[dict]:
    trains = [f"T{rng.randint(10, 99)}" for _ in range(4)]
    while len(set(trains)) < 4:
        trains = [f"T{rng.randint(10, 99)}" for _ in range(4)]
    times = sorted(rng.sample(range(6 * 60, 12 * 60, 15), 4))
    stops = ["中央站", "滨江站", "北门站", "老城站", "机场站", "港口站"]
    stop_sets = []
    for _ in trains:
        n = rng.randint(2, 3)
        stop_sets.append(set(rng.sample(stops, n)))
        if not stop_sets[-1]:
            stop_sets[-1] = {stops[0]}
    st = Style(rng, zh=True)
    img, d = canvas(st, "发车时刻")
    y = 116
    d.text((st.margin, y), "车次", font=st.small, fill=st.dim())
    d.text((st.margin + 130, y), "发车", font=st.small, fill=st.dim())
    d.text((st.margin + 260, y), "经停", font=st.small, fill=st.dim())
    y += 44
    for tr, tm, ss in zip(trains, times, stop_sets):
        d.text((st.margin, y), tr, font=st.body, fill=st.ink)
        d.text((st.margin + 130, y), _fmt_hhmm(tm), font=st.body,
               fill=st.ink)
        d.text((st.margin + 260, y), "、".join(sorted(ss)), font=st.small,
               fill=st.ink)
        y += 74
    rid = _rid("sched", rng)
    i, j = rng.sample(range(4), 2)
    ask_stop = rng.choice(sorted(set.intersection(set(stop_sets[i]),
                                                 stop_sets[j])
                                 or {sorted(stop_sets[i])[0]}))
    thr = times[j] + rng.choice([-15, 15, -30, 30])
    rows = [
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一块发车时刻表。",
                   "哪趟车最先发车？", trains, 0, "schedule_table"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一块发车时刻表。",
                 f"{trains[i]} 次列车经停{ask_stop}吗？",
                 f"该车次经停该站", "该车次不经停该站",
                 int(ask_stop in stop_sets[i]), "schedule_table"),
        noul_row(f"synth3zh-{rid}-n2", f"{rid}.png", "一块发车时刻表。",
                 f"{trains[j]} 在 {_fmt_hhmm(thr)} 之前发车吗？",
                 "发车时间早于该时刻", "发车时间不早于该时刻",
                 int(times[j] < thr), "schedule_table"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 6 ------------------------------------------------------------ attendance
def gen_attendance_zh(rng: random.Random) -> list[dict]:
    people = rng.sample(["李华", "王芳", "张伟", "刘洋", "陈静", "杨帆"], 4)
    days = DAYS_ZH
    while True:
        mark = [[rng.random() < 0.6 for _ in days] for _ in people]
        counts = [sum(r) for r in mark]
        if counts.count(max(counts)) == 1 and max(counts) > 0:
            break
    st = Style(rng, zh=True)
    img, d = canvas(st, "考勤表")
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
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张考勤表。",
                 f"{people[pi]}在{days[di]}出勤了吗？", "当天已标记出勤",
                 "当天未标记出勤", int(mark[pi][di]), "attendance_sheet"),
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张考勤表。",
                   "谁的出勤天数最多？", people, top, "attendance_sheet"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png", "一张考勤表。",
                  f"{people[pi]}出勤了几天？",
                  ["零天", "一天", "两天", "三天", "四天", "五天"],
                  counts[pi], "attendance_sheet"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 7 ------------------------------------------------------------ sales table
def gen_table_chart_zh(rng: random.Random) -> list[dict]:
    prods = rng.sample(["甲系列", "乙系列", "丙系列", "丁系列"], 3)
    qs = ["一季度", "二季度", "三季度"]
    while True:
        vals = {p: rng.sample(range(20, 200, 5), 3) for p in prods}
        q2 = [vals[p][1] for p in prods]
        if len(set(q2)) == len(q2):
            break
    st = Style(rng, zh=True)
    img, d = canvas(st, "季度销量（件）")
    y = 130
    d.text((st.margin + 170, y), "    ".join(qs), font=st.body, fill=st.dim())
    y += 54
    for p in prods:
        d.text((st.margin, y), p, font=st.body, fill=st.ink)
        for j, v in enumerate(vals[p]):
            d.text((st.margin + 190 + j * 140, y), str(v), font=st.body,
                   fill=st.ink)
        y += 74
    rid = _rid("tab", rng)
    a, b = rng.sample(prods, 2)
    q2_top = max(prods, key=lambda p: vals[p][1])
    rows = [
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张销量表。",
                   "哪个产品的三季度销量最高？", prods, prods.index(q2_top),
                   "table_chart"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张销量表。",
                 f"{a}在三季度的销量比{b}高吗？", f"{a}该季度销量更高",
                 f"{b}的销量不低于{a}", int(vals[a][2] > vals[b][2]),
                 "table_chart"),
        noul_row(f"synth3zh-{rid}-n2", f"{rid}.png", "一张销量表。",
                 f"{a}的销量从二季度到三季度增长了吗？", "数量上升了",
                 "数量持平或下降", int(vals[a][2] > vals[a][1]),
                 "table_chart"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 8 ----------------------------------------------------------------- hours
def gen_sign_zh(rng: random.Random) -> list[dict]:
    days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    durs = rng.sample(range(360, 780, 30), 7)
    opens = {}
    k = 0
    for day in days:
        if day == "周日" and rng.random() < 0.5:
            opens[day] = None
        else:
            o = rng.randint(7, 10) * 60
            opens[day] = (o, o + durs[k])
            k += 1
    open_days = [x for x in days if opens[x] is not None]
    st = Style(rng, zh=True)
    img, d = canvas(st, "营业时间")
    y = 124
    for day in days:
        d.text((st.margin, y), day, font=st.body, fill=st.ink)
        v = opens[day]
        txt = "休息" if v is None else f"{_fmt_hhmm(v[0])} - {_fmt_hhmm(v[1])}"
        d.text((st.margin + 150, y), txt, font=st.body, fill=st.ink)
        y += 64
    rid = _rid("sign", rng)
    open_days = [x for x in days if opens[x] is not None]
    longest = max(open_days, key=lambda x: opens[x][1] - opens[x][0])
    ask = rng.choice(days)
    v = opens[ask]
    rows = [
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张营业时间牌。",
                 f"{ask}营业吗？", "当天列出了营业时间", "牌上写着休息",
                 int(v is not None), "sign_poster"),
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张营业时间牌。",
                   "哪一天的营业时间最长？", open_days,
                   open_days.index(longest), "sign_poster"),
    ]
    if v is not None:
        t = rng.choice([v[0] - 30, v[0], v[0] + 30])
        rows.append(noul_row(
            f"synth3zh-{rid}-n2", f"{rid}.png", "一张营业时间牌。",
            f"{ask}在 {_fmt_hhmm(t)} 之前（含）开门吗？", "开门时间不晚于该时刻",
            "开门更晚", int(v[0] <= t), "sign_poster"))
    return [{"img": img, "rid": rid, "rows": rows}]


# 9 ---------------------------------------------------------------- ticket
def gen_ticket_zh(rng: random.Random) -> list[dict]:
    gate = rng.choice(["A", "B", "C", "D"])
    row_n = rng.randint(3, 25)
    seat = rng.randint(1, 40)
    dom = rng.randint(2, 27)
    hour = rng.randint(13, 22)
    minute = rng.choice([0, 15, 30, 45])
    st = Style(rng, zh=True)
    img, d = canvas(st, "入场券")
    fields = [("活动", "春季音乐会"), ("日期", f"2026年6月{dom}日"),
              ("时间", _fmt_hhmm(hour * 60 + minute)),
              ("入口", gate), ("排", str(row_n)), ("座", str(seat))]
    y = 128
    for label, val in fields:
        d.text((st.margin, y), label, font=st.small, fill=st.dim())
        d.text((st.margin + 110, y), val, font=st.body, fill=st.ink)
        y += 74
    rid = _rid("tix", rng)
    thr = rng.choice([row_n - 1, row_n, row_n + 1])
    rows = [
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张活动门票。",
                   "门票上写的入口是哪一个？", ["A", "B", "C", "D"],
                   "ABCD".index(gate), "ticket_stub"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张活动门票。",
                 f"座位排号不低于第 {thr} 排吗？", "排号达到或超过该排",
                 "排号低于该排", int(row_n >= thr), "ticket_stub"),
        noul_row(f"synth3zh-{rid}-n2", f"{rid}.png", "一张活动门票。",
                 "活动日期在上半月吗？", "日期在 1 日至 15 日之间",
                 "日期在 16 日或之后", int(dom <= 15), "ticket_stub"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 10 ----------------------------------------------------------- key/value
def gen_key_value_zh(rng: random.Random) -> list[dict]:
    weight = rng.randint(5, 95)
    colors = rng.sample(["红色", "蓝色", "绿色", "黑色", "白色"], 3)
    ordered = list(colors)
    rng.shuffle(ordered)
    dims = (rng.randint(20, 40), rng.randint(15, 35), rng.randint(15, 30))
    st = Style(rng, zh=True)
    img, d = canvas(st, "产品规格")
    fields = [("重量", f"{weight / 10:.1f} kg"),
              ("尺寸", f"{dims[0]} x {dims[1]} x {dims[2]} cm"),
              ("颜色", "、".join(ordered))]
    y = 130
    for label, val in fields:
        d.text((st.margin, y), label, font=st.small, fill=st.dim())
        d.text((st.margin + 130, y), val, font=st.body, fill=st.ink)
        y += 84
    rid = _rid("spec", rng)
    t = weight + rng.choice([-10, 10, -5, 5])
    all_colors = ["红色", "蓝色", "绿色", "黑色", "白色"]
    rows = [
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张产品规格表。",
                 f"重量超过 {t / 10:.1f} kg 吗？", "所列重量高于该值",
                 "所列重量不高于该值", int(weight > t), "key_value_panel"),
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张产品规格表。",
                   "颜色一栏中排在第一位的颜色是哪个？", all_colors,
                   all_colors.index(ordered[0]), "key_value_panel"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 11 -------------------------------------------------------------- calendar
def gen_calendar_zh(rng: random.Random) -> list[dict]:
    m = rng.randint(0, 11)
    days = rng.choice([28, 30, 31])
    e1 = rng.randint(1, days)
    e2 = e1
    while abs(e2 - e1) < 4:
        e2 = rng.randint(1, days)
    st = Style(rng, allow_dark=False, zh=True)
    img, d = canvas(st, MONTHS_ZH[m])
    y0 = 116
    for col, wd in enumerate(WEEKDAYS_ZH):
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
           f"实心 = 团队午餐（第 {e1} 天）；方框 = 复盘（第 {e2} 天）",
           font=st.small, fill=st.dim())
    rid = f"zh-cal-{m:02d}-{days}-{e1:02d}-{e2:02d}"
    week_of = lambda day: (day - 1) // 7
    rows = [
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png",
                 "一页带有两个标记事件的日历。",
                 "实心标记的团队午餐安排在当月 15 号之后吗？",
                 "活动日期在 15 号之后", "活动日期在 15 号当天或之前",
                 int(e1 > 15), "calendar_card"),
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png",
                   "一页带有两个标记事件的日历。",
                   "实心团队午餐在当月第几周？",
                   ["第1周", "第2周", "第3周", "第4周", "第5周"],
                   week_of(e1), "calendar_card"),
        noul_row(f"synth3zh-{rid}-n2", f"{rid}.png",
                 "一页带有两个标记事件的日历。",
                 "实心团队午餐发生在方框复盘之前吗？", "实心事件日期更早",
                 "实心事件同日或更晚", int(e1 < e2), "calendar_card"),
        noul_row(f"synth3zh-{rid}-n3", f"{rid}.png",
                 "一页带有两个标记事件的日历。",
                 "方框复盘与实心团队午餐在同一周吗？", "两者落在同一周行",
                 "两者不在同一周行", int(week_of(e1) == week_of(e2)),
                 "calendar_card"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 12 ------------------------------------------------------------- bar chart
def gen_bar_chart_zh(rng: random.Random) -> list[dict]:
    cats = rng.sample(["甲", "乙", "丙", "丁", "戊"], 4)
    while True:
        vals = rng.sample(range(10, 100, 5), 4)
        if len(set(vals)) == 4:
            break
    st = Style(rng, allow_dark=False, zh=True)
    img, d = canvas(st, "季度出货量")
    x0, y0, y1 = st.margin + 40, 460, 150
    for gv in range(0, 101, 25):
        y = y0 - (y0 - y1) * gv / 100
        d.line((x0 - 14, y, SIZE - st.margin, y), fill=st.dim(), width=1)
        d.text((x0 - 20, y), str(gv), font=st.small, fill=st.dim(),
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
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张柱状图。",
                   "哪根柱子最高？", cats, vals.index(max(vals)),
                   "bar_chart"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张柱状图。",
                 f"{cats[a]}的数值比{cats[b]}高吗？", f"{cats[a]}的柱子更高",
                 f"{cats[b]}的柱子不低于{cats[a]}",
                 int(vals[a] > vals[b]), "bar_chart"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png", "一张柱状图。",
                  "数值超过 50 的柱子有几根？",
                  ["没有", "一根", "两根", "三根", "四根"], over,
                  "bar_chart"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 13 --------------------------------------------------------------- seating
def gen_seating_zh(rng: random.Random) -> list[dict]:
    names = ["陈立", "王芳", "刘洋", "张敏", "李强"]
    grid = [[rng.random() < 0.5 for _ in range(5)] for _ in range(4)]
    while True:
        row_counts = [sum(r) for r in grid]
        if row_counts and row_counts.count(max(row_counts)) == 1 \
                and sum(row_counts) < 20:
            break
        grid = [[rng.random() < 0.5 for _ in range(5)] for _ in range(4)]
    st = Style(rng, zh=True)
    img, d = canvas(st, "工位图")
    x0, y0 = st.margin + 60, 140
    for r in range(4):
        for c in range(5):
            x, y = x0 + c * 100, y0 + r * 104
            d.rectangle((x, y, x + 80, y + 78), outline=st.ink, width=3)
            if grid[r][c]:
                d.text((x + 40, y + 39), names[(r * 5 + c) % 5],
                       font=st.small, fill=st.accent, anchor="mm")
            else:
                d.text((x + 40, y + 39), "空位", font=st.small,
                       fill=st.dim(), anchor="mm")
    for r in range(4):
        d.text((x0 - 10, y0 + r * 104 + 39), f"第{r + 1}排",
               font=st.small, fill=st.ink, anchor="rm")
    rid = _rid("seat", rng)
    empty = sum(1 for r in grid for v in r if not v)
    pi, pj = rng.randrange(4), rng.randrange(5)
    top_row = row_counts.index(max(row_counts))
    rows = [
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张办公室工位图。",
                 f"第{pi + 1}排第{pj + 1}个工位有人使用吗？", "该工位显示了姓名",
                 "该工位标为空位", int(grid[pi][pj]), "seating_chart"),
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张办公室工位图。",
                   "哪一排占用的工位最多？",
                   ["第1排", "第2排", "第3排", "第4排"], top_row,
                   "seating_chart"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png", "一张办公室工位图。",
                 "空工位有多少个？",
                 ["没有", "1-5个", "6-10个", "11-15个", "16-20个"],
                 min(4, (empty + 4) // 5), "seating_chart"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 14 --------------------------------------------------------------- weather
def gen_weather_zh(rng: random.Random) -> list[dict]:
    days = DAYS_ZH
    while True:
        temps = rng.sample(range(-2, 28), 5)
        if temps.count(max(temps)) == 1 and temps.count(min(temps)) == 1:
            break
    icons = ["晴", "多云", "雨", "雪"]
    wx = [rng.choice(icons) for _ in days]
    st = Style(rng, zh=True)
    img, d = canvas(st, "五日天气")
    for j, (day, t, w) in enumerate(zip(days, temps, wx)):
        x = st.margin + 40 + j * 112
        d.text((x + 40, 130), day, font=st.body, fill=st.ink, anchor="ma")
        d.text((x + 40, 220), w, font=st.small, fill=st.dim(), anchor="ma")
        d.text((x + 40, 330), f"{t}\u00b0", font=st.head, fill=st.accent,
               anchor="ma")
    rid = _rid("wx", rng)
    a, b = rng.sample(range(5), 2)
    cold = sum(1 for t in temps if t < 10)
    rows = [
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张天气预报。",
                   "哪一天最暖和？", days, temps.index(max(temps)),
                   "weather_panel"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张天气预报。",
                 f"{days[a]}比{days[b]}暖和吗？", f"{days[a]}的气温更高",
                 f"{days[b]}至少同样暖和", int(temps[a] > temps[b]),
                 "weather_panel"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png", "一张天气预报。",
                  "低于 10 度的有几天？",
                  ["没有", "一天", "两天", "三天", "四天", "五天"], cold,
                  "weather_panel"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 15 ---------------------------------------------------------- notifications
def gen_notification_zh(rng: random.Random) -> list[dict]:
    apps = rng.sample(["邮件", "聊天", "新闻", "相册", "银行", "音乐",
                       "商店", "健身"], 5)
    counts = rng.sample(range(1, 20), 5)
    st = Style(rng, allow_dark=True, zh=True)
    img, d = canvas(st, "通知")
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
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png",
                   "一组带角标的应用列表。",
                   "哪个应用的未读通知最多？", apps, top, "notification_card"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png",
                 "一组带角标的应用列表。",
                 f"{apps[a]}的未读通知超过 9 条吗？", "角标数字达到 10 或以上",
                 "角标数字不超过 9", int(counts[a] >= 10),
                 "notification_card"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png",
                  "一组带角标的应用列表。",
                  "未读通知达到 10 条以上的应用有几个？",
                  ["没有", "一个", "两个", "三个", "四个", "五个"], many,
                  "notification_card"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 16 ----------------------------------------------------------------- kanban
def gen_kanban_zh(rng: random.Random) -> list[dict]:
    cols = ["待办", "进行中", "已完成"]
    tasks = ["登录流程", "深色模式", "修复崩溃", "文档", "计费",
             "搜索", "导出", "移动端"]
    rng.shuffle(tasks)
    board = {c: [] for c in cols}
    for t in tasks:
        board[rng.choice(cols)].append(t)
    while any(len(v) == 0 for v in board.values()):
        board = {c: [] for c in cols}
        for t in tasks:
            board[rng.choice(cols)].append(t)
    st = Style(rng, zh=True)
    img, d = canvas(st, "迭代看板")
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
    n_todo = len(board["待办"])
    rows = [
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一张任务看板。",
                   f"“{ask}”在哪一列？", cols,
                   next(i for i, c in enumerate(cols) if ask in board[c]),
                   "kanban_board"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一张任务看板。",
                 "任务“计费”在“进行中”列吗？", "它在进行中列",
                 "它在其他列", int("计费" in board["进行中"]),
                 "kanban_board"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png", "一张任务看板。",
                  "“待办”列里有几张卡片？",
                  ["没有", "一张", "两张", "三张", "四张及以上"],
                  min(4, n_todo), "kanban_board"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 17 ------------------------------------------------------------ grid cells
def gen_grid_pattern_zh(rng: random.Random) -> list[dict]:
    nums = [rng.randint(10, 99) for _ in range(16)]
    while len(set(nums)) != 16:
        nums = [rng.randint(10, 99) for _ in range(16)]
    colored = set(rng.sample(range(16), rng.randint(2, 6)))
    st = Style(rng, allow_dark=False, zh=True)
    img, d = canvas(st, "编号网格")
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
    corner = nums[3]
    r2c3 = nums[6]
    opts = [corner] + rng.sample([x for x in nums if x != corner], 3)
    rng.shuffle(opts)
    rows = [choice_row(f"synth3zh-{rid}-c", f"{rid}.png",
                       "一个 4x4 的数字网格。",
                       "右上角单元格里的数字是多少？",
                       [str(x) for x in opts], opts.index(corner),
                       "grid_pattern"),
            noul_row(f"synth3zh-{rid}-n", f"{rid}.png",
                     "一个 4x4 的数字网格。",
                     "第2行第3列（从左上角起数，从一开始）的数字大于 55 吗？",
                     "该数字超过 55", "该数字不超过 55", int(r2c3 > 55),
                     "grid_pattern"),
            score_row(f"synth3zh-{rid}-s", f"{rid}.png",
                      "一个 4x4 的数字网格。",
                      "用强调色高亮的单元格有几个？",
                      ["没有", "两个", "三个或四个", "五个或六个"],
                      0 if not colored else (1 if len(colored) == 2 else
                                             (2 if len(colored) <= 4 else 3)),
                      "grid_pattern")]
    return [{"img": img, "rid": rid, "rows": rows}]


# 18 ------------------------------------------------------------ analog clock
def gen_clock_zh(rng: random.Random) -> list[dict]:
    hour = rng.randint(1, 12)
    minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    st = Style(rng, allow_dark=False, zh=True)
    img, d = canvas(st, None)
    cx, cy, R = SIZE // 2, SIZE // 2 + 20, 230
    d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=st.ink, width=8)
    for i in range(12):
        ang = i * 30 - 90
        x1 = cx + (R - 30) * math.cos(math.radians(ang))
        y1 = cy + (R - 30) * math.sin(math.radians(ang))
        x2 = cx + (R - 12) * math.cos(math.radians(ang))
        y2 = cy + (R - 12) * math.sin(math.radians(ang))
        d.line((x1, y1, x2, y2), fill=st.ink, width=6)
    ang_h = (hour % 12 + minute / 60) * 30 - 90
    ang_m = minute * 6 - 90
    d.line((cx, cy, cx + (R - 90) * math.cos(math.radians(ang_h)),
            cy + (R - 90) * math.sin(math.radians(ang_h))),
           fill=st.ink, width=14)
    d.line((cx, cy, cx + (R - 40) * math.cos(math.radians(ang_m)),
            cy + (R - 40) * math.sin(math.radians(ang_m))),
           fill=st.accent, width=9)
    d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=st.ink)
    total = hour * 60 + minute
    rid = f"zh-clock-{hour:02d}{minute:02d}"
    thr_h, thr_m = rng.choice([(hour, (minute + 30) % 60),
                               (hour % 12 + 1, minute), (hour, minute)])
    thr_total = thr_h * 60 + thr_m if thr_h else 720
    if abs(thr_total - total) < 3:
        thr_total += 30
    rows = [
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一个模拟时钟表盘。",
                 "表盘显示的时间在中午之前吗？",
                 "时针在上午半区（12:00 之前，12:00 算中午）",
                 "时间是正午 12:00 或之后", int(hour < 12), "clock_face"),
        noul_row(f"synth3zh-{rid}-n2", f"{rid}.png", "一个模拟时钟表盘。",
                 "分针越过刻度 6 了吗（即超过 30 分钟）？",
                 "分针已越过 6", "分针在 6 或之前", int(minute > 30),
                 "clock_face"),
        noul_row(f"synth3zh-{rid}-n3", f"{rid}.png", "一个模拟时钟表盘。",
                 "显示的时间在 "
                 f"{((thr_total // 60 - 1) % 12) + 1}:{thr_total % 60:02d}"
                 " 之前吗？", "所示时间更早", "所示时间达到或晚于它",
                 int(total < thr_total), "clock_face"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 19 ------------------------------------------------------------ email card
def gen_email_card_zh(rng: random.Random) -> list[dict]:
    senders = ["billing@acme-corp.com", "no-reply@shop-now.example",
               "team@projectflow.io", "alerts@bankmail.example"]
    sender = rng.choice(senders)
    has_attachment = rng.random() < 0.5
    folder = rng.choice(["收件箱", "工作", "订阅"])
    subj = rng.choice(["您八月的账单", "每周摘要", "回复：合同", "款项已到账",
                       "新的登录提醒"])
    st = Style(rng, zh=True)
    img, d = canvas(st, "邮件")
    y = 130
    d.text((st.margin, y), "发件人：", font=st.small, fill=st.dim())
    d.text((st.margin + 130, y), sender, font=st.body, fill=st.ink)
    y += 62
    d.text((st.margin, y), "主题：", font=st.small, fill=st.dim())
    d.text((st.margin + 130, y), subj, font=st.body, fill=st.ink)
    y += 62
    d.text((st.margin, y), "文件夹：", font=st.small, fill=st.dim())
    d.text((st.margin + 130, y), folder, font=st.body, fill=st.ink)
    y += 80
    if has_attachment:
        d.rectangle((st.margin, y, st.margin + 220, y + 64),
                    outline=st.ink, width=3)
        d.text((st.margin + 16, y + 18), "附件.pdf", font=st.body,
               fill=st.ink)
    rid = _rid("mail", rng)
    domain = sender.split("@")[1]
    ask_folder = rng.choice(["收件箱", "工作", "订阅"])
    rows = [
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一个邮件标头。",
                 "这封邮件有附件吗？", "显示了附件标识", "没有附件标识",
                 int(has_attachment), "email_card"),
        noul_row(f"synth3zh-{rid}-n2", f"{rid}.png", "一个邮件标头。",
                 f"这封邮件归档在“{ask_folder}”文件夹吗？",
                 "文件夹标签一致", "文件夹标签不同",
                 int(ask_folder == folder), "email_card"),
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一个邮件标头。",
                   "发件人的地址属于哪个域名？",
                   sorted({s.split('@')[1] for s in senders}),
                   sorted({s.split('@')[1] for s in senders}).index(domain),
                   "email_card"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


# 20 ------------------------------------------------------------- shapes
def gen_shape_scene_zh(rng: random.Random) -> list[dict]:
    kinds = ["三角形", "圆形", "正方形"]
    colors = ["红", "蓝", "绿", "琥珀"]
    while True:
        counts = {k: rng.randint(0, 5) for k in kinds}
        vals = list(counts.values())
        if sum(vals) >= 4 and vals.count(max(vals)) == 1:
            break
    st = Style(rng, allow_dark=False, zh=True)
    img, d = canvas(st, "图形场景")
    pen = {"红": "#c0392b", "蓝": "#2471a3", "绿": "#1e8449",
           "琥珀": "#b9770e"}
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
            if kind == "圆形":
                d.ellipse((x - 26, y - 26, x + 26, y + 26),
                          outline=pen[col], width=6)
            elif kind == "正方形":
                d.rectangle((x - 24, y - 24, x + 24, y + 24),
                            outline=pen[col], width=6)
            else:
                p = [(x, y - 30), (x + 28, y + 22), (x - 28, y + 22)]
                d.polygon(p, outline=pen[col], width=6)
    rid = _rid("shape", rng)
    top_kind = max(kinds, key=lambda k: counts[k])
    a, b = rng.sample(kinds, 2)
    rows = [
        choice_row(f"synth3zh-{rid}-c", f"{rid}.png", "一个图形场景。",
                   "哪种图形出现得最多？", kinds, kinds.index(top_kind),
                   "shape_scene"),
        noul_row(f"synth3zh-{rid}-n", f"{rid}.png", "一个图形场景。",
                 f"{a}比{b}多吗？", f"{a}更多", f"{b}至少同样多",
                 int(counts[a] > counts[b]), "shape_scene"),
        score_row(f"synth3zh-{rid}-s", f"{rid}.png", "一个图形场景。",
                  f"图中有几个{a}？", ["没有", "一两个", "三四个", "五个"],
                  min(3, counts[a]), "shape_scene"),
    ]
    return [{"img": img, "rid": rid, "rows": rows}]


GENS_ZH = {
    "menu_board": gen_menu_zh, "price_tag": gen_price_tag_zh,
    "invoice_receipt": gen_invoice_zh, "form_sheet": gen_form_zh,
    "schedule_table": gen_schedule_zh, "attendance_sheet": gen_attendance_zh,
    "table_chart": gen_table_chart_zh, "sign_poster": gen_sign_zh,
    "ticket_stub": gen_ticket_zh, "key_value_panel": gen_key_value_zh,
    "bar_chart": gen_bar_chart_zh, "calendar_card": gen_calendar_zh,
    "seating_chart": gen_seating_zh, "weather_panel": gen_weather_zh,
    "notification_card": gen_notification_zh, "kanban_board": gen_kanban_zh,
    "grid_pattern": gen_grid_pattern_zh, "clock_face": gen_clock_zh,
    "email_card": gen_email_card_zh, "shape_scene": gen_shape_scene_zh,
}