"""Synthetic v2-zh: the synth2 skill families, Chinese surface edition.

Same three families (temporal_numeric, multi_hop, long_policy), same
gold-by-construction logic (imported from synth2, never re-implemented),
fully Chinese state values, instructions, and criteria. State KEYS stay
English — they are wire-format identifiers, exactly as in the OCNLI
gate rows; only values and templates localize. Answer tokens stay
yes/no + letters per the v1.1 pre-registration.

    python3 -m kapteeni.synth2zh selfcheck
    python3 -m kapteeni.synth2zh synth --n 4200 \
        --out data_cache/rows_synth2zh.jsonl \
        --criteria-out data_cache/synth2zh_criteria.json
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from kapteeni.synth2 import (
    _holiday_set, _months_between, _rand_date, _shift_days_for_months,
    add_bd, month_add,
)

WD_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def fmt_date_zh(d: date, rng: random.Random) -> str:
    """5 natural Chinese date formats (weekday-styled included)."""
    style = rng.choice(["ymd", "slash", "noyear", "zero", "wd"])
    if style == "ymd":
        return f"{d.year}年{d.month}月{d.day}日"
    if style == "slash":
        return f"{d.year}/{d.month}/{d.day}"
    if style == "noyear":
        return f"{d.month}月{d.day}日"
    if style == "zero":
        return f"{d.year}年{d.month:02d}月{d.day:02d}日"
    return f"{d.year}年{d.month}月{d.day}日（{WD_ZH[d.weekday()]}）"


def _holiday_str_zh(holidays: set) -> str:
    return "、".join(sorted(f"{h.year}年{h.month}月{h.day}日"
                           for h in holidays))


# ------------------------------------------------------------- temporal_numeric

def gen_date_arith_zh(rng: random.Random) -> dict:
    base = _rand_date(rng)
    unit = rng.choice(["days", "days", "weeks", "months"])
    unit_zh = {"days": "天", "weeks": "周", "months": "个月"}[unit]
    n = {"days": rng.randint(2, 45), "weeks": rng.randint(1, 8),
         "months": rng.randint(1, 3)}[unit]
    if unit == "months" and base.day > 28:
        base = base.replace(day=rng.randint(1, 28))
    true_event = (base + timedelta(days=n) if unit == "days" else
                  base + timedelta(weeks=n) if unit == "weeks" else
                  month_add(base, n))
    roll = rng.random()
    if roll < 0.5:
        claimed = true_event
    elif roll < 0.8:
        claimed = true_event + timedelta(days=rng.choice([-3, -2, -1, 1, 2, 3, 7]))
    else:
        claimed = true_event + timedelta(weeks=rng.choice([-2, -1, 1, 2]))
    gold = int(claimed == true_event)
    neg = rng.random() < 0.15
    thing = rng.choice(["设备巡检", "年度审计", "董事会会议", "交付验收",
                        "员工培训", "产品发布"])
    phr = rng.choice([
        f"{thing}恰好在 `reference_date` 之后 {n} {unit_zh}进行。",
        f"根据日程安排，{thing}恰好在 `reference_date` 之后 {n} {unit_zh}，"
        "不早也不晚。",
    ])
    state = {"reference_date": fmt_date_zh(base, rng), "schedule": phr,
             "claimed_date": fmt_date_zh(claimed, rng)}
    if rng.random() < 0.3:
        state["reminder_sent_on"] = fmt_date_zh(
            base + timedelta(days=rng.randint(1, 10)), rng)
    if not neg:
        return {"row_id": None, "primitive": "noul", "state": state,
                "instructions": "`claimed_date` 是不是 `schedule` 所描述"
                                "的那一天？",
                "criteria": {"true": "声称的日期就是所描述的日期",
                             "false": "声称的日期是另一个日期"},
                "label": gold, "meta": {"family": "temporal_numeric"}}
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "`claimed_date` 和 `schedule` 所描述的日期"
                            "不一样吗？",
            "criteria": {"true": "声称的日期与所描述的日期不同",
                         "false": "声称的日期就是所描述的日期"},
            "label": 1 - gold, "meta": {"family": "temporal_numeric"}}


def gen_business_days_zh(rng: random.Random) -> dict:
    start = _rand_date(rng)
    n = rng.choice([3, 5, 10, 15])
    hol = _holiday_set(rng)
    mode = rng.choice(["window", "due"])
    if mode == "window":
        k = rng.choice([n - 1, n, n, n + 1, n + 2])
        sent = add_bd(start, k, hol)
        gold = int(k <= n)
        state = {"received_on": fmt_date_zh(start, rng),
                 "rule": f"须在 `received_on` 之后 {n} 个工作日内作出回复。"
                         "工作日为星期一至星期五，且不含所列节假日。",
                 "holidays": _holiday_str_zh(hol),
                 "responded_on": fmt_date_zh(sent, rng)}
        return {"row_id": None, "primitive": "noul", "state": state,
                "instructions": "`responded_on` 中的回复是否在 `rule` "
                                "规定的期限内？",
                "criteria": {"true": "回复在期限内",
                             "false": "回复超期"},
                "label": gold, "meta": {"family": "temporal_numeric"}}
    due = add_bd(start, n, hol)
    roll = rng.random()
    if roll < 0.5:
        claimed = due
    elif roll < 0.85:
        claimed = due + timedelta(days=rng.choice([-3, -1, 1, 2, 3]))
    else:
        claimed = due + timedelta(weeks=1)
    gold = int(claimed == due)
    state = {"received_on": fmt_date_zh(start, rng),
             "rule": f"截止日为 `received_on` 之后恰好 {n} 个工作日。"
                     "工作日为星期一至星期五，且不含所列节假日。",
             "holidays": _holiday_str_zh(hol),
             "claimed_due_date": fmt_date_zh(claimed, rng)}
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "按照 `rule`，`claimed_due_date` 是正确的"
                            "截止日吗？",
            "criteria": {"true": "声称的截止日正确",
                         "false": "声称的截止日至少差了一个工作日"},
            "label": gold, "meta": {"family": "temporal_numeric"}}


def gen_deadline_bounds_zh(rng: random.Random) -> dict:
    start = _rand_date(rng)
    n = rng.choice([3, 7, 10, 14, 30, 60])
    rel = rng.choice(["window", "no_later", "lead_time", "strict", "after"])
    if rel == "window":
        delta = rng.choice([n - 1, n, n, n + 1, n + 2])
        act = start + timedelta(days=delta)
        gold = int(delta <= n)
        state = {"incident_on": fmt_date_zh(start, rng),
                 "policy": f"索赔须在 `incident_on` 起 {n} 天内提出"
                           "（事发当日计为第 0 天）。",
                 "filed_on": fmt_date_zh(act, rng)}
        q = "`filed_on` 中的索赔按 `policy` 是否及时？"
        crit = {"true": "索赔在期限内", "false": "索赔超期"}
    elif rel == "no_later":
        due = start + timedelta(days=n)
        delta = rng.choice([-2, -1, 0, 0, 1, 2])
        act = due + timedelta(days=delta)
        gold = int(act <= due)
        state = {"deadline": fmt_date_zh(due, rng),
                 "action_on": fmt_date_zh(act, rng)}
        q = "`action_on` 满足 `deadline` 的要求吗？"
        crit = {"true": "该行为发生在截止日当天或之前",
                "false": "该行为发生在截止日之后"}
    elif rel == "lead_time":
        ev = start + timedelta(days=n + rng.choice([0, 1, 1, 2]))
        delta = rng.choice([n - 1, n, n, n + 1])
        act = ev - timedelta(days=delta)
        gold = int((ev - act).days >= n)
        state = {"event_on": fmt_date_zh(ev, rng),
                 "notice_policy": f"通知须至少在 `event_on` 之前 {n} 天"
                                  "发出。",
                 "notice_given_on": fmt_date_zh(act, rng)}
        q = "`notice_given_on` 中的通知符合 `notice_policy` 吗？"
        crit = {"true": "通知满足最低提前量", "false": "通知发出得太晚"}
    elif rel == "strict":
        due = start + timedelta(days=n)
        delta = rng.choice([1, 1, 2, 0])
        act = due - timedelta(days=delta)
        gold = int(act < due)
        state = {"deadline": fmt_date_zh(due, rng),
                 "rule": "申请必须在 `deadline` 之前提交（当天提交视为"
                         "超时）。",
                 "submitted_on": fmt_date_zh(act, rng)}
        q = "`submitted_on` 满足 `rule` 吗？"
        crit = {"true": "提交发生在截止日之前",
                "false": "提交发生在截止日当天或之后"}
    else:
        fdate = start
        delta = rng.choice([0, 1, 1, 2, 3])
        act = fdate + timedelta(days=delta)
        gold = int(act > fdate)
        state = {"filing_date": fmt_date_zh(fdate, rng),
                 "rule": "仅在 `filing_date` 之后受理修改申请。",
                 "amendment_on": fmt_date_zh(act, rng)}
        q = "`amendment_on` 的修改按 `rule` 会被受理吗？"
        crit = {"true": "修改发生在备案日之后", "false": "修改不晚于备案日"}
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": q, "criteria": crit, "label": gold,
            "meta": {"family": "temporal_numeric"}}


def gen_overlap_zh(rng: random.Random) -> dict:
    s1 = _rand_date(rng)
    len1 = rng.randint(3, 12)
    e1 = s1 + timedelta(days=len1)
    off = rng.choice([-len1, -3, -2, -1, -1, 0, 0, 1, 2, len1 - 1, len1 + 2])
    s2 = s1 + timedelta(days=off)
    e2 = s2 + timedelta(days=rng.randint(2, 10))
    gold = int(max(s1, s2) <= min(e1, e2))
    a = rng.choice(["值班安排", "维护窗口", "促销活动", "审计外勤", "培训批次"])
    b = rng.choice(["系统升级", "封板期", "库存盘点", "灰度发布", "办公室搬迁"])
    state = {
        "window_a": {"name": f"{a}一期",
                     "from": fmt_date_zh(s1, rng), "through": fmt_date_zh(e1, rng)},
        "window_b": {"name": f"{b}",
                     "from": fmt_date_zh(s2, rng), "through": fmt_date_zh(e2, rng)},
        "note": "两个时间段的起止日期均含当天。",
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "`window_a` 和 `window_b` 是否至少有一天"
                            "重合？",
            "criteria": {"true": "至少有一个自然日在两个时间段内",
                         "false": "两个时间段没有重合的日子"},
            "label": gold, "meta": {"family": "temporal_numeric"}}


def gen_month_arith_zh(rng: random.Random) -> dict:
    base = _rand_date(rng)
    base = base.replace(day=rng.randint(1, 28))
    n = rng.choice([1, 2, 3, 5, 6])
    true_event = month_add(base, n)
    if rng.random() < 0.55:
        claimed = true_event
    else:
        claimed = month_add(true_event, rng.choice([-1, 1])) \
            if rng.random() < 0.5 else true_event + timedelta(
                days=rng.choice([-2, 2]))
    gold = int(claimed == true_event)
    state = {"agreement_signed_on": fmt_date_zh(base, rng),
             "term": f"续约在 `agreement_signed_on` 之后 {n} 个月生效。",
             "claimed_effective_date": fmt_date_zh(claimed, rng)}
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "按照 `term`，`claimed_effective_date` 是正确"
                            "的生效日吗？",
            "criteria": {"true": "声称的日期正确",
                         "false": "声称的日期有误"},
            "label": gold, "meta": {"family": "temporal_numeric"}}


def gen_status_choice_zh(rng: random.Random) -> dict:
    today = _rand_date(rng)
    delta = rng.choice([-5, -1, 0, 0, 1, 2, 3, 5, 7, 8, 12, 30])
    due = today + timedelta(days=delta)
    thing = rng.choice(["发票", "订阅续费", "许可证到期", "保修到期", "报告截止"])
    if delta < 0:
        gold = "overdue"
    elif delta == 0:
        gold = "due_today"
    elif delta <= 7:
        gold = "due_within_7_days"
    else:
        gold = "due_later"
    opts = ["overdue", "due_today", "due_within_7_days", "due_later"]
    state = {"today": fmt_date_zh(today, rng),
             "due_date": fmt_date_zh(due, rng),
             "subject": f"所指的{thing}。"}
    return {"row_id": None, "primitive": "choice", "state": state,
            "instructions": "相对于 `today`，`due_date` 处于什么状态？",
            "criteria": None, "label": gold,
            "meta": {"options": opts, "family": "temporal_numeric"}}


def gen_urgency_score_zh(rng: random.Random) -> dict:
    today = _rand_date(rng)
    delta = rng.choice([-3, -1, 0, 0, 1, 2, 2, 4, 6, 7, 8, 10, 15, 21])
    deadline = today + timedelta(days=delta)
    if delta <= 0:
        gold = 0
    elif delta <= 2:
        gold = 1
    elif delta <= 7:
        gold = 2
    else:
        gold = 3
    state = {"today": fmt_date_zh(today, rng),
             "deadline": fmt_date_zh(deadline, rng),
             "scale": "紧急度分档：紧急——截止日为今天或已经过去；"
                      "高——距截止日还有 1-2 天；中——距截止日还有 3-7 天；"
                      "低——距截止日还有 7 天以上。"}
    return {"row_id": None, "primitive": "score", "state": state,
            "instructions": "按照 `scale` 中的分档，`deadline` 的紧急度"
                            "是多少？",
            "criteria": None, "label": gold,
            "meta": {"attribute": "deadline_urgency",
                     "family": "temporal_numeric"}}


# ------------------------------------------------------------------- multi_hop

def gen_chain_rules_zh(rng: random.Random) -> dict:
    t_req = rng.choice([6, 12, 18, 24])
    s_req = rng.choice([250, 500, 1000, 2000])
    b_max = rng.choice([15, 30, 45])
    plan = rng.choice(["standard", "standard", "elite"])
    plan_zh = {"standard": "标准套餐", "elite": "尊享套餐"}[plan]
    today = _rand_date(rng)
    tenure = rng.choice([t_req - 1, t_req - 1, t_req, t_req + 1])
    joined = _shift_days_for_months(today, tenure)
    spend = s_req + rng.choice([-50, -10, 0, 0, 10, 50])
    overdue = rng.choice([b_max - 1, b_max, b_max + 1, b_max + 15, -1])
    ok_tenure = tenure >= t_req or plan == "elite"
    ok_spend = spend >= s_req
    ok_overdue = overdue <= b_max
    gold = int(ok_tenure and ok_spend and ok_overdue)
    state = {
        "today": fmt_date_zh(today, rng),
        "plan": plan_zh,
        "account_opened_on": fmt_date_zh(joined, rng),
        "trailing_12m_spend_usd": spend,
        "oldest_unpaid_invoice_age_days": overdue,
        "policy": "满足以下条件时账户可享受忠诚折扣：开户已满 "
                  f"{t_req} 个完整月（{plan_zh}账户可免开户时长要求）；"
                  f"近 12 个月消费不低于 ¥{s_req}；且没有任何一笔未付账单"
                  f"超过 {b_max} 天。“开户已满 N 个完整月”按 "
                  "`account_opened_on` 至 `today` 计算。未付账单账龄为 -1 "
                  "表示没有未付账单。",
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "按照 `policy`，该账户有资格获得忠诚折扣吗？"
                            "未能证实满足的条件视为不满足。",
            "criteria": {"true": "所有适用条件均满足",
                         "false": "至少有一个适用条件不满足"},
            "label": gold, "meta": {"family": "multi_hop"}}


def gen_process_chain_zh(rng: random.Random) -> dict:
    order = _rand_date(rng)
    hol = _holiday_set(rng, rng.randint(3, 6))
    k, v, r = (rng.choice([2, 3, 4]), rng.choice([1, 2, 3]),
               rng.choice([2, 3, 5]))
    ship = add_bd(order, k, hol)
    deliv = add_bd(ship, v, hol)
    post = add_bd(deliv, r, hol)
    target = post + timedelta(days=rng.choice([-2, -1, 0, 0, 1, 2]))
    gold = int(post <= target)
    state = {
        "order_placed_on": fmt_date_zh(order, rng),
        "process": f"订单在下单后恰好 {k} 个工作日发货；发货后恰好 {v} "
                   f"个工作日送达；送达后恰好 {r} 个工作日完成退款。"
                   "工作日为星期一至星期五，不含所列节假日。",
        "holidays": _holiday_str_zh(hol),
        "refund_needed_by": fmt_date_zh(target, rng),
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "严格按照 `process` 推算，退款会在 "
                            "`refund_needed_by` 当天或之前到账吗？",
            "criteria": {"true": "退款在该日期当天或之前到账",
                         "false": "退款在该日期之后到账"},
            "label": gold, "meta": {"family": "multi_hop"}}


def gen_fee_math_zh(rng: random.Random) -> dict:
    price = rng.choice([120, 260, 480, 900, 1500])
    warranty = price + rng.choice([-price // 3, -price // 10, 0,
                                   price // 10, price // 3])
    pct = rng.choice([0, 5, 10, 15])
    cap = rng.choice([200, 400, 1000])
    base_amt = min(price, warranty)
    after_pct = round(base_amt * (100 - pct) / 100, 2)
    refund = min(after_pct, cap)
    if rng.random() < 0.6:
        q_level = refund
        gold = 1
    else:
        q_level = refund + rng.choice([10, 50])
        gold = 0
    state = {
        "purchase_price_usd": price,
        "warranty_remaining_usd": warranty,
        "policy": "退款金额取购买价与剩余保修价值中的较低者，再从中扣除 "
                  f"{pct}% 的手续费，且退款总额不超过 ¥{cap}。",
        "question_amount_usd": q_level,
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "按照 `policy` 规定的计算顺序，退款金额"
                            "达到 `question_amount_usd` 了吗？",
            "criteria": {"true": "计算出的退款不低于该金额",
                         "false": "计算出的退款低于该金额"},
            "label": gold, "meta": {"family": "multi_hop"}}


def gen_deep_logic_zh(rng: random.Random) -> dict:
    conds = [
        ("理赔表单填写完整", "理赔表单不完整"),
        ("已附上购买凭证", "缺少购买凭证"),
        ("商品在保障期内", "商品已过保障期"),
        ("账户状态正常", "账户处于欠费状态"),
    ]
    rng.shuffle(conds)
    k = 3
    chosen = conds[:k]
    exception_applies = rng.random() < 0.5
    carveout_applies = exception_applies and rng.random() < 0.5
    base_ok = rng.random() < 0.7
    gold = int(base_ok and not (exception_applies and not carveout_applies))
    facts = [name for name, _ in chosen]
    if not base_ok:
        facts[rng.randrange(k)] = chosen[rng.randrange(k)][1]
    state = {
        "policy": "同时满足以下全部条件时，理赔予以赔付："
                   + "；".join(name for name, _ in chosen)
                   + "。例外：被标记为重复提交的理赔将被拒绝，"
                     "除非客户声明原始订单从未送达。",
        "situation": "。".join(facts) + "。"
                     + ("该理赔被标记为重复提交。" if exception_applies
                        else "该理赔未被标记为重复提交。")
                     + ("客户声明原始订单从未送达。" if carveout_applies
                        else ("客户未就原始订单作出任何声明。"
                              if exception_applies else "")),
    }
    return {"row_id": None, "primitive": "noul", "state": state,
            "instructions": "按照 `policy`，`situation` 中的理赔"
                            "可以获得赔付吗？",
            "criteria": {"true": "所有条件成立且没有例外适用",
                         "false": "有条件不成立或例外适用"},
            "label": gold, "meta": {"family": "multi_hop"}}


def gen_doc_conflict_zh(rng: random.Random) -> dict:
    general = rng.choice([3, 4, 5, 10])
    specific = rng.choice([1, 2, 3])
    while specific == general:
        specific = rng.choice([1, 2, 3])
    plan = rng.choice(["premium", "enterprise", "standard", "basic"])
    plan_zh = {"premium": "尊享", "enterprise": "企业",
              "standard": "标准", "basic": "基础"}[plan]
    if plan in ("premium", "enterprise"):
        gold = f"sla_{specific}_business_days"
    else:
        gold = f"sla_{general}_business_days"
    opts = [f"sla_{general}_business_days", f"sla_{specific}_business_days",
            "policy_silent"]
    state = {
        "clause_general": f"所有账户：客服在 {general} 个工作日内回复。",
        "clause_specific": f"{plan_zh}账户：客服在 {specific} 个工作日内回复。"
        if plan in ("premium", "enterprise") else
        f"尊享账户：客服在 {specific} 个工作日内回复。",
        "account": {"plan": plan_zh},
        "note": "当更具体的条款明确覆盖该账户时，其效力高于一般条款。",
    }
    return {"row_id": None, "primitive": "choice", "state": state,
            "instructions": "哪一项响应时限适用于该 `account`？",
            "criteria": None, "label": gold,
            "meta": {"options": opts, "family": "multi_hop"}}


def gen_eligibility_tree_zh(rng: random.Random) -> dict:
    t_req = rng.choice([12, 24, 36])
    s_req = rng.choice([500, 1000, 2500])
    c_req = rng.choice([1, 3, 5])
    names = ["tenure_requirement", "spend_requirement",
             "case_volume_requirement", "meets_all_requirements"]
    fail_idx = rng.choice([-1, -1, 0, 1, 2])
    today = _rand_date(rng)
    tenure = t_req + rng.choice([0, 1, 3])
    spend = s_req + rng.choice([0, 100, 500])
    cases = c_req + rng.choice([0, 1, 2])
    if fail_idx == 0:
        tenure = t_req - rng.choice([1, 2])
    elif fail_idx == 1:
        spend = s_req - rng.choice([50, 100])
    elif fail_idx == 2:
        cases = max(0, c_req - 1)
    joined = _shift_days_for_months(today, tenure)
    if fail_idx == -1:
        gold = "meets_all_requirements"
    else:
        gold = names[fail_idx]
    state = {
        "today": fmt_date_zh(today, rng),
        "requirements": {
            "tenure_requirement": f"自 `joined_on` 起已满 {t_req} 个完整月",
            "spend_requirement": f"近 12 个月消费不低于 ¥{s_req}",
            "case_volume_requirement": f"本季度提交工单不少于 {c_req} 件",
        },
        "application": {
            "joined_on": fmt_date_zh(joined, rng),
            "trailing_12m_spend_usd": spend,
            "cases_filed_this_quarter": cases,
        },
    }
    return {"row_id": None, "primitive": "choice", "state": state,
            "instructions": "该计划只有在 `requirements` 中的每一条都被 "
                            "`application` 满足时才授予访问权限。"
                            "`application` 未满足哪一条要求"
                            "——还是全部满足？",
            "criteria": None, "label": gold,
            "meta": {"options": names, "family": "multi_hop"}}


# ---------------------------------------------------------------- long_policy

def gen_long_policy_zh(rng: random.Random) -> list:
    plans = ["basic", "standard", "premium"]
    plan_zh = {"basic": "基础", "standard": "标准", "premium": "尊享"}
    wins = {"basic": rng.choice([14, 15, 21]),
            "standard": rng.choice([21, 30, 45]),
            "premium": rng.choice([45, 60, 90])}
    fees = {"basic": rng.choice([15, 20]), "standard": rng.choice([10, 15]),
            "premium": rng.choice([0, 5])}
    appeal_days = rng.choice([10, 14, 30])
    proc_days = rng.choice([3, 5, 7])
    today = _rand_date(rng)
    eff = _shift_days_for_months(today, rng.choice([2, 3, 6]))

    sec_scope = ("2. 适用范围。本政策覆盖美国、欧洲和亚太地区的基础、"
                 "标准、尊享套餐购买行为。内部测试账户不在覆盖范围内。")
    sec_windows = ("3. 退货窗口。基础套餐须在购买后 "
                   f"{wins['basic']} 天内发起退货，标准套餐为 "
                   f"{wins['standard']} 天，尊享套餐为 {wins['premium']} 天。"
                   "窗口按购买日起经过的自然日数计算（购买当日计为第 0 天）；"
                   "在第 N 天当天或之前发起的退货视为在窗口内。")
    sec_excl = ("4. 排除条款。最终清仓商品不可退货。数字商品一经打开下载"
                "链接即不可退货。因使用不当损坏的商品不可退货；但存在"
                "制造缺陷的商品不受窗口限制，始终可以退货。")
    sec_fees = ("5. 重新上架费。批准的退货将扣除重新上架费：基础套餐 "
                f"{fees['basic']}%，标准套餐 {fees['standard']}%，"
                f"尊享套餐 {fees['premium']}%。因制造缺陷引起的退货"
                "免收该费用。")
    sec_appeal = (f"6. 申诉。被拒绝的退货可在拒绝之日起 {appeal_days} "
                  "天内申诉。窗口按拒绝日起经过的自然日数计算（拒绝当日"
                  "计为第 0 天）；第 N 天当天或之前提交的申诉视为及时。"
                  "申诉由退款团队裁定。")
    sec_proc = (f"7. 处理。批准的退款将在批准后 {proc_days} 个工作日内"
                "发出。")
    decoy = ("8. 内部测试账户。内部测试账户可随时退货且免收任何费用。"
             "其他类型的账户不受本条影响。")
    middle = [sec_scope, sec_windows, sec_excl, sec_fees, sec_appeal,
              sec_proc]
    rng.shuffle(middle)
    doc = (f"供应商服务协议——退款政策（自 {fmt_date_zh(eff, rng)} 起生效）"
           "\n\n"
           "1. 目的。本文档规定何种购买可以退货以及退款如何计算。\n\n"
           + "\n\n".join(middle)
           + f"\n\n{decoy}\n\n"
             "9. 定义。工作日指星期一至星期五，不含公共节假日。制造缺陷"
             "指交付时已存在的故障。")
    doc_state = {"policy_document": doc, "today": fmt_date_zh(today, rng)}

    rows = []

    plan = rng.choice(plans)
    w = wins[plan]
    delta = rng.choice([w - 1, w, w, w + 1, w + 5])
    purchased = today - timedelta(days=delta)
    gold = int(delta <= w)
    rows.append({
        "row_id": None, "primitive": "noul",
        "state": {**doc_state,
                  "case": {"plan": plan_zh[plan],
                           "purchased_on": fmt_date_zh(purchased, rng),
                           "return_initiated_on": fmt_date_zh(today, rng)}},
        "instructions": "按照该政策，`case` 中的退货是否在对应套餐的"
                        "退货窗口内？",
        "criteria": {"true": "退货在对应套餐的窗口内",
                     "false": "退货超出了对应套餐的窗口"},
        "label": gold, "meta": {"family": "long_policy"},
    })

    plan2 = rng.choice(plans)
    defect = rng.random() < 0.35
    gold2 = "no_fee" if defect else f"fee_{fees[plan2]}_percent"
    opts = ["no_fee", f"fee_{fees['basic']}_percent",
            f"fee_{fees['standard']}_percent",
            f"fee_{fees['premium']}_percent"]
    opts = list(dict.fromkeys(opts))
    rows.append({
        "row_id": None, "primitive": "choice",
        "state": {**doc_state,
                  "case": {"plan": plan_zh[plan2],
                           "reason": "制造缺陷" if defect else "不想要了",
                           "item_condition": "未使用，原包装完好"}},
        "instructions": "按照该政策，这笔退货适用哪一档重新上架费？",
        "criteria": None, "label": gold2,
        "meta": {"options": opts, "family": "long_policy"},
    })

    denied = today - timedelta(days=rng.choice(
        [appeal_days - 1, appeal_days, appeal_days, appeal_days + 1,
         appeal_days + 3]))
    gold3 = int((today - denied).days <= appeal_days)
    rows.append({
        "row_id": None, "primitive": "noul",
        "state": {**doc_state,
                  "case": {"denial_date": fmt_date_zh(denied, rng),
                           "appeal_filed_on": fmt_date_zh(today, rng)}},
        "instructions": "按照该政策，`case` 中的申诉是否及时？",
        "criteria": {"true": "申诉在申诉窗口内", "false": "申诉超时"},
        "label": gold3, "meta": {"family": "long_policy"},
    })

    if rng.random() < 0.6:
        plan4 = rng.choice(plans)
        w4 = wins[plan4]
        over = rng.choice([0, 0, 1, 2, 5, 10, 20])
        purchased4 = today - timedelta(days=w4 + over)
        if over == 0:
            gold4 = 0
        elif over <= 3:
            gold4 = 1
        elif over <= 14:
            gold4 = 2
        else:
            gold4 = 3
        rows.append({
            "row_id": None, "primitive": "score",
            "state": {**doc_state,
                      "case": {"plan": plan_zh[plan4],
                               "purchased_on": fmt_date_zh(purchased4, rng),
                               "return_initiated_on": fmt_date_zh(today, rng)},
                      "severity_scale": "按政策自身窗口条款划分的严重度："
                                        "无——在对应套餐窗口内；轻微——"
                                        "超期 1-3 天；严重——超期 4-14 天；"
                                        "极严重——超期 14 天以上。"},
            "instructions": "按照 `severity_scale` 的分档，这笔退货相对"
                            "其套餐窗口的超期有多严重？",
            "criteria": None, "label": gold4,
            "meta": {"attribute": "window_violation_severity",
                     "family": "long_policy"},
        })
    return rows


# ----------------------------------------------------------------------- cli

def _status_descs_zh() -> dict:
    return {
        "overdue": "到期日早于今天",
        "due_today": "到期日恰为今天",
        "due_within_7_days": "到期日在今天之后 1-7 天内",
        "due_later": "到期日在今天之后 7 天以上",
        "sla_1_business_days": "客服在 1 个工作日内回复",
        "sla_2_business_days": "客服在 2 个工作日内回复",
        "sla_3_business_days": "客服在 3 个工作日内回复",
        "sla_4_business_days": "客服在 4 个工作日内回复",
        "sla_5_business_days": "客服在 5 个工作日内回复",
        "sla_10_business_days": "客服在 10 个工作日内回复",
        "policy_silent": "政策未对该类账户作出承诺",
        "tenure_requirement": "未满足开户时长要求",
        "spend_requirement": "未满足消费金额要求",
        "case_volume_requirement": "未满足工单数量要求",
        "meets_all_requirements": "所有要求均已满足",
        "no_fee": "不收取重新上架费",
        "fee_0_percent": "适用 0% 的重新上架费",
        "fee_5_percent": "适用 5% 的重新上架费",
        "fee_10_percent": "适用 10% 的重新上架费",
        "fee_15_percent": "适用 15% 的重新上架费",
        "fee_20_percent": "适用 20% 的重新上架费",
    }


def _score_levels_zh() -> dict:
    return {
        "deadline_urgency": [
            "紧急——截止日为今天或已经过去",
            "高——距截止日还有 1-2 天",
            "中——距截止日还有 3-7 天",
            "低——距截止日还有 7 天以上",
        ],
        "window_violation_severity": [
            "无——退货在对应套餐窗口内",
            "轻微——退货超期 1-3 天",
            "严重——退货超期 4-14 天",
            "极严重——退货超期 14 天以上",
        ],
    }


GENS_ZH = [
    (gen_date_arith_zh, 500),
    (gen_business_days_zh, 400),
    (gen_deadline_bounds_zh, 500),
    (gen_overlap_zh, 350),
    (gen_month_arith_zh, 300),
    (gen_status_choice_zh, 400),
    (gen_urgency_score_zh, 300),
    (gen_chain_rules_zh, 450),
    (gen_process_chain_zh, 350),
    (gen_fee_math_zh, 400),
    (gen_deep_logic_zh, 400),
    (gen_doc_conflict_zh, 250),
    (gen_eligibility_tree_zh, 300),
    (gen_long_policy_zh, 1200),
]


def _selfcheck() -> int:
    from datetime import date as D
    import random as R
    # the imported gold logic is already selfchecked by synth2; here we
    # verify the Chinese surfaces only.
    r = R.Random(3)
    got = set()
    for _ in range(40):
        got.add(fmt_date_zh(D(2025, 3, 9), r))
    assert any("星期日" in s for s in got), got  # 2025-03-09 is a Sunday
    assert any("2025年03月09日" in s for s in got)
    for gen, _ in GENS_ZH:
        for seed in (0, 1):
            g1 = gen(R.Random(seed))
            g2 = gen(R.Random(seed))
            l1 = g1 if isinstance(g1, list) else [g1]
            l2 = g2 if isinstance(g2, list) else [g2]
            strip = lambda rs: [{k: v for k, v in row.items()
                                 if k != "row_id"} for row in rs]
            assert strip(l1) == strip(l2), f"{gen.__name__} not deterministic"
            for row in l1:
                assert "family" in row["meta"]
                if row["primitive"] == "noul":
                    assert set(row["criteria"]) == {"true", "false"}
    # label balance sanity: date_arith produces both polarities across seeds
    r = R.Random(9)
    labs = [gen_date_arith_zh(r)["label"] for _ in range(200)]
    assert 0.25 < sum(labs) / len(labs) < 0.75
    print("selfcheck: all zh generators hold their invariants and are "
          "deterministic")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synth", "selfcheck"])
    ap.add_argument("--n", type=int, default=4200)
    ap.add_argument("--out", default="")
    ap.add_argument("--criteria-out", default="")
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args(argv)

    if args.cmd == "selfcheck":
        return _selfcheck()
    assert args.out, "--out is required for synth"

    rng = random.Random(args.seed)
    rows: list[dict] = []
    total_quota = sum(q for _, q in GENS_ZH)
    for gen, quota in GENS_ZH:
        take = round(quota * args.n / total_quota)
        produced = 0
        while produced < take:
            out = gen(rng)
            if isinstance(out, list):
                rows.extend(out)
                produced += len(out)
            else:
                rows.append(out)
                produced += 1
    for i, r in enumerate(rows):
        r["row_id"] = f"synth2zh-{i:05d}"
        if r["primitive"] == "choice":
            r["label"] = r["meta"]["options"].index(r["label"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    fam = Counter(r["meta"]["family"] for r in rows)
    prim = Counter(r["primitive"] for r in rows)
    pos = [r["label"] for r in rows if r["primitive"] == "noul"]
    avg_len = sum(len(json.dumps(r["state"])) for r in rows) / len(rows)
    print(f"synth2zh: wrote {len(rows)} rows -> {args.out}")
    print(f"  families: {dict(fam)}")
    print(f"  primitives: {dict(prim)}")
    print(f"  noul positive rate: {round(sum(pos) / len(pos), 3)}")
    print(f"  avg state bytes: {avg_len:.0f}")

    if args.criteria_out:
        crit = _status_descs_zh()
        crit.update(_score_levels_zh())
        Path(args.criteria_out).write_text(
            json.dumps(crit, indent=2, ensure_ascii=False))
        print(f"wrote {args.criteria_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())