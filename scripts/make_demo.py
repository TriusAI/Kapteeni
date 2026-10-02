"""Builds the kapteeni-v1.1c demo website's pre-configured cases.

Every case is generated FRESH from the repo's own gold-by-construction
generators (deterministic seeds; no train/val row is reused — the demo
never touches the held-out gate slices). The sampled facts drive both
the rendered document and the gold answer, so each case carries a
trustworthy gold annotation that the page shows next to the model's
distribution.

    PYTHONPATH=. python3 scripts/make_demo.py

Output: kapteeni/demo/{images/*.png, cases.json}
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kapteeni import synth2, synth2zh  # noqa: E402
from kapteeni import (synth3_families, synth3_families_v,  # noqa: E402
                      synth3_families_zh)

DEMO = ROOT / "kapteeni" / "demo"
CRIT2 = json.loads((ROOT / "data_cache" / "synth2_criteria.json").read_text())
CRIT2ZH = json.loads((ROOT / "data_cache" / "synth2zh_criteria.json").read_text())
MODEL = "kapteeni-v1.1c"


# --------------------------------------------------------------- image cases

def img_case(cid, title, blurb, group, specs, seed_note, gold_label="construction"):
    """group: a gen_* output dict {img, rid, rows}; specs: [(row, qid)]."""
    (DEMO / "images").mkdir(parents=True, exist_ok=True)
    group["img"].save(DEMO / "images" / f"{cid}.png")
    questions, gold = {}, {}
    for row, qid in specs:
        q = {"noul": lambda r: {"type": "noul",
                                "instructions": r["instructions"],
                                "criteria": r["criteria"]},
             "choice": lambda r: {"type": "choice",
                                  "instructions": r["instructions"],
                                  "criteria": {o: None for o in r["options"]}},
             "score": lambda r: {"type": "score",
                                 "instructions": r["instructions"],
                                 "criteria": r["levels"]}}[row["primitive"]](row)
        questions[qid] = q
        if row["primitive"] == "noul":
            gold[qid] = "yes" if row["label"] == 1 else "no"
        elif row["primitive"] == "choice":
            gold[qid] = row["options"][row["label"]]
        else:
            gold[qid] = str(row["levels"][row["label"]])
    return {
        "id": cid, "title": title, "blurb": blurb,
        "family": group["rows"][0]["meta"]["family"],
        "image": f"images/{cid}.png", "gold": gold,
        "gold_label": "construction",
        "request": {
            "state": {"image": f"@file:images/{cid}.png",
                      "note": group["rows"][0]["state_text"]},
            "model": MODEL, "questions": questions,
        },
        "provenance": seed_note,
    }


def build_image_cases():
    cases = []
    g = synth3_families.gen_menu(random.Random(1101))[0]
    rows = g["rows"]
    ch = next(r for r in rows if r["primitive"] == "choice"
              and "expensive" in r["instructions"])
    ch2 = next(r for r in rows if r["primitive"] == "choice"
               and "cheapest" in r["instructions"])
    no = next(r for r in rows if r["primitive"] == "noul")
    cases.append(img_case(
        "menu-en", "Cafe menu — one image, three questions",
        "One attached image, three independent questions (two choice, one "
        "noul). Independence is structural: every question is judged in "
        "its own pass group against the same state, so adding or removing "
        "a question never changes another answer.",
        g, [(ch, "priciest"), (ch2, "cheapest"), (no, "price_gate")],
        "synth3 menu_board, seed 1101 — fresh render, not a dataset row"))

    g = synth3_families.gen_invoice(random.Random(1102))[0]
    no = next(r for r in g["rows"] if r["primitive"] == "noul")
    cases.append(img_case(
        "invoice-en", "Invoice — noul over a document",
        "A yes/no judgment over a rendered invoice. Noul is ABSOLUTE "
        "probability (not a complement-consistent pair): P(true) is what "
        "your code branches on.",
        g, [(no, "invoice_check")],
        "synth3 invoice_receipt, seed 1102"))

    g = synth3_families_v.gen_seating(random.Random(1103))[0]
    ch = next(r for r in g["rows"] if r["primitive"] == "choice")
    cases.append(img_case(
        "seating-en", "Seating chart — choice over positions",
        "A relative decision over a visual layout: probabilities over the "
        "seat options sum to exactly 1, with the contract's confidence "
        "derived from the distribution's shape.",
        g, [(ch, "seat")],
        "synth3 seating_map, seed 1103"))

    g = synth3_families_v.gen_weather(random.Random(1104))[0]
    sc = next(r for r in g["rows"] if r["primitive"] == "score")
    cases.append(img_case(
        "weather-en", "Weather forecast — scored levels",
        "A scored question: each level is judged independently (does the "
        "state match this level?), normalized at the API layer; the score "
        "is the 0-based expectation and can be fractional.",
        g, [(sc, "days")],
        "synth3 weather, seed 1104"))

    g = synth3_families_zh.gen_menu_zh(random.Random(1105))[0]
    ch = next(r for r in g["rows"] if r["primitive"] == "choice")
    cases.append(img_case(
        "menu-zh", "Chinese menu — 中文图像决策",
        "同一张图像、中文的问题与选项 — the trained answer space stays "
        "yes/no + option names in every language; Chinese enters through "
        "the state, instructions, and criteria, never the answer tokens.",
        g, [(ch, "最贵")],
        "synth3zh menu_board, seed 1105"))
    return cases


# ---------------------------------------------------------------- text cases

def text_case(cid, title, blurb, row, crit_map, seed_note):
    qid = "q"
    lab = row["label"]  # synth2 labels: int index or the option name
    if row["primitive"] == "noul":
        question = {"type": "noul", "instructions": row["instructions"],
                     "criteria": row["criteria"]}
        gold = "yes" if int(lab) == 1 else "no"
    elif row["primitive"] == "choice":
        opts = row["meta"]["options"]
        question = {"type": "choice", "instructions": row["instructions"],
                    "criteria": {o: crit_map.get(o) for o in opts}}
        gold = lab if isinstance(lab, str) else opts[lab]
    else:
        levels = crit_map[row["meta"]["attribute"]]
        question = {"type": "score", "instructions": row["instructions"],
                    "criteria": levels}
        gold = str(levels[lab])
    return {
        "id": cid, "title": title, "blurb": blurb,
        "family": row["meta"]["family"], "image": None,
        "gold": {qid: gold},
        "request": {"state": row["state"], "model": MODEL,
                    "questions": {qid: question}},
        "provenance": seed_note,
    }


def build_text_cases():
    cases = []
    cases.append(text_case(
        "temporal-en", "Date arithmetic — text noul",
        "The temporal_numeric skill family: exact date arithmetic with "
        "weekday-consistent trap dates. No image — decisions over "
        "structured state alone.",
        synth2.gen_date_arith(random.Random(1201)), CRIT2,
        "synth2 temporal_numeric, seed 1201"))

    cases.append(text_case(
        "eligibility-en", "Eligibility chain — multi-step text choice",
        "The multi_hop family: three requirements, one question — which "
        "single requirement fails (or none)? Each option is judged "
        "against the state in its own pass; the softmax over the group is "
        "shaped in-loss during training.",
        synth2.gen_eligibility_tree(random.Random(1202)), CRIT2,
        "synth2 multi_hop, seed 1202"))

    cases.append(text_case(
        "urgency-en", "Deadline urgency — scored text",
        "A scored decision with explicitly binned levels stated in the "
        "state; gold is the bin the deadline falls into.",
        synth2.gen_urgency_score(random.Random(1203)), CRIT2,
        "synth2 temporal_numeric (score), seed 1203"))

    cases.append(text_case(
        "temporal-zh", "中文日期运算 — text noul",
        "同一技能家族的中文版本：中文日期格式、中文指令与判据 — the "
        "readout heads are language-agnostic by construction.",
        synth2zh.gen_date_arith_zh(random.Random(1204)), CRIT2ZH,
        "synth2zh temporal_numeric, seed 1204"))
    return cases


# -------------------------------------------------- real photographs
# (the user's own photos; hand-labeled — NO gold-by-construction claim.
# The model trained solely on synthetic renders, so these are honest
# out-of-distribution probes shown as such on the page. The downscaled
# copies in demo/images/ are committed; regeneration only refreshes the
# downscale when the originals are present in the repo root.)

REAL_PHOTOS = (  # (repo-root original, demo copy, longest edge)
    ("sign_test_1.jpg", "images/sign-1.jpg"),
    ("sign_test_2.jpg", "images/sign-2.jpg"),
    ("sign_test_3.png", "images/sign-3.jpg"),
)


def refresh_real_images():
    from PIL import Image
    (DEMO / "images").mkdir(parents=True, exist_ok=True)
    for src_name, dst_rel in REAL_PHOTOS:
        dst = DEMO / dst_rel
        if dst.exists():
            continue  # committed copy present; regeneration-safe
        src = ROOT / src_name
        img = Image.open(src)
        img = img.convert("RGB")
        w, h = img.size
        s = 640 / max(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        img.save(dst, "JPEG", quality=88)


def real_case(cid, title, blurb, image_rel, note, questions, gold,
              provenance):
    return {
        "id": cid, "title": title, "blurb": blurb, "family": "real_photo",
        "image": image_rel, "gold": gold, "gold_label": "hand",
        "request": {
            "state": {"image": f"@file:{image_rel}", "note": note},
            "model": MODEL, "questions": questions,
        },
        "provenance": provenance,
    }


def build_real_cases():
    cases = []
    # 1 — the horses sign (English, on a green palisade fence)
    cases.append(real_case(
        "horses-sign", "Warning sign, real photograph (OOD probe)",
        "A real photograph — the model trained ONLY on synthetic renders, "
        "so this is out-of-distribution by construction: watch how "
        "confident the calibration stays (or doesn't) on pixels it has "
        "never seen the like of. Gold below is hand-read from the photo.",
        "images/sign-1.jpg", "A red-lettered warning sign mounted on a "
        "green metal palisade fence.",
        {"happens": {"type": "choice",
                     "instructions": "What does the sign say will happen "
                                     "to horses found on these lands?",
                     "criteria": {
                         "found horses are impounded": None,
                         "found horses are sold at auction": None,
                         "horse riding is welcome on these lands": None,
                         "found horses are put up for adoption": None}},
         "fence": {"type": "noul",
                   "instructions": "Is the sign mounted on a green metal "
                                   "fence?",
                   "criteria": {"true": "the sign is mounted on a green "
                                        "metal fence",
                                "false": "the sign is mounted on "
                                         "something else"}}},
        {"happens": "found horses are impounded", "fence": "yes"},
        "real photograph, hand-checked (never in any training split)"))
    # 2 — the church hours sign (Chinese)
    cases.append(real_case(
        "church-zh", "教堂告示，真实照片（中文 OOD 探测）",
        "一张真实的照片：训练只用过合成渲染，因此这类真实照片在 "
        "构造上就是分布外样本——观察模型在从未见过的真实像素上的"
        "校准表现。Gold 为人工从照片读出（hand-checked）。",
        "images/sign-2.jpg", "一张教堂门口的弥撒与参观时间告示。",
        {"closed_day": {"type": "choice",
                        "instructions": "按照告示，教堂哪一天休息、"
                                        "不对外开放？",
                        "criteria": {"星期一": None, "星期日": None,
                                     "星期六": None, "星期二": None}},
         "sun_mass": {"type": "noul",
                      "instructions": "按照告示，主日弥撒时间包含"
                                      "下午15:00，对吗？",
                      "criteria": {"true": "主日弥撒包含下午15:00",
                                   "false": "主日弥撒不包含下午15:00"}}},
        {"closed_day": "星期一", "sun_mass": "yes"},
        "真实照片，人工核对（从未进入任何训练划分）"))
    # 3 — the delivery-app screenshot (zh + Uyghur script)
    cases.append(real_case(
        "app-zh", "外卖截图，真实照片（最分布外的一种输入）",
        "手机点餐页面的截图——不是文档、不是街景。物品名混排中文与"
        "维吾尔文。与上两例一样是分布外探测；gold 为人工从截图读出。",
        "images/sign-3.jpg", "一份手机外卖点餐页面（烤肉类）的截图。",
        {"price_1133": {"type": "choice",
                        "instructions": "截图中哪个商品的价格是 ¥11.33？",
                        "criteria": {"鸭肠": None, "烤鸡中翅": None,
                                     "烤香芋": None, "牛板筋": None}},
         "price_compare": {"type": "noul",
                           "instructions": "截图中「烤香芋」的价格比"
                                           "「烤鸡中翅」低，对吗？",
                           "criteria": {"true": "烤香芋价格更低",
                                        "false": "烤香芋价格不低于"
                                                 "烤鸡中翅"}}},
        {"price_1133": "鸭肠", "price_compare": "yes"},
        "真实照片（截图），人工核对（从未进入任何训练划分）"))
    return cases


def main() -> int:
    refresh_real_images()
    cases = build_image_cases() + build_real_cases() + build_text_cases()
    (DEMO / "images").mkdir(parents=True, exist_ok=True)
    (DEMO / "cases.json").write_text(json.dumps(
        {"model": MODEL, "cases": cases}, ensure_ascii=False, indent=1))
    print(f"{len(cases)} cases ({sum(1 for c in cases if c['image'])} with "
          f"images; {sum(1 for c in cases if c.get('gold_label') == 'hand')} "
          f"real-photo hand-labeled) -> {DEMO}/cases.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())