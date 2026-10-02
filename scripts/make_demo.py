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

def img_case(cid, title, blurb, group, specs, seed_note):
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


def main() -> int:
    cases = build_image_cases() + build_text_cases()
    (DEMO / "images").mkdir(parents=True, exist_ok=True)
    (DEMO / "cases.json").write_text(json.dumps(
        {"model": MODEL, "cases": cases}, ensure_ascii=False, indent=1))
    print(f"{len(cases)} cases ({sum(1 for c in cases if c['image'])} with "
          f"images) -> {DEMO}/cases.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())