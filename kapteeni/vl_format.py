"""Kapteeni-V shared format: the one-pass lettered decision prompt.

Used by the V0 probe, V2 training, gate evaluation, and (later) serving,
so every stage sees byte-identical prompts.

Layout of one pass (the image enters through the chat template's vision
placeholder; its textual slot is noted in the state):

    <state json text>

    [noul] <instructions>
    criteria - true: ... | false: ...
    answer:

    [choice] <instructions>          (score likewise, with levels)
    A. <option 1>
    B. <option 2>
    ...
    answer:

Answer token: "yes"/"no" for noul; " A", " B", ... (leading space) for
options/levels. Options are subsampled to <= 16 per question,
deterministically per row (gold always kept), mirroring the text
pipeline's choice-subset logic.
"""

from __future__ import annotations

import hashlib
import json
import random

from kapteeni.serialize import state_text

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_OPTS = 16


def lettered_lines(options: list) -> str:
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def build_vl_prompt(state, primitive: str, instructions, criteria=None,
                    options=None, levels=None) -> str:
    st = state_text(state)
    if primitive == "noul":
        body = (f"[noul] {instructions}\n"
                f"criteria - true: {criteria['true']} | "
                f"false: {criteria['false']}\nanswer:")
    else:
        opts = options if primitive == "choice" else levels
        assert opts, "choice/score needs options/levels"
        assert len(opts) <= MAX_OPTS, f"{len(opts)} options > {MAX_OPTS}"
        body = (f"[{primitive}] {instructions}\n"
                f"{lettered_lines(opts)}\nanswer:")
    return st + "\n\n" + body


def answer_token(primitive: str, label: int) -> str:
    """The supervised target token for a gold label."""
    if primitive == "noul":
        return "yes" if label == 1 else "no"
    return " " + LETTERS[label]


def option_subset(options: list, gold: int, max_opts: int = MAX_OPTS,
                  row_id: str = "") -> tuple[list, int]:
    """Gold + deterministic distractors (the text pipeline's rule)."""
    if len(options) <= max_opts:
        return options, gold
    rng = random.Random(int(hashlib.sha256(row_id.encode())
                             .hexdigest()[:8], 16))
    keep = {gold} | set(rng.sample([i for i in range(len(options))
                                    if i != gold], max_opts - 1))
    idxs = sorted(keep)
    return [options[i] for i in idxs], idxs.index(gold)


# ----------------------------------------------------------------- examples

def synth3_example(row: dict, img_dir: str) -> dict:
    """A synth3 row -> one training/eval example."""
    prompt = build_vl_prompt(
        {"image": "<attached>", "note": row["state_text"]},
        row["primitive"], row["instructions"],
        criteria=row.get("criteria"), options=row.get("options"),
        levels=row.get("levels"))
    return {"prompt": prompt, "image": f"{img_dir}/{row['image']}",
            "answer": answer_token(row["primitive"], row["label"]),
            "primitive": row["primitive"], "gold": row["label"],
            "n_cands": (1 if row["primitive"] == "noul" else
                        len(row.get("options") or row["levels"])),
            "source": "synth3", "row_id": row["row_id"],
            "family": row["meta"]["family"]}


def text_example(row: dict, criteria_map: dict,
                 max_opts: int = MAX_OPTS) -> dict:
    """A text pipeline row (rows_*.jsonl) -> one lettered example."""
    prim = row["primitive"]
    if prim == "noul":
        prompt = build_vl_prompt(row["state"], "noul", row["instructions"],
                                  criteria=row["criteria"])
        n = 1
    elif prim == "choice":
        opts, gold = option_subset(row["meta"]["options"], row["label"],
                                   max_opts, row["row_id"])
        descs = [criteria_map.get(o, o) for o in opts]
        prompt = build_vl_prompt(row["state"], "choice",
                                 row["instructions"], options=descs)
        row = dict(row, label=gold)
        n = len(opts)
    else:
        levels = criteria_map[row["meta"]["attribute"]]
        prompt = build_vl_prompt(row["state"], "score",
                                  row["instructions"], levels=levels)
        n = len(levels)
    return {"prompt": prompt, "image": None,
            "answer": answer_token(prim, row["label"]),
            "primitive": prim, "gold": row["label"], "n_cands": n,
            "source": "text", "row_id": row["row_id"], "family": None}


def load_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]