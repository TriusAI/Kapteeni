import random

import pytest

from kapteeni.vl_format import (
    MAX_OPTS,
    answer_token,
    build_vl_prompt,
    lettered_lines,
    option_subset,
    synth3_example,
    text_example,
)


def test_lettered_lines_and_prompt_shapes():
    assert lettered_lines(["a", "b"]) == "A. a\nB. b"
    p = build_vl_prompt({"s": "x"}, "noul", "Is it?",
                        criteria={"true": "t", "false": "f"})
    assert "[noul] Is it?" in p and "criteria - true: t" in p
    assert p.endswith("answer:")
    p = build_vl_prompt({"s": "x"}, "choice", "Which?", options=["u", "v"])
    assert "A. u\nB. v" in p
    with pytest.raises(AssertionError):
        build_vl_prompt({"s": "x"}, "choice", "Which?",
                        options=[str(i) for i in range(MAX_OPTS + 1)])


def test_answer_tokens():
    assert answer_token("noul", 1) == "yes"
    assert answer_token("noul", 0) == "no"
    assert answer_token("choice", 0) == " A"
    assert answer_token("score", 7) == " H"


def test_option_subset_deterministic_and_keeps_gold():
    opts = list(range(100))
    a = option_subset(opts, 42, 8, "rid-1")
    b = option_subset(opts, 42, 8, "rid-1")
    assert a == b
    keep, new_gold = a
    assert len(keep) == 8 and keep[new_gold] == 42
    c = option_subset(opts, 42, 8, "rid-2")
    assert c != a  # different row -> different distractors
    small, gold = option_subset([1, 2, 3], 1, 8, "x")
    assert small == [1, 2, 3] and gold == 1


def _synth3_row(prim):
    base = {"state_text": "A render.", "instructions": "Q?",
            "meta": {"family": "menu_board"}, "row_id": "r1"}
    if prim == "noul":
        return {**base, "primitive": "noul", "image": "x.png",
                "criteria": {"true": "t", "false": "f"}, "label": 1}
    if prim == "choice":
        return {**base, "primitive": "choice", "image": "x.png",
                "options": ["a", "b", "c"], "label": 2}
    return {**base, "primitive": "score", "image": "x.png",
            "levels": ["lo", "mid", "hi"], "label": 0}


@pytest.mark.parametrize("prim", ["noul", "choice", "score"])
def test_synth3_example(prim):
    ex = synth3_example(_synth3_row(prim), "imgdir")
    assert ex["image"] == "imgdir/x.png"
    assert ex["source"] == "synth3"
    assert ex["answer"] in ("yes", "no", " A", " B", " C")
    assert ex["prompt"].endswith("answer:")


def test_text_example_noul_and_choice():
    r = {"row_id": "b-1", "primitive": "noul", "state": {"q": "hi"},
         "instructions": "Yes?", "criteria": {"true": "t", "false": "f"},
         "label": 1}
    ex = text_example(r, {})
    assert ex["image"] is None and ex["answer"] == "yes"
    r = {"row_id": "c-1", "primitive": "choice", "state": {"q": "hi"},
         "instructions": "Which?", "label": 3,
         "meta": {"options": [f"o{i}" for i in range(40)]}}
    ex = text_example(r, {"o3": "desc3"})
    assert ex["n_cands"] == MAX_OPTS
    assert ex["gold"] < MAX_OPTS
    # the gold description is among the lettered options
    assert "desc3" in ex["prompt"]
    # deterministic: same row -> same example
    assert text_example(r, {"o3": "desc3"}) == ex