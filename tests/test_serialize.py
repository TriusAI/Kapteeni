import json

from kapteeni.serialize import (
    choice_option_pass,
    instructions_text,
    noul_pass,
    score_level_pass,
    state_text,
)


def test_state_string_passthrough():
    assert state_text("hello") == "hello"


def test_state_object_becomes_json_text_with_field_names():
    s = state_text({"ticket": {"subject": "Duplicate charge"}})
    assert json.loads(s) == {"ticket": {"subject": "Duplicate charge"}}
    assert "ticket" in s and "subject" in s


def test_noul_pass_layout():
    text = noul_pass(
        {"claim": "A", "evidence": "B"},
        "Is `claim` supported by `evidence`?",
        {"true": "supported", "false": "contradicted"},
    )
    assert text.startswith('{"claim": "A", "evidence": "B"}')
    assert "[noul] Is `claim` supported by `evidence`?" in text
    assert "criteria - true: supported | false: contradicted" in text
    assert text.endswith("answer:")


def test_noul_pass_without_criteria():
    text = noul_pass("state", "Is this true?")
    assert "criteria" not in text


def test_choice_option_pass_focuses_one_option():
    text = choice_option_pass({"query": "I lost my card"}, "What is the intent?", "card_lost", "User reports a lost or stolen card")
    assert "[choice] What is the intent?" in text
    assert "option - card_lost: User reports a lost or stolen card" in text


def test_choice_null_description_ok():
    text = choice_option_pass("s", "q", "other", None)
    assert "option - other\n" in text or "option - other" in text
    assert ":" not in text.split("option - other")[1].split("\n")[0]


def test_score_level_pass_numbered_and_described():
    text = score_level_pass("essay", "Rate the quality", 2, 4, "solid work")
    assert "[score] Rate the quality" in text
    assert "level 3 of 4 - solid work" in text
    assert text.endswith("answer:")


def test_structured_instructions_prefer_question_key():
    t = instructions_text({"question": "Is it the same person?", "record": {"name": "J"}})
    assert t.startswith("Is it the same person?")
    assert "record" in t  # data rides along


def test_array_instructions_are_json():
    assert instructions_text(["multi", "part"]) == json.dumps(["multi", "part"])