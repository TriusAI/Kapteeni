import json

from kapteeni.build_data import expand_passes, is_val
from kapteeni.distill import judge_prompt
from kapteeni.ollama import parse_json_loose

OPTS = ["card_lost", "card_stolen", "transfer", "balance", "statement", "atm"]


def _noul_row(rid, label=1):
    return {
        "row_id": rid, "primitive": "noul",
        "state": {"claim": "A", "evidence": "B"},
        "instructions": "Is `claim` supported by `evidence`?",
        "criteria": {"true": "supported", "false": "contradicted"},
        "label": label,
    }


def _choice_row(rid, label=1, opts=None):
    return {
        "row_id": rid, "primitive": "choice",
        "state": {"query": "I lost my card"},
        "instructions": "What is the customer's intent?",
        "criteria": None, "label": label,
        "meta": {"options": opts or OPTS},
    }


def _score_row(rid, label=2):
    return {
        "row_id": rid, "primitive": "score",
        "state": {"response": "text"},
        "instructions": "Rate the response",
        "criteria": None, "label": label,
        "meta": {"attribute": "helpfulness"},
    }


def test_is_val_is_deterministic_and_balanced():
    ids = [f"boolq-{i}" for i in range(500)]
    assert [is_val(i) for i in ids] == [is_val(i) for i in ids]
    frac = sum(is_val(i) for i in ids) / len(ids)
    assert 0.05 < frac < 0.2


def test_noul_train_uses_soft_label_val_uses_gold():
    soft = {"boolq-5": {"p": 0.72, "weight": 0.8}}
    rows = [_noul_row("boolq-5", label=1)]
    # force train: pick an id that is not val
    rid = next(i for i in (f"boolq-{j}" for j in range(500)) if not is_val(i))
    rows = [_noul_row(rid, label=1), _noul_row("boolq-5", label=1)]
    soft = {rid: {"p": 0.72, "weight": 0.8}}
    passes = expand_passes(rows, {}, soft)
    tr = [p for p in passes if p["row_id"] == rid][0]
    assert tr["target"] == 0.72 and tr["weight"] == 0.8 and tr["gold"] == 1.0
    va = [p for p in passes if p["row_id"] == "boolq-5"][0]
    assert va["target"] == 1.0 and va["gold"] == 1.0  # val: gold target


def test_choice_train_subset_keeps_gold_and_caps():
    rid = next(i for i in (f"bank77-{j}" for j in range(500)) if not is_val(i))
    passes = expand_passes([_choice_row(rid, label=2)], {}, {}, opts_per_row=4)
    assert len(passes) == 4
    gold_present = any(p["target"] == 1.0 for p in passes)
    assert gold_present
    assert sum(p["target"] for p in passes) == 1.0


def test_choice_val_gets_full_option_set():
    rid = next(i for i in (f"bank77-{j}" for j in range(500)) if is_val(i))
    passes = expand_passes([_choice_row(rid, label=1)], {}, {}, opts_per_row=4)
    assert len(passes) == len(OPTS)  # all options, not capped


def test_score_expands_per_level_one_hot():
    rid = next(i for i in (f"help-{j}" for j in range(500)) if not is_val(i))
    levels = ["awful", "weak", "fine", "good", "great"]
    passes = expand_passes([_score_row(rid, label=2)],
                           {"helpfulness": levels}, {})
    assert len(passes) == 5
    assert [p["target"] for p in passes] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert passes[2]["key"] == 2
    assert "level 3 of 5 - fine" in passes[2]["text"]


def test_choice_pass_carries_criteria_text():
    rid = next(i for i in (f"bank77-{j}" for j in range(500)) if not is_val(i))
    crit = {"transfer": "Customer wants to move money between accounts"}
    passes = expand_passes([_choice_row(rid, label=2)], crit, {}, opts_per_row=6)
    tr = [p for p in passes if p["key"] == "transfer"][0]
    assert "Customer wants to move money between accounts" in tr["text"]


def test_judge_prompt_contains_state_and_question():
    p = judge_prompt({"claim": "A"}, "Is it true?", {"true": "yes"})
    assert '"claim": "A"' in p
    assert "Is it true?" in p
    assert '{"p": <0..1>}' in p


def test_parse_json_loose_fences():
    assert parse_json_loose('{"p": 0.5}') == {"p": 0.5}
    assert parse_json_loose('```json\n{"p": 0.5}\n```') == {"p": 0.5}
    assert parse_json_loose('Sure! ```\n{"p": 0.5}\n``` hope that helps') == {"p": 0.5}
    assert parse_json_loose('garbage') is None
    assert parse_json_loose('[1,2]') is None