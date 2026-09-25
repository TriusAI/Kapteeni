import json

from kapteeni.familyval import build_questions, score_row


def _noul_row(label=1):
    return {"row_id": "x", "primitive": "noul", "state": {"a": "b"},
            "instructions": "Is `a` true?", "criteria": {"true": "t",
                                                         "false": "f"},
            "label": label, "meta": {"family": "temporal_numeric"}}


def _choice_row(label=1):
    return {"row_id": "x", "primitive": "choice", "state": {"a": "b"},
            "instructions": "Which?", "criteria": None, "label": label,
            "meta": {"options": ["overdue", "due_today"],
                     "family": "temporal_numeric"}}


def _score_row(label=2):
    return {"row_id": "x", "primitive": "score", "state": {"a": "b"},
            "instructions": "How bad?", "criteria": None, "label": label,
            "meta": {"attribute": "deadline_urgency",
                     "family": "temporal_numeric"}}


def test_build_questions_shapes():
    crit = {"overdue": "past", "due_today": "today",
            "deadline_urgency": ["a", "b", "c", "d"]}
    q = build_questions(_noul_row(), crit)
    assert q["type"] == "noul" and q["criteria"]["true"] == "t"
    q = build_questions(_choice_row(), crit)
    assert q["type"] == "choice" and set(q["criteria"]) == {"overdue",
                                                            "due_today"}
    q = build_questions(_score_row(), crit)
    assert q["type"] == "score" and q["criteria"] == ["a", "b", "c", "d"]


def test_score_row_all_primitives():
    assert score_row({"noul": 0.7}, _noul_row(label=1))
    assert not score_row({"noul": 0.7}, _noul_row(label=0))
    assert score_row({"choice": "due_today"}, _choice_row(label=1))
    assert not score_row({"choice": "overdue"}, _choice_row(label=1))
    # score answers come back with stringified level keys
    assert score_row({"probabilities": {"0": 0.1, "1": 0.2, "2": 0.6,
                                        "3": 0.1}}, _score_row(label=2))
    assert not score_row({"probabilities": {"0": 0.1, "1": 0.7, "2": 0.1,
                                            "3": 0.1}}, _score_row(label=2))