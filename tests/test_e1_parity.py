"""E1 — batching parity.

Reference invariants (docs + consistency cookbook):
  - batched (N questions, 1 call) == single (N calls, 1 question each)
  - adding a question never changes another question's answer
  - repeated identical calls are stable (our model is fully deterministic;
    the reference shows std ~= 0.01 — a documented divergence we accept)
  - question ids never reach the model (identical content, different ids
    -> identical answers)
"""

import copy

STATE = {
    "ticket": {
        "subject": "Duplicate charge",
        "messages": [
            {"from": "customer", "text": "I was charged twice for order A-104."},
        ],
    },
    "order": {"id": "A-104", "charges": [
        {"amount_usd": 49, "status": "captured"},
        {"amount_usd": 49, "status": "captured"},
    ]},
    "refund_policy": "Duplicate charges are eligible for a refund.",
}

QUESTIONS = {
    "refund_requested": {
        "type": "noul",
        "instructions": "Does `ticket.messages[0].text` indicate a duplicate charge?",
    },
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "Payment, invoicing, refund issues",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer appear?",
        "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry"],
    },
    "policy_supports": {
        "type": "noul",
        "instructions": "Does `refund_policy` cover this case given `order.charges`?",
    },
}


def _answers(model, questions):
    answers, usage = model.evaluate(copy.deepcopy(STATE), copy.deepcopy(questions))
    assert isinstance(usage, dict) and usage["input_tokens"] > 0 and usage["output_tokens"] >= 0
    return answers


def test_batched_equals_single(model):
    batched = _answers(model, QUESTIONS)
    for qid, q in QUESTIONS.items():
        single = _answers(model, {qid: q})
        assert batched[qid] == single[qid], qid


def test_adding_question_never_changes_others(model):
    base = {k: v for k, v in list(QUESTIONS.items())[:2]}
    extra = dict(base)
    extra.update({
        "added_noul": {"type": "noul", "instructions": "Is `order.id` mentioned?"},
        "added_score": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["can wait", "this week", "now"],
        },
    })
    a_base = _answers(model, base)
    a_extra = _answers(model, extra)
    for qid in base:
        assert a_base[qid] == a_extra[qid], qid


def test_identical_across_repeats(model):
    a1 = _answers(model, QUESTIONS)
    a2 = _answers(model, QUESTIONS)
    assert a1 == a2


def test_ids_never_reach_the_model(model):
    """Same content, different ids -> identical answers."""
    q = {"type": "noul", "instructions": "Is a refund being requested?"}
    a = _answers(model, {"first_id": q})
    b = _answers(model, {"a_completely_different_id": q})
    assert a["first_id"] == b["a_completely_different_id"]


def test_all_three_primitives_answered(model):
    a = _answers(model, QUESTIONS)
    assert set(a) == set(QUESTIONS)
    assert a["refund_requested"]["type"] == "noul"
    assert a["department"]["type"] == "choice"
    assert a["frustration"]["type"] == "score"