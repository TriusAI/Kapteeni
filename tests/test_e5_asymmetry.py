"""E5 — Noul/Choice asymmetry.

Reference-documented behavior:
  - Noul is ABSOLUTE: P(A) + P(not-A) != 1 (their example: 0.72 + 0.47 = 1.19)
  - Noul and Choice on the same yes/no question give DIFFERENT numbers
    (their example: noul 0.22 vs choice-yes 0.01)
  - Choice is RELATIVE: probabilities sum to exactly 1

We assert the structural identities hold for every evaluate() implementation.
"""

STATE = "The customer writes: my card was charged twice this month and I want the duplicate removed."
Q = "Does the message request a refund?"
Q_NOT = "Does the message decline a refund?"


def test_noul_is_not_complement_consistent(model):
    a_yes, _ = model.evaluate(STATE, {"q": {"type": "noul", "instructions": Q}})
    a_no, _ = model.evaluate(STATE, {"q": {"type": "noul", "instructions": Q_NOT}})
    s = a_yes["q"]["noul"] + a_no["q"]["noul"]
    assert abs(s - 1.0) > 1e-6, (
        f"P(A)+P(not-A) = {s} -- a complement-consistent Noul readout is "
        "exactly the bug the reference docs warn against"
    )


def test_choice_sums_to_exactly_one(model):
    a, _ = model.evaluate(
        STATE,
        {"yn": {
            "type": "choice",
            "instructions": Q,
            "criteria": {"yes": "A refund is requested", "no": "No refund requested"},
        }},
    )
    probs = a["yn"]["probabilities"]
    assert set(probs) == {"yes", "no"}
    assert sum(probs.values()) == 1.0  # exact decimal sum


def test_noul_and_choice_differ(model):
    a_noul, _ = model.evaluate(STATE, {"n": {"type": "noul", "instructions": Q}})
    a_choice, _ = model.evaluate(
        STATE,
        {"c": {
            "type": "choice",
            "instructions": Q,
            "criteria": {"yes": "A refund is requested", "no": "No refund requested"},
        }},
    )
    assert a_noul["n"]["noul"] != a_choice["c"]["probabilities"]["yes"]


def test_noul_has_no_confidence_field(model):
    a, _ = model.evaluate(STATE, {"n": {"type": "noul", "instructions": Q}})
    assert "confidence" not in a["n"]
    assert set(a["n"]) == {"type", "noul"}


def test_negation_moves_noul(model):
    """Sanity: a clear question and its negation should not both sit at 0.5."""
    state = "The sky on a clear day"
    a_pos, _ = model.evaluate(state, {"p": {"type": "noul", "instructions": "Is the sky blue?"}})
    a_neg, _ = model.evaluate(state, {"n": {"type": "noul", "instructions": "Is the sky green?"}})
    assert a_pos["p"]["noul"] != a_neg["n"]["noul"]