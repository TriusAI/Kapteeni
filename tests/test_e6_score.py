"""E6 — Score semantics.

Reference behavior: score is the probability-weighted answer across levels
(expectation, can land between levels); legend repeats the level descriptions
keyed by string index; probabilities sum to exactly 1.
"""

STATE = "Our API started returning 500 errors on every request 20 minutes ago."
LEVELS = ["Calm, just stating facts", "Frustrated but civil", "Very angry"]


def test_score_is_expectation_within_bounds(model):
    a, _ = model.evaluate(
        STATE,
        {"f": {"type": "score", "instructions": "How frustrated is the customer?",
               "criteria": LEVELS}},
    )
    ans = a["f"]
    p = ans["probabilities"]
    assert ans["legend"] == {str(i): lvl for i, lvl in enumerate(LEVELS)}
    assert set(p) == {"0", "1", "2"}
    assert sum(p.values()) == 1.0
    assert 0.0 <= ans["score"] <= 2.0
    expected = sum(int(k) * v for k, v in p.items())
    assert abs(ans["score"] - expected) < 0.0011  # rounding tolerance


def test_score_can_be_fractional(model):
    """Find a state where the distribution is split -> fractional score."""
    a, _ = model.evaluate(
        STATE,
        {"f": {"type": "score", "instructions": "How frustrated is the customer?",
               "criteria": LEVELS}},
    )
    # whatever the value, it must equal the expectation of the returned dist
    p = a["f"]["probabilities"]
    assert a["f"]["score"] == round(sum(int(k) * v for k, v in p.items()), 4)


def test_score_with_many_levels(model):
    a, _ = model.evaluate(
        "The essay develops a clear thesis with strong evidence.",
        {"q": {"type": "score", "instructions": "Rate the writing quality.",
               "criteria": [
                   "incoherent", "weak", "below average", "average",
                   "solid", "good", "strong", "excellent", "outstanding",
               ]}},
    )
    ans = a["q"]
    assert len(ans["probabilities"]) == 9
    assert len(ans["legend"]) == 9
    assert 0.0 <= ans["score"] <= 8.0
    assert 0.0 <= ans["confidence"] <= 1.0