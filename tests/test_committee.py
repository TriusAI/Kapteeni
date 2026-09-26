from kapteeni.committee import CommitteeModel, merge_answers
from kapteeni.contract import confidence


def _noul(p):
    return {"type": "noul", "noul": p}


def _choice(probs):
    top = max(probs, key=probs.get)
    return {"type": "choice", "choice": top, "probabilities": probs,
            "confidence": confidence(list(probs.values()))}


def _score(probs, legend=None):
    keys = sorted(probs, key=lambda s: int(s))
    vals = [probs[k] for k in keys]
    return {"type": "score",
            "score": sum(int(k) * probs[k] for k in keys),
            "legend": legend or {k: f"level {k}" for k in keys},
            "probabilities": probs,
            "confidence": confidence(vals)}


def test_merge_noul_averages():
    m = merge_answers(_noul(0.7), _noul(0.4))
    assert m["type"] == "noul"
    assert abs(m["noul"] - 0.55) < 1e-9


def test_merge_choice_averages_and_reargmaxes():
    a = _choice({"billing": 0.2, "technical": 0.7, "sales": 0.1})
    b = _choice({"billing": 0.4, "technical": 0.1, "sales": 0.5})
    m = merge_answers(a, b)
    # averaged: billing 0.3, technical 0.4, sales 0.3
    assert m["choice"] == "technical"
    assert abs(sum(m["probabilities"].values()) - 1.0) < 1e-6
    assert abs(m["probabilities"]["technical"] - 0.4) < 1e-6
    # confidence derives from the MERGED (flatter) distribution
    assert m["confidence"] == confidence(list(m["probabilities"].values()))


def test_merge_score_recomputes_expectation():
    a = _score({"0": 0.1, "1": 0.8, "2": 0.1})
    b = _score({"0": 0.5, "1": 0.3, "2": 0.2})
    m = merge_answers(a, b)
    # averaged: 0.3 / 0.55 / 0.15 -> expectation 0.85
    assert abs(m["score"] - 0.85) < 1e-6
    assert m["legend"] == a["legend"]
    assert abs(m["probabilities"]["1"] - 0.55) < 1e-6


def test_merge_is_pure_and_type_safe():
    assert merge_answers(_noul(0.5), _noul(0.5))["noul"] == 0.5
    import pytest
    with pytest.raises(ValueError):
        merge_answers({"type": "weird"}, {"type": "weird"})


class _Stub:
    def __init__(self, answers, out_tokens=100):
        self.answers = answers
        self.out = out_tokens

    def evaluate(self, state, questions):
        return ({q: self.answers[q] for q in questions},
                {"input_tokens": 42, "output_tokens": self.out})


def test_committee_model_usage_and_merge():
    a = _Stub({"q": _noul(0.8)}, out_tokens=120)
    b = _Stub({"q": _noul(0.2)}, out_tokens=80)
    cm = CommitteeModel(a, b)
    answers, usage = cm.evaluate({}, {"q": None})
    assert abs(answers["q"]["noul"] - 0.5) < 1e-9
    # input counted once (the decision's); output counts BOTH models
    assert usage == {"input_tokens": 42, "output_tokens": 200}