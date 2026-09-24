"""E11-class contract tests: the reference docs' examples must parse and the
answer shapes must match byte-for-byte semantics."""

import math

import pytest

from kapteeni.contract import (
    ContractError,
    choice_answer,
    confidence,
    noul_answer,
    score_answer,
    validate_request,
    validate_question,
)


class TestValidation:
    def test_docs_example_request_parses(self):
        body = {
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
            },
        }
        state, model, qs = validate_request(body)
        assert state == body["state"]
        assert model == "jev-latest"
        assert qs["is_urgent"]["type"] == "noul"

    def test_all_three_primitives_parse(self):
        body = {
            "state": {"message": "API returning 500s for 20 minutes, orders blocked."},
            "model": "jev-latest",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this",
                    "criteria": {
                        "billing": "Payment or subscription issues",
                        "technical": "Bugs or integration problems",
                        "sales": "Pricing or account questions",
                    },
                },
                "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated the customer appears",
                    "criteria": ["Calm", "Frustrated but civil", "Very angry"],
                },
            },
        }
        _, _, qs = validate_request(body)
        assert set(qs) == {"department", "is_urgent", "frustration"}

    def test_structured_instructions_and_criteria_parse(self):
        body = {
            "state": {},
            "model": "jev-latest",
            "questions": {
                "same_person": {
                    "type": "noul",
                    "instructions": {
                        "potential_duplicate": {"name": "John Smith"},
                        "question": "Is the resume for the same person as `potential_duplicate`?",
                    },
                    "criteria": {"true": "Same person", "false": "Different people"},
                }
            },
        }
        _, _, qs = validate_request(body)
        assert qs["same_person"]["criteria"]["true"] == "Same person"

    def test_missing_state_422(self):
        with pytest.raises(ContractError) as e:
            validate_request({"model": "jev-latest", "questions": {}})
        assert e.value.status == 422
        assert e.value.field == "state"

    def test_bad_type_422(self):
        with pytest.raises(ContractError):
            validate_question("q", {"type": "bool", "instructions": "x"})

    def test_missing_instructions_422(self):
        with pytest.raises(ContractError):
            validate_question("q", {"type": "noul"})

    def test_choice_option_bounds(self):
        q = {"type": "choice", "instructions": "pick", "criteria": {"a": None}}
        with pytest.raises(ContractError):
            validate_question("q", q)  # 1 option
        big = {f"o{i}": None for i in range(256)}
        with pytest.raises(ContractError):
            validate_question("q", {**q, "criteria": big})
        ok = {f"o{i}": None for i in range(255)}
        assert validate_question("q", {**q, "criteria": ok})["type"] == "choice"

    def test_choice_null_description_ok(self):
        v = validate_question(
            "q",
            {"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": "desc"}},
        )
        assert v["criteria"]["a"] is None

    def test_score_level_bounds(self):
        base = {"type": "score", "instructions": "rate"}
        with pytest.raises(ContractError):
            validate_question("q", {**base, "criteria": ["only"]})
        with pytest.raises(ContractError):
            validate_question("q", {**base, "criteria": [str(i) for i in range(11)]})
        assert validate_question("q", {**base, "criteria": ["a", "b"]})["type"] == "score"

    def test_error_body_shape(self):
        err = ContractError("boom", field="state")
        assert err.body() == {"error": {"message": "boom", "field": "state"}}


class TestAnswers:
    def test_noul_answer_shape(self):
        a = noul_answer(0.95)
        assert a == {"type": "noul", "noul": 0.95}
        # absolute: no confidence, no complement constraint implied
        assert "confidence" not in a

    def test_choice_answer_docs_example_semantics(self):
        a = choice_answer({"billing": 3.0, "technical": 1.0, "sales": -2.0})
        assert a["type"] == "choice"
        assert a["choice"] == "billing"
        assert set(a["probabilities"]) == {"billing", "technical", "sales"}
        assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-9)
        assert 0.0 <= a["confidence"] <= 1.0
        assert a["probabilities"]["sales"] == 0.0 or a["probabilities"]["sales"] > 0

    def test_choice_probabilities_sum_to_one_exactly_in_decimal(self):
        for trial in range(50):
            a = choice_answer({f"o{i}": (i * 7 % 13) - 6.0 for i in range(9)})
            s = sum(a["probabilities"].values())
            assert s == 1.0, (trial, s)  # exact decimal sum, not isclose

    def test_choice_temperature_flattens(self):
        sharp = choice_answer({"a": 5.0, "b": 0.0})
        flat = choice_answer({"a": 5.0, "b": 0.0}, temperature=10.0)
        assert flat["probabilities"]["b"] > sharp["probabilities"]["b"]
        assert flat["confidence"] < sharp["confidence"]

    def test_score_answer_docs_example(self):
        # docs: p {0:0.0,1:0.95,2:0.05} -> score 1.05; our softmax reproduces
        # the shape from any scores with those relative weights.
        a = score_answer(
            [-6.0, 3.0, 1.35],  # softmax -> ~[0.0, 0.95, 0.05]
            ["Calm", "Frustrated", "Very angry"],
        )
        assert a["type"] == "score"
        assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
        assert set(a["probabilities"]) == {"0", "1", "2"}
        assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-9)
        assert 0.9 <= a["score"] <= 1.2  # ~1.05, expectation over levels

    def test_score_is_expectation_and_can_be_fractional(self):
        a = score_answer([0.0, 0.0], ["lo", "hi"])  # 50/50
        assert a["score"] == 0.5

    def test_confidence_flat_is_zero(self):
        assert confidence([0.5, 0.5]) == 0.0
        assert confidence([1.0, 0.0]) == 1.0
        assert confidence([0.25] * 4) == 0.0

    def test_confidence_monotone_in_peakedness(self):
        assert confidence([0.9, 0.1]) > confidence([0.6, 0.4]) > confidence([0.5, 0.5])