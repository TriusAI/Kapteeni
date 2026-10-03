"""Ollama letter-format renderer tests — byte-exactness against the
runner contract pinned in docs/OLLAMA-PROMPT-SPEC.md (v0.35.1 source)."""

import pytest

from kapteeni.ollama_format import (decision_messages, msg_qtype,
                                    go_json, letter_index,
                                    teacher_to_letters)


def state_str():
    return "Our checkout has returned 500 errors since 9am."


def choice_q():
    return {"name": "label", "type": "choice",
            "instructions": "Which label fits this ticket?",
            "criteria": {"billing": "Payments and refunds",
                         "bug": "Software errors",
                         "account": "Login and account access"}}


class TestPayload:
    def test_exact_payload_and_message(self):
        """Hand-computed expectation of Go json.Marshal output for the
        API reference's own example request."""
        msgs = decision_messages(state_str(), [choice_q()])
        expected_payload = (
            '{"context":"Our checkout has returned 500 errors since 9am.",'
            '"schema":[{"name":"label","description":"Which label fits '
            'this ticket?","choices":[{"code":"A","value":"billing",'
            '"description":"Payments and refunds"},{"code":"B","value":'
            '"bug","description":"Software errors"},{"code":"C","value":'
            '"account","description":"Login and account access"}]}]}')
        assert len(msgs) == 1
        assert msgs[0]["message"] == expected_payload + \
            '\n\nRequested field: "label"'
        assert [c["code"] for c in msgs[0]["candidates"]] == ["A", "B", "C"]

    def test_multi_question_shared_payload(self):
        q2 = {"name": "urgent", "type": "noul",
              "instructions": "Is this urgent?"}
        msgs = decision_messages(state_str(), [choice_q(), q2])
        assert len(msgs) == 2
        # both messages share the SAME schema payload (all fields listed)
        payload1 = msgs[0]["message"].split("\n\nRequested field: ")[0]
        payload2 = msgs[1]["message"].split("\n\nRequested field: ")[0]
        assert payload1 == payload2
        assert msgs[0]["message"].endswith('Requested field: "label"')
        assert msgs[1]["message"].endswith('Requested field: "urgent"')
        # the schema lists BOTH fields in each message
        assert '"name":"urgent"' in msgs[0]["message"]
        assert '"name":"label"' in msgs[1]["message"]

    def test_state_object_compact_json_key_order(self):
        # an object state is embedded as an ESCAPED JSON string inside
        # the payload (context is a string field in the runner's
        # marshal struct) — compact, key order preserved, escaped
        msgs = decision_messages({"ticket": "refund me", "sev": 2},
                                  [choice_q()])
        payload = msgs[0]["message"].split("\n\n")[0]
        assert '"context":"{\\"ticket\\":\\"refund me\\",\\"sev\\":2}"' \
            in payload

    def test_go_html_escapes(self):
        assert go_json({"a": "<b>&x"}) == '{"a":"\\u003cb\\u003e\\u0026x"}'

    def test_noul_letters_and_criteria(self):
        q = {"name": "refund", "type": "noul",
             "instructions": "Is a refund requested?",
             "criteria": {"false": "No refund is requested",
                          "true": "A refund is requested"}}
        (m,) = decision_messages(state_str(), [q])
        vals = [(c["code"], c["value"]) for c in m["candidates"]]
        assert vals == [("A", False), ("B", True)]
        assert m["candidates"][0]["description"] == "No refund is requested"
        # default criteria when omitted
        (m2,) = decision_messages(state_str(), [
            {"name": "x", "type": "noul", "instructions": "ok?"}])
        assert m2["candidates"][0]["description"] == "No"
        assert m2["candidates"][1]["description"] == "Yes"

    def test_score_letters_and_values(self):
        q = {"name": "urg", "type": "score", "instructions": "How urgent?",
             "criteria": ["Routine", "Soon", "Immediate"]}
        (m,) = decision_messages(state_str(), [q])
        assert [(c["code"], c["value"]) for c in m["candidates"]] == \
            [("A", "0"), ("B", "1"), ("C", "2")]

    def test_instructions_object_becomes_compact_json(self):
        q = dict(choice_q(), instructions={"question": "Which label?",
                                          "context": "support triage"})
        (m,) = decision_messages(state_str(), [q])
        assert '"description":"{\\"question\\":\\"Which label?\\",' \
            '\\"context\\":\\"support triage\\"}"' in m["message"]

    def test_choice_null_description_uses_key(self):
        q = dict(choice_q(), criteria={"one": None, "two": None})
        (m,) = decision_messages(state_str(), [q])
        assert m["candidates"][0]["description"] == "one"

    def test_candidate_bounds(self):
        q = dict(choice_q(), criteria={k: None for k in "AB"})
        with pytest.raises(ValueError, match="2-26"):
            decision_messages(state_str(), [dict(q, name="bad",
                                                 criteria={"solo": None})])


class TestTargets:
    def test_letter_index_all_primitives(self):
        msgs = decision_messages(state_str(), [
            choice_q(),
            {"name": "yn", "type": "noul", "instructions": "?"},
            {"name": "sc", "type": "score", "instructions": "?",
             "criteria": ["a", "b", "c"]}])
        assert letter_index(msgs[0], "bug") == 1
        assert letter_index(msgs[1], True) == 1
        assert letter_index(msgs[1], False) == 0
        assert letter_index(msgs[2], 2) == 2

    def test_field_qtype_recovery(self):
        msgs = decision_messages(state_str(), [
            choice_q(),
            {"name": "yn", "type": "noul", "instructions": "?"},
            {"name": "sc", "type": "score", "instructions": "?",
             "criteria": ["a", "b", "c"]}])
        assert [msg_qtype(m) for m in msgs] == ["choice", "noul", "score"]

    def test_teacher_to_letters(self):
        msgs = decision_messages(state_str(), [
            choice_q(),
            {"name": "yn", "type": "noul", "instructions": "?"},
            {"name": "sc", "type": "score", "instructions": "?",
             "criteria": ["a", "b", "c"]}])
        # choice: dict keyed by option
        assert teacher_to_letters(msgs[0], {"billing": 0.1, "bug": 0.6,
                                           "account": 0.3}) == [0.1, 0.6,
                                                               0.3]
        # noul: P(true)
        assert teacher_to_letters(msgs[1], 0.8) == pytest.approx([0.2, 0.8])
        # score: level keys, int or str, renormalized
        got = teacher_to_letters(msgs[2], {"0": 0.2, "1": 0.3, "2": 0.5})
        assert got == [0.2, 0.3, 0.5]
        renorm = teacher_to_letters(msgs[0], {"billing": 2, "bug": 1,
                                             "account": 1})
        assert sum(renorm) == pytest.approx(1.0)

    def test_teacher_missing_key_raises(self):
        msgs = decision_messages(state_str(), [choice_q()])
        with pytest.raises(ValueError, match="missing"):
            teacher_to_letters(msgs[0], {"bug": 1.0})