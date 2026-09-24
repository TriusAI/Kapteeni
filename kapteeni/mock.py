"""Deterministic, torch-free mock model.

Implements the same evaluate(state, questions) -> (answers, usage) interface
as SystemOneModel so the E1/E5/E6 interface suites run before any engine or
training exists (the Kapteeni approach; kept here). Answers derive from
CONTENT ONLY (question ids never influence the mock, mirroring the reference:
ids are not sent to the model).
"""

from __future__ import annotations

import hashlib
import json

from kapteeni import contract
from kapteeni.serialize import instructions_text, state_text


def _unit(*parts: str) -> float:
    """Deterministic [0,1) from content hash."""
    h = hashlib.sha256("\x1f".join(p for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _noul_key(q: dict) -> str:
    crit = json.dumps(q.get("criteria"), sort_keys=True, ensure_ascii=False)
    return crit


class MockModel:
    def __init__(self, seed: str = "kapteeni-mock"):
        self.seed = seed

    def evaluate(self, state, questions: dict):
        st = state_text(state)
        answers: dict = {}
        for qid, q in questions.items():
            ins = instructions_text(q["instructions"])
            if q["type"] == "noul":
                u = _unit(self.seed, "noul", st, ins, _noul_key(q))
                answers[qid] = contract.noul_answer(0.05 + 0.9 * u)
            elif q["type"] == "choice":
                scores = {}
                for opt, desc in q["criteria"].items():
                    u = _unit(self.seed, "choice", st, ins, opt,
                              json.dumps(desc, ensure_ascii=False))
                    scores[opt] = (u - 0.5) * 8.0
                answers[qid] = contract.choice_answer(scores)
            else:
                scores = []
                for i, lvl in enumerate(q["criteria"]):
                    u = _unit(self.seed, "score", st, ins, str(i),
                              json.dumps(lvl, ensure_ascii=False))
                    scores.append((u - 0.5) * 8.0)
                answers[qid] = contract.score_answer(scores, q["criteria"])

        in_tok = len(st) // 4 + sum(
            len(instructions_text(q["instructions"])) // 4 for q in questions.values()
        )
        out_tok = len(json.dumps(answers)) // 4
        return answers, {"input_tokens": in_tok, "output_tokens": out_tok}