"""Drop-in contract for the TypeSafe System One API (`POST /v1/systemone`).

Field names, bounds, and answer shapes mirror docs.typesafe.ai/api (read
2026-09-24). This module is deliberately framework-free and torch-free: it
is the executable spec of the wire format.

Documented rules implemented here:
  - state: string | object | array (required)
  - model: string (required); "jev-latest" is the flagship alias
  - questions: map<qid, Question>, required, non-empty; qids never reach the model
  - noul:   instructions required; criteria optional {true?, false?}
  - choice: instructions required; criteria required, map<option, desc|null>,
            2..255 options
  - score:  instructions required; criteria required, ordered array of level
            descriptions, 2..10 levels
  - Answers:
      noul   -> {type, noul}                     (absolute; NOT complement-consistent)
      choice -> {type, choice, probabilities, confidence}
      score  -> {type, score, legend, probabilities, confidence}
        score = sum(level_index * p_level) — the docs' example 1.05 =
        0*0.0 + 1*0.95 + 2*0.05 confirms 0-based expectation.
        legend/probabilities keys are STRING level indices.
  - confidence on choice/score only; derived from the distribution shape.
    The reference's exact formula is not public; ours (documented divergence,
    see README) is 1 - H(p)/ln(K), floored at 0.
  - probabilities are floats that sum to exactly 1 (we round to 4 decimals and
    repair the argmax entry so the sum is exact in decimal, like the
    reference's published examples).
"""

from __future__ import annotations

import math
from typing import Any

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
DECIMALS = 4


class ContractError(Exception):
    """Validation failure -> HTTP 422 with a JSON body naming the field."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.status = 422

    def body(self) -> dict:
        err: dict[str, Any] = {"message": self.message}
        if self.field is not None:
            err["field"] = self.field
        return {"error": err}


# ----------------------------------------------------------------- validation


def _is_jsony(v: Any) -> bool:
    """string | object | array — the docs' three allowed instruction/criteria shapes."""
    return isinstance(v, (str, dict, list))


def validate_question(qid: str, q: Any) -> dict:
    if not isinstance(q, dict):
        raise ContractError("question must be an object", field=f"questions.{qid}")
    qtype = q.get("type")
    if qtype not in ("noul", "choice", "score"):
        raise ContractError(
            "type must be one of noul, choice, score", field=f"questions.{qid}.type"
        )
    instructions = q.get("instructions")
    if not _is_jsony(instructions):
        raise ContractError(
            "instructions are required (string, object, or array)",
            field=f"questions.{qid}.instructions",
        )
    criteria = q.get("criteria")

    if qtype == "noul":
        if criteria is not None:
            if not isinstance(criteria, dict):
                raise ContractError(
                    "noul criteria must be an object with true/false descriptions",
                    field=f"questions.{qid}.criteria",
                )
            for side in ("true", "false"):
                if side in criteria and not _is_jsony(criteria[side]):
                    raise ContractError(
                        f"noul criteria.{side} must be a string, object, or array",
                        field=f"questions.{qid}.criteria.{side}",
                    )
    elif qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ContractError(
                "choice criteria must be a map of at least 2 options",
                field=f"questions.{qid}.criteria",
            )
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise ContractError(
                f"choice supports at most {MAX_CHOICE_OPTIONS} options",
                field=f"questions.{qid}.criteria",
            )
        for opt, desc in criteria.items():
            if not isinstance(opt, str) or not opt:
                raise ContractError(
                    "choice options must be non-empty strings",
                    field=f"questions.{qid}.criteria",
                )
            if desc is not None and not _is_jsony(desc):
                raise ContractError(
                    "choice criteria values must be a string, object, array, or null",
                    field=f"questions.{qid}.criteria.{opt}",
                )
    else:  # score
        if not isinstance(criteria, list) or not (
            MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS
        ):
            raise ContractError(
                f"score criteria must be an ordered array of "
                f"{MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} level descriptions",
                field=f"questions.{qid}.criteria",
            )
        for i, lvl in enumerate(criteria):
            if not _is_jsony(lvl):
                raise ContractError(
                    "score levels must be strings, objects, or arrays",
                    field=f"questions.{qid}.criteria.{i}",
                )

    return {"type": qtype, "instructions": instructions, "criteria": criteria}


def validate_request(body: Any) -> tuple[Any, str, dict[str, dict]]:
    """Returns (state, model, {qid: normalized question}); raises ContractError."""
    if not isinstance(body, dict):
        raise ContractError("request body must be a JSON object")
    if "state" not in body:
        raise ContractError("state is required", field="state")
    state = body["state"]
    if not _is_jsony(state):
        raise ContractError(
            "state must be a string, object, or array", field="state"
        )
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ContractError("model is required", field="model")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ContractError(
            "questions must be a non-empty map of question id to question",
            field="questions",
        )
    normalized: dict[str, dict] = {}
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            raise ContractError(
                "question ids must be non-empty strings", field="questions"
            )
        normalized[qid] = validate_question(qid, q)
    return state, model, normalized


# -------------------------------------------------------------------- answers


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def _round_repair(probs: list[float]) -> list[float]:
    """Round to DECIMALS and repair the largest entry so the decimal sum is 1."""
    r = [round(p, DECIMALS) for p in probs]
    k = max(range(len(r)), key=lambda i: probs[i])
    r[k] = round(r[k] + (1.0 - sum(r)), DECIMALS)
    # guard against fp artifacts like -0.0 or 1.0000000002
    r = [min(1.0, max(0.0, p)) for p in r]
    if abs(sum(r) - 1.0) > 1e-9:
        r[k] = round(1.0 - sum(p for i, p in enumerate(r) if i != k), DECIMALS)
    return r


def confidence(probs: list[float]) -> float:
    """1 - H(p)/ln(K), floored at 0 (our documented definition)."""
    k = len(probs)
    if k <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, round(1.0 - h / math.log(k), DECIMALS))


def noul_answer(p: float) -> dict:
    """Noul is absolute: no renormalization, no complement constraint (P(A)+
    P(not-A) != 1 is reference behavior), no confidence field."""
    return {"type": "noul", "noul": round(float(p), DECIMALS)}


def choice_answer(scores: dict[str, float], temperature: float = 1.0) -> dict:
    """scores: per-option logit (or any monotone score). Softmax over the full
    option set -> relative distribution that sums to exactly 1."""
    opts = list(scores)
    probs = _softmax([float(scores[o]) / max(1e-9, temperature) for o in opts])
    probs = _round_repair(probs)
    top = max(range(len(opts)), key=lambda i: probs[i])
    return {
        "type": "choice",
        "choice": opts[top],
        "probabilities": {o: probs[i] for i, o in enumerate(opts)},
        "confidence": confidence(probs),
    }


def score_answer(
    level_scores: list[float], levels: list, temperature: float = 1.0
) -> dict:
    """level_scores: per-level independent score; API layer normalizes (softmax)
    per the documented 'independent levels' behavior; score = expectation."""
    n = len(level_scores)
    probs = _softmax([float(s) / max(1e-9, temperature) for s in level_scores])
    probs = _round_repair(probs)
    score = sum(i * p for i, p in enumerate(probs))
    return {
        "type": "score",
        "score": round(score, DECIMALS),
        "legend": {str(i): levels[i] for i in range(n)},
        "probabilities": {str(i): probs[i] for i in range(n)},
        "confidence": confidence(probs),
    }