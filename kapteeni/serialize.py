"""Pass serialization: how a (state, question[, option|level]) becomes model input.

Every question — and, for choice, every option; for score, every level — is its
own pass against the same state, per the reference's documented independence
(no cross-question attention, adding questions doesn't change answers).

Layout of one pass (v0, plain text; no chat template so behavior is
template-independent and fully deterministic):

    {state text}

    [noul] {instructions}
    criteria - true: {t} | false: {f}
    answer:

The final token's hidden state is the readout position (backbone.py).

State is JSON text with field names (reference recommendation); dot-paths in
instructions (e.g. `ticket.messages[0].text`) refer to these fields verbatim.
Question ids are never serialized. Instructions/criteria may be structured
(string | object | array); an object with a "question" key gets its question
line plus the remaining fields as data, matching the reference's structured
instructions feature.
"""

from __future__ import annotations

import json
from typing import Any


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _jsony_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def instructions_text(instructions: Any) -> str:
    """Structured instructions: prefer a `question` field, keep the rest as data."""
    if isinstance(instructions, dict) and "question" in instructions:
        data = {k: v for k, v in instructions.items() if k != "question"}
        line = _jsony_text(instructions["question"])
        if data:
            line += "\n" + json.dumps(data, ensure_ascii=False)
        return line
    return _jsony_text(instructions)


def _criteria_pair(criteria: Any) -> str:
    if not isinstance(criteria, dict):
        return ""
    parts = []
    if "true" in criteria:
        parts.append(f"true: {_jsony_text(criteria['true'])}")
    if "false" in criteria:
        parts.append(f"false: {_jsony_text(criteria['false'])}")
    return " | ".join(parts)


def noul_pass(state: Any, instructions: Any, criteria: Any = None) -> str:
    parts = [f"[noul] {instructions_text(instructions)}"]
    crit = _criteria_pair(criteria)
    if crit:
        parts.append(f"criteria - {crit}")
    parts.append("answer:")
    return state_text(state) + "\n\n" + "\n".join(parts)


def choice_option_pass(
    state: Any, instructions: Any, option: str, description: Any
) -> str:
    line = f"option - {option}"
    if description is not None:
        line += f": {_jsony_text(description)}"
    parts = [
        f"[choice] {instructions_text(instructions)}",
        line,
        "answer:",
    ]
    return state_text(state) + "\n\n" + "\n".join(parts)


def score_level_pass(
    state: Any, instructions: Any, level_idx: int, n_levels: int, level: Any
) -> str:
    parts = [
        f"[score] {instructions_text(instructions)}",
        f"level {level_idx + 1} of {n_levels} - {_jsony_text(level)}",
        "answer:",
    ]
    return state_text(state) + "\n\n" + "\n".join(parts)