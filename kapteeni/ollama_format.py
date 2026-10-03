"""Ollama System One letter-format renderer — the port's data surface.

Implements, in Python, the EXACT prompt payload Ollama's decision
runner builds at runtime (runner source decision/systemone.go at
v0.35.1; the normative contract is docs/OLLAMA-PROMPT-SPEC.md §2).
Used by the port's training-data conversion and its pre-registered
evaluation; at serve time Ollama itself does the rendering — this
module exists so training prompts and runtime prompts agree byte for
byte.

Contract notes that are easy to get wrong (each is asserted in tests):

- `context` (the state): a string passes verbatim; an object/array is
  rendered as COMPACT JSON, key order preserved. The runner json.Compacts
  whatever the client sent; our convention (train + wire client) is
  `json.dumps(state, ensure_ascii=False, separators=(",", ":"))`.
- the payload is Go's json.Marshal of {"context": ..., "schema": [...]}:
  compact, keys in struct order, and HTML-ESCAPED by Go default
  (< > & -> \\u003c \\u003e \\u0026). `go_json` replicates it.
- each schema field: {"name", "description", "choices": [{"code",
  "value", "description"}...]} with letters A.. in option order:
  noul A=false B=true; choice A.. by criteria order; score A=level 0.
- one user message per question: payload + "\\n\\nRequested field: "
  + the JSON-quoted question name.
- `description` for instructions follows the same content rule as the
  state (string verbatim; object/array compact JSON).

Candidates return answers, never questions: this module renders one
row's questions independently — no answer or question text crosses
question boundaries (the runtime contract; ours since v0).
"""

from __future__ import annotations

import json
from typing import Any

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_ESC = {"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"}


def _html_escape(s: str) -> str:
    """Go's json.Marshal escapes <, > and & by default; replicate it so
    training payloads match the runtime byte for byte."""
    return "".join(_ESC.get(ch, ch) for ch in s)


def go_json(value: Any) -> str:
    """Compact JSON the way the runner emits it: no spaces, sorted keys
    are NOT applied (struct order is preserved by construction here),
    HTML-escaped, ensure_ascii=False equivalent (Go emits raw UTF-8)."""
    return _html_escape(json.dumps(value, ensure_ascii=False,
                                   separators=(",", ":")))


def _content(v: Any) -> str:
    """The runner's content() rule: strings verbatim, objects/arrays as
    compact JSON."""
    if isinstance(v, str):
        return v
    return go_json(v)


def _field(name: str, q: dict) -> dict:
    """One schema field, mirroring decision/systemone.go compileField."""
    f: dict[str, Any] = {"name": name, "description": _content(
        q["instructions"]), "choices": []}
    qt = q["type"]
    if qt not in ("noul", "choice", "score"):
        raise ValueError(f"question type {qt!r}")
    if qt == "noul":
        crit = q.get("criteria") or {}
        no = crit.get("false", "No")
        yes = crit.get("true", "Yes")
        f["choices"] = [{"code": "A", "value": False,
                         "description": _content(no)},
                        {"code": "B", "value": True,
                         "description": _content(yes)}]
    elif qt == "choice":
        for i, (key, desc) in enumerate(q["criteria"].items()):
            if desc is None:
                desc = key
            f["choices"].append({"code": LETTERS[i], "value": key,
                                 "description": _content(desc)})
    else:  # score
        for i, desc in enumerate(q["criteria"]):
            f["choices"].append({"code": LETTERS[i], "value": str(i),
                                 "description": _content(desc)})
    if not 2 <= len(f["choices"]) <= 26:
        raise ValueError(f"{name}: need 2-26 candidates")
    return f


def decision_messages(state: Any, questions: list[dict]) -> list[dict]:
    """The per-question user messages the runner renders.

    questions: [{"name", "type", "instructions", "criteria"}...] in
    request order. Returns one dict per question:
      {"name", "message": <user content>, "candidates":
       [{"code", "value", "description"}...]}
    The caller applies the model's chat template to each message —
    the runtime renders through the model's own template, so training
    must too.
    """
    fields = [_field(q["name"], q) for q in questions]
    payload = go_json({"context": _content(state), "schema": fields})
    out = []
    for f in fields:
        out.append({
            "name": f["name"],
            "message": payload + "\n\nRequested field: " + go_json(
                f["name"]),
            "candidates": f["choices"],
        })
    return out


def letter_index(msg: dict, target: Any) -> int:
    """Index of the candidate whose `value` matches the gold target
    (noul: bool; choice: the option key; score: the zero-based level).
    `msg` is one entry of decision_messages()'s output."""
    qt = msg_qtype(msg)
    for i, ch in enumerate(msg["candidates"]):
        v = ch["value"]
        if qt == "noul":
            if bool(v) == bool(target):
                return i
        elif qt == "score":
            if v == str(int(target)):
                return i
        else:
            if v == target:
                return i
    raise ValueError(f"target {target!r} is not one of the candidates")


def msg_qtype(msg: dict) -> str:
    """Recover the primitive from a compiled message (noul has exactly
    the two booleans, score values are digit strings)."""
    vals = [c["value"] for c in msg["candidates"]]
    if all(isinstance(v, bool) for v in vals):
        return "noul"
    if all(isinstance(v, str) and v.isdigit() for v in vals):
        return "score"
    return "choice"


def teacher_to_letters(msg: dict, probs: dict | float) -> list[float]:
    """Map a teacher's per-candidate output distribution onto the
    message's letter order (the distillation target).

    noul: probs = P(true) -> [1-P, P]
    choice: probs = {option_key: p}
    score: probs = {level_index: p} (keys int or str)
    Renormalized to sum 1 within the message's candidates.
    """
    qt = msg_qtype(msg)
    if qt == "noul":
        p = float(probs)
        out = [1.0 - p, p]
    else:
        out = []
        for ch in msg["candidates"]:
            key = ch["value"]
            if qt == "score":
                key = int(key)
            p = probs.get(key)
            if p is None and qt == "score":
                p = probs.get(str(key))
            if p is None:
                raise ValueError(f"teacher distribution missing {key!r}")
            out.append(float(p))
    s = sum(out)
    return [p / s for p in out] if s > 0 else out