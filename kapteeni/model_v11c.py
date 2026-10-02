"""The kapteeni-v1.1c served model: the multimodal System One.

`evaluate(state, questions) -> (answers, usage)` over the same wire
contract as the text model (kapteeni.contract), backed by the v1.1c
artifacts: Qwen3.5-4B + the trained LoRA adapter + the per-primitive
PassMLP heads, read through the multi-pass judgment structure the model
was trained and gated on (docs/PREREG-KAPTEENI-V11C.md — every gate
passed 2026-10-02).

Wire extension (the pre-registered ship step): when `state` is an
object carrying an `image` field — base64 PNG/JPEG bytes, bare or with
a data-URI prefix — the image is attached to every pass of every
question and the field is serialized as the "<attached>" placeholder
(exactly the training-state layout: {"image": "<attached>", "note":
...}); all other state fields pass through verbatim.

E1 independence is structural here in a stronger sense than batching:
each QUESTION gets its own forward pass over its own passes (a choice
question's options share one forward for the group softmax), so adding
or removing questions cannot perturb another question's h_last at any
decimal. Fitted serving constants: the per-primitive temperatures from
final_gates.json (the pre-registered combined-val fit).

    python3 -m kapteeni.serve_v11c            # served as kapteeni-v1.1c
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from typing import Any

from kapteeni import contract
from kapteeni.contract import ContractError
from kapteeni.serialize import (choice_option_pass, instructions_text,
                                noul_pass, score_level_pass, state_text)

IMAGE_PLACEHOLDER = "<attached>"
_DATA_URI = re.compile(r"^data:image/(?:png|jpeg|jpg|webp);base64,", re.I)
MAX_IMAGE_BYTES = 8 * 2**20


def extract_image(state: Any):
    """Split the wire state into (pass-state, PIL image | None).

    The image leaves the serialized state (it enters the model through
    the vision path, not the text), replaced by the placeholder the
    training states used. Returns (state, None) unchanged when there is
    no image. Raises ContractError on a present-but-invalid image.
    """
    if not (isinstance(state, dict) and "image" in state):
        return state, None
    raw = state["image"]
    if not isinstance(raw, str) or not raw:
        raise ContractError(
            "state.image must be a base64 string (PNG/JPEG)",
            field="state.image")
    b64 = _DATA_URI.sub("", raw)
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ContractError(
            "state.image is not valid base64", field="state.image")
    if len(data) > MAX_IMAGE_BYTES:
        raise ContractError(
            f"state.image exceeds the {MAX_IMAGE_BYTES // 2**20} MiB limit",
            field="state.image")
    try:
        from io import BytesIO

        from PIL import Image
        img = Image.open(BytesIO(data))
        img.load()
    except Exception:
        raise ContractError(
            "state.image must be decodable image bytes (PNG/JPEG)",
            field="state.image")
    pass_state = dict(state)
    pass_state["image"] = IMAGE_PLACEHOLDER
    return pass_state, img


def question_passes(q: dict, pass_state: Any):
    """One question -> its pass records (text + shared image). The
    multi-pass layout is byte-identical to training (serialize.py)."""
    ins = q["instructions"]
    if q["type"] == "noul":
        return [{"text": noul_pass(pass_state, ins, q.get("criteria")),
                 "qtype": "noul", "n": 1}]
    if q["type"] == "choice":
        out = []
        for opt, desc in q["criteria"].items():
            out.append({"text": choice_option_pass(pass_state, ins, opt, desc),
                        "qtype": "choice", "n": len(q["criteria"])})
        return out
    levels = q["criteria"]
    return [{"text": score_level_pass(pass_state, ins, i, len(levels), lvl),
             "qtype": "score", "n": len(levels)}
            for i, lvl in enumerate(levels)]


class SystemOneV11C:
    """The trained multimodal model (torch loads lazily; GPU at serve time)."""

    def __init__(self, base: str, adapter: str, heads_path: str,
                 temps: dict[str, float], device: str = "cuda"):
        self.base = base
        self.adapter = adapter
        self.heads_path = heads_path
        self.temps = temps
        self.device = device
        self._loaded = False

    def _load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
        from kapteeni.heads import PassMLP
        from kapteeni.v11c import _inner_of, _last_h, vl_inputs

        self._torch = torch
        self._vl_inputs = vl_inputs
        self._last_h = _last_h
        proc = AutoProcessor.from_pretrained(self.base)
        model = AutoModelForImageTextToText.from_pretrained(
            self.base, dtype=torch.bfloat16).to(self.device)
        model = PeftModel.from_pretrained(model, self.adapter)
        model.eval()
        # the fp32 tower (the NaN-diagnosis fix; see the pre-reg's
        # engineering note) — serving uses the same numerics as training
        inner = _inner_of(model)
        if hasattr(inner, "visual"):
            inner.visual.float()
        sd = torch.load(self.heads_path, weights_only=True)
        heads = {}
        for qt, h_sd in sd.items():
            head = PassMLP(h_sd["net.0.weight"].shape[1]).to(self.device)
            head.load_state_dict(h_sd)
            head.eval()
            heads[qt] = head
        self.proc, self.inner, self.heads = proc, inner, heads
        self.tok = proc
        self._loaded = True

    def _h(self, records):
        """(N, hidden) h_last for one question's passes (one forward)."""
        torch = self._torch
        with torch.no_grad():
            inputs = self._vl_inputs(self.proc, self.tok, records)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            hidden = self.inner(**inputs, use_cache=False).last_hidden_state
            return self._last_h(hidden, inputs).float()

    def evaluate(self, state, questions: dict):
        if not self._loaded:
            self._load()
        pass_state, img = extract_image(state)
        st = state_text(pass_state)
        answers: dict = {}
        in_tok = 0
        for qid, q in questions.items():
            records = question_passes(q, pass_state)
            for r in records:
                r["image"] = img
            h = self._h(records)
            z = self.heads[q["type"]](h).tolist()  # (N,) per-pass scores
            T = self.temps[q["type"]]
            if q["type"] == "noul":
                p = 1.0 / (1.0 + math.exp(-(z[0] / T)))
                answers[qid] = contract.noul_answer(p)
            elif q["type"] == "choice":
                scores = {opt: z[i] / T for i, opt in enumerate(q["criteria"])}
                answers[qid] = contract.choice_answer(scores)
            else:
                answers[qid] = contract.score_answer(
                    [zi / T for zi in z], q["criteria"])
            # usage: state once + each question once (the serialized
            # instruction + answer-space surface), output = answers
            surface = instructions_text(q["instructions"])
            crit = q.get("criteria")
            if isinstance(crit, dict):
                surface += " ".join(str(k) for k in crit)
            elif isinstance(crit, list):
                surface += " ".join(str(x) for x in crit)
            in_tok += len(surface) // 4
        in_tok += len(st) // 4 + (399 if img is not None else 0)
        out_tok = len(json.dumps(answers)) // 4
        return answers, {"input_tokens": in_tok, "output_tokens": out_tok}