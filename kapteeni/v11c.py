"""Kapteeni v1.1c shared machinery: the v1 phase pipeline's multi-pass
data form on the multimodal base, plus the h_last extraction and heads
readout used by P1 (precompute + heads), P2 (LoRA + heads) and the gates.

Governing document: docs/PREREG-KAPTEENI-V11C.md (committed before any
v1.1c measurement). Design notes:

- Pass records are the v1 pipeline's schema (train_p2.load_passes shape:
  row_id/qtype/kind/key/target/weight/gold/soft/text) plus "image"
  (a render path, or None). The v1.2 text passes
  (data_cache/passes_p2.jsonl + passes_synth2.jsonl) are REUSED
  byte-identical; only synth2zh and the synth3/synth3zh image passes are
  newly expanded.
- The chat template is the serving surface of the v1.1 line (its probe
  baselines and gates are template-based); every pass -- text or image --
  is rendered through it identically.
- The heads readout is answer-vocabulary-free: noul = sigmoid(z/T),
  choice/score = softmax over the row's pass group. Identical machinery
  for every language and modality (the pre-registered mechanism).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from kapteeni.build_data import expand_passes, is_val
from kapteeni.serialize import (choice_option_pass, noul_pass,
                                score_level_pass)

# ------------------------------------------------------------------ sources

P2_REUSE = [  # the v1.2 union, byte-identical (train rows only inside)
    "data_cache/passes_p2.jsonl",
    "data_cache/passes_synth2.jsonl",
]
# passes_p2 row prefixes that P1 does NOT see (v0's P1 had four sources;
# these joined at P2 in the v1 lineage and do the same here)
P2_ONLY_PREFIXES = ("clinc150", "goemo", "synth-")
SOFT = {"boolq": "data_cache/soft_boolq.jsonl",
        "fever": "data_cache/soft_fever.jsonl"}
CRITERIA = {"banking77": "data_cache/crit_banking77.json",
            "helpsteer2": "data_cache/crit_helpsteer2.json",
            "synth2": "data_cache/synth2_criteria.json",
            "synth2zh": "data_cache/synth2zh_criteria.json"}
IMG_DIRS = {"synth3": "data_cache/synth3/images",
            "synth3zh": "data_cache/synth3zh/images"}
# P1's sources (v0's four + the unified-line families); val slices for the
# P1 temperature fit use the same per-source conventions (val rows carry
# gold and FULL option sets)
P1_TEXT_SOURCES = ("boolq", "fever", "banking77", "helpsteer2",
                   "synth2", "synth2zh")


def _load_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _load_json(path: str):
    return json.loads(open(path, encoding="utf-8").read())


def _soft_map(name: str) -> dict:
    if name not in SOFT:
        return {}
    return {r["row_id"]: r for r in _load_rows(SOFT[name])}


def _criteria_map(name: str) -> dict:
    return _load_json(CRITERIA[name]) if name in CRITERIA else {}


def _add_image(rec: dict, image) -> dict:
    rec = dict(rec)
    rec["image"] = image
    return rec


def image_row_passes(row: dict, img_dir: str) -> list[dict]:
    """One synth3/synth3zh row -> its multi-pass records (image attached to
    every pass). Full option/level sets (they are small: <= 7)."""
    rid, qt = row["row_id"], row["primitive"]
    img = f"{img_dir}/{row['image']}"
    state = {"image": "<attached>", "note": row["state_text"]}
    gold = float(row["label"])
    if qt == "noul":
        return [dict(row_id=rid, qtype="noul", kind="noul", key=None,
                     target=gold, weight=1.0, gold=gold, soft=None,
                     text=noul_pass(state, row["instructions"],
                                    row.get("criteria")), image=img)]
    if qt == "choice":
        out = []
        for oi, opt in enumerate(row["options"]):
            out.append(dict(row_id=rid, qtype="choice", kind="choice_opt",
                            key=opt, target=1.0 if oi == row["label"] else 0.0,
                            weight=1.0, gold=gold, soft=None,
                            text=choice_option_pass(state, row["instructions"],
                                                    opt, None), image=img))
        return out
    levels = row["levels"]
    return [dict(row_id=rid, qtype="score", kind="score_lvl", key=i,
                 target=1.0 if i == row["label"] else 0.0, weight=1.0,
                 gold=gold, soft=None,
                 text=score_level_pass(state, row["instructions"], i,
                                       len(levels), lvl), image=img)
            for i, lvl in enumerate(levels)]


def synth3_passes(ds: str, val: bool | None = None) -> list[dict]:
    """All synth3/synth3zh passes, or one split of them."""
    out = []
    for row in _load_rows(f"data_cache/{ds}/items.jsonl"):
        rv = is_val(row["row_id"])
        if val is None or rv == val:
            out.extend(image_row_passes(row, IMG_DIRS[ds]))
    return out


def text_source_passes(name: str, val: bool | None = None) -> list[dict]:
    """Multi-pass expansion of one text source (rows_*.jsonl). Val rows
    carry gold + full option sets; train rows follow expand_passes' rules
    (soft teacher labels where they exist, 24-option deterministic
    subsets otherwise)."""
    rows = [r for r in _load_rows(f"data_cache/rows_{name}.jsonl")
            if val is None or is_val(r["row_id"]) == val]
    return [_add_image(r, None)
            for r in expand_passes(rows, _criteria_map(name), _soft_map(name))]


def reuse_passes(p1_only: bool = False, train_only: bool = True) -> list[dict]:
    """The v1.2 union, byte-identical, split-filtered."""
    out = []
    for path in P2_REUSE:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if train_only and is_val(r["row_id"]):
                continue
            if p1_only and r["row_id"].startswith(P2_ONLY_PREFIXES):
                continue
            out.append(_add_image(r, None))
    return out


# --------------------------------------------------------------- train/val

def p1_train_passes() -> list[dict]:
    """P1 head-training passes: v0's four sources + synth2 + synth2zh +
    the synth3/synth3zh image families (train rows only)."""
    out = reuse_passes(p1_only=True)
    for name in ("synth2zh",):
        out += text_source_passes(name, val=False)
    for ds in ("synth3", "synth3zh"):
        out += synth3_passes(ds, val=False)
    return out


def p1_val_passes() -> list[dict]:
    """P1 temperature-fit slices (val rows, v0 conventions)."""
    out = []
    for name in P1_TEXT_SOURCES:
        out += text_source_passes(name, val=True)
    for ds in ("synth3", "synth3zh"):
        out += synth3_passes(ds, val=True)
    return out


def p2_train_passes() -> list[dict]:
    """The P2 training set: the full v1.2 union + synth2zh + image
    families (train rows only)."""
    out = reuse_passes(p1_only=False)
    out += text_source_passes("synth2zh", val=False)
    for ds in ("synth3", "synth3zh"):
        out += synth3_passes(ds, val=False)
    return out


# ------------------------------------------------------------------- gates

def gate_slice(name: str, limit: int = 0) -> list[dict]:
    """Pre-registered gate/monitor slices, as multi-pass records.

    mnli/ocnli: first `limit` eval-only rows (never trained on) as noul
    passes. synth3/synth3zh/synth2zh/synth2-en: the first `limit` VAL rows
    (generation order, the v1.1-line convention); 0 = all of them.
    """
    if name in ("mnli", "ocnli"):
        rows = _load_rows(f"data_cache/rows_{name}.jsonl")[:limit or 10**9]
        out = []
        for r in rows:
            out.append(dict(row_id=r["row_id"], qtype="noul", kind="noul",
                            key=None, target=float(r["label"]), weight=1.0,
                            gold=float(r["label"]), soft=None,
                            text=noul_pass(r["state"], r["instructions"],
                                           r["criteria"]), image=None))
        return out
    if name in ("synth3", "synth3zh"):
        rows = [r for r in _load_rows(f"data_cache/{name}/items.jsonl")
                if is_val(r["row_id"])][: limit or 10**9]
        out = []
        for row in rows:
            out.extend(image_row_passes(row, IMG_DIRS[name]))
        return out
    if name in ("synth2zh", "synth2"):
        rows = [r for r in _load_rows(f"data_cache/rows_{name}.jsonl")
                if is_val(r["row_id"])][: limit or 10**9]
        return [_add_image(r, None)
                for r in expand_passes(rows, _criteria_map(name), {})]
    raise KeyError(name)


MONITOR_SLICES = (("mnli", 150), ("ocnli", 150), ("synth3", 300),
                  ("synth2zh", 200), ("synth2", 200))
FINAL_GATES = (("synth3", 0), ("mnli", 150), ("ocnli", 150),
               ("synth2zh", 0), ("synth2", 0))
# combined val for the pre-registered temperature refit: the five
# mixed-domain text sources + synth3/synth3zh/synth2/synth2zh val
COMBINED_VAL = (("boolq", 0), ("fever", 0), ("banking77", 0),
                ("clinc150", 0), ("helpsteer2", 0), ("synth3", 0),
                ("synth3zh", 0), ("synth2", 0), ("synth2zh", 0))


# ------------------------------------------------------- VL encode + h_last

def render_prompts(proc, passes) -> tuple[list[str], list]:
    """Chat-template render of every pass; images collected positionally
    (the processor maps them to the vision placeholders in order)."""
    texts, images = [], []
    for p in passes:
        content = ([{"type": "image", "image": p["image"]}] if p["image"]
                   else []) + [{"type": "text", "text": p["text"]}]
        texts.append(proc.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False))
        if p["image"]:
            images.append(p["image"])
    return texts, images


def vl_inputs(proc, tok, passes):
    """Batched processor call; asserts right padding (the h extraction and
    the training label math both index the first pad slot / last real
    token)."""
    texts, images = render_prompts(proc, passes)
    inputs = proc(text=texts, images=images or None, padding=True,
                  return_tensors="pt")
    if getattr(tok.tokenizer, "padding_side", "right") != "right":
        raise RuntimeError("h extraction assumes right padding")
    return inputs


def _last_h(hidden: torch.Tensor, inputs) -> torch.Tensor:
    """Final REAL token's hidden state per row (the readout position,
    identical to the text pipeline's convention)."""
    last = inputs["attention_mask"].sum(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device),
                  last.to(hidden.device)]


def extract_h(inner, proc, tok, passes, device, batch: int = 8) -> torch.Tensor:
    """(N, hidden) h_last, no grad, bf16 model -> float32 readout."""
    out = []
    with torch.no_grad():
        for i in range(0, len(passes), batch):
            chunk = passes[i: i + batch]
            inputs = vl_inputs(proc, tok, chunk)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = inner(**inputs, use_cache=False).last_hidden_state
            out.append(_last_h(hidden, inputs).float().cpu())
    return torch.cat(out)


def forward_h(inner, proc, tok, passes, device) -> torch.Tensor:
    """extract_h's training twin: gradients flow (P2's heads-on-h_last
    objective). Callers must have enabled input grads + checkpointing."""
    inputs = vl_inputs(proc, tok, passes)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hidden = inner(**inputs, use_cache=False).last_hidden_state
    return _last_h(hidden, inputs).float()


def token_cost(tok, p) -> int:
    """Real-token packing cost: text tokens + the measured 399 vision
    tokens/image + a fixed template/answer margin (the v1.1 convention)."""
    n = len(tok.tokenizer(p["text"], add_special_tokens=False)["input_ids"])
    return n + (399 + 24 if p["image"] else 24)


# ------------------------------------------------------------ heads readout

def group_rows(records: list[dict]) -> list[tuple[str, list[int]]]:
    """(row_id, indices into records), file order preserved (a source's
    passes are emitted row-consecutively by the builders)."""
    by_row: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_row[r["row_id"]].append(i)
    return list(by_row.items())


def read_rows(head, T: float, h: torch.Tensor, records: list[dict]):
    """Per-row readout for one primitive's records: noul -> (p, gold);
    choice/score -> (top-label confidence, correct)."""
    head.eval()
    h = h.to(next(head.parameters()).device)
    with torch.no_grad():
        z = head(h).cpu().tolist()
    out = []
    for _, g in group_rows(records):
        if records[g[0]]["qtype"] == "noul":
            p = 1 / (1 + math.exp(-(z[g[0]] / T)))
            gold = records[g[0]]["gold"]
            out.append((p, gold))
        else:
            s = [z[i] / T for i in g]
            m = max(s)
            e = [math.exp(x - m) for x in s]
            soft = [x / sum(e) for x in e]
            top = max(range(len(s)), key=lambda i: s[i])
            gold = next((pos for pos, i in enumerate(g)
                         if records[i]["target"] == 1.0), -1)
            out.append((soft[top], 1.0 if top == gold else 0.0))
    return out


def slice_accuracy(head, T: float, h: torch.Tensor, records: list[dict]) -> float:
    """Accuracy through the heads readout: noul = P >= 0.5 vs gold;
    choice/score = argmax over the row's group vs the gold pass."""
    if not records:
        return float("nan")
    rows = read_rows(head, T, h, records)
    if records[0]["qtype"] == "noul":
        return sum((p >= 0.5) == bool(g) for p, g in rows) / len(rows)
    return sum(c for _, c in rows) / len(rows)


def _inner_of(model):
    """Any wrapper spelling (PeftModel, PeftModel(CausalLM), the bare
    multimodal ForConditionalGeneration, dist-style merged load) ->
    the model whose forward takes input_ids + pixel_values and carries
    .layers (decoder) + .visual (the vision tower). Walks down through
    .model / .base_model indirections until it lands there."""
    m = model
    for _ in range(5):
        if hasattr(m, "layers") and hasattr(m, "visual"):
            return m
        nxt = getattr(m, "model", None)
        if nxt is None:
            nxt = getattr(m, "base_model", None)
        if nxt is None or nxt is m:
            return m
        m = nxt
    return m