"""V0 probe: frozen Qwen3-VL-4B-Instruct readout on the synth3 probe items.

Zero training — the djev-style feasibility check. One forward pass per
question; the image enters the state prefix through the vision tower;
candidate probabilities are read from the LM head at the `answer:`
position (lettered options for choice/score, yes/no for noul).

    python3 scripts/vprobe.py            # -> data_cache/vprobe/report.json

Pre-registered gate (docs/PREREG-KAPTEENI-V.md): >= 70% accuracy on the
27-item probe, deterministic across two runs, every answer schema-valid.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kapteeni.serialize import instructions_text, state_text  # noqa: E402

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
LETTERS = "ABCDEFGH"


def letter_token_ids(tok):
    """Ids for ' A', ' B', ... (the tokens written right after 'answer:')."""
    return [tok.tokenizer(" " + L, add_special_tokens=False)["input_ids"][0]
            for L in LETTERS]


def yes_no_token_ids(tok):
    return (tok.tokenizer("yes", add_special_tokens=False)["input_ids"][0],
            tok.tokenizer("no", add_special_tokens=False)["input_ids"][0])


def build_prompt(row) -> str:
    st = state_text({"image": "<attached>", "note": row["state_text"]})
    if row["primitive"] == "noul":
        body = (f"[noul] {instructions_text(row['instructions'])}\n"
                f"criteria - true: {row['criteria']['true']} | "
                f"false: {row['criteria']['false']}\nanswer:")
    else:
        opts = row.get("options") or row.get("levels")
        lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        kind = "choice" if row["primitive"] == "choice" else "score"
        body = (f"[{kind}] {instructions_text(row['instructions'])}\n"
                f"{lines}\nanswer:")
    return st + "\n\n" + body


@torch.no_grad()
def readout(model, proc, tok, row, img_path, device):
    prompt = build_prompt(row)
    # the chat template inserts the vision placeholder tokens into the
    # text; the raw text= API does not (transformers 5.15)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": prompt},
    ]}]
    inputs = proc.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0, -1, :]
    if row["primitive"] == "noul":
        y_id, n_id = yes_no_token_ids(tok)
        return torch.softmax(logits[[y_id, n_id]], dim=0)[0].item()
    ids = letter_token_ids(tok)[:len(row.get("options") or row["levels"])]
    return torch.softmax(logits[ids], dim=0).tolist()


def score_answer(probs, row):
    if row["primitive"] == "noul":
        return (probs >= 0.5) == bool(row["label"])
    pred = probs.index(max(probs))
    return pred == row["label"]


def main() -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda"
    tok = AutoProcessor.from_pretrained(MODEL)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device).eval()

    items = [json.loads(l) for l in open(ROOT / "data_cache/vprobe/items.jsonl")]
    imgdir = ROOT / "data_cache/vprobe/images"

    # determinism: run the first item twice
    p1 = readout(model, proc, tok, items[0],
                 str(imgdir / items[0]["image"]), device)
    p2 = readout(model, proc, tok, items[0],
                 str(imgdir / items[0]["image"]), device)
    deterministic = p1 == p2
    print(f"determinism: {p1} == {p2} -> {deterministic}")

    fam: dict[str, list] = defaultdict(list)
    for row in items:
        probs = readout(model, proc, tok, row,
                        str(imgdir / row["image"]), device)
        ok = score_answer(probs, row)
        fam[row["meta"]["family"]].append(ok)
        conf = max(probs) if isinstance(probs, list) else max(probs, 1 - probs)
        print(f"  {row['row_id']:<28} {row['primitive']:<6} "
              f"correct={ok} conf={conf:.3f}")

    report = {"families": {f: sum(v) / len(v) for f, v in fam.items()},
              "overall": sum(sum(v) for v in fam.values())
              / sum(len(v) for v in fam.values()),
              "deterministic": deterministic, "n_items": len(items)}
    print("\n" + json.dumps(report, indent=2))
    (ROOT / "data_cache/vprobe/report.json").write_text(json.dumps(report,
                                                                   indent=2))
    print("wrote data_cache/vprobe/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())