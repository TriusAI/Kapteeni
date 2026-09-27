"""V2 trainer: Kapteeni-V — LoRA on Qwen3-VL-4B-Instruct's language
projections (vision tower frozen), answer-token supervision through the
LM head (docs/PREREG-KAPTEENI-V.md + the 2026-09-27 readout amendment).

    python3 -m kapteeni.train_v --out model_cache/kapteeni_v2_1

Pre-registered V2.1 recipe (token-parity replay; fixed before the run):
  - image data: ALL synth3 train rows (5,412 after the deterministic
    val split), one pass per question, lettered options (~4.3M tokens)
  - text replay, same lettered format, ALL non-val rows from BoolQ,
    FEVER, Banking77, CLINC150, HelpSteer2 and synth2 (~12.8k rows,
    ~4.3M tokens) — text:image ~1:1 in real tokens (V2's ~1:2 ratio
    failed the MNLI gate; see the PREREG V2 OUTCOME)
  - MNLI: NEVER trained on (the text OOD gate, >= 0.88 final)
  - LoRA r=32 alpha=64 dropout 0.05 on language-model projections only,
    lr 1e-4, one epoch, REAL-token budget 4096, grad checkpointing
  - in-loop gates: MNLI noul slice + synth3 val slice every 300 steps;
    final gates measured on the completed adapter
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from kapteeni.build_data import is_val
from kapteeni.vl_format import LETTERS, synth3_example, text_example, load_rows

MODEL = "Qwen/Qwen3-VL-4B-Instruct"

TEXT_REPLAY = {  # rows per source; "all" = every non-val row (V2.1 parity recipe)
    "boolq": "all", "fever": "all", "banking77": "all", "clinc150": "all",
    "helpsteer2": "all",
}
TEXT_REPLAY_V2 = {  # the failed first recipe, kept for the record
    "boolq": 1500, "fever": 800, "banking77": 1200, "clinc150": 800,
    "helpsteer2": 400,
}

CRITERIA = {
    "banking77": "data_cache/crit_banking77.json",
    "clinc150": "data_cache/crit_clinc150.json",
    "helpsteer2": "data_cache/crit_helpsteer2.json",
}


def build_examples(seed: int = 7) -> tuple[list[dict], list[dict], list[dict]]:
    """(train examples, val examples) per the pre-registered recipe."""
    rng = random.Random(seed)
    train, val = [], []

    s3 = load_rows("data_cache/synth3/items.jsonl")
    img_dir = "data_cache/synth3/images"
    for r in s3:
        ex = synth3_example(r, img_dir)
        (val if is_val(r["row_id"]) else train).append(ex)

    for name, n in TEXT_REPLAY.items():
        rows = load_rows(f"data_cache/rows_{name}.jsonl")
        crit = json.loads(open(CRITERIA[name]).read()) \
            if name in CRITERIA else {}
        rows = [r for r in rows if not is_val(r["row_id"])]
        rng.shuffle(rows)
        take = rows if n == "all" else rows[:n]
        for r in take:
            train.append(text_example(r, crit))
    # synth2 text replay at full non-val volume (V2.1 parity recipe)
    s2 = [r for r in load_rows("data_cache/rows_synth2.jsonl")
          if not is_val(r["row_id"])]
    rng.shuffle(s2)
    crit2 = json.loads(open("data_cache/synth2_criteria.json").read())
    for r in s2:
        train.append(text_example(r, crit2))
    rng.shuffle(train)
    # MNLI gate slice (eval only, never trained on)
    mnli = load_rows("data_cache/rows_mnli.jsonl")[:150]
    gate_val = [text_example(r, {}) for r in mnli]
    return train, val, gate_val


def make_lora(model, r: int = 32, alpha: int = 64):
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|"
                       r"o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def encode_batch(proc, tok, examples):
    """Official batched processor call (placeholder expansion happens
    inside); answer supervision via a label placed at the first pad
    position of each row, whose logits come from that row's last real
    token (the model's internal label shift)."""
    texts, images = [], []
    for ex in examples:
        content = ([{"type": "image", "image": ex["image"]}] if ex["image"]
                   else []) + [{"type": "text", "text": ex["prompt"]}]
        texts.append(proc.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False))
        if ex["image"]:
            images.append(ex["image"])
    inputs = proc(text=texts, images=images or None, padding=True,
                  return_tensors="pt")
    if getattr(tok.tokenizer, "padding_side", "right") != "right":
        raise RuntimeError("this label placement assumes right padding")
    labels = torch.full_like(inputs["input_ids"], -100)
    ans_ids = [tok.tokenizer.encode(ex["answer"],
                                    add_special_tokens=False)[0]
               for ex in examples]
    lens = inputs["attention_mask"].sum(1)
    old_w = inputs["input_ids"].shape[1]
    if int(lens.max()) == old_w:
        # the longest row has no pad slot; add one column to every
        # sequence-width tensor (input_ids, attention_mask,
        # mm_token_type_ids, ...)
        pad_id = tok.tokenizer.pad_token_id or tok.tokenizer.eos_token_id
        n = labels.shape[0]
        for k in list(inputs):
            v = inputs[k]
            if v.ndim == 2 and v.shape[1] == old_w:
                fill = pad_id if k == "input_ids" else 0
                inputs[k] = torch.cat(
                    [v, torch.full((n, 1), fill, dtype=v.dtype)], dim=1)
        labels = torch.full_like(inputs["input_ids"], -100)
    for i, a in enumerate(ans_ids):
        labels[i, lens[i]] = a  # first pad slot == the answer position
    inputs["labels"] = labels
    return inputs


@torch.no_grad()
def run_gate(model, proc, tok, examples, device, limit=120):
    """Accuracy of the argmax readout on a gate slice."""
    model.eval()
    by = defaultdict(lambda: [0, 0])
    for i in range(0, min(limit, len(examples)), 8):
        batch = examples[i:i + 8]
        # prompts only, no answer appended
        texts, images = [], []
        for ex in batch:
            messages = [{"role": "user", "content": (
                [{"type": "image", "image": ex["image"]}] if ex["image"]
                else []) + [{"type": "text", "text": ex["prompt"]}]}]
            texts.append(proc.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False))
            if ex["image"]:
                images.append(ex["image"])
        inputs = proc(text=texts, images=images or None,
                      return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        for j, ex in enumerate(batch):
            L = int(inputs["attention_mask"][j].sum())
            row = logits[j, L - 1, :]
            if ex["primitive"] == "noul":
                y = tok.tokenizer.encode("yes", add_special_tokens=False)[0]
                n = tok.tokenizer.encode("no", add_special_tokens=False)[0]
                pred = 1 if row[y] > row[n] else 0
                ok = pred == ex["gold"]
            else:
                ids = [tok.tokenizer.encode(" " + LETTERS[k],
                                             add_special_tokens=False)[0]
                       for k in range(ex["n_cands"])]
                pred = int(torch.argmax(row[ids]))
                ok = pred == ex["gold"]
            src = ex["source"]
            by[src][0] += int(ok)
            by[src][1] += 1
    model.train()
    return {k: round(v[0] / v[1], 4) for k, v in by.items() if v[1]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_cache/kapteeni_v2")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--token-budget", type=int, default=6144)
    ap.add_argument("--gate-every", type=int, default=300)
    ap.add_argument("--ckpt-every", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frozen-baseline", action="store_true",
                    help="no training: measure the frozen backbone on the "
                         "V2 gate slices (the reference the trained model "
                         "must beat)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    ckpt = out_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)

    train, val, gate_val = build_examples()
    if args.limit:
        train = train[:args.limit]
    print(f"train examples: {len(train)} "
          f"({Counter(e['source'] for e in train)})")
    print(f"synth3 val (image gate): {len(val)}; MNLI gate slice: "
          f"{len(gate_val)}")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device)

    if args.frozen_baseline:
        print("frozen baseline on the V2 gate slices (no training):",
              flush=True)
        model.eval()
        g1 = run_gate(model, proc, tok, gate_val, args.device, limit=150)
        g2 = run_gate(model, proc, tok, val, args.device, limit=100000)
        out = {"mnli_noul": g1.get("text"),
               "synth3_val": g2.get("synth3")}
        (out_dir / "frozen_baseline.json").write_text(json.dumps(out,
                                                                indent=2))
        Path("data_cache/v2_frozen_baseline.json").write_text(
            json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return 0

    model = make_lora(model)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    prompt_lens = tok.tokenizer(
        [e["prompt"] for e in train], add_special_tokens=False)["input_ids"]

    # batch by REAL token count (prompt tokens + chat-template overhead
    # + ~400 vision tokens per image); the fake-unit budget of the first
    # attempt packed 3x heavier batches than intended and OOM'd
    def cost(i):
        c = len(prompt_lens[i]) + 24
        if train[i]["image"]:
            c += 400
        return c
    order = sorted(range(len(train)), key=cost)
    batches, cur, cur_c = [], [], 0
    for i in order:
        c = cost(i)
        if cur and (cur_c + c > args.token_budget or len(cur) >= 12):
            batches.append(cur)
            cur, cur_c = [], 0
        cur.append(i)
        cur_c += c
    if cur:
        batches.append(cur)
    print(f"batches: {len(batches)} "
          f"(real-token budget {args.token_budget})", flush=True)

    # resume: adapter + optimizer + step from the checkpoint dir
    step = 0
    if (ckpt / "adapter").exists() and (ckpt / "opt.pt").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model.base_model.model,
                                          str(ckpt / "adapter"))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        opt.load_state_dict(torch.load(ckpt / "opt.pt", weights_only=True))
        step = json.loads((ckpt / "state.json").read_text())["step"]
        model.train()
        print(f"resumed at step {step}", flush=True)

    step, t0, done = step, time.time(), 0
    for bi in range(step, len(batches)):
        batch = [train[i] for i in batches[bi]]
        inputs = encode_batch(proc, tok, batch)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        out = model(**inputs)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        done += sum(cost(i) for i in batches[bi])
        if step % 20 == 0:
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"  step {step}/{len(batches)} loss {float(out.loss):.4f}"
                  f" ({done / (time.time() - t0):.0f} real tok/s, "
                  f"peak {mem:.1f}G)", flush=True)
            torch.cuda.reset_peak_memory_stats()
        if step % args.gate_every == 0:
            g1 = run_gate(model, proc, tok, gate_val, args.device)
            g2 = run_gate(model, proc, tok, val, args.device)
            print(f"  [gate {step}] MNLI noul acc={g1.get('text')} "
                  f"synth3-val acc={g2.get('synth3')}", flush=True)
        if step % args.ckpt_every == 0:
            model.save_pretrained(str(ckpt / "adapter"))
            torch.save(opt.state_dict(), ckpt / "opt.pt")
            (ckpt / "state.json").write_text(json.dumps({"step": step}))

    model.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "recipe.json").write_text(json.dumps({
        "model": args.model, "lora_r": 32, "alpha": 64, "lr": args.lr,
        "token_budget": args.token_budget, "epochs": 1,
        "text_replay": TEXT_REPLAY, "train_examples": len(train),
    }, indent=2))
    print(f"done: {step} steps -> {out_dir}/adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())