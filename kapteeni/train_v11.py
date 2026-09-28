"""Kapteeni v1.1 trainer: the unified image + English + Chinese model.

Governing document: docs/PREREG-KAPTEENI-V11 (gates fixed there; this
file implements exactly that recipe — one run, no bench-gated tuning).

Backbone: Qwen3.5-4B (natively multimodal; the V-track's text-erosion
diagnosis says the bolted-on Qwen3-VL tower + full-projection LoRA was
the likely culprit, so this recipe narrows to attention-only LoRA and
halves the LR). Vision tower frozen.

Mixture: synth3 + synth3zh image rows, the six English replay sources
at full non-val volume, synth2zh. Answers stay yes/no + letters in all
languages (the wire answer space is fixed by the pre-registration).

In-loop monitors (NOT gates), every --gate-every steps: MNLI-noul[150],
OCNLI-noul[150], synth3-val[:300], synth2zh-val[:200]. Final gates run
via --gates-only on the completed adapter.

  HF_HUB_OFFLINE=1 python3 -m kapteeni.train_v11 --out model_cache/kapteeni_v11
  HF_HUB_OFFLINE=1 python3 -m kapteeni.train_v11 --gates-only \
      --adapter model_cache/kapteeni_v11/adapter
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from kapteeni.build_data import is_val
from kapteeni.train_v import CRITERIA, TEXT_REPLAY, encode_batch, run_gate
from kapteeni.vl_format import synth3_example, text_example, load_rows

MODEL = "model_cache/qwen3.5-4b"
IMG_TOKENS = 399  # measured: the Qwen3.5-4B processor emits 399 image
# placeholder tokens for a 640x640 render (the 400/image constant was
# Qwen3-VL-specific; re-measured per the pre-registration's engineering
# clause — nearly unchanged).


def build_examples(seed: int = 7):
    """(train, image-val, mnli-gate, ocnli-gate) per the v1.1 recipe."""
    rng = random.Random(seed)
    train, val = [], []

    for ds, img_dir in (("synth3", "data_cache/synth3/images"),
                        ("synth3zh", "data_cache/synth3zh/images")):
        for r in load_rows(f"data_cache/{ds}/items.jsonl"):
            ex = synth3_example(r, img_dir)
            if ds == "synth3zh":
                ex["source"] = "synth3zh"
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

    crit2 = json.loads(open("data_cache/synth2_criteria.json").read())
    for r in load_rows("data_cache/rows_synth2.jsonl"):
        if not is_val(r["row_id"]):
            train.append(text_example(r, crit2))

    crit2zh = json.loads(open("data_cache/synth2zh_criteria.json").read())
    n_zh = 0
    for r in load_rows("data_cache/rows_synth2zh.jsonl"):
        if not is_val(r["row_id"]):
            ex = text_example(r, crit2zh)
            ex["source"] = "synth2zh"
            train.append(ex)
            n_zh += 1
    rng.shuffle(train)

    # gate slices (eval only, never trained on): the pre-registered
    # n=150 NLI slices
    mnli = load_rows("data_cache/rows_mnli.jsonl")[:150]
    ocnli = load_rows("data_cache/rows_ocnli.jsonl")[:150]
    return train, val, [text_example(r, {}) for r in mnli], \
        [text_example(r, {}) for r in ocnli]


def make_lora(model, r: int = 32, alpha: int = 64):
    """Attention projections only (q/k/v/o of the language model) — the
    narrower update footprint the V2/V2.1 diagnosis calls for. The
    Gated DeltaNet layers have no q/k/v/o projections and stay untouched,
    as does the vision tower (no LoRA targets there at all)."""
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|"
                       r"o_proj)$",
    )
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def final_gates(model, proc, tok, val, gate_mnli, gate_ocnli, zh_val,
                device) -> dict:
    """The five pre-registered gates, full slices (ECE fitting is a
    post-gate finalize step, not run here)."""
    s3 = [e for e in val if e["source"] == "synth3"]
    s3zh = [e for e in val if e["source"] == "synth3zh"]
    out = {
        "synth3_val_n": len(s3),
        "synth3_val": run_gate(model, proc, tok, s3, device,
                               limit=10**9, pad_to=512),
        "synth3zh_val_n": len(s3zh),
        "synth3zh_val": run_gate(model, proc, tok, s3zh, device,
                                 limit=10**9, pad_to=512),
        "mnli_noul_n": len(gate_mnli),
        "mnli_noul": run_gate(model, proc, tok, gate_mnli, device,
                              limit=10**9, pad_to=512),
        "ocnli_noul_n": len(gate_ocnli),
        "ocnli_noul": run_gate(model, proc, tok, gate_ocnli, device,
                               limit=10**9, pad_to=512),
        "synth2zh_val_n": len(zh_val),
        "synth2zh_val": run_gate(model, proc, tok, zh_val, device,
                                 limit=10**9, pad_to=512),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_cache/kapteeni_v11")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--gate-every", type=int, default=300)
    ap.add_argument("--ckpt-every", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gates-only", action="store_true",
                    help="no training: run the five final gates on a "
                         "completed adapter")
    args = ap.parse_args(argv)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device)

    if args.gates_only:
        assert args.adapter, "--adapter is required with --gates-only"
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        train, val, gate_mnli, gate_ocnli = build_examples()
        crit2zh = json.loads(open("data_cache/synth2zh_criteria.json").read())
        zh_val = [text_example(r, crit2zh)
                  for r in load_rows("data_cache/rows_synth2zh.jsonl")
                  if is_val(r["row_id"])]
        for e in zh_val:
            e["source"] = "synth2zh"
        res = final_gates(model, proc, tok, val, gate_mnli, gate_ocnli,
                          zh_val, args.device)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "final_gates.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return 0

    out_dir = Path(args.out)
    ckpt = out_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)

    train, val, gate_mnli, gate_ocnli = build_examples()
    if args.limit:
        train = train[:args.limit]
    print(f"train examples: {len(train)} "
          f"({Counter(e['source'] for e in train)})")
    print(f"image val: synth3 {sum(1 for e in val if e['source'] == 'synth3')}"
          f" + synth3zh "
          f"{sum(1 for e in val if e['source'] == 'synth3zh')}; "
          f"gate slices: MNLI {len(gate_mnli)}, OCNLI {len(gate_ocnli)}",
          flush=True)

    # in-loop monitors exactly as pre-registered (not gates)
    s3_mon = [e for e in val if e["source"] == "synth3"][:300]
    crit2zh = json.loads(open("data_cache/synth2zh_criteria.json").read())
    zh_train_val = [text_example(r, crit2zh)
                    for r in load_rows("data_cache/rows_synth2zh.jsonl")
                    if is_val(r["row_id"])]
    for e in zh_train_val:
        e["source"] = "synth2zh"
    s2zh_mon = zh_train_val[:200]

    model = make_lora(model)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    prompt_lens = tok.tokenizer(
        [e["prompt"] for e in train], add_special_tokens=False)["input_ids"]

    def cost(i):
        c = len(prompt_lens[i]) + 24
        if train[i]["image"]:
            c += IMG_TOKENS
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

    t0 = time.time()
    for bi in range(step, len(batches)):
        batch = [train[i] for i in batches[bi]]
        inputs = encode_batch(proc, tok, batch, pad_to=512)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        out = model(**inputs)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0:
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"  step {step}/{len(batches)} loss {float(out.loss):.4f}"
                  f" (peak {mem:.1f}G)", flush=True)
            torch.cuda.reset_peak_memory_stats()
        if step % args.gate_every == 0:
            # monitors are pre-sliced to the registered sizes; no
            # run_gate-side truncation
            g_m = run_gate(model, proc, tok, gate_mnli, args.device,
                           limit=10**9, pad_to=512)
            g_o = run_gate(model, proc, tok, gate_ocnli, args.device,
                           limit=10**9, pad_to=512)
            g_i = run_gate(model, proc, tok, s3_mon, args.device,
                           limit=10**9, pad_to=512)
            g_z = run_gate(model, proc, tok, s2zh_mon, args.device,
                           limit=10**9, pad_to=512)
            print(f"  [monitor {step}] MNLI={g_m.get('text')} "
                  f"OCNLI={g_o.get('text')} "
                  f"synth3-val={g_i.get('synth3')} "
                  f"synth2zh-val={g_z.get('synth2zh')}", flush=True)
        if step % args.ckpt_every == 0:
            model.save_pretrained(str(ckpt / "adapter"))
            torch.save(opt.state_dict(), ckpt / "opt.pt")
            (ckpt / "state.json").write_text(json.dumps({"step": step}))

    model.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "recipe.json").write_text(json.dumps({
        "model": args.model, "lora_r": 32, "alpha": 64,
        "lora_targets": "language_model q/k/v/o only",
        "vision_tower": "frozen", "lr": args.lr,
        "token_budget": args.token_budget, "epochs": 1,
        "img_tokens_measured": IMG_TOKENS,
        "text_replay": TEXT_REPLAY, "train_examples": len(train),
    }, indent=2))
    print(f"done: {step} steps -> {out_dir}/adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())