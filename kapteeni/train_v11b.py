"""Kapteeni v1.1b trainer: warm-start from the v1.1 adapter, then
2 additional synth2-only epochs (EN + ZH, no images, no replay).

Governing document: docs/PREREG-KAPTEENI-V11B (committed before this
run; gates + abort rule frozen there). The b-run tests the volume
hypothesis: the v1.1 unified recipe failed its synth2 mastery gate
because ONE epoch cannot reconstitute the synth2 rule skills (the
diagnosis shows the same weakness on EN val and across backbones).

Mechanics (encode_batch / run_gate) come from train_v — the shared,
smoke-tested path. Differences from train_v11: warm start, synth2-only
mixture, a fifth monitor (synth2-EN-val), the pre-registered abort
rule, and monitor rounds appended to a JSONL so history survives log
loss (the first v1.1 attempt lost its monitor log with the process).

  HF_HUB_OFFLINE=1 python3 -m kapteeni.train_v11b --out model_cache/kapteeni_v11b
  HF_HUB_OFFLINE=1 python3 -m kapteeni.train_v11b --gates-only \
      --adapter model_cache/kapteeni_v11b/adapter
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
from kapteeni.train_v import encode_batch, run_gate
from kapteeni.train_v11 import build_examples as v11_build_examples
from kapteeni.vl_format import text_example, load_rows

WARM_BASE = "model_cache/qwen3.5-4b"
WARM_ADAPTER = "model_cache/kapteeni_v11/adapter"
EPOCHS = 2
ABORT_DROP = 0.10  # pre-registered (amended rule): catastrophic-only


def build_synth2_examples(seed: int = 11) -> list[dict]:
    """One synth2-heavy epoch: EN + ZH non-val rows, shuffled together."""
    rng = random.Random(seed)
    crit2 = json.loads(open("data_cache/synth2_criteria.json").read())
    crit2zh = json.loads(open("data_cache/synth2zh_criteria.json").read())
    rows = [r for r in load_rows("data_cache/rows_synth2.jsonl")
            if not is_val(r["row_id"])]
    rows += [r for r in load_rows("data_cache/rows_synth2zh.jsonl")
             if not is_val(r["row_id"])]
    rng.shuffle(rows)
    out = []
    for r in rows:
        is_zh = r["row_id"].startswith("synth2zh-")
        ex = text_example(r, crit2zh if is_zh else crit2)
        ex["source"] = "synth2zh" if is_zh else "synth2"
        out.append(ex)
    return out


def monitor_slices(val, gate_mnli, gate_ocnli):
    """The five pre-registered monitor lists (sliced to registered sizes)."""
    s3_mon = [e for e in val if e["source"] == "synth3"][:300]
    crit2zh = json.loads(open("data_cache/synth2zh_criteria.json").read())
    s2zh_mon = []
    for r in load_rows("data_cache/rows_synth2zh.jsonl"):
        if is_val(r["row_id"]):
            ex = text_example(r, crit2zh)
            ex["source"] = "synth2zh"
            s2zh_mon.append(ex)
    crit2 = json.loads(open("data_cache/synth2_criteria.json").read())
    s2_mon = []
    for r in load_rows("data_cache/rows_synth2.jsonl"):
        if is_val(r["row_id"]):
            ex = text_example(r, crit2)
            ex["source"] = "synth2"
            s2_mon.append(ex)
    return s3_mon, s2zh_mon[:200], s2_mon[:200], gate_mnli, gate_ocnli


def append_monitor(path: Path, step: int, accs: dict) -> None:
    """data_cache/v11b_monitors.jsonl (also prints the line)."""
    line = json.dumps({"step": step, "ts": time.time(),
                       "accs": accs}, ensure_ascii=False)
    with open(path, "a") as f:
        f.write(line + "\n")
    print(f"  [monitor {step}] {json.dumps(accs, ensure_ascii=False)}",
          flush=True)


def final_gates(model, proc, tok, val, gate_mnli, gate_ocnli, device) -> dict:
    """The six pre-registered gates, full slices."""
    s3 = [e for e in val if e["source"] == "synth3"]
    s3zh = [e for e in val if e["source"] == "synth3zh"]
    crit2 = json.loads(open("data_cache/synth2_criteria.json").read())
    crit2zh = json.loads(open("data_cache/synth2zh_criteria.json").read())
    s2 = []
    for r in load_rows("data_cache/rows_synth2.jsonl"):
        if is_val(r["row_id"]):
            ex = text_example(r, crit2)
            ex["source"] = "synth2"
            s2.append(ex)
    s2zh = []
    for r in load_rows("data_cache/rows_synth2zh.jsonl"):
        if is_val(r["row_id"]):
            ex = text_example(r, crit2zh)
            ex["source"] = "synth2zh"
            s2zh.append(ex)
    return {
        "synth3_val_n": len(s3),
        "synth3_val": run_gate(model, proc, tok, s3, device, limit=10**9),
        "synth3zh_val_n": len(s3zh),
        "synth3zh_val": run_gate(model, proc, tok, s3zh, device, limit=10**9),
        "mnli_noul_n": len(gate_mnli),
        "mnli_noul": run_gate(model, proc, tok, gate_mnli, device,
                              limit=10**9),
        "ocnli_noul_n": len(gate_ocnli),
        "ocnli_noul": run_gate(model, proc, tok, gate_ocnli, device,
                               limit=10**9),
        "synth2zh_val_n": len(s2zh),
        "synth2zh_val": run_gate(model, proc, tok, s2zh, device, limit=10**9),
        "synth2_en_val_n": len(s2),
        "synth2_en_val": run_gate(model, proc, tok, s2, device, limit=10**9),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_cache/kapteeni_v11b")
    ap.add_argument("--base", default=WARM_BASE)
    ap.add_argument("--warm", default=WARM_ADAPTER)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--gate-every", type=int, default=300)
    ap.add_argument("--ckpt-every", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--adapter", default="model_cache/kapteeni_v11b/adapter")
    args = ap.parse_args(argv)

    if args.gates_only:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
        proc = AutoProcessor.from_pretrained(args.base)
        tok = proc
        model = AutoModelForImageTextToText.from_pretrained(
            args.base, dtype=torch.bfloat16).to(args.device)
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        train_unused, val, gate_mnli, gate_ocnli = v11_build_examples()
        res = final_gates(model, proc, tok, val, gate_mnli, gate_ocnli,
                          args.device)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "final_gates.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return 0

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(args.base)
    tok = proc
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16).to(args.device)

    # warm start: the v1.1 adapter (or a local checkpoint on resume)
    ckpt = Path(args.out) / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    warm = str(ckpt / "adapter") \
        if (ckpt / "adapter").exists() and (ckpt / "opt.pt").exists() \
        else args.warm
    model = PeftModel.from_pretrained(model, warm, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    warm_tag = "LOCAL CHECKPOINT (resumed)" if warm.startswith(str(ckpt)) \
        else f"v1.1 adapter ({args.warm})"
    print(f"warm start: {warm_tag}", flush=True)
    step = 0
    if warm.startswith(str(ckpt)):
        step = json.loads((ckpt / "state.json").read_text())["step"]
        print(f"resumed at step {step}", flush=True)

    train, val, gate_mnli, gate_ocnli = v11_build_examples()
    if args.limit:
        epochs = [build_synth2_examples(seed=11 + e)[:args.limit]
                  for e in range(args.epochs)]
    else:
        epochs = [build_synth2_examples(seed=11 + e)
                  for e in range(args.epochs)]
    print(f"b-epochs: {args.epochs} x "
          f"{Counter(e['source'] for e in epochs[0])}", flush=True)
    slices = monitor_slices(val, gate_mnli, gate_ocnli)
    mon_path = Path("data_cache/v11b_monitors.jsonl")

    # optimizer is fresh for the b-run (AdamW, same LR);
    # the resume path reloads it from the local checkpoint
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    opt_state_loaded = False
    if warm.startswith(str(ckpt)):
        opt.load_state_dict(torch.load(ckpt / "opt.pt", weights_only=True))
        opt_state_loaded = True
        print("resumed optimizer state", flush=True)

    prompt_lens = tok.tokenizer(
        [e["prompt"] for e in epochs[0]], add_special_tokens=False)["input_ids"]

    def cost(i):
        return len(prompt_lens[i]) + 24

    prev = {}  # previous monitor accs, for the abort rule

    def batches_for(epoch_rows):
        order = sorted(range(len(epoch_rows)), key=cost)
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
        return batches

    t0 = time.time()
    base = 0  # global batch index of the next epoch's first batch
    for ep, rows in enumerate(epochs):
        batches = batches_for(rows)
        if ep == 0 and step == 0:
            print(f"batches per epoch: {len(batches)} "
                  f"(budget {args.token_budget})", flush=True)
        for bi, b in enumerate(batches):
            if base + bi < step:
                continue  # already done before the resume point
            batch = [rows[i] for i in b]
            inputs = encode_batch(proc, tok, batch)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            out_ = model(**inputs)
            out_.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  step {step} ep{ep + 1}/{args.epochs} "
                      f"loss {float(out_.loss):.4f} (peak {mem:.1f}G)",
                      flush=True)
                torch.cuda.reset_peak_memory_stats()
            if step % args.gate_every == 0:
                accs = {}
                g_m = run_gate(model, proc, tok, slices[3], args.device,
                               limit=10**9)
                g_o = run_gate(model, proc, tok, slices[4], args.device,
                               limit=10**9)
                g_i = run_gate(model, proc, tok, slices[0], args.device,
                               limit=10**9)
                g_z = run_gate(model, proc, tok, slices[1], args.device,
                               limit=10**9)
                g_e = run_gate(model, proc, tok, slices[2], args.device,
                               limit=10**9)
                accs = {"MNLI": g_m.get("text"), "OCNLI": g_o.get("text"),
                        "synth3_val": g_i.get("synth3"),
                        "synth2zh_val": g_z.get("synth2zh"),
                        "synth2_en_val": g_e.get("synth2")}
                append_monitor(mon_path, step, accs)
                # pre-registered abort rule (amended: catastrophic-only)
                for k, v in accs.items():
                    if v is None:
                        continue
                    best = max(v, prev.get(k, v))
                    prev[k] = best
                for k, v in accs.items():
                    if v is not None and prev[k] - v > ABORT_DROP:
                        print(f"ABORT RULE TRIGGERED: {k} "
                              f"{prev[k]} -> {v} (drop > {ABORT_DROP})",
                              flush=True)
                        model.save_pretrained(str(ckpt / "adapter"))
                        torch.save(opt.state_dict(), ckpt / "opt.pt")
                        (ckpt / "state.json").write_text(
                            json.dumps({"step": step}))
                        return 1
                prev = accs
            if step % args.ckpt_every == 0:
                model.save_pretrained(str(ckpt / "adapter"))
                torch.save(opt.state_dict(), ckpt / "opt.pt")
                (ckpt / "state.json").write_text(json.dumps({"step": step}))
        base += len(batches)  # global index steps past this epoch

    model.save_pretrained(str(Path(args.out) / "adapter"))
    (Path(args.out) / "recipe.json").write_text(json.dumps({
        "model": args.base, "warm_start": args.warm, "lora_r": 32,
        "alpha": 64, "lora_targets": "language_model q/k/v/o only",
        "lr": args.lr, "token_budget": args.token_budget,
        "b_epochs": args.epochs, "epoch_rows": len(epochs[0]),
        "steps": step, "prereg": "docs/PREREG-KAPTEENI-V11B.md",
    }, indent=2))
    print(f"done: {step} steps -> {args.out}/adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())