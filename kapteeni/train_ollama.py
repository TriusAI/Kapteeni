"""Student trainer for the Ollama port (PREREG-KAPTEENI-OLLAMA).

Trains a fresh LoRA on the teacher's merged backbone (the published
kapteeni-v1-meticulous pack) to reproduce the teacher's served
probability distributions as NEXT-TOKEN LETTER distributions — the
exact objective Ollama's runtime scores (softmax over the candidate
letters at T=1; docs/OLLAMA-PROMPT-SPEC.md).

Loss (frozen in the pre-reg): 0.5 * CE over the restricted letter
softmax against the teacher's distribution + 0.5 * hard CE against
the argmax letter on the FULL vocabulary (guarantees the letters
outrank every other token — the runner's outrank check).

  HF_HUB_OFFLINE=1 setsid nohup python3 -m kapteeni.train_ollama \
      --out model_cache/kapteeni_ollama \
      > /tmp/opencode/train_ollama.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

TEACHER_DIST = "../kapteeni-v1-meticulous-dist"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Run-2 recipe (PREREG-KAPTEENI-OLLAMA.md, frozen before the run):
# cuts are deterministic (row_id-sorted first-N of the non-val
# lessons); oversample replicates lessons within the single epoch.
CUTS = {"banking77": 600, "clinc150": 1000, "synth": 2500}
REBALANCE = {"goemotions": 3, "helpsteer2": 3, "fever": 2, "boolq": 2}


def load_examples(data_dir: str, train_only: bool = True) -> list[dict]:
    out = []
    for p in sorted(Path(data_dir).glob("teach_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line (generation may be running)
            if train_only and r["is_val"]:
                continue
            out.append(r)
    return out


def load_train(data_dir: str, extra_dir: str = "") -> list[dict]:
    """Non-val lessons from the main dir + optional extra dir,
    rebalanced per the frozen Run-2 recipe."""
    rows = load_examples(data_dir, train_only=True)
    if extra_dir:
        rows += load_examples(extra_dir, train_only=True)
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    out: list[dict] = []
    for src, rs in sorted(by_src.items()):
        if src in CUTS:
            rs = sorted(rs, key=lambda r: r["row_id"])[: CUTS[src]]
        out += rs * REBALANCE.get(src, 1)
    return out


def batches_by_cost(lengths: list[int], budget: int, cap: int = 12):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_c = [], [], 0
    for i in order:
        c = lengths[i]
        if cur and (cur_c + c > budget or len(cur) >= cap):
            batches.append(cur)
            cur, cur_c = [], 0
        cur.append(i)
        cur_c += c
    if cur:
        batches.append(cur)
    return batches


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_cache/kapteeni_ollama")
    ap.add_argument("--base", default=TEACHER_DIST)
    ap.add_argument("--data", default="data_cache/ollama_port")
    ap.add_argument("--extra-data", default="",
                    help="extra lessons dir (Run-2 fresh synth2x)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-every", type=int, default=300)
    ap.add_argument("--ckpt-every", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hard-weight", type=float, default=0.5,
                    help="loss weight of the hard argmax-CE term (the "
                    "outrank auxiliary); the rest is the soft "
                    "teacher-distribution CE. Run-2's diagnosed value "
                    "for shape fidelity: 0.1")
    ap.add_argument("--no-ckpt", action="store_true",
                    help="disable gradient checkpointing (fast config: "
                    "the 96G box fits the 4B text model at budget 8192 "
                    "without recompute)")
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16).to(args.device)
    model.config.use_cache = False

    from peft import LoraConfig, PeftModel, get_peft_model
    ckpt = Path(args.out) / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    if (ckpt / "adapter").exists() and (ckpt / "state.json").exists():
        model = PeftModel.from_pretrained(model, str(ckpt / "adapter"),
                                          is_trainable=True)
        step = json.loads((ckpt / "state.json").read_text())["step"]
        print(f"resumed at step {step}", flush=True)
    else:
        lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=r".*\.(q|k|v|o|gate|up|down)_proj$")
        model = get_peft_model(model, lcfg)
        step = 0
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.no_ckpt:
        # unwrap to the base transformer and disable recompute
        target = model.base_model.model if hasattr(
            model, "base_model") else model
        if hasattr(target, "gradient_checkpointing_disable"):
            target.gradient_checkpointing_disable()
    model.train()
    n_train = sum(p.numel() for p in model.parameters()
                  if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.2f}M", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    if step:
        opt.load_state_dict(torch.load(ckpt / "opt.pt",
                                        weights_only=True))

    # letter token ids: single-token A..Z (asserted; the runtime
    # requires candidates to append exactly one ordinary token)
    letter_ids = []
    for ch in LETTERS:
        ids = tok(ch, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"letter {ch} is not a single token"
        letter_ids.append(ids[0])
    letter_ids_t = torch.tensor(letter_ids, device=args.device)

    train = load_train(args.data, args.extra_data)
    if args.limit:
        train = train[: args.limit]
    # per-source val slices (Run 1's process fix: monitor every
    # source, 40 val rows each, not a biased global slice)
    val_by_src: dict[str, list[dict]] = {}
    for r in load_examples(args.data, train_only=False):
        if r["is_val"]:
            val_by_src.setdefault(r["source"], [])
            if len(val_by_src[r["source"]]) < 40:
                val_by_src[r["source"]].append(r)
    val = [r for rs in val_by_src.values() for r in rs]
    print(f"train: {len(train)} (rebalanced) | "
          f"val monitor: {len(val)} across {len(val_by_src)} sources",
          flush=True)
    from collections import Counter
    print("  per-source:", dict(Counter(r["source"] for r in train)),
          flush=True)

    lengths = [len(tok(r["prompt"], add_special_tokens=False)
                   ["input_ids"]) + 2 for r in train]
    batches = batches_by_cost(lengths, args.token_budget)
    print(f"batches/epoch: {len(batches)} x {args.epochs}", flush=True)

    def encode(rows: list[dict]):
        texts = [r["prompt"] for r in rows]
        enc = tok(texts, add_special_tokens=False, padding=True,
                  return_tensors="pt")
        n = enc["attention_mask"].sum(1)  # real lengths (right padding)
        return enc, n

    def letter_loss(rows, logits, n):
        # last REAL token position per row; logits there predict the
        # letter
        idx = (n - 1).to(args.device)
        last_logits = logits[torch.arange(logits.shape[0],
                                          device=args.device), idx]
        soft_loss = hard_loss = None
        for bi, r in enumerate(rows):
            k = len(r["candidates"])
            lids = letter_ids_t[:k]
            restricted = last_logits[bi][lids]
            logq = F.log_softmax(restricted, dim=-1)
            tgt = torch.tensor(r["target_probs"], device=args.device)
            s = -(tgt * logq).sum()
            hard = F.cross_entropy(
                last_logits[bi].unsqueeze(0),
                lids[int(tgt.argmax())].unsqueeze(0))
            soft_loss = s if soft_loss is None else soft_loss + s
            hard_loss = hard if hard_loss is None else hard_loss + hard
        b = len(rows)
        hw = args.hard_weight  # closure over main()'s args
        return ((1 - hw) * soft_loss + hw * hard_loss) / b

    def monitor() -> dict:
        """Per-source argmax-letter agreement with the teacher."""
        model.eval()
        agree = {s: 0 for s in val_by_src}
        n = {s: 0 for s in val_by_src}
        with torch.no_grad():
            for src, rows in val_by_src.items():
                for i in range(0, len(rows), 8):
                    chunk = rows[i:i + 8]
                    enc, lens = encode(chunk)
                    inputs = {k: v.to(args.device)
                              for k, v in enc.items()}
                    logits = model(**inputs).logits
                    idx = (lens - 1).to(args.device)
                    for bi, r in enumerate(chunk):
                        k = len(r["candidates"])
                        lids = letter_ids_t[:k]
                        pred = logits[torch.arange(
                            len(chunk), device=args.device)[bi],
                            idx[bi]][lids]
                        t = torch.tensor(r["target_probs"],
                                         device=args.device)
                        agree[src] += int(pred.argmax() == t.argmax())
                        n[src] += 1
        model.train()
        return {s: agree[s] / max(n[s], 1) for s in agree}

    base_step = step
    t0 = time.time()
    for ep in range(args.epochs):
        for bi, b in enumerate(batches):
            gidx = base_step + ep * len(batches) + bi
            if gidx < step:
                continue  # resume fast-forward
            rows = [train[i] for i in b]
            enc, lens = encode(rows)
            inputs = {k: v.to(args.device) for k, v in enc.items()}
            logits = model(**inputs).logits
            loss = letter_loss(rows, logits, lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  step {step} loss {float(loss):.4f} "
                      f"(peak {mem:.1f}G)", flush=True)
                torch.cuda.reset_peak_memory_stats()
            if step % args.gate_every == 0:
                m = monitor()
                print(f"  [monitor {step}] " + " ".join(
                    f"{s[:6]}={v:.3f}" for s, v in sorted(m.items())),
                    flush=True)
            if step % args.ckpt_every == 0:
                model.save_pretrained(str(ckpt / "adapter"))
                torch.save(opt.state_dict(), ckpt / "opt.pt")
                (ckpt / "state.json").write_text(json.dumps(
                    {"step": step}))

    out = Path(args.out)
    model.save_pretrained(str(out / "adapter"))
    (out / "recipe.json").write_text(json.dumps({
        "base": args.base, "prereg": "docs/PREREG-KAPTEENI-OLLAMA.md",
        "lora_r": 32, "alpha": 64, "lr": args.lr, "epochs": args.epochs,
        "budget": args.token_budget, "seed": args.seed,
        "train_examples": len(train), "steps": step}, indent=2))
    print(f"done: {step} steps -> {out}/adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())