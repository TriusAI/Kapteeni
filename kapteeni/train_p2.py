"""P2 trainer: LoRA on the backbone + trained heads, joint, on the broadened
decision mix.

    python3 -m kapteeni.train_p2 --passes data_cache/passes_p2.jsonl \
        --bundle model_cache/kapteeni_v0.pt --out model_cache/kapteeni_p2 \
        --token-budget 7168 --lr 1e-4 --epochs 1

    (v1's actual recipe: token-budget 8192 — 1205 steps, 8.7M tokens, peak
    ~21G; the argparse default of 12288 predates v1 and yields fewer,
    bigger batches. Soup seeds must match v1 at 8192.)

Design (plan §6 P2, adjusted by today's findings):
  - LoRA r=32/alpha=64 on all attention+MLP projections (peft), bf16,
    gradient checkpointing; heads warm-started from the P1 bundle and
    trainable (they must re-learn on the adapted features).
  - Batches are ROWS (a choice row's option passes stay together for the
    group softmax loss), packed to a token budget, length-sorted.
  - Losses: noul BCE vs soft teacher labels (weight = agreement), choice
    group-CE, score per-level BCE — proper scoring rules only.
  - OOD gate: MNLI is EXCLUDED from training; every eval interval the head's
    MNLI accuracy/ECE is logged — generalization must not regress.
  - Checkpoint every N steps (adapter+heads), resumable.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from kapteeni.backbone import DEFAULT_MODEL, load_backbone
from kapteeni.build_data import is_val
from kapteeni.heads import PassMLP


def load_passes(paths: list[str]) -> list[dict]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            rows[r["row_id"]].append(r)
    out = []
    for rid, ps in rows.items():
        first = ps[0]
        out.append({
            "row_id": rid, "qtype": first["qtype"], "kind": first["kind"],
            "target": [p["target"] for p in ps],
            "weight": [p.get("weight", 1.0) for p in ps],
            "texts": [p["text"] for p in ps],
        })
    return out


def pack_batches(rows: list[dict], token_budget: int, tok) -> list[list[dict]]:
    costs = []
    for r in rows:
        n = sum(len(x) for x in tok(r["texts"], add_special_tokens=False)["input_ids"])
        costs.append((n, r))
    costs.sort(key=lambda x: x[0])
    batches, cur, cur_tok = [], [], 0
    for n, r in costs:
        if cur and cur_tok + n > token_budget:
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append((n, r))
        cur_tok += n
    if cur:
        batches.append(cur)
    return batches


def make_lora_model(model_name: str, device: str, r: int = 32, alpha: int = 64):
    from peft import LoraConfig, get_peft_model

    model, tok = load_backbone(model_name, device)
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tok


def get_inner_model(model):
    """PeftModel(CausalLM) -> inner Qwen3Model (no LM head, no logits).

    LoRA adapters live inside the decoder layers, so driving the inner model
    still routes through them; avoids the full-vocab logits and all-layer
    hidden states that OOM'd the CausalLM path.
    """
    m = model.base_model.model if hasattr(model, "base_model") else model
    return m.model if hasattr(m, "model") else m


def forward_h(model, tok, texts: list[str], device: str) -> torch.Tensor:
    """(N, hidden) final-token hidden states, gradients flowing."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    ids = torch.full((len(enc), max(len(x) for x in enc)), pad_id,
                     dtype=torch.long)
    attn = torch.zeros(ids.shape, dtype=torch.long)
    for r, x in enumerate(enc):
        ids[r, : len(x)] = torch.tensor(x)
        attn[r, : len(x)] = 1
    o = model(input_ids=ids.to(device), attention_mask=attn.to(device),
              use_cache=False)
    h = o.last_hidden_state
    last = attn.sum(1) - 1
    return h[torch.arange(len(enc), device=device), last.to(device)].float()


def build_mnli_gate(limit: int = 150) -> list[dict]:
    rows = [json.loads(l) for l in open("data_cache/rows_mnli.jsonl",
                                         encoding="utf-8")][:limit]
    from kapteeni.serialize import noul_pass

    out = []
    for r in rows:
        out.append({
            "text": noul_pass(r["state"], r["instructions"], r["criteria"]),
            "gold": float(r["label"]),
        })
    return out


@torch.no_grad()
def run_mnli_gate(model, tok, head, gate_rows, T, device) -> dict:
    was_training = model.training
    model.eval()
    h = forward_h(model, tok, [r["text"] for r in gate_rows], device)
    p = torch.sigmoid(head(h) / T).tolist()
    model.train() if was_training else model.eval()
    gold = [r["gold"] for r in gate_rows]
    acc = sum((pi >= 0.5) == bool(g) for pi, g in zip(p, gold)) / len(gold)
    from kapteeni.metrics import ece
    return {"acc": round(acc, 4), "ece": round(ece(p, gold), 4)}


def row_loss(head, h_rows, batch, device):
    """Per-row losses on the row's passes; batch = [(cost, row), ...]."""
    qt = batch[0][1]["qtype"]
    losses = []
    offs = 0
    for n, r in batch:
        h = h_rows[offs: offs + len(r["texts"])]
        offs += len(r["texts"])
        t = torch.tensor(r["target"], device=device, dtype=torch.float32)
        w = torch.tensor(r["weight"], device=device, dtype=torch.float32)
        if qt == "noul":
            losses.append(F.binary_cross_entropy_with_logits(
                head(h), t, weight=w))
        elif qt == "choice":
            s = head(h)
            losses.append(-(F.log_softmax(s, dim=0) * t).sum() / len(t))
        else:
            losses.append(F.binary_cross_entropy_with_logits(head(h), t))
    return torch.stack(losses).mean()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", nargs="+", required=True)
    ap.add_argument("--bundle", default="model_cache/kapteeni_v0.pt")
    ap.add_argument("--out", default="model_cache/kapteeni_p2")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=12288)
    ap.add_argument("--limit", type=int, default=0, help="debug: N rows only")
    ap.add_argument("--ckpt-every", type=int, default=150)
    ap.add_argument("--gate-every", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rows = load_passes(args.passes)
    # val rows are NEVER trained on (deterministic split, same as P1)
    rows = [r for r in rows if not is_val(r["row_id"])
            and not r["row_id"].startswith("mnli")]  # OOD gate stays clean
    if args.limit:
        rows = rows[: args.limit]
    print(f"training rows: {len(rows)} "
          f"({sum(len(r['texts']) for r in rows)} passes)")

    model, tok = make_lora_model(args.model, args.device)
    blob = torch.load(args.bundle, weights_only=False)
    heads = {}
    for qt in ("noul", "choice", "score"):
        head = PassMLP(blob[qt]["in_dim"]).to(args.device)
        head.load_state_dict(blob[qt]["head"])  # warm start from P1
        heads[qt] = head

    params = [p for p in model.parameters() if p.requires_grad]
    for h in heads.values():
        params += list(h.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    # train mode REQUIRED for gradient checkpointing to engage (transformers
    # checks self.training); everything was in eval since load_backbone
    model.train()
    for h in heads.values():
        h.train()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "ckpt"
    step = 0
    if (ckpt / "adapter").exists():  # resume
        from peft import PeftModel

        model = PeftModel.from_pretrained(model.base_model.model, str(ckpt / "adapter"))
        for qt in ("noul", "choice", "score"):
            heads[qt].load_state_dict(torch.load(ckpt / f"head_{qt}.pt",
                                                 weights_only=True))
        opt.load_state_dict(torch.load(ckpt / "opt.pt", weights_only=True))
        step = json.loads((ckpt / "state.json").read_text())["step"]
        print(f"resumed at step {step}")

    rng = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    done_tok = 0
    batches = pack_batches(rows, args.token_budget, tok)
    total_steps = args.epochs * len(batches)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(
            0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    inner = get_inner_model(model)
    gate_rows = build_mnli_gate()
    gate_T = blob["noul"]["T"]
    print(f"batches/epoch: {len(batches)} (~{total_steps} steps); "
          f"MNLI gate: {len(gate_rows)} rows every {args.gate_every}")
    for ep in range(args.epochs):
        order = torch.randperm(len(batches), generator=rng).tolist()
        for bi in order:
            batch = batches[bi]
            qt = batch[0][1]["qtype"]
            texts = [t for _, r in batch for t in r["texts"]]
            with torch.enable_grad():
                h = forward_h(inner, tok, texts, args.device)
            loss = row_loss(heads[qt], h, batch, args.device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            done_tok += sum(n for n, _ in batch)
            if step % args.log_every == 0:
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  step {step} ep {ep} loss {float(loss):.4f} "
                      f"({done_tok/dt:.0f} tok/s train, "
                      f"{done_tok/step/1000:.1f}k tok/step, peak {mem:.1f}G)",
                      flush=True)
                torch.cuda.reset_peak_memory_stats()
            if step % args.gate_every == 0:
                g = run_mnli_gate(inner, tok, heads["noul"], gate_rows,
                                  gate_T, args.device)
                print(f"  [gate step {step}] MNLI head acc={g['acc']} "
                      f"ece={g['ece']}", flush=True)
            if step % args.ckpt_every == 0:
                model.save_pretrained(str(ckpt / "adapter"))
                for qt in ("noul", "choice", "score"):
                    torch.save(heads[qt].state_dict(), ckpt / f"head_{qt}.pt")
                torch.save(opt.state_dict(), ckpt / "opt.pt")
                (ckpt / "state.json").write_text(json.dumps({"step": step}))

    model.save_pretrained(str(out_dir / "adapter"))
    torch.save({qt: heads[qt].state_dict() for qt in heads},
                out_dir / "heads.pt")
    torch.save({qt: blob[qt]["T"] for qt in ("noul", "choice", "score")},
               out_dir / "temps.pt")
    print(f"done: {step} steps, {done_tok/1e6:.1f}M train tokens -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())