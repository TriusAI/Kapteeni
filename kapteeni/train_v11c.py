"""Kapteeni v1.1c trainer: the v1 phase pipeline on Qwen3.5-4B.

Governing document: docs/PREREG-KAPTEENI-V11C.md. Three modes:

  # P1: heads on frozen features (consumes the v11c precompute)
  python3 -m kapteeni.train_v11c --phase1 \
      --emb data_cache/v11c_emb.pt --out model_cache/kapteeni_v11c_p1.pt

  # P2: LoRA + heads, joint, one epoch, budget 8196 (the v1.2 recipe)
  HF_HUB_OFFLINE=1 setsid nohup python3 -m kapteeni.train_v11c --p2 \
      --bundle model_cache/kapteeni_v11c_p1.pt \
      --out model_cache/kapteeni_v11c > /tmp/opencode/v11c_p2.log 2>&1 &

  # final gates (one reading, full slices, on the completed model)
  HF_HUB_OFFLINE=1 python3 -m kapteeni.train_v11c --gates-only \
      --out model_cache/kapteeni_v11c

Ports: kapteeni/train.py (P1 head training + temperature fit, hyperparams
verbatim) and kapteeni/train_p2.py (P2: LoRA config, budget packing by
rows, row-grouped proper-scoring losses, warmup+cosine, MNLI-style OOD
monitor) onto the multimodal base, with the v1.1b monitor/abort/ckpt
safeguards (JSONL monitors, catastrophic-only abort, 150-step resume).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from kapteeni.build_data import data_dir

from kapteeni.build_data import is_val
from kapteeni.heads import PassMLP
from kapteeni.metrics import ece, fit_temperature
from kapteeni.v11c import (MONITOR_SLICES, FINAL_GATES, COMBINED_VAL,
                           extract_h, forward_h, gate_slice, group_rows,
                           p2_train_passes, read_rows, slice_accuracy,
                           synth3_passes, text_source_passes, token_cost,
                           _inner_of)

BASE = "model_cache/qwen3.5-4b"
LORA_TARGETS = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj"
                r"|gate_proj|up_proj|down_proj)$")
ABORT_DROP = 0.10  # carried verbatim from v1.1b's amended rule


# ---------------------------------------------------------------------- P1

def _row_groups(records: list[dict]) -> list[list[int]]:
    by_row: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_row[r["row_id"]].append(i)
    return list(by_row.values())


def _train_noul(h, tr, device, epochs=40, lr=1e-3):
    head = PassMLP(h.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ht = h[[r["_i"] for r in tr]].to(device)
    tt = torch.tensor([r["target"] for r in tr], device=device)
    wt = torch.tensor([r.get("weight", 1.0) for r in tr], device=device)
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(
            head(ht), tt, weight=wt).backward()
        opt.step()
    return head


def _train_choice(h, tr, device, epochs=12, lr=1e-3, chunk_rows=64):
    head = PassMLP(h.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ht = h[[r["_i"] for r in tr]].to(device)
    tt = torch.tensor([r["target"] for r in tr], device=device)
    groups = _row_groups(tr)
    for _ in range(epochs):
        head.train()
        rng = random.Random(0)
        order = list(range(len(groups)))
        rng.shuffle(order)
        for c0 in range(0, len(order), chunk_rows):
            chunk = [groups[i] for i in order[c0: c0 + chunk_rows]]
            flat = [i for g in chunk for i in g]
            opt.zero_grad()
            s = head(ht[flat])
            loss = s.new_zeros(())
            off = 0
            for g in chunk:
                k = len(g)
                loss = loss - (F.log_softmax(s[off: off + k], dim=0)
                               * tt[flat[off: off + k]]).sum() / k
                off += k
            loss.backward()
            opt.step()
    return head


def _train_score(h, tr, device, epochs=30, lr=1e-3):
    head = PassMLP(h.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ht = h[[r["_i"] for r in tr]].to(device)
    tt = torch.tensor([r["target"] for r in tr], device=device)
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(head(ht), tt).backward()
        opt.step()
    return head


def _fit_T(head, h, va, device):
    hv = h[[r["_i"] for r in va]].to(device)

    def readout(T):
        head.eval()
        with torch.no_grad():
            z = head(hv).cpu().tolist()
        rows = read_rows(head, T, hv, va)
        if va[0]["qtype"] == "noul":
            return [p for p, _ in rows], [float(g) for _, g in rows]
        return [c for c, _ in rows], [c for _, c in rows]

    return fit_temperature(readout)


def phase1(args) -> int:
    blob = torch.load(args.emb, weights_only=False)
    h_all, records = blob["h"].float(), blob["records"]
    tr = [r for r in records if not is_val(r["row_id"])]
    va = [r for r in records if is_val(r["row_id"])]
    print(f"P1: {len(tr)} train / {len(va)} val passes", flush=True)
    out = {}
    for qt, train_fn in (("noul", _train_noul), ("choice", _train_choice),
                         ("score", _train_score)):
        tr_q = [r for r in tr if r["qtype"] == qt]
        va_q = [r for r in va if r["qtype"] == qt]
        if not tr_q:
            continue
        t0 = time.time()
        head = train_fn(h_all, tr_q, args.device)
        T = _fit_T(head, h_all, va_q, args.device) if va_q else 1.0
        rows = read_rows(head, T,
                         h_all[[r["_i"] for r in va_q]].to(args.device), va_q)
        if qt == "noul":
            e = ece([p for p, _ in rows], [float(g) for _, g in rows])
            acc = sum((p >= 0.5) == bool(g) for p, g in rows) / max(1, len(rows))
        else:
            e = ece([c for c, _ in rows], [c for _, c in rows])
            acc = sum(c for _, c in rows) / max(1, len(rows))
        out[qt] = {"head": head.state_dict(), "T": T,
                   "in_dim": h_all.shape[1],
                   "val_ece": round(e, 4), "val_acc": round(acc, 4),
                   "n_train": len(tr_q), "n_val": len(va_q)}
        print(f"  {qt}: T={T} val_acc={acc:.4f} val_ece={e:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(out, args.out)
    print(f"P1 bundle -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------------- P2

def load_row_groups(passes: list[dict]) -> list[dict]:
    """Flat pass records -> row records (a row's passes stay together for
    the group losses; train_p2.load_passes' shape + image)."""
    rows: dict[str, list[dict]] = defaultdict(list)
    for p in passes:
        rows[p["row_id"]].append(p)
    out = []
    for rid, ps in rows.items():
        first = ps[0]
        out.append({
            "row_id": rid, "qtype": first["qtype"],
            "target": [p["target"] for p in ps],
            "weight": [p.get("weight", 1.0) for p in ps],
            "texts": [p["text"] for p in ps],
            "image": first["image"],
        })
    return out


def pack_batches(rows: list[dict], budget: int, tok) -> list[list[dict]]:
    costs = [(sum(token_cost(tok, {"text": t, "image": r["image"]})
                  for t in r["texts"]), r) for r in rows]
    costs.sort(key=lambda x: x[0])
    batches, cur, cur_tok = [], [], 0
    for n, r in costs:
        if cur and cur_tok + n > budget:
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(r)
        cur_tok += n
    if cur:
        batches.append(cur)
    return batches


def row_loss(head, h_rows: torch.Tensor, batch: list[dict], device):
    """train_p2.row_loss verbatim: weighted BCE (noul), group-CE (choice),
    per-level BCE (score); mean over the batch's rows."""
    losses = []
    offs = 0
    for r in batch:
        h = h_rows[offs: offs + len(r["texts"])]
        offs += len(r["texts"])
        t = torch.tensor(r["target"], device=device, dtype=torch.float32)
        w = torch.tensor(r["weight"], device=device, dtype=torch.float32)
        if r["qtype"] == "noul":
            losses.append(F.binary_cross_entropy_with_logits(
                head(h), t, weight=w))
        elif r["qtype"] == "choice":
            s = head(h)
            losses.append(-(F.log_softmax(s, dim=0) * t).sum() / len(t))
        else:
            losses.append(F.binary_cross_entropy_with_logits(head(h), t))
    return torch.stack(losses).mean()


def pooled_accuracy(heads, Ts, h, records) -> float:
    """Row-level accuracy over a MIXED-primitive slice (split by qtype,
    pooled by rows)."""
    n, correct = 0, 0.0
    for qt in ("noul", "choice", "score"):
        idx = [i for i, r in enumerate(records) if r["qtype"] == qt]
        if not idx:
            continue
        sub = [records[i] for i in idx]
        acc = slice_accuracy(heads[qt], Ts[qt], h[idx], sub)
        n_rows = len(group_rows(sub))
        n += n_rows
        correct += acc * n_rows
    return correct / max(1, n)


def load_model(args, trainable: bool):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16).to(args.device)
    if trainable:
        from peft import (LoraConfig, PeftModel, get_peft_model)
        if args.adapter and Path(args.adapter).exists():
            # resume: load the checkpointed adapter directly (no double wrap)
            model = PeftModel.from_pretrained(model, args.adapter,
                                              is_trainable=True)
        else:
            lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=LORA_TARGETS)
            model = get_peft_model(model, lcfg)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
    # Numerical-stabilization fix (2026-09-30 NaN diagnosis, see the
    # pre-reg's ENGINEERING NOTE): the bf16 vision tower produces
    # non-finite embeds on borderline image batches under training-mode
    # perturbation (reproduced twice, localized to the tower output; the
    # frozen-feature precompute and every eval-mode forward are finite).
    # The tower runs in fp32 (the multimodal merge casts inputs to the
    # tower's dtype and its fp32 embeds flow back into the bf16 stream);
    # the LM stays bf16. Full-fp32 also clears the repro but costs ~2x.
    inner = _inner_of(model)
    if hasattr(inner, "visual"):
        inner.visual.float()
    return model, proc, proc


def run_monitors(inner, proc, tok, heads, Ts, slices, device) -> dict:
    """One monitor round: per-slice pooled accuracy through the heads."""
    accs = {}
    was_training = inner.training
    inner.eval()
    for name, limit in slices:
        records = gate_slice(name, limit)
        h = extract_h(inner, proc, tok, records, device)
        accs[name] = round(pooled_accuracy(heads, Ts, h, records), 4)
    if was_training:
        inner.train()
    return accs


def p2(args) -> int:
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    ckpt = out_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    step = 0
    resumed = (ckpt / "adapter").exists()
    if resumed:
        args.adapter = str(ckpt / "adapter")
    model, proc, tok = load_model(args, trainable=True)
    blob = torch.load(args.bundle, weights_only=False)
    heads, Ts = {}, {}
    for qt in ("noul", "choice", "score"):
        if qt not in blob:
            continue
        head = PassMLP(blob[qt]["in_dim"]).to(args.device)
        head.load_state_dict(blob[qt]["head"])  # warm start from P1
        heads[qt] = head
        Ts[qt] = blob[qt]["T"]
    for h in heads.values():
        h.train()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    for h in heads.values():
        params += list(h.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    if resumed:
        for qt, h in heads.items():
            h.load_state_dict(torch.load(ckpt / f"head_{qt}.pt",
                                        weights_only=True))
        opt.load_state_dict(torch.load(ckpt / "opt.pt", weights_only=True))
        step = json.loads((ckpt / "state.json").read_text())["step"]
        print(f"resumed at step {step}", flush=True)
    inner = _inner_of(model)

    passes = p2_train_passes()
    rows = load_row_groups(passes)
    if args.limit:
        rows = rows[: args.limit]
    print(f"P2 rows: {len(rows)} "
          f"({Counter(r['qtype'] for r in rows)}) "
          f"{sum(len(r['texts']) for r in rows)} passes", flush=True)
    batches = pack_batches(rows, args.token_budget, tok)
    total_steps = args.epochs * len(batches)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(
            0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    if resumed:
        for _ in range(step):  # fast-forward the LR schedule
            sched.step()
    print(f"batches/epoch: {len(batches)} (~{total_steps} steps)", flush=True)

    mon_path = Path(f"{data_dir()}/v11c_monitors.jsonl")
    best: dict[str, float] = {}
    t0, done_tok, base = time.time(), 0, 0
    n_skip = 0  # non-finite-h batches skipped (backstop; expect 0 with the
    # fp32 tower; > 20 aborts — see the pre-reg ENGINEERING NOTE)

    def save_ckpt():
        model.save_pretrained(str(ckpt / "adapter"))
        for qt, h in heads.items():
            torch.save(h.state_dict(), ckpt / f"head_{qt}.pt")
        torch.save(opt.state_dict(), ckpt / "opt.pt")
        (ckpt / "state.json").write_text(json.dumps({"step": step}))

    for ep in range(args.epochs):
        rng = torch.Generator().manual_seed(args.seed + ep)
        order = torch.randperm(len(batches), generator=rng).tolist()
        it = 0  # batches CONSUMED this epoch (shuffled order; the resume
        for bi in order:  # point is the first `step` of them, in order)
            if base + it < step:
                it += 1
                continue  # already trained before the resume point
            it += 1
            batch = batches[bi]
            flat = [{"text": t, "image": r["image"]}
                    for r in batch for t in r["texts"]]
            h = forward_h(inner, proc, tok, flat, args.device)
            if not bool(torch.isfinite(h).all()):
                n_skip += 1
                print(f"  step {step + 1}: SKIP batch {bi} "
                      f"(non-finite h; skip {n_skip})", flush=True)
                opt.zero_grad(set_to_none=True)
                if n_skip > 20:
                    print("ABORT: > 20 non-finite batches", flush=True)
                    save_ckpt()
                    return 1
                continue
            loss = row_loss(heads[batch[0]["qtype"]], h, batch, args.device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            done_tok += sum(token_cost(tok, p) for p in flat)
            if step % 20 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  step {step} loss {float(loss):.4f} "
                      f"({done_tok/(time.time()-t0):.0f} tok/s, "
                      f"peak {mem:.1f}G)", flush=True)
                torch.cuda.reset_peak_memory_stats()
            if step % args.gate_every == 0:
                accs = run_monitors(inner, proc, tok, heads, Ts,
                                    MONITOR_SLICES, args.device)
                line = json.dumps({"step": step, "ts": time.time(),
                                   "accs": accs})
                with open(mon_path, "a") as f:
                    f.write(line + "\n")
                print(f"  [monitor {step}] {json.dumps(accs)}", flush=True)
                for k, v in accs.items():
                    if v is None or math.isnan(v):
                        continue
                    best[k] = max(v, best.get(k, v))
                for k, v in accs.items():
                    if (v is not None and not math.isnan(v)
                            and best[k] - v > ABORT_DROP):
                        print(f"ABORT RULE TRIGGERED: {k} "
                              f"{best[k]} -> {v} (drop > {ABORT_DROP})",
                              flush=True)
                        save_ckpt()
                        return 1
            if step % args.ckpt_every == 0:
                save_ckpt()
        base += len(batches)

    model.save_pretrained(str(out_dir / "adapter"))
    torch.save({qt: h.state_dict() for qt, h in heads.items()},
               out_dir / "heads.pt")
    (out_dir / "recipe.json").write_text(json.dumps({
        "base": args.base, "bundle": args.bundle, "lora_r": 32, "alpha": 64,
        "lora_targets": "language_model q/k/v/o/gate/up/down (tower frozen)",
        "lr": args.lr, "token_budget": args.token_budget,
        "epochs": args.epochs, "steps": step,
        "train_rows": len(rows), "prereg": "docs/PREREG-KAPTEENI-V11C.md",
    }, indent=2))
    print(f"done: {step} steps, {done_tok/1e6:.1f}M tokens -> {args.out}",
          flush=True)
    return 0


# ------------------------------------------------------------------- gates

GATE_TABLE = {"synth3": ("synth3_val", 0.8068, ">"),
              "mnli": ("mnli_noul", 0.84, ">="),
              "ocnli": ("ocnli_noul", 0.83, ">="),
              "synth2zh": ("synth2zh_val", 0.90, ">="),
              "synth2": ("synth2_en_val", 0.90, ">=")}


def combined_val_records() -> list[dict]:
    """The pre-registered combined held-out val for the temperature refit:
    five mixed-domain text sources + synth3/synth3zh + synth2/synth2zh."""
    out = []
    for name in ("boolq", "fever", "banking77", "clinc150", "helpsteer2"):
        out += text_source_passes(name, val=True)
    out += text_source_passes("synth2", val=True)
    out += text_source_passes("synth2zh", val=True)
    out += synth3_passes("synth3", val=True)
    out += synth3_passes("synth3zh", val=True)
    return out


def gates_only(args) -> int:
    model, proc, tok = load_model(args, trainable=False)
    inner = _inner_of(model)
    heads_sd = torch.load(Path(args.out) / "heads.pt", weights_only=True)
    heads = {}
    for qt, sd in heads_sd.items():
        head = PassMLP(sd["net.0.weight"].shape[1]).to(args.device)
        head.load_state_dict(sd)
        heads[qt] = head

    # 1. per-primitive temperature refit on the combined val (frozen
    #    procedure; the only constants fitted anywhere in v1.1c)
    cv = combined_val_records()
    print(f"combined val: {len(cv)} passes", flush=True)
    Ts = {}
    for qt in ("noul", "choice", "score"):
        sub = [r for r in cv if r["qtype"] == qt]
        h = extract_h(inner, proc, tok, sub, args.device)

        def readout(T, qt=qt, h=h, sub=sub):
            rows = read_rows(heads[qt], T, h, sub)
            if qt == "noul":
                return ([p for p, _ in rows], [float(g) for _, g in rows])
            return ([c for c, _ in rows], [c for _, c in rows])

        Ts[qt] = fit_temperature(readout)
        rows = read_rows(heads[qt], Ts[qt], h, sub)
        if qt == "noul":
            e = ece([p for p, _ in rows], [float(g) for _, g in rows])
        else:
            e = ece([c for c, _ in rows], [c for _, c in rows])
        print(f"  refit {qt}: T={Ts[qt]} ECE={e:.4f} (n={len(sub)} passes)",
              flush=True)

    # 2. the six gates + the reported-not-gated synth3zh slice
    res = {"fit_temps": Ts}
    gate_ece = {}
    for qt in ("noul", "choice", "score"):
        sub = [r for r in cv if r["qtype"] == qt]
        h = extract_h(inner, proc, tok, sub, args.device)
        rows = read_rows(heads[qt], Ts[qt], h, sub)
        if qt == "noul":
            gate_ece[qt] = round(ece([p for p, _ in rows],
                                     [float(g) for _, g in rows]), 4)
        else:
            gate_ece[qt] = round(ece([c for c, _ in rows],
                                     [c for _, c in rows]), 4)
    res["fitted_ece"] = gate_ece
    res["ece_gate"] = {"req": "<= 0.10 every primitive",
                       "pass": all(v <= 0.10 for v in gate_ece.values())}

    verdicts = {}
    for name, limit in FINAL_GATES:
        records = gate_slice(name, limit)
        h = extract_h(inner, proc, tok, records, args.device)
        acc = pooled_accuracy(heads, Ts, h, records)
        key, bar, op = GATE_TABLE[name]
        ok = acc > bar if op == ">" else acc >= bar
        verdicts[key] = {"acc": round(acc, 4), "n_rows":
                         len(group_rows(records)), "req": f"{op} {bar}",
                         "pass": ok}
    res["gates"] = verdicts
    zh = gate_slice("synth3zh", 0)
    h = extract_h(inner, proc, tok, zh, args.device)
    res["synth3zh_val_reported"] = {
        "acc": round(pooled_accuracy(heads, Ts, h, zh), 4),
        "n_rows": len(group_rows(zh))}
    res["all_pass"] = (all(v["pass"] for v in verdicts.values())
                       and res["ece_gate"]["pass"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_gates.json").write_text(json.dumps(res, indent=2,
                                                    ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    return 0


# ----------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--p2", action="store_true")
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--emb", default=f"{data_dir()}/v11c_emb.pt")
    ap.add_argument("--bundle", default="model_cache/kapteeni_v11c_p1.pt")
    ap.add_argument("--out", default="model_cache/kapteeni_v11c")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--gate-every", type=int, default=300)
    ap.add_argument("--ckpt-every", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.phase1:
        return phase1(args)
    if args.p2:
        if not args.adapter:
            args.adapter = str(Path(args.out) / "ckpt/adapter")
        return p2(args)
    if args.gates_only:
        args.adapter = args.adapter or str(Path(args.out) / "adapter")
        return gates_only(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())