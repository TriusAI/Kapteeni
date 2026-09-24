"""P1 trainer: train the three readout heads on precomputed h_last tensors.

    python3 -m kapteeni.train --emb data_cache/emb.pt --out model_cache/kapteeni_v0.pt

Splits by row (a question's passes stay together; no leakage). Fits a
temperature per head on the val split (metrics.py), then reports the E2
calibration numbers (ECE/Brier per primitive; bar: ECE <= 0.06 after temp).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from kapteeni.heads import PassMLP, choice_loss, noul_loss, score_level_loss
from kapteeni.build_data import is_val
from kapteeni.metrics import brier, ece, fit_temperature


def load_emb(path: str):
    blob = torch.load(path, weights_only=False)
    return blob["h"].float(), blob["records"]


def split_by_row(records: list[dict], val_frac: float = 0.1, seed: int = 0):
    """Same deterministic rule as expansion/distillation (build_data.is_val):
    sha256(row_id) % 10 == 0 -> val. Val rows carry gold targets and full
    option sets; train rows carry teacher soft labels."""
    tr = [r for r in records if not is_val(r["row_id"])]
    va = [r for r in records if is_val(r["row_id"])]
    return tr, va


def _h(h_all: torch.Tensor, records: list[dict], device: str) -> torch.Tensor:
    return h_all[[r["_i"] for r in records]].to(device)


def _row_groups(records: list[dict]) -> list[list[int]]:
    """Indices (into `records`) of one row's passes, rows ordered by row_id."""
    by_row: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_row[r["row_id"]].append(i)
    return [by_row[k] for k in sorted(by_row)]


def _soft(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def _gold_pos(records: list[dict], g: list[int]) -> int:
    for pos, i in enumerate(g):
        if records[i]["target"] == 1.0:
            return pos
    return -1


# ---------------------------------------------------------------------- noul


def train_noul(h_all: torch.Tensor, tr: list[dict], va: list[dict], epochs: int,
               lr: float, device: str):
    head = PassMLP(h_all.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ht = _h(h_all, tr, device)
    tt = torch.tensor([r["target"] for r in tr], device=device)
    wt = torch.tensor([r.get("weight", 1.0) for r in tr], device=device)
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        noul_loss(head(ht), tt, wt).backward()
        opt.step()

    hv = _h(h_all, va, device)

    def readout(T: float):
        head.eval()
        with torch.no_grad():
            z = head(hv).cpu()
        p = [1 / (1 + math.exp(-(zi / T))) for zi in z.tolist()]
        return p, [r["target"] for r in va]

    T = fit_temperature(readout)
    p, y = readout(T)
    head.eval()
    with torch.no_grad():
        z_tr = head(ht).cpu()
    p_tr = [1 / (1 + math.exp(-zi)) for zi in z_tr.tolist()]
    return head, T, {
        "val_ece": round(ece(p, y), 4),
        "val_brier": round(brier(p, y), 4),
        "train_ece": round(ece(p_tr, [r["target"] for r in tr]), 4),
        "n_val": len(va),
    }


# ---------------------------------------------------------------- grouped
# (choice and score share the grouped readout; their losses differ)


def _train_grouped(qtype: str, h_all: torch.Tensor, tr: list[dict], va: list[dict],
                   epochs: int, lr: float, device: str, chunk_rows: int = 64):
    head = PassMLP(h_all.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ht = _h(h_all, tr, device)
    tt = torch.tensor([r["target"] for r in tr], device=device)
    groups = _row_groups(tr)

    if qtype == "choice":
        for _ in range(epochs):
            head.train()
            rng = random.Random(0)
            order = list(range(len(groups)))
            rng.shuffle(order)
            for c0 in range(0, len(order), chunk_rows):
                chunk = [groups[i] for i in order[c0 : c0 + chunk_rows]]
                flat = [i for g in chunk for i in g]
                opt.zero_grad()
                choice_loss(head(ht[flat]), [len(g) for g in chunk], tt[flat]).backward()
                opt.step()
    else:  # score: per-level BCE, full batch
        for _ in range(epochs):
            head.train()
            opt.zero_grad()
            score_level_loss(head(ht), tt).backward()
            opt.step()

    va_groups = _row_groups(va)
    hv = _h(h_all, va, device)

    def readout(T: float):
        head.eval()
        with torch.no_grad():
            z = head(hv).cpu()
        conf, correct, exp_mae = [], [], []
        for g in va_groups:
            s = [z[i].item() / T for i in g]
            p = _soft(s)
            top = max(range(len(s)), key=lambda i: s[i])
            gold = _gold_pos(va, g)
            correct.append(1.0 if top == gold else 0.0)
            conf.append(p[top])
            exp_mae.append(abs(sum(i * p[i] for i in range(len(p))) - gold))
        return conf, correct

    T = fit_temperature(readout)
    conf, correct = readout(T)
    acc = sum(correct) / max(1, len(correct))
    return head, T, {
        "val_ece": round(ece(conf, correct), 4),
        "val_top1": round(acc, 4),
        "n_val_rows": len(va_groups),
    }


def train_choice(h_all, tr, va, epochs, lr, device):
    return _train_grouped("choice", h_all, tr, va, epochs, lr, device)


def train_score(h_all, tr, va, epochs, lr, device):
    return _train_grouped("score", h_all, tr, va, epochs, lr, device)


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs-noul", type=int, default=40)
    ap.add_argument("--epochs-choice", type=int, default=12)
    ap.add_argument("--epochs-score", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    h_all, records = load_emb(args.emb)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for i, r in enumerate(records):
        r["_i"] = i
        by_type[r["qtype"]].append(r)

    bundle: dict = {
        "meta": {"emb": args.emb, "seed": args.seed,
                 "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    }
    for qt in ("noul", "choice", "score"):
        recs = by_type.get(qt, [])
        if not recs:
            print(f"[{qt}] no passes; skipping")
            continue
        tr, va = split_by_row(recs)
        if not va:
            va = tr  # degenerate debug case
        t0 = time.time()
        if qt == "noul":
            head, T, m = train_noul(h_all, tr, va, args.epochs_noul, args.lr, args.device)
        elif qt == "choice":
            head, T, m = train_choice(h_all, tr, va, args.epochs_choice, args.lr, args.device)
        else:
            head, T, m = train_score(h_all, tr, va, args.epochs_score, args.lr, args.device)
        bundle[qt] = {"head": head.state_dict(), "T": T, "metrics": m,
                      "in_dim": h_all.shape[1]}
        print(f"[{qt}] {time.time()-t0:.1f}s  T={T}  {json.dumps(m)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())