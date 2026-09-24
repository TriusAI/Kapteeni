"""Verbalizer readout (P0/Design-B style): native next-token logits.

The pass texts already end with "answer:" (serialize.py), so the backbone's
final-position logits give a natural yes/no score per pass:

    s = logit(" yes") - logit(" no")

- noul:   p_yes = sigmoid(s / T)                     (restricted softmax)
- choice: softmax over the row's option s_i values    (relative, sums to 1)
- score:  softmax over the row's level s_l values

This is the "Raw Qwen3-4B direct logits" baseline class that scored
Intelligence 46.4 on the public board (vs our trained heads' 30.9). Blending
it with the trained heads is Phase 1 of the improvement plan.
"""

from __future__ import annotations

import torch

from kapteeni.backbone import load_backbone


def yes_no_ids(tok) -> tuple[int, int]:
    """Single-token ids for ' yes'/' no' (Qwen3 tokenizer: both are single)."""
    yes = tok.encode(" yes", add_special_tokens=False)
    no = tok.encode(" no", add_special_tokens=False)
    assert len(yes) == 1 and len(no) == 1, (yes, no)
    return yes[0], no[0]


@torch.no_grad()
def verbalizer_scores(
    model, tok, texts: list[str], batch_tokens: int = 16384, device: str = "cuda",
) -> torch.Tensor:
    """(N,) yes-minus-no logit at each pass's final token position."""
    yes_id, no_id = yes_no_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    out = torch.empty(len(texts), dtype=torch.float32)
    batch: list[int] = []

    def flush(batch: list[int]):
        if not batch:
            return
        ids = [enc[i] for i in batch]
        maxlen = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids), maxlen), dtype=torch.long)
        for r, x in enumerate(ids):
            input_ids[r, : len(x)] = torch.tensor(x)
            attn[r, : len(x)] = 1
        o = model(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            use_cache=False,
        )
        logits = o.logits  # (B, L, V) — logits[t] predict token t+1
        last = attn.sum(1) - 1  # final real token position
        rows = torch.arange(len(ids), device=device)
        yes = logits[rows, last.to(device), yes_id]
        no = logits[rows, last.to(device), no_id]
        out[torch.tensor(batch)] = (yes - no).float().cpu()

    for i in order:
        need = max(max(len(enc[j]) for j in batch) if batch else 0, len(enc[i])) * (
            len(batch) + 1
        )
        if batch and need > batch_tokens:
            flush(batch)
            batch = []
        batch.append(i)
    flush(batch)
    return out


def load_for_verbalizer(model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                        device: str = "cuda"):
    model, tok = load_backbone(model_name, device)
    return model, tok