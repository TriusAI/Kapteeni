"""Frozen backbone: Qwen3-4B-Instruct-2507 (bf16) + batched h_last extraction.

One pass = state text + question/option/level suffix (serialize.py); the
readout position is the FINAL token's last-layer hidden state. Passes are
independent, so batching them changes nothing (E1 parity is structural).

Precompute CLI (crash-safe: saves emb.pt incrementally every --chunk passes;
resume skips passes already present in the output):

    python3 -m kapteeni.backbone --passes data_cache/passes.jsonl \
        --out data_cache/emb.pt --batch-tokens 16384

Output (.pt): {"h": fp16 tensor (N, hidden), "records": [pass records]}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
STATE_CHAR_CAP = 32_000  # v0 pilot: cap serialized state before suffix


def load_backbone(model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@torch.no_grad()
def _batched_forward(model, tok, texts: list[str], batch_tokens: int, device: str,
                    want_verb: bool):
    """Shared batching: final-token hidden states (+ yes/no logits if asked)."""
    from kapteeni.verbalizer import yes_no_ids

    yes_id, no_id = yes_no_ids(tok) if want_verb else (None, None)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    order = sorted(range(len(texts)), key=lambda i: len(enc[i]))
    out = torch.empty(len(texts), model.config.hidden_size, dtype=torch.float16)
    out_v = torch.empty(len(texts), dtype=torch.float32) if want_verb else None
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
            output_hidden_states=True,
            use_cache=False,
        )
        h = o.hidden_states[-1]  # (B, L, hidden)
        last = attn.sum(1) - 1  # position of each sequence's final token
        rows = torch.arange(len(ids), device=device)
        last_d = last.to(device)
        hs = h[rows, last_d]
        out[torch.tensor(batch)] = hs.to(torch.float16).cpu()
        if want_verb:
            logits = o.logits
            out_v[torch.tensor(batch)] = (
                logits[rows, last_d, yes_id] - logits[rows, last_d, no_id]
            ).float().cpu()

    for i in order:
        need = max(max(len(enc[j]) for j in batch) if batch else 0, len(enc[i])) * (
            len(batch) + 1
        )
        if batch and need > batch_tokens:
            flush(batch)
            batch = []
        batch.append(i)
    flush(batch)
    return out, out_v


@torch.no_grad()
def extract_h_last(
    model, tok, texts: list[str], batch_tokens: int = 32768, device: str = "cuda",
    progress_every: int = 0,
) -> torch.Tensor:
    """Final-layer hidden state at each text's last token. (N, hidden) fp16."""
    h, _ = _batched_forward(model, tok, texts, batch_tokens, device, want_verb=False)
    return h


@torch.no_grad()
def extract_h_last_verb(
    model, tok, texts: list[str], batch_tokens: int = 32768, device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """h_last AND the yes-minus-no verbalizer logit per pass — same forward,
    ~free. Returns (h (N, hidden) fp16, s_verb (N,) fp32)."""
    return _batched_forward(model, tok, texts, batch_tokens, device, want_verb=True)


KV_BUDGET_BYTES = 8.0e9  # conservative transient budget for the repeated cache


@torch.no_grad()
def extract_shared_prefix(
    model, tok, state_text: str, suffixes: list[str], device: str = "cuda",
    kv_budget_bytes: float = KV_BUDGET_BYTES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared-prefix serving path: prefill the state ONCE, run every suffix
    against the repeated KV cache (the reference repo's §8.2 serving model).

    Cost = state_prefill + n_suffixes x (suffix_len through the cache) instead
    of n_suffixes x full prefill — 4-6x fewer tokens on long-state
    multi-option decisions. The state/suffix tokenization split is exact (the
    "\n\n" join is a BPE boundary; verified empirically).

    Returns (h (n, hidden) fp16, s_verb (n,) fp32) aligned with `suffixes`.
    """
    from kapteeni.verbalizer import yes_no_ids

    yes_id, no_id = yes_no_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    state_ids = tok(state_text, add_special_tokens=False)["input_ids"]
    suf_ids = [tok(s, add_special_tokens=False)["input_ids"] for s in suffixes]
    n = len(suf_ids)
    cfg = model.config
    kv_per_tok = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * (
        cfg.hidden_size // cfg.num_attention_heads) * 2
    state_kv = len(state_ids) * kv_per_tok
    # memory budget governs the repeat; no arbitrary cap beyond 512. NOTE the
    # suffix forward MUTATES the cache (transformers updates it despite
    # use_cache=False), so multi-chunk reuse is impossible — if the budget
    # cannot hold all n suffix copies at once, fall back to full-forward.
    m = max(1, min(n, int(kv_budget_bytes / max(1, state_kv)), 512))
    if m < n:
        full = [state_text + "\n\n" + s for s in suffixes]
        return extract_h_last_verb(model, tok, full, device=device)

    # 1) prefill the state once — base model only (skips the vocab projection)
    p = torch.tensor([state_ids], device=device)
    base = model.model(input_ids=p, use_cache=True)
    h_prefix = base.last_hidden_state[0, -1]

    # 2) repeat the batch-1 cache to m; suffix chunks of size m reuse it
    # (use_cache=False below, so the repeated cache is never mutated)
    past = base.past_key_values
    if m > 1:
        past.batch_repeat_interleave(m)
    cache_len = past.get_seq_length() if hasattr(past, "get_seq_length") \
        else len(state_ids)

    h_out = torch.empty(n, cfg.hidden_size, dtype=torch.float16)
    v_out = torch.empty(n, dtype=torch.float32)
    lm_head = model.lm_head
    for c0 in range(0, n, m):
        chunk = suf_ids[c0 : c0 + m]
        cs = len(chunk)
        max_s = max(len(s) for s in chunk)
        ids = torch.full((cs, max_s), pad_id, dtype=torch.long)
        attn = torch.zeros((cs, max_s), dtype=torch.long)
        last = []
        for r, s in enumerate(chunk):
            ids[r, : len(s)] = torch.tensor(s)
            attn[r, : len(s)] = 1
            last.append(len(s) - 1)
        pos = torch.arange(cache_len, cache_len + max_s, device=device) \
            .unsqueeze(0).expand(cs, -1)
        # full-coverage 2D mask: (cs, cache_len + max_s). A query-only mask
        # gets RIGHT-padded with zeros by masking_utils.prepare_padding_mask,
        # which masks out the suffix tokens themselves (verified the hard way).
        full_attn = torch.cat([
            torch.ones(cs, cache_len, dtype=torch.long), attn
        ], dim=1)
        out = model.model(
            input_ids=ids.to(device),
            attention_mask=full_attn.to(device),
            position_ids=pos,
            past_key_values=past,
            use_cache=False,
        )
        hh = out.last_hidden_state  # (cs, max_s, hidden)
        rows = torch.arange(cs, device=device)
        last_t = torch.tensor(last, device=device)
        h_final = hh[rows, last_t]  # (cs, hidden)
        logits = lm_head(h_final)  # only cs positions through the vocab head
        h_out[torch.arange(c0, c0 + cs)] = h_final.to(torch.float16).cpu()
        v_out[torch.arange(c0, c0 + cs)] = (
            logits[:, yes_id] - logits[:, no_id]
        ).float().cpu()
        del out, hh, h_final, logits
    del base, past
    return h_out, v_out


# ------------------------------------------------------------------ precompute


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", required=True, help="jsonl of pass records with text")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N passes")
    ap.add_argument("--chunk", type=int, default=6000,
                    help="save incrementally every N passes (crash-safe)")
    args = ap.parse_args(argv)

    passes = [json.loads(l) for l in open(args.passes, encoding="utf-8")]
    if args.limit:
        passes = passes[: args.limit]

    records: list[dict] = []
    h_known: torch.Tensor | None = None

    def save(h_new: torch.Tensor, new_recs: list[dict]) -> torch.Tensor:
        merged = torch.cat([h_known, h_new]) if h_known is not None else h_new
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out + ".tmp"
        torch.save({"h": merged, "records": records + new_recs}, tmp)
        Path(tmp).rename(args.out)
        return merged

    if Path(args.out).exists():  # resume from incremental saves
        blob = torch.load(args.out, weights_only=False)
        records, h_known = blob["records"], blob["h"]
        done_keys = {(r["row_id"], r["kind"], r["key"]) for r in records}
        passes = [p for p in passes
                  if (p["row_id"], p["kind"], p["key"]) not in done_keys]
        print(f"resume: {len(records)} already done, {len(passes)} to go")
    if not passes:
        print("nothing to do")
        return 0

    model, tok = load_backbone(args.model, args.device)
    print(f"extracting h_last for {len(passes)} passes "
          f"(incremental saves every {args.chunk}) ...", flush=True)
    t0 = time.time()
    done = 0
    for c0 in range(0, len(passes), args.chunk):
        chunk = passes[c0 : c0 + args.chunk]
        texts = [p["text"][: STATE_CHAR_CAP + 2000] for p in chunk]
        h_new = extract_h_last(model, tok, texts, args.batch_tokens, args.device)
        new_recs = [{k: v for k, v in p.items() if k != "text"} for p in chunk]
        h_known = save(h_new, new_recs)
        records += new_recs
        done += len(chunk)
        dt = time.time() - t0
        eta = dt / done * (len(passes) - done)
        print(f"  {done}/{len(passes)} passes ({done/dt:.1f}/s, "
              f"eta {eta/60:.0f} min, saved)", flush=True)

    dt = time.time() - t0
    ntok = sum(len(x) for x in tok(
        [p["text"] for p in passes], add_special_tokens=False)["input_ids"])
    print(f"done in {dt:.0f}s  ({ntok/dt:.0f} tok/s, {len(passes)/dt:.1f} pass/s)")
    print(f"wrote {args.out}: h {tuple(h_known.shape)}, {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())