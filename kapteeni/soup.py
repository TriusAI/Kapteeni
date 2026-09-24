"""Model soup: elementwise average of MERGED seed models (basis-independent).

Averaging merged (base + scale*B@A) weights is exact regardless of each
seed's random adapter basis; averaging raw adapters would not be
(mean of B@A != mean(B) @ mean(A)). Each seed is merged from a FRESH base
(peft's merge mutates the base in place), pulled to CPU fp32 key-by-key,
accumulated, divided, and written back into a final skeleton that is saved
as a plain merged model dir (serve --model / p2_finalize --merged-dir).

Heads are averaged too: all seeds warm-start from the same P1 bundle, so
they plausibly stay in one basin. The finalize step's mixed-val report
validates this; if head accuracy degrades, retrain heads on soup features
instead (see docs/PREREG-V1.1-V1.2.md).

    python3 -m kapteeni.soup \
      --adapters model_cache/kapteeni_p2/adapter \
                 model_cache/kapteeni_p2_s1/adapter \
                 model_cache/kapteeni_p2_s2/adapter \
      --heads model_cache/kapteeni_p2/heads.pt \
              model_cache/kapteeni_p2_s1/heads.pt \
              model_cache/kapteeni_p2_s2/heads.pt \
      --out model_cache/kapteeni_soup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kapteeni.backbone import DEFAULT_MODEL, load_backbone  # noqa: E402


def merge_seed(model_name: str, adapter: str, device: str) -> dict:
    """Fresh base + adapter merge -> merged state dict on CPU bf16."""
    from peft import PeftModel

    model, _ = load_backbone(model_name, device)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    sd = {k: v.detach().to("cpu", torch.bfloat16).clone()
          for k, v in model.state_dict().items()}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return sd


def average_heads(sds: list[dict]) -> dict:
    """Elementwise mean of head state dicts (all warm-started from the same
    P1 bundle, so basin mismatch is the risk to watch at finalize time)."""
    keys = set(sds[0])
    for s in sds[1:]:
        if set(s) != keys:
            raise ValueError("head files have different key sets")
    for qt in keys:
        layers = set(sds[0][qt])
        for s in sds[1:]:
            if set(s[qt]) != layers:
                raise ValueError(
                    f"head files have different layer keys for {qt!r}")
    out = {}
    for qt in sds[0]:
        layers = set(sds[0][qt])
        out[qt] = {
            kk: torch.stack([s[qt][kk].float() for s in sds]).mean(0)
            .to(sds[0][qt][kk].dtype)
            for kk in layers
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--heads", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    n = len(args.adapters)
    if len(set(args.adapters)) != n:
        print("error: duplicate adapters listed")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    acc: dict[str, torch.Tensor] = {}
    for i, ad in enumerate(args.adapters):
        print(f"[{i+1}/{n}] merging {ad} ...", flush=True)
        sd = merge_seed(args.model, ad, args.device)
        if i == 0:
            acc = {k: v.float() for k, v in sd.items()}
        else:
            missing = set(acc) ^ set(sd)
            if missing:
                print(f"error: key mismatch across seeds: {missing}")
                return 1
            for k, v in sd.items():
                acc[k] += v.float()
        del sd

    print("building averaged skeleton and saving ...", flush=True)
    model, tok = load_backbone(args.model, args.device)
    ref = model.state_dict()
    if set(ref) != set(acc):
        print(f"error: skeleton keys differ: {set(ref) ^ set(acc)}")
        return 1
    with torch.no_grad():
        for k, t in ref.items():
            t.copy_((acc[k] / n).to(t.dtype, t.device))
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"wrote soup of {n} seeds -> {out}")

    if args.heads:
        sds = [torch.load(h, weights_only=True) for h in args.heads]
        soup_heads = average_heads(sds)
        torch.save(soup_heads, out / "heads_soup.pt")
        print(f"wrote averaged heads ({len(args.heads)} files) -> "
              f"{out / 'heads_soup.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())