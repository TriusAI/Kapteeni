"""Teacher soft-label distillation for noul rows (k-sample agreement).

For each train row the teacher judges P(yes) k times (temperature sampling);
soft = mean, weight = 1 - std clamped to [0.25, 1]. Ambiguous rows (teacher
disagreement) are KEPT and down-weighted — the calibration band is learned
from them, not despite them (reference data plan §5.2).

Val rows are never soft-labeled (their ECE is measured against gold).

    python3 -m kapteeni.distill --rows data_cache/rows_boolq.jsonl \
        --out data_cache/soft_boolq.jsonl --k 5 --conc 24
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from kapteeni.build_data import is_val
from kapteeni.ollama import content_hash, ollama_chat, parse_json_loose
from kapteeni.serialize import instructions_text, state_text

CACHE_DIR = Path("data_cache/distill_cache")


def judge_prompt(state, instructions, criteria) -> str:
    crit = ""
    if isinstance(criteria, dict):
        crit = "\n".join(
            f"{k}: {v}" for k, v in criteria.items() if isinstance(v, str)
        )
    return (
        "You are labeling a calibration dataset. Judge the question against "
        "the state and reply with the probability that the answer is YES, as "
        'JSON {"p": <number 0..1>}.\n\nSTATE:\n'
        + state_text(state)
        + "\n\nQUESTION:\n"
        + instructions_text(instructions)
        + (("\n\nYes means: " + crit.split("\n")[0].split(": ", 1)[-1]) if crit else "")
        + '\n\nReply with JSON only: {"p": <0..1>}'
    )


def label_row(row: dict, teacher: str, k: int, temp: float,
              client: httpx.Client) -> dict | None:
    prompt = judge_prompt(row["state"], row["instructions"], row["criteria"])
    key = content_hash(teacher, prompt, f"k{k}")
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        vals = json.loads(cache_path.read_text())["vals"]
    else:
        vals = []
        for _ in range(k):
            try:
                text = ollama_chat(teacher, prompt, temperature=temp,
                                   think=False, client=client)
                v = parse_json_loose(text)
                if v is None or "p" not in v:
                    continue
                p = float(v["p"])
                if 0.0 <= p <= 1.0:
                    vals.append(p)
            except Exception:
                continue
        if vals:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"vals": vals}))
    if not vals:
        return None
    p = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "row_id": row["row_id"],
        "p": round(p, 4),
        "std": round(std, 4),
        "weight": round(max(0.25, min(1.0, 1.0 - std)), 3),
        "k_ok": len(vals),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="deepseek-v4-pro:cloud")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--conc", type=int, default=24)
    ap.add_argument("--include-val", action="store_true",
                    help="also label val rows (default: train rows only)")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    rows = [r for r in rows if r["primitive"] == "noul"]
    if not args.include_val:
        rows = [r for r in rows if not is_val(r["row_id"])]

    done: set[str] = set()
    out_path = Path(args.out)
    if out_path.exists():  # resume
        for l in open(out_path, encoding="utf-8"):
            done.add(json.loads(l)["row_id"])
    rows = [r for r in rows if r["row_id"] not in done]
    print(f"{len(rows)} rows to label ({len(done)} resumed, "
          f"teacher={args.teacher}, k={args.k})")

    out_f = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    n_ok = n_fail = 0
    lock_write = __import__("threading").Lock()

    with httpx.Client(trust_env=False, timeout=180.0) as client:
        with ThreadPoolExecutor(max_workers=args.conc) as ex:
            futs = {ex.submit(label_row, r, args.teacher, args.k, args.temp, client): r
                    for r in rows}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec is None:
                    n_fail += 1
                    continue
                n_ok += 1
                with lock_write:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                if (n_ok + n_fail) % 250 == 0:
                    dt = time.time() - t0
                    print(f"  {n_ok + n_fail}/{len(rows)} "
                          f"({(n_ok + n_fail)/dt:.1f}/s, {n_fail} failed)", flush=True)
    out_f.close()
    print(f"done: {n_ok} labeled, {n_fail} failed, "
          f"{time.time()-t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())