"""Dataset builders: HF datasets -> training rows -> expanded passes.

Row: {"row_id", "primitive", "state", "instructions", "criteria", "label", "meta"}
Pass: {"row_id", "qtype", "kind", "key", "target", "weight", "gold", "soft", "text"}

Verified dataset facts (mirrors that work with `datasets` 5.x, label semantics
empirically checked in the Kapteeni reference build):
  - BoolQ   aps/super_glue:boolq         label 1 = yes
  - FEVER   Dzeniks/fever_2way           label 0 = SUPPORTS (we invert to 1=true)
  - MNLI    nyu-mll/glue:mnli            label 0 = entailment (we map to 1)
  - Bank77  legacy-datasets/banking77    (parquet mirror; PolyAI repo uses a script)
  - HelpSteer2 nvidia/HelpSteer2         attributes 0..4 (5 levels, card-verified)

CLI:
    python3 -m kapteeni.build_data boolq   --n 2500 --out data_cache/rows_boolq.jsonl
    python3 -m kapteeni.build_data fever   --n 1500 --out data_cache/rows_fever.jsonl
    python3 -m kapteeni.build_data mnli    --n 500  --out data_cache/rows_mnli.jsonl
    python3 -m kapteeni.build_data banking77 --n 1800 --out data_cache/rows_banking77.jsonl
    python3 -m kapteeni.build_data helpsteer2 --n 700 --out data_cache/rows_helpsteer2.jsonl
    python3 -m kapteeni.build_data passes  --rows ... [--soft ...] [--opts 24] --out data_cache/passes.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from kapteeni.serialize import (
    choice_option_pass,
    noul_pass,
    score_level_pass,
    state_text,
)

RESPONSE_CHAR_CAP = 1200  # HelpSteer2 responses can be very long; cap for v0
SCORE_ROWS = 500          # v0 pilot size (500 rows x 3 attributes)


def is_val(row_id: str, frac: int = 10) -> bool:
    """Deterministic row-level split (same rule everywhere: expansion, trainer,
    distiller). sha256(row_id) % frac == 0 -> validation row."""
    return int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) % frac == 0


# ------------------------------------------------------------------ builders


def build_boolq(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("aps/super_glue", "boolq", split="train")
    rows = []
    for i, r in enumerate(ds):
        if i >= n:
            break
        rows.append({
            "row_id": f"boolq-{i}",
            "primitive": "noul",
            "state": {"passage": r["passage"]},
            "instructions": f"Does `passage` answer `{r['question']}` with yes?",
            "criteria": {
                "true": "the passage answers the question with yes",
                "false": "the passage does not support a yes answer",
            },
            "label": int(r["label"]),
        })
    return rows


def build_fever(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Dzeniks/fever_2way", split="train")
    rows, seen = [], set()
    for r in ds:
        claim = (r.get("claim") or "").strip()
        ev = (r.get("evidence") or "").strip()
        if not claim or not ev or claim in seen:
            continue
        seen.add(claim)
        i = len(rows)
        rows.append({
            "row_id": f"fever-{i}",
            "primitive": "noul",
            "state": {"claim": claim, "evidence": ev},
            "instructions": "Is `claim` supported by `evidence`?",
            "criteria": {
                "true": "the claim is supported by the evidence",
                "false": "the claim is contradicted by or unrelated to the evidence",
            },
            "label": 1 if int(r["label"]) == 0 else 0,  # repo: 0 = SUPPORTS
        })
        if len(rows) >= n:
            break
    return rows


def build_mnli(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    rows = []
    for i, r in enumerate(ds):
        if i >= n or r["label"] == -1:
            continue
        rows.append({
            "row_id": f"mnli-{i}",
            "primitive": "noul",
            "state": {"premise": r["premise"]},
            "instructions": f"Is `hypothesis` true given `premise`? Hypothesis: {r['hypothesis']}",
            "criteria": {
                "true": "the hypothesis is entailed by the premise",
                "false": "the hypothesis is not entailed by the premise",
            },
            "label": 1 if r["label"] == 0 else 0,  # 0=entailment
        })
    return rows


def build_banking77(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("legacy-datasets/banking77", split="train")
    names = ds.features["label"].names
    rows = []
    for i, r in enumerate(ds):
        if i >= n:
            break
        rows.append({
            "row_id": f"bank77-{i}",
            "primitive": "choice",
            "state": {"query": r["text"]},
            "instructions": "What is the customer's intent?",
            "criteria": None,  # filled from criteria cache at expansion
            "label": int(r["label"]),
            "meta": {"options": names},
        })
    return rows


def build_helpsteer2(n: int, attrs: list[str]) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("nvidia/HelpSteer2", split="train")
    rows = []
    for i, r in enumerate(ds):
        if i >= n:
            break
        for attr in attrs:
            v = r.get(attr)
            if v is None:
                continue
            rows.append({
                "row_id": f"helpsteer2-{i}-{attr}",
                "primitive": "score",
                "state": {
                    "prompt": r["prompt"],
                    "response": r["response"][:RESPONSE_CHAR_CAP],
                },
                "instructions": f"How would you rate `response` on {attr}?",
                "criteria": None,  # filled from criteria cache at expansion
                "label": int(v),
                "meta": {"attribute": attr},
            })
    return rows


def build_clinc150(n: int) -> list[dict]:
    """CLINC150 (plus config, train): 150 intents + oos -> `other` option.
    Cached mirror verified in the reference repo (PolyAI repo is script-based).
    Label semantics: intent index into features['intent'].names; oos is last.
    """
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    feats = ds.features["intent"].names
    out = []
    for i, r in enumerate(ds):
        if i >= n:
            break
        out.append({
            "row_id": f"clinc150-{i}",
            "primitive": "choice",
            "state": {"query": r["text"]},
            "instructions": "What is the user's intent? Options include an out-of-scope `other`.",
            "criteria": None,  # filled from criteria cache at expansion
            "label": int(r["intent"]),
            "meta": {"options": feats},
        })
    return out


def build_goemotions(n: int) -> list[dict]:
    """GoEmotions (simplified): 27 emotions + neutral. Multi-label -> single
    strongest label per row (first listed), full label set as options."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified",
                      split="train")
    feats = ds.features["labels"].feature.names
    out = []
    for r in ds:
        if not r["labels"]:
            continue
        if len(out) >= n:
            break
        out.append({
            "row_id": f"goemo-{len(out)}",
            "primitive": "choice",
            "state": {"comment": r["text"]},
            "instructions": "Which emotion does `comment` most express?",
            "criteria": None,
            "label": int(r["labels"][0]),
            "meta": {"options": feats},
        })
    return out


BUILDERS = {
    "boolq": lambda n: build_boolq(n),
    "fever": lambda n: build_fever(n),
    "mnli": lambda n: build_mnli(n),
    "banking77": lambda n: build_banking77(n),
    "clinc150": lambda n: build_clinc150(n),
    "goemotions": lambda n: build_goemotions(n),
    "helpsteer2": lambda n: build_helpsteer2(
        min(n, SCORE_ROWS), ["helpfulness", "correctness", "coherence"]
    ),
}


# ------------------------------------------------------------------ expansion


def _choice_option_subset(row: dict, opts_per_row: int, rng: random.Random):
    """Train rows: gold + sampled distractors (precompute explosion cap).
    Val rows: the full option set (honest top-1 accuracy)."""
    options = row["meta"]["options"]
    gold = row["label"]
    if is_val(row["row_id"]) or len(options) <= opts_per_row:
        return list(range(len(options)))
    distractors = [i for i in range(len(options)) if i != gold]
    rng.shuffle(distractors)
    keep = set([gold] + distractors[: opts_per_row - 1])
    # canonical dataset order so pass ordering is deterministic
    return [i for i in range(len(options)) if i in keep]


def expand_passes(rows: list[dict], criteria: dict, soft: dict,
                  opts_per_row: int = 24, seed: int = 7) -> list[dict]:
    """rows + criteria + teacher soft labels -> flat pass records with text."""
    rng = random.Random(seed)
    passes: list[dict] = []
    for row in rows:
        rid, qt = row["row_id"], row["primitive"]
        val = is_val(rid)
        if qt == "noul":
            gold = float(row["label"])
            s = soft.get(rid)
            target = s["p"] if (s and not val) else gold
            passes.append({
                "row_id": rid, "qtype": "noul", "kind": "noul", "key": None,
                "target": target,
                "weight": (s["weight"] if s else 1.0) if not val else 1.0,
                "gold": gold,
                "soft": s["p"] if s else None,
                "text": noul_pass(row["state"], row["instructions"], row["criteria"]),
            })
        elif qt == "choice":
            opts = _choice_option_subset(row, opts_per_row, rng)
            names = row["meta"]["options"]
            for oi in opts:
                desc = criteria.get(names[oi]) if criteria else names[oi]
                passes.append({
                    "row_id": rid, "qtype": "choice", "kind": "choice_opt",
                    "key": names[oi], "target": 1.0 if oi == row["label"] else 0.0,
                    "weight": 1.0, "gold": float(row["label"]), "soft": None,
                    "text": choice_option_pass(
                        row["state"], row["instructions"], names[oi], desc),
                })
        else:  # score
            levels = criteria[row["meta"]["attribute"]]
            n_levels = len(levels)
            for i, lvl in enumerate(levels):
                passes.append({
                    "row_id": rid, "qtype": "score", "kind": "score_lvl",
                    "key": i, "target": 1.0 if i == row["label"] else 0.0,
                    "weight": 1.0, "gold": float(row["label"]), "soft": None,
                    "text": score_level_pass(
                        row["state"], row["instructions"], i, n_levels, lvl),
                })
    return passes


# ----------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=[*BUILDERS, "passes"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", nargs="+", default=[], help="passes: row jsonl files")
    ap.add_argument("--criteria", nargs="+", default=[],
                    help="passes: criteria json files (choice/score)")
    ap.add_argument("--soft", nargs="+", default=[],
                    help="passes: teacher soft-label jsonl files (noul)")
    ap.add_argument("--opts", type=int, default=24)
    args = ap.parse_args(argv)

    if args.cmd in BUILDERS:
        rows = BUILDERS[args.cmd](args.n)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{args.cmd}: wrote {len(rows)} rows -> {args.out}")
        return 0

    rows: list[dict] = []
    for p in args.rows:
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    criteria: dict = {}
    for p in args.criteria:
        criteria.update(json.load(open(p, encoding="utf-8")))
    soft: dict = {}
    for p in args.soft:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            soft[r["row_id"]] = {"p": r["p"], "weight": r.get("weight", 1.0)}

    passes = expand_passes(rows, criteria, soft, opts_per_row=args.opts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in passes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_val_rows = len({p["row_id"] for p in passes if is_val(p["row_id"])})
    print(f"passes: {len(rows)} rows -> {len(passes)} passes "
          f"({n_val_rows} val rows) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())