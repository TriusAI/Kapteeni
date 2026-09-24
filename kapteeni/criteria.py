"""Teacher-written criteria (rubric text) for choice options and score levels.

Bare label strings fail (the reference's documented numbers-only-levels
failure); every option/level carries a description. The teacher writes one
compact description per option (grounded with real examples from the dataset)
and one description per level per attribute.

    python3 -m kapteeni.criteria banking77 --rows data_cache/rows_banking77.jsonl \
        --out data_cache/crit_banking77.json
    python3 -m kapteeni.criteria helpsteer2 --rows data_cache/rows_helpsteer2.jsonl \
        --out data_cache/crit_helpsteer2.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from kapteeni.ollama import ollama_chat, parse_json_loose

CACHE_DIR = Path("data_cache/criteria_cache")
TEACHER = "deepseek-v4-pro:cloud"


def _cached(name: str, kind: str):
    p = CACHE_DIR / f"{kind}_{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _store(name: str, kind: str, val: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{kind}_{name}.json").write_text(json.dumps(val, ensure_ascii=False))


OPTION_PROMPTS = {
    "banking77": (
        'Banking customer-intent classification has an option "{name}". '
        "Write ONE compact rubric description (max 30 words) of what it "
        "covers, what it does NOT cover, and a 2-3 example phrases. "
        "Ground it in these real examples:\n- {examples}\n\n"
        'Reply as JSON {{"desc": "<one paragraph>"}}'
    ),
    "clinc150": (
        'A voice-assistant intent classifier has an option "{name}". '
        "Write ONE compact rubric description (max 30 words) of what it "
        "covers, what it does NOT cover, and 2-3 example queries. "
        "Ground it in these real examples:\n- {examples}\n\n"
        'Reply as JSON {{"desc": "<one paragraph>"}}'
    ),
    "goemotions": (
        'A Reddit-comment emotion classifier has an emotion option "{name}". '
        "Write ONE compact rubric description (max 25 words) of what kind of "
        "comment expresses this emotion and what it is NOT. "
        "Ground it in these real examples:\n- {examples}\n\n"
        'Reply as JSON {{"desc": "<one paragraph>"}}'
    ),
}


def option_description(dataset: str, name: str, examples: list[str], client) -> str:
    cached = _cached(name, f"opt_{dataset}")
    if cached:
        return cached["desc"]
    for attempt in range(3):
        prompt = OPTION_PROMPTS[dataset].format(
            name=name, examples="\n- ".join(examples[:5]))
        try:
            v = parse_json_loose(ollama_chat(TEACHER, prompt, think=False, client=client))
            if v and isinstance(v.get("desc"), str) and 10 < len(v["desc"]) < 500:
                _store(name, f"opt_{dataset}", {"desc": v["desc"]})
                return v["desc"]
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"criteria for option {name!r} failed after retries")


def attribute_levels(attr: str, client) -> list[str]:
    cached = _cached(attr, "levels")
    if cached:
        return cached["levels"]
    what = {
        "helpfulness": "how well the response addresses the user's request",
        "correctness": "whether the response is factually and technically correct",
        "coherence": "how well-structured, clear, and logically consistent the response is",
    }.get(attr, f"the quality of the response on {attr}")
    for attempt in range(3):
        prompt = (
            f'Rate an AI assistant response on "{attr}" ({what}), on a 0-4 scale. '
            "Write one short description (max 12 words each) of what each level "
            "looks like. 0 is the worst, 4 is the best.\n"
            'Reply as JSON {"levels": ["<level 0>", "<level 1>", "<level 2>", '
            '"<level 3>", "<level 4>"]}'
        )
        try:
            v = parse_json_loose(ollama_chat(TEACHER, prompt, think=False, client=client))
            if v and isinstance(v.get("levels"), list) and len(v["levels"]) == 5 \
                    and all(isinstance(x, str) and x for x in v["levels"]):
                _store(attr, "levels", {"levels": v["levels"]})
                return v["levels"]
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"levels for {attr!r} failed after retries")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["banking77", "clinc150", "goemotions",
                                    "helpsteer2"])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8")]
    out_path = Path(args.out)
    result: dict = json.loads(out_path.read_text()) if out_path.exists() else {}

    with httpx.Client(trust_env=False, timeout=180.0) as client:
        if args.cmd in ("banking77", "clinc150", "goemotions"):
            options = rows[0]["meta"]["options"]
            by_label: dict[int, list[str]] = {}
            for r in rows:
                by_label.setdefault(r["label"], []).append(
                    r["state"]["query" if args.cmd != "goemotions" else "comment"])
            todo = [o for i, o in enumerate(options) if o not in result]
            print(f"{len(todo)} option descriptions to write "
                  f"({len(options) - len(todo)} cached)")
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=12) as ex:
                futs = {ex.submit(option_description, args.cmd, o,
                                 by_label.get(options.index(o), []), client): o
                        for o in todo}
                done = 0
                for fut in as_completed(futs):
                    result[futs[fut]] = fut.result()
                    done += 1
                    if done % 25 == 0:
                        print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}s)",
                              flush=True)
                        out_path.write_text(
                            json.dumps(result, ensure_ascii=False, indent=1))
        else:  # helpsteer2
            attrs = sorted({r["meta"]["attribute"] for r in rows})
            for a in attrs:
                if a in result:
                    continue
                result[a] = attribute_levels(a, client)
                print(f"levels for {a}: {result[a]}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote {args.out} ({len(result)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())