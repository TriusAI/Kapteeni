import random

import pytest

from kapteeni.synth3 import GENS, _emit
from kapteeni.synth3_gfx import validate_rows


@pytest.mark.parametrize("name", sorted(GENS))
def test_family_emits_valid_rows(name):
    rng = random.Random(7)
    rows = _emit(GENS[name], rng, 0)
    assert rows
    validate_rows([{k: v for k, v in r.items()
                   if not k.startswith("_")} for r in rows])
    for r in rows:
        assert r["meta"]["family"] == name
        if r["primitive"] == "choice":
            assert len(set(r["options"])) == len(r["options"])
        # one image per render bundle
        assert r["_rid"].endswith(".png") is False and r["_rid"]


@pytest.mark.parametrize("name", sorted(GENS))
def test_family_is_deterministic(name):
    def run(seed):
        rows = _emit(GENS[name], random.Random(seed), 0)
        return [{k: v for k, v in rows_i.items()
                 if not k.startswith("_")} for rows_i in rows]
    assert run(3) == run(3)
    a, b = run(3), run(4)
    assert a and b  # different seeds still emit (no infinite loops)


def test_row_ids_globally_unique_across_renders():
    seen = set()
    rng = random.Random(11)
    seq = 0
    for name in list(GENS)[:5]:
        for r in _emit(GENS[name], rng, seq):
            assert r["row_id"] not in seen
            seen.add(r["row_id"])
        seq += 1


def test_gen_cli_deterministic(tmp_path):
    import subprocess, sys
    outs = []
    for d in ("a", "b"):
        out = tmp_path / d
        subprocess.run([sys.executable, "-m", "kapteeni.synth3", "gen",
                        "--rows", "400", "--seed", "5", "--out", str(out)],
                       check=True, capture_output=True)
        outs.append((out / "items.jsonl").read_text())
    assert outs[0] == outs[1]
    import json
    rows = [json.loads(l) for l in outs[0].splitlines()]
    assert len(rows) >= 400
    for r in rows:
        assert r["image"].endswith(".png")
        assert (tmp_path / "a" / "images" / r["image"]).exists()