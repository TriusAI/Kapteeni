import json
import random
from datetime import date

import pytest

from kapteeni.synth2 import (
    GENS,
    _months_between,
    _ord,
    _shift_days_for_months,
    _selfcheck,
    add_bd,
    fmt_date,
    gen_status_choice,
    gen_urgency_score,
    main as synth2_main,
    month_add,
)


def test_helpers_invariants():
    assert _selfcheck() == 0


def test_month_shift_roundtrip_all_month_ends():
    # the day==1 edge that the selfcheck originally caught
    for day in (1, 2, 15, 28):
        today = date(2025, 3, day)
        for n in (1, 2, 6, 12):
            j = _shift_days_for_months(today, n)
            assert _months_between(j, today) == n, (today, n, j)


def test_add_bd_known_dates():
    from datetime import date as D
    # Mon Jan 1 2024 holiday; 3 business days after Mon Jan 1 -> Thu Jan 4
    assert add_bd(D(2024, 1, 1), 3, {D(2024, 1, 1)}) == D(2024, 1, 4)
    # Fri + 1 business day -> Mon
    assert add_bd(D(2024, 1, 5), 1, set()) == D(2024, 1, 8)
    assert month_add(D(2024, 11, 10), 3) == D(2025, 2, 10)
    assert _ord(21) == "21st" and _ord(11) == "11th"


def test_fmt_date_styles_are_all_parseable_weekday_style_correct():
    rng = random.Random(2)
    seen = set()
    for _ in range(30):
        seen.add(fmt_date(date(2025, 3, 9), rng))
    assert any("Sunday" in s for s in seen)   # 2025-03-09 is a Sunday
    assert any("2025-03-09" == s for s in seen)  # iso style appears


@pytest.mark.parametrize("gen,quota", GENS)
def test_generators_emit_valid_rows(gen, quota):
    rng = random.Random(7)
    out = gen(rng)
    rows = out if isinstance(out, list) else [out]
    assert rows
    for r in rows:
        assert r["primitive"] in ("noul", "choice", "score")
        assert r["meta"]["family"] in ("temporal_numeric", "multi_hop",
                                       "long_policy")
        if r["primitive"] == "noul":
            assert r["label"] in (0, 1)
            assert set(r["criteria"]) == {"true", "false"}
        elif r["primitive"] == "choice":
            assert r["label"] in r["meta"]["options"]
        else:
            assert isinstance(r["label"], int)
            assert r["meta"]["attribute"]


def test_status_choice_covers_all_bins():
    rng = random.Random(3)
    seen = {gen_status_choice(rng)["label"] for _ in range(200)}
    assert seen == {"overdue", "due_today", "due_within_7_days",
                    "due_later"}


def test_urgency_score_covers_all_levels():
    rng = random.Random(4)
    seen = {gen_urgency_score(rng)["label"] for _ in range(200)}
    assert seen == {0, 1, 2, 3}


def test_full_run_deterministic_and_labeled(tmp_path):
    common = ["synth", "--n", "400", "--seed", "23"]
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    crit = tmp_path / "crit.json"
    assert synth2_main([*common, "--out", str(a),
                       "--criteria-out", str(crit)]) == 0
    assert synth2_main([*common, "--out", str(b)]) == 0
    ra = a.read_text().splitlines()
    rb = b.read_text().splitlines()
    assert ra == rb
    rows = [json.loads(l) for l in ra]
    # choice rows store the option INDEX (build_data contract)
    for r in rows:
        if r["primitive"] == "choice":
            assert isinstance(r["label"], int)
            assert 0 <= r["label"] < len(r["meta"]["options"])
    # every criteria reference resolves
    critd = json.loads(crit.read_text())
    for r in rows:
        if r["primitive"] == "score":
            assert critd[r["meta"]["attribute"]]
        if r["primitive"] == "choice":
            for o in r["meta"]["options"]:
                assert o in critd, o