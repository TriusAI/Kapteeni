"""v1.1c unit tests (no GPU): the pass builders (determinism, splits, gold
kept, byte-identical v1.2 reuse, image attach, gate-slice shapes) and the
heads readout / packing logic on synthetic tensors.

The pass builders read the real data_cache files; these tests assert the
invariants the pre-registration (docs/PREREG-KAPTEENI-V11C.md) fixes:
- v1.2 text passes reused byte-identical, train rows only in the P2 set;
- val/gate rows never in any train set; MNLI/OCNLI never trained on;
- choice expansions keep the gold option; synth3 image passes attach the
  render to every pass of the row.
"""

import json
import math

import pytest
import torch

from kapteeni.build_data import is_val
from kapteeni.heads import PassMLP
from kapteeni.v11c import (gate_slice, group_rows, image_row_passes,
                           p1_train_passes, p1_val_passes, p2_train_passes,
                           read_rows, slice_accuracy, synth3_passes,
                           text_source_passes, token_cost)


def test_reuse_byte_identical():
    """P2's reused text passes are the v1.2 files' records verbatim
    (plus image=None), train rows only."""
    p2_passes = p2_train_passes()
    by_id = {}
    for p in p2_passes:
        by_id.setdefault(p["row_id"], []).append(p)
    orig = [json.loads(l) for l in open("data_cache/passes_p2.jsonl")]
    for r in orig[:5000]:
        recs = by_id[r["row_id"]]
        ours = next((p for p in recs if p["text"] == r["text"]), None)
        assert ours is not None, r["row_id"]
        for k in ("qtype", "kind", "key", "target", "weight", "gold"):
            assert ours[k] == r[k]
        assert ours["image"] is None


def test_train_val_disjoint():
    train_ids = {p["row_id"] for p in p2_train_passes()}
    val_ids = {p["row_id"] for p in p1_val_passes()}
    assert not (train_ids & val_ids), "val rows leaked into the P2 train set"
    for rid in train_ids:
        assert not rid.startswith(("mnli", "ocnli"))
    for rid in val_ids:
        assert not rid.startswith(("mnli", "ocnli"))


def test_p1_subset_of_p2():
    """P1's sources are P2's minus the v1-lineage P2-only families."""
    p1_ids = {p["row_id"] for p in p1_train_passes()}
    p2_ids = {p["row_id"] for p in p2_train_passes()}
    assert p1_ids <= p2_ids
    assert not any(rid.startswith(("clinc150", "goemo", "synth-"))
                   for rid in p1_ids)


def test_image_passes_attach_render_and_keep_gold():
    rows = [json.loads(l) for l in open("data_cache/synth3/items.jsonl")]
    row = next(r for r in rows if r["primitive"] == "choice")
    passes = image_row_passes(row, "data_cache/synth3/images")
    assert len(passes) == len(row["options"])
    assert sum(p["target"] for p in passes) == 1.0
    gold = next(p for p in passes if p["target"] == 1.0)
    assert gold["key"] == row["options"][row["label"]]
    assert all(p["image"].endswith(row["image"]) for p in passes)
    assert all(p["text"].startswith('{"image": "<attached>", "note"')
               for p in passes)


def test_synth3_val_split():
    val = synth3_passes("synth3", val=True)
    train = synth3_passes("synth3", val=False)
    assert len({p["row_id"] for p in val}) == 590
    assert len({p["row_id"] for p in train}) == 5412
    assert all(is_val(p["row_id"]) for p in val)
    assert not any(is_val(p["row_id"]) for p in train)


def test_gate_slices():
    mnli = gate_slice("mnli", 150)
    assert len(mnli) == 150 and all(p["qtype"] == "noul" for p in mnli)
    assert all(p["image"] is None for p in mnli)
    ocnli = gate_slice("ocnli", 150)
    assert len(ocnli) == 150
    s3 = gate_slice("synth3", 300)
    assert len({p["row_id"] for p in s3}) == 300
    assert all(is_val(p["row_id"]) for p in s3)
    s3_full = gate_slice("synth3", 0)
    assert len({p["row_id"] for p in s3_full}) == 590
    s2zh = gate_slice("synth2zh", 200)
    assert len({p["row_id"] for p in s2zh}) == 200
    with pytest.raises(KeyError):
        gate_slice("nope", 1)


def test_val_choice_full_option_sets():
    """Val choice rows keep the FULL option set (the v0 convention)."""
    val = text_source_passes("synth2zh", val=True)
    rows = [json.loads(l) for l in open("data_cache/rows_synth2zh.jsonl")]
    row = next(r for r in rows
               if is_val(r["row_id"]) and r["primitive"] == "choice")
    recs = [p for p in val if p["row_id"] == row["row_id"]]
    assert len(recs) == len(row["meta"]["options"])


def test_determinism():
    assert p1_val_passes() == p1_val_passes()
    assert synth3_passes("synth3zh", val=True) == synth3_passes(
        "synth3zh", val=True)


def test_group_rows_and_readout():
    """Synthetic-h readout: choice argmax over the row group; noul
    threshold; the temperature divides z before softmax."""
    torch.manual_seed(0)
    head = PassMLP(4)
    recs = [
        {"row_id": "r1", "qtype": "choice", "target": 1.0},
        {"row_id": "r1", "qtype": "choice", "target": 0.0},
        {"row_id": "r2", "qtype": "noul", "target": 0.0, "gold": 0.0},
        {"row_id": "r3", "qtype": "choice", "target": 0.0},
        {"row_id": "r3", "qtype": "choice", "target": 1.0},
    ]
    groups = group_rows(recs)
    assert [g for _, g in groups] == [[0, 1], [2], [3, 4]]
    h = torch.randn(5, 4)
    # force the head's ordering: make h[0] > h[1] and h[4] > h[3] clearly
    with torch.no_grad():
        head.net[0].weight.zero_()
        head.net[0].bias.zero_()
        head.net[3].weight.zero_()
        head.net[3].bias.zero_()  # z == 0 -> ties: argmax picks index 0
    rows = read_rows(head, 1.0, h, recs)
    # z all zero: choice top = first pass, conf = 0.5; noul p = 0.5
    assert math.isclose(rows[0][0], 0.5)
    assert math.isclose(rows[2][0], 0.5)
    choice_recs = [recs[0], recs[1], recs[3], recs[4]]
    acc = slice_accuracy(head, 1.0, h[[0, 1, 3, 4]], choice_recs)
    assert acc == 0.5  # r1 correct under ties (gold first), r3 not


def test_token_cost():
    def fake_tokenizer(text, add_special_tokens=False):
        return {"input_ids": [1] * 17}

    class FakeTok:
        tokenizer = staticmethod(fake_tokenizer)

    tok = FakeTok()
    assert token_cost(tok, {"text": "x", "image": None}) == 17 + 24
    assert token_cost(tok, {"text": "x",
                            "image": "img.png"}) == 17 + 24 + 399