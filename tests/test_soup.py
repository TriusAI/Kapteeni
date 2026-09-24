import pytest
import torch

from kapteeni.soup import average_heads


def _fake_head(scale: float) -> dict:
    return {
        qt: {"net.0.weight": torch.full((8, 4), scale),
             "net.0.bias": torch.zeros(8),
             "net.3.weight": torch.full((1, 8), scale),
             "net.3.bias": torch.zeros(1)}
        for qt in ("noul", "choice", "score")
    }


def test_average_heads_is_elementwise_mean():
    a, b, c = _fake_head(1.0), _fake_head(2.0), _fake_head(6.0)
    soup = average_heads([a, b, c])
    t = soup["noul"]["net.0.weight"]
    assert t.shape == (8, 4)
    assert torch.allclose(t, torch.full((8, 4), 3.0))


def test_average_heads_keeps_dtype():
    a, b = _fake_head(1.0), _fake_head(2.0)
    for k in a["noul"]:
        a["noul"][k] = a["noul"][k].half()
    soup = average_heads([a, b])
    assert soup["noul"]["net.0.weight"].dtype == torch.float16


def test_average_heads_rejects_key_mismatch():
    a, b = _fake_head(1.0), _fake_head(2.0)
    del b["score"]["net.3.bias"]
    with pytest.raises(ValueError):
        average_heads([a, b])