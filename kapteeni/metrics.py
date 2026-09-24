"""Calibration metrics + temperature scaling (no scipy dependency).

Proper-scoring-rule view: for soft targets t in [0,1] the empirical frequency
in a bin is mean(t), so ECE generalizes to soft labels unchanged.
"""

from __future__ import annotations

import math


def _reliability(probs: list[float], targets: list[float], bins: int = 15):
    n = len(probs)
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if lo <= probs[i] < hi or (b == bins - 1 and probs[i] >= hi)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        freq = sum(targets[i] for i in idx) / len(idx)
        out.append((conf, freq, len(idx)))
    return out


def ece(probs: list[float], targets: list[float], bins: int = 15) -> float:
    n = len(probs)
    if n == 0:
        return float("nan")
    rel = _reliability(probs, targets, bins)
    return sum(count * abs(conf - freq) for conf, freq, count in rel) / n


def brier(probs: list[float], targets: list[float]) -> float:
    n = len(probs)
    if n == 0:
        return float("nan")
    return sum((p - t) ** 2 for p, t in zip(probs, targets)) / n


def fit_temperature(
    probs_of_temp, lo: float = 0.05, hi: float = 8.0, iters: int = 40
) -> float:
    """Golden-section search for the T minimizing ECE.

    probs_of_temp(T) -> (probs, targets): caller supplies the readout for a
    given T (sigmoid/softmax are cheap; recomputing keeps this torch-free).
    """
    gr = (math.sqrt(5) - 1) / 2
    a, b = math.log(lo), math.log(hi)
    c, d = b - gr * (b - a), a + gr * (b - a)

    def score(x: float) -> float:
        t = math.exp(x)
        p, y = probs_of_temp(t)
        return ece(p, y)

    fc, fd = score(c), score(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = score(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = score(d)
    return round(math.exp((a + b) / 2), 4)


def softmax(xs: list[float], temperature: float = 1.0) -> list[float]:
    m = max(xs)
    exps = [math.exp(x / max(1e-9, temperature) - m / max(1e-9, temperature)) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]