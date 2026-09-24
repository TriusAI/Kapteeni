import math
import random

from kapteeni.metrics import brier, ece, fit_temperature, softmax


def _sigmoid(z, t=1.0):
    return [1 / (1 + math.exp(-(zi / t))) for zi in z]


def test_perfect_calibration_ece_zero():
    probs = [0.0] * 50 + [1.0] * 50
    targets = [0.0] * 50 + [1.0] * 50
    assert ece(probs, targets) == 0.0


def test_miscalibration_detected():
    # confidently wrong
    assert ece([1.0] * 100, [0.0] * 100) == 1.0


def test_ece_20_percent_bucket():
    # 100 predictions at p=0.2; 20 fire -> ECE 0
    probs = [0.2] * 100
    targets = [1.0] * 20 + [0.0] * 80
    assert ece(probs, targets) == 0.0


def test_soft_targets_generalize():
    probs = [0.2, 0.2]
    targets = [0.3, 0.1]  # mean 0.2 -> perfectly calibrated bin
    assert ece(probs, targets) == 0.0


def test_brier():
    assert brier([0.0], [0.0]) == 0.0
    assert brier([1.0], [0.0]) == 1.0


def test_fit_temperature_recovers_flattening():
    # logits are over-sharp (T_true=0.25); fit_temperature must find ~0.25
    rng = random.Random(0)
    z, y = [], []
    for _ in range(400):
        t = rng.random()
        z.append((t - 0.5) * 16)  # sharp
        y.append(t)
    # label: 1 with prob sigmoid(z/T_true), T_true = 0.25
    labels = [1.0 if rng.random() < _sigmoid([zi], 0.25)[0] else 0.0 for zi in z]

    def readout(T):
        return _sigmoid(z, T), labels

    T = fit_temperature(readout)
    assert 0.15 <= T <= 0.45, T


def test_softmax_temperature():
    p = softmax([3.0, 0.0], 1.0)
    assert math.isclose(sum(p), 1.0)
    hotter = softmax([3.0, 0.0], 0.1)
    assert hotter[0] > p[0]  # lower T -> sharper