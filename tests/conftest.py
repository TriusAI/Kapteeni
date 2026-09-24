"""Shared model fixture: the E-suites run against ANY evaluate() implementation.

Default: MockModel (deterministic, no torch). Set KAPTEENI_TEST_BUNDLE=<path>
to run the same suites against a trained SystemOneModel; add
KAPTEENI_TEST_LORA=<adapter dir> for a LoRA-merged (P2) model.
"""

import os

import pytest


def _make_model():
    bundle = os.environ.get("KAPTEENI_TEST_BUNDLE", "")
    if bundle:
        pytest.importorskip("torch")
        from kapteeni.model import SystemOneModel

        return SystemOneModel(bundle,
                             lora=os.environ.get("KAPTEENI_TEST_LORA") or None)
    from kapteeni.mock import MockModel

    return MockModel()


@pytest.fixture(scope="session")
def model():
    return _make_model()


@pytest.fixture(scope="session")
def model_kind():
    return "trained" if os.environ.get("KAPTEENI_TEST_BUNDLE", "") else "mock"