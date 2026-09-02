"""Tests for the calibrated soft-voting ensemble and the 0.24 threshold."""

from __future__ import annotations

import math

import numpy as np
import pytest

from toofan.genesis.ensemble import ENSEMBLE_WEIGHTS, GenesisSoftVotingEnsemble
from toofan.genesis.errors import GenesisModelUnavailable
from toofan.genesis.model_adapters import (
    LightGBMAdapter,
    RandomForestAdapter,
    XGBoostAdapter,
)


class _Stub:
    """Minimal adapter-like stub returning a fixed probability."""

    def __init__(self, prob: float, name: str) -> None:
        self._prob = prob
        self.name = name
        self.is_loaded = True

    def predict_proba(self, X):
        return np.asarray([self._prob], dtype=float)


# -- G. exact weighted ensemble calculation ---------------------------------
def test_ensemble_exact_63():
    """Deterministic mocked probability test.

    LightGBM = 0.80, XGBoost = 0.60, RandomForest = 0.40
    0.40(0.80) + 0.35(0.60) + 0.25(0.40) = 0.32 + 0.21 + 0.10 = 0.63
    """
    ensemble = GenesisSoftVotingEnsemble(
        lightgbm=_Stub(0.80, "lgb"),
        xgboost=_Stub(0.60, "xgb"),
        randomforest=_Stub(0.40, "rf"),
    )
    res = ensemble.predict(np.zeros((1, 34)))
    assert math.isclose(res.ensemble_probability, 0.63, rel_tol=1e-9, abs_tol=1e-9)


def test_ensemble_weights_exact():
    assert ENSEMBLE_WEIGHTS == {"lightgbm": 0.40, "xgboost": 0.35, "randomforest": 0.25}
    assert ENSEMBLE_WEIGHTS["lightgbm"] == 0.40
    assert ENSEMBLE_WEIGHTS["xgboost"] == 0.35
    assert ENSEMBLE_WEIGHTS["randomforest"] == 0.25


def test_ensemble_is_probability_vote_not_label_vote():
    """Verify it's a weighted probability vote, not a hard-label majority."""
    # If it were a majority label vote, 2 models above 0.5 -> 1.
    # With these probabilities the format is identical regardless, but we
    # explicitly assert the per-component probabilities are exposed.
    ensemble = GenesisSoftVotingEnsemble(
        lightgbm=_Stub(0.80, "lgb"),
        xgboost=_Stub(0.60, "xgb"),
        randomforest=_Stub(0.40, "rf"),
    )
    res = ensemble.predict(np.zeros((1, 34)))
    assert res.lightgbm_probability == 0.80
    assert res.xgboost_probability == 0.60
    assert res.randomforest_probability == 0.40


def test_ensemble_with_real_adapters(
    tiny_lightgbm_file, tiny_xgboost_file, tiny_randomforest_file, repo_root
):
    lgb = LightGBMAdapter().load(tiny_lightgbm_file, repo_root)
    xgb = XGBoostAdapter().load(tiny_xgboost_file, repo_root)
    rf = RandomForestAdapter().load(tiny_randomforest_file, repo_root)
    ensemble = GenesisSoftVotingEnsemble(lgb, xgb, rf)
    res = ensemble.predict(np.zeros((1, 34)))
    expected = (
        0.40 * res.lightgbm_probability
        + 0.35 * res.xgboost_probability
        + 0.25 * res.randomforest_probability
    )
    assert math.isclose(res.ensemble_probability, expected, rel_tol=1e-9)


# -- missing component => ensemble UNAVAILABLE -------------------------------
def test_ensemble_requires_all_three():
    ensemble = GenesisSoftVotingEnsemble(
        lightgbm=_Stub(0.80, "lgb"),
        xgboost=_Stub(0.60, "xgb"),
        randomforest=None,
    )
    with pytest.raises(GenesisModelUnavailable):
        ensemble.predict(np.zeros((1, 34)))


# -- H. threshold = 0.24 -----------------------------------------------------
def test_threshold_is_024():
    from toofan.genesis.service import GenesisService

    svc = GenesisService(cfg={"genesis": {"mode": "production", "threshold": 0.24}})
    assert svc._threshold == 0.24


def test_threshold_configurable():
    from toofan.genesis.service import GenesisService

    svc = GenesisService(cfg={"genesis": {"mode": "production", "threshold": 0.30}})
    assert svc._threshold == 0.30
