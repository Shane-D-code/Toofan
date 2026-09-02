"""Original-vs-adapter fidelity verification.

There is no pre-existing separate "original inference script" for Genesis in
this repo (it does not exist yet — this integration creates it). Therefore the
meaningful fidelity check is:

    adapter.predict_proba(X)   ==   raw_framework.predict(X)

on identical input.  The adapter must NOT alter the underlying prediction,
preprocessing, or feature ordering.  We also verify the exact framework-level
prediction for each of the three approved models matches within tolerance.
"""

from __future__ import annotations

import numpy as np

import lightgbm as lgb
import xgboost as xgb

from toofan.genesis.model_adapters import (
    LightGBMAdapter,
    RandomForestAdapter,
    XGBoostAdapter,
)

TOL = 1e-6


def test_lightgbm_original_vs_adapter(tiny_lightgbm_file, repo_root, feature_vector):
    raw = lgb.Booster(model_file=str(tiny_lightgbm_file))
    adapter = LightGBMAdapter().load(tiny_lightgbm_file, repo_root)

    raw_p = np.asarray(raw.predict(feature_vector), dtype=float)
    adapter_p = adapter.predict_proba(feature_vector)
    assert np.allclose(raw_p, adapter_p, atol=TOL)
    # same hard prediction under the 0.24 threshold
    assert ((raw_p >= 0.24).astype(int) == adapter.predict(feature_vector)).all()


def test_xgboost_original_vs_adapter(tiny_xgboost_file, repo_root, feature_vector):
    raw = xgb.XGBClassifier()
    raw.load_model(str(tiny_xgboost_file))
    adapter = XGBoostAdapter().load(tiny_xgboost_file, repo_root)

    raw_p = raw.predict_proba(feature_vector)[:, 1]
    adapter_p = adapter.predict_proba(feature_vector)
    assert np.allclose(raw_p, adapter_p, atol=TOL)


def test_randomforest_original_vs_adapter(tiny_randomforest_file, repo_root, feature_vector):
    import joblib

    raw = joblib.load(str(tiny_randomforest_file))
    adapter = RandomForestAdapter().load(tiny_randomforest_file, repo_root)

    raw_p = raw.predict_proba(feature_vector)[:, 1]
    adapter_p = adapter.predict_proba(feature_vector)
    assert np.allclose(raw_p, adapter_p, atol=TOL)
