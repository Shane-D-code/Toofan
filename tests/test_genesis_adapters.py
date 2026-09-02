"""Tests for the three approved Genesis adapters (loading + prediction)."""

from __future__ import annotations

import numpy as np
import pytest

from toofan.genesis.adapter import (
    APPROVED_GENESIS_FRAMEWORKS,
    DISALLOWED_ARTIFACT_TOKENS,
)
from toofan.genesis.errors import GenesisArtifactMissing
from toofan.genesis.model_adapters import (
    LightGBMAdapter,
    RandomForestAdapter,
    XGBoostAdapter,
)


# -- A. LightGBM loading ----------------------------------------------------
def test_lightgbm_loads(tiny_lightgbm_file, repo_root):
    adapter = LightGBMAdapter().load(tiny_lightgbm_file, repo_root)
    assert adapter.is_loaded
    assert adapter.name == "genesis_lightgbm"
    assert adapter.framework == "lightgbm"


# -- B. XGBoost loading -----------------------------------------------------
def test_xgboost_loads(tiny_xgboost_file, repo_root):
    adapter = XGBoostAdapter().load(tiny_xgboost_file, repo_root)
    assert adapter.is_loaded
    assert adapter.name == "genesis_xgboost"
    assert adapter.framework == "xgboost"


# -- C. RandomForest loading ------------------------------------------------
def test_randomforest_loads(tiny_randomforest_file, repo_root):
    adapter = RandomForestAdapter().load(tiny_randomforest_file, repo_root)
    assert adapter.is_loaded
    assert adapter.name == "genesis_randomforest"
    assert adapter.framework == "sklearn"


# -- D. LightGBM prediction -------------------------------------------------
def test_lightgbm_predicts(tiny_lightgbm_file, repo_root, feature_vector):
    adapter = LightGBMAdapter().load(tiny_lightgbm_file, repo_root)
    p = adapter.predict_proba(feature_vector)
    assert p.shape == (1,)
    assert 0.0 <= p[0] <= 1.0


# -- E. XGBoost prediction --------------------------------------------------
def test_xgboost_predicts(tiny_xgboost_file, repo_root, feature_vector):
    adapter = XGBoostAdapter().load(tiny_xgboost_file, repo_root)
    p = adapter.predict_proba(feature_vector)
    assert p.shape == (1,)
    assert 0.0 <= p[0] <= 1.0


# -- F. RandomForest prediction ---------------------------------------------
def test_randomforest_predicts(tiny_randomforest_file, repo_root, feature_vector):
    adapter = RandomForestAdapter().load(tiny_randomforest_file, repo_root)
    p = adapter.predict_proba(feature_vector)
    assert p.shape == (1,)
    assert 0.0 <= p[0] <= 1.0


# -- I. probability bounded [0,1] -------------------------------------------
@pytest.mark.parametrize(
    "adapter_fixture",
    ["tiny_lightgbm_file", "tiny_xgboost_file", "tiny_randomforest_file"],
)
def test_probabilities_bounded(adapter_fixture, repo_root, feature_vector, request):
    fpath = request.getfixturevalue(adapter_fixture)
    if "lightgbm" in adapter_fixture:
        adapter = LightGBMAdapter().load(fpath, repo_root)
    elif "xgboost" in adapter_fixture:
        adapter = XGBoostAdapter().load(fpath, repo_root)
    else:
        adapter = RandomForestAdapter().load(fpath, repo_root)
    p = adapter.predict_proba(feature_vector)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


# -- J. no model substitution ------------------------------------------------
def test_framework_allowlist_only():
    assert APPROVED_GENESIS_FRAMEWORKS == {"lightgbm", "xgboost", "sklearn"}
    assert "catboost" not in APPROVED_GENESIS_FRAMEWORKS


def test_disallowed_artifact_tokens_present():
    for token in ("catboost", "extratree", "gradientboosting", "best_model"):
        assert token in DISALLOWED_ARTIFACT_TOKENS


def test_lightgbm_rejects_nonjoblib_or_disallowed(tmp_path, repo_root):
    # A generic ".pkl" / "best_model" artifact token must be rejected.
    bad = tmp_path / "tc_genesis_BEST_MODEL_300.joblib"
    bad.write_bytes(b"fake")
    with pytest.raises(GenesisArtifactMissing):
        LightGBMAdapter().load(bad, repo_root)


def test_missing_artifact_fails_honestly(tmp_path, repo_root):
    missing = tmp_path / "does_not_exist.joblib"
    with pytest.raises(GenesisArtifactMissing):
        LightGBMAdapter().load(missing, repo_root)
