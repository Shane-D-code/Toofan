"""Tests for the GenesisService (modes, availability, feature handling)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from toofan.core.schemas import ForecastContext
from toofan.genesis.errors import GenesisInsufficientInput, GenesisModelUnavailable
from toofan.genesis.factory import ModelFactory
from toofan.genesis.service import GenesisService


def _ctx(storm_id="TEST_01", lat=10.0, lon=88.0):
    return ForecastContext(
        storm_id=storm_id,
        timestamp=datetime(2024, 8, 1, 0, 0, tzinfo=timezone.utc),
        candidate_latitude=lat,
        candidate_longitude=lon,
    )


# -- M. missing artifact handling (production) -------------------------------
def test_production_missing_lightgbm_returns_unavailable(tmp_path, feature_dict):
    cfg = {"genesis": {"mode": "production", "threshold": 0.24}}
    # artifacts_dir points to an empty temp dir -> LightGBM missing.
    svc = GenesisService(cfg=cfg, artifacts_dir=tmp_path)
    pred = svc.predict(_ctx(), feature_dict, mode="production")
    assert pred.status == "UNAVAILABLE"
    assert "lightgbm" not in svc._factory._adapters  # no substitute loaded


# -- ensemble missing one component -> UNAVAILABLE ---------------------------
def test_ensemble_missing_component_returns_unavailable(
    tmp_path, tiny_lightgbm_file, tiny_xgboost_file, feature_dict
):
    cfg = {"genesis": {"mode": "ensemble", "threshold": 0.24}}
    # Copy LightGBM + XGBoost into a dedicated subdir, leave RandomForest out.
    import shutil

    sub = tmp_path / "artifacts"
    sub.mkdir()
    shutil.copy(tiny_lightgbm_file, sub / tiny_lightgbm_file.name)
    shutil.copy(tiny_xgboost_file, sub / tiny_xgboost_file.name)

    svc = GenesisService(cfg=cfg, artifacts_dir=sub)
    pred = svc.predict(_ctx(), feature_dict, mode="ensemble")
    assert pred.status == "UNAVAILABLE"


# -- N. invalid feature handling ---------------------------------------------
def test_insufficient_feature_dict_returns_unavailable(tmp_path, feature_dict):
    cfg = {"genesis": {"mode": "production", "threshold": 0.24}}
    svc = GenesisService(cfg=cfg, artifacts_dir=tmp_path)
    bad = dict(feature_dict)
    bad.pop(next(iter(bad)))
    pred = svc.predict(_ctx(), bad, mode="production")
    assert pred.status == "UNAVAILABLE"


def test_build_feature_vector_raises_insufficient(tmp_path, feature_dict):
    from toofan.genesis.preprocess import GenesisPreprocessor

    pre = GenesisPreprocessor()
    bad = dict(feature_dict)
    bad.pop("sst")
    with pytest.raises(GenesisInsufficientInput):
        pre.build_feature_vector(bad)


def test_wrong_shape_rejected(tiny_lightgbm_file, repo_root):
    from toofan.genesis.errors import GenesisFeatureError
    from toofan.genesis.model_adapters import LightGBMAdapter

    adapter = LightGBMAdapter().load(tiny_lightgbm_file, repo_root)
    with pytest.raises(GenesisFeatureError):
        adapter.predict_proba(np.zeros((1, 5)))


# -- invalid mode -------------------------------------------------------------
def test_invalid_mode_rejected(feature_dict):
    cfg = {"genesis": {"mode": "bogus", "threshold": 0.24}}
    with pytest.raises(ValueError):
        GenesisService(cfg=cfg)


# -- explicit mode override ----------------------------------------------------
def test_mode_override_does_not_silently_switch(tmp_path, feature_dict):
    # default mode production but call with ensemble -> ensemble used
    cfg = {"genesis": {"mode": "production", "threshold": 0.24}}
    svc = GenesisService(cfg=cfg, artifacts_dir=tmp_path)
    pred = svc.predict(_ctx(), feature_dict, mode="ensemble")
    assert pred.model_name == "genesis_soft_voting_ensemble"
    assert pred.status == "UNAVAILABLE"  # artifacts missing in tmp


# -- availability reporting ----------------------------------------------------
def test_availability_report(tmp_path):
    cfg = {"genesis": {"mode": "production", "threshold": 0.24}}
    svc = GenesisService(cfg=cfg, artifacts_dir=tmp_path)
    avail = svc.availability()
    assert avail["production"] == "UNAVAILABLE"
    assert avail["ensemble"] == "UNAVAILABLE"
    assert avail["lightgbm"] == "MISSING"
