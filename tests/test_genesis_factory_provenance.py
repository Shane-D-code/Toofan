"""Tests for ModelFactory registration, provenance and SHA-256 recording."""

from __future__ import annotations

import hashlib

import pytest

from toofan.core.schemas import sha256_of_file
from toofan.genesis.errors import GenesisModelUnavailable
from toofan.genesis.factory import ModelFactory
from toofan.genesis.model_adapters import (
    LightGBMAdapter,
    RandomForestAdapter,
    XGBoostAdapter,
)


# -- O. ModelFactory registration -------------------------------------------
def test_factory_known_keys():
    factory = ModelFactory()
    assert "genesis_lightgbm" in factory._registry
    assert "genesis_xgboost" in factory._registry
    assert "genesis_randomforest" in factory._registry
    # Exactly the approved set.
    assert set(factory._registry) == {
        "genesis_lightgbm",
        "genesis_xgboost",
        "genesis_randomforest",
    }


def test_factory_registration_set_approved():
    from toofan.genesis.adapter import APPROVED_GENESIS_MODELS

    assert APPROVED_GENESIS_MODELS == {"lightgbm", "xgboost", "randomforest"}


def test_factory_load_lightgbm(tiny_lightgbm_file, repo_root):
    factory = ModelFactory(repo_root=repo_root)
    adapter = factory.load("genesis_lightgbm", tiny_lightgbm_file)
    assert isinstance(adapter, LightGBMAdapter)
    assert adapter.is_loaded


def test_factory_load_xgboost(tiny_xgboost_file, repo_root):
    factory = ModelFactory(repo_root=repo_root)
    adapter = factory.load("genesis_xgboost", tiny_xgboost_file)
    assert isinstance(adapter, XGBoostAdapter)
    assert adapter.is_loaded


def test_factory_load_randomforest(tiny_randomforest_file, repo_root):
    factory = ModelFactory(repo_root=repo_root)
    adapter = factory.load("genesis_randomforest", tiny_randomforest_file)
    assert isinstance(adapter, RandomForestAdapter)
    assert adapter.is_loaded


def test_factory_unknown_key_rejected(repo_root):
    factory = ModelFactory(repo_root=repo_root)
    with pytest.raises(GenesisModelUnavailable):
        factory.load("genesis_catboost", "x.joblib")


def test_factory_registers_ensemble_key():
    factory = ModelFactory()
    assert "genesis_soft_voting_ensemble" in factory._composite_registry


def test_factory_loads_ensemble_composite(
    tiny_lightgbm_file, tiny_xgboost_file, tiny_randomforest_file, repo_root, feature_vector
):
    """ModelFactory must support loading the soft-voting ensemble composite."""
    from toofan.genesis.ensemble import GenesisSoftVotingEnsemble

    import shutil

    sub = tiny_lightgbm_file.parent
    # already in the same tmp dir; use it directly as artifacts dir
    factory = ModelFactory(
        artifacts_dir=sub, repo_root=repo_root,
        artifact_filenames={
            "lightgbm": tiny_lightgbm_file.name,
            "xgboost": tiny_xgboost_file.name,
            "randomforest": tiny_randomforest_file.name,
        },
    )
    ensemble = factory.load("genesis_soft_voting_ensemble")
    assert isinstance(ensemble, GenesisSoftVotingEnsemble)
    res = ensemble.predict(feature_vector)
    assert 0.0 <= res.ensemble_probability <= 1.0


def test_factory_ensemble_missing_component(
    tiny_lightgbm_file, tiny_xgboost_file, tmp_path, repo_root
):
    import shutil

    sub = tmp_path / "arts"
    sub.mkdir()
    shutil.copy(tiny_lightgbm_file, sub / tiny_lightgbm_file.name)
    shutil.copy(tiny_xgboost_file, sub / tiny_xgboost_file.name)
    factory = ModelFactory(artifacts_dir=sub, repo_root=repo_root)
    with pytest.raises(GenesisModelUnavailable):
        factory.load("genesis_soft_voting_ensemble")


def test_factory_missing_artifact_fails(tiny_lightgbm_file, repo_root):
    factory = ModelFactory(repo_root=repo_root)
    with pytest.raises(GenesisModelUnavailable):
        factory.load("genesis_lightgbm", "nonexistent.joblib")


# -- K. provenance correctness ----------------------------------------------
@pytest.mark.parametrize(
    "fixture_name,expected_name",
    [
        ("tiny_lightgbm_file", "tc_genesis_lightgbm_300_OPTIMIZED.joblib"),
        ("tiny_xgboost_file", "tc_genesis_xgboost_300_OPTIMIZED.joblib"),
        ("tiny_randomforest_file", "tc_genesis_randomforest_300_OPTIMIZED.joblib"),
    ],
)
def test_provenance_correct(
    fixture_name, expected_name, request, repo_root
):
    fpath = request.getfixturevalue(fixture_name)
    from toofan.genesis.adapter import ModelAdapter
    from toofan.genesis.model_adapters import (
        LightGBMAdapter,
        RandomForestAdapter,
        XGBoostAdapter,
    )

    cls = {
        "tiny_lightgbm_file": LightGBMAdapter,
        "tiny_xgboost_file": XGBoostAdapter,
        "tiny_randomforest_file": RandomForestAdapter,
    }[fixture_name]

    adapter = cls().load(fpath, repo_root)
    prov = adapter.provenance
    assert prov["artifact_filename"] == fpath.name
    assert prov["target"] == "genesis_24h"
    assert prov["feature_count"] == 34
    assert len(prov["feature_schema"]) == 34
    assert isinstance(prov["artifact_path"], str)


# -- L. SHA-256 recorded ------------------------------------------------------
@pytest.mark.parametrize(
    "fixture_name,cls",
    [
        ("tiny_lightgbm_file", LightGBMAdapter),
        ("tiny_xgboost_file", XGBoostAdapter),
        ("tiny_randomforest_file", RandomForestAdapter),
    ],
)
def test_sha256_recorded(fixture_name, cls, request, repo_root):
    fpath = request.getfixturevalue(fixture_name)
    adapter = cls().load(fpath, repo_root)
    on_disk = sha256_of_file(fpath)
    recorded = adapter.artifact_hash
    assert recorded == on_disk
    assert adapter.provenance["artifact_hash_sha256"] == on_disk
    assert len(recorded) == 64  # sha256 hex


def test_sha256_of_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello genesis")
    expected = hashlib.sha256(b"hello genesis").hexdigest()
    assert sha256_of_file(p) == expected
