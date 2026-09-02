"""Shared pytest fixtures / helpers for Genesis tests.

Two strategies are used so the tests remain meaningful even when real
trained artifacts aren't present in the repo:

1. Mocked-probability tests (deterministic) — verify the exact ensemble math
   and threshold logic without any on-disk model.
2. Inline-model artifact tests — build tiny real LightGBM / XGBoost /
   RandomForest classifiers, serialize them to ``.joblib`` (or JSON/model
   file) in a tmp artifacts dir, and load them through the real adapters /
   ModelFactory. This exercises actual artifact loading + SHA-256
   provenance without depending on the production 300-sample artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def feature_vector() -> np.ndarray:
    """A fixed, deterministic 34-feature row derived from GENESIS_FEATURES."""
    from toofan.genesis.features import GENESIS_FEATURES

    n = len(GENESIS_FEATURES)
    v = np.linspace(0.1, 0.9, n)
    return v.reshape(1, n)


@pytest.fixture
def feature_dict() -> dict:
    """Deterministic dict of the 34 canonical feature names."""
    from toofan.genesis.features import GENESIS_FEATURES

    n = len(GENESIS_FEATURES)
    vals = np.linspace(0.1, 0.9, n)
    return {name: float(vals[i]) for i, name in enumerate(GENESIS_FEATURES)}


@pytest.fixture
def repo_root() -> Path:
    from toofan.core.config import REPO_ROOT

    return REPO_ROOT


@pytest.fixture
def tiny_lightgbm_file(tmp_path: Path):
    """Serialize a tiny LightGBM classifier to a .joblib-compatible file.

    LightGBM's native serializer uses its own format; we write the booster
    via ``save_model`` to a file ending .joblib to emulate the expected
    artifact.  (The OPTIMIZED production artifacts are LightGBM Boosters
    saved via booster.save_model; the adapter uses Booster(model_file=...).)
    """
    import lightgbm as lgb

    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 34))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=120) > 0).astype(int)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 7,
        "learning_rate": 0.1,
        "verbosity": -1,
        "seed": 42,
    }
    d = lgb.Dataset(X, label=y)
    booster = lgb.train(params, d, num_boost_round=5)
    path = tmp_path / "tc_genesis_lightgbm_300_OPTIMIZED.joblib"
    booster.save_model(str(path))
    return path


@pytest.fixture
def tiny_xgboost_file(tmp_path: Path):
    """Serialize a tiny XGBoost classifier to a .joblib file (using its model format)."""
    import xgboost as xgb

    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 34))
    y = (X[:, 2] + 0.4 * X[:, 3] + rng.normal(size=120) > 0).astype(int)

    clf = xgb.XGBClassifier(
        n_estimators=5, max_depth=2, learning_rate=0.2,
        eval_metric="logloss", use_label_encoder=False,
    )
    clf.fit(X, y)
    path = tmp_path / "tc_genesis_xgboost_300_OPTIMIZED.joblib"
    clf.save_model(str(path))
    return path


@pytest.fixture
def tiny_randomforest_file(tmp_path: Path):
    """Serialize a tiny RandomForest classifier via joblib."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 34))
    y = (X[:, 4] + rng.normal(size=120) > 0).astype(int)

    clf = RandomForestClassifier(n_estimators=5, random_state=42, max_depth=3)
    clf.fit(X, y)
    path = tmp_path / "tc_genesis_randomforest_300_OPTIMIZED.joblib"
    joblib.dump(clf, path)
    return path
