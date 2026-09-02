"""Concrete Genesis ModelAdapter implementations.

Only the three approved models are implemented here:

    LightGBM     (PRIMARY / PRODUCTION)
    XGBoost      (ensemble)
    RandomForest (ensemble)

Each adapter loads the exact OPTIMIZED artifact, hashes it, records
provenance, and exposes ``predict_proba`` for the positive class
(genesis).  No retraining/fine-tuning/modification occurs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .adapter import ModelAdapter
from .errors import GenesisArtifactMissing
from .provenance import build_provenance

logger = logging.getLogger(__name__)


class LightGBMAdapter(ModelAdapter):
    """Wraps the trained LightGBM Genesis model (PRIMARY)."""

    def __init__(self, name: str = "genesis_lightgbm") -> None:
        super().__init__(name=name, framework="lightgbm", model_type="gradient_boosting")

    def load(self, artifact_path: str | Path, repo_root: Path) -> "LightGBMAdapter":
        path = self._guard_artifact(artifact_path)
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - env dependent
            raise GenesisArtifactMissing(
                f"LightGBM not installed; cannot load '{path.name}'."
            ) from exc

        loader = getattr(self, "_raw_model", None)
        if loader is not None:
            # Some environments provide the artifact through a custom loader.
            self._model = loader
        else:
            try:
                # A LightGBM text/model file saved via booster.save_model.
                self._model = lgb.Booster(model_file=str(path))
            except Exception as lgb_exc:  # noqa: BLE001
                # Fallback: a pickled Booster / sklearn wrapper saved via joblib.
                try:
                    import joblib

                    obj = joblib.load(str(path))
                    if isinstance(obj, lgb.Booster):
                        self._model = obj
                    elif hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
                        self._model = obj
                    else:
                        raise GenesisArtifactMissing(
                            f"'{path.name}' did not load as a usable LightGBM model."
                        ) from lgb_exc
                except GenesisArtifactMissing:
                    raise
                except Exception as jl_exc:  # noqa: BLE001
                    raise GenesisArtifactMissing(
                        f"Failed to load LightGBM artifact '{path.name}': {jl_exc}"
                    ) from jl_exc
        self._artifact_path = path
        self._provenance = build_provenance(
            path, framework="lightgbm", model_type="gradient_boosting",
            repo_root=repo_root,
        )
        self._artifact_hash = self._provenance["artifact_hash_sha256"]
        logger.info(
            "Loaded LightGBM Genesis '%s' from %s (%s)",
            self.name, path.name, self._artifact_hash[:12],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._validate_features(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if hasattr(self._model, "predict_proba"):
            p = self._model.predict_proba(X)
            if p.ndim == 2:
                return np.asarray(p[:, 1], dtype=float)
            return np.asarray(p, dtype=float)
        raw = self._model.predict(X)  # Booster returns +class probabilities
        return np.asarray(raw, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.24).astype(int)


class XGBoostAdapter(ModelAdapter):
    """Wraps the trained XGBoost Genesis model (ensemble)."""

    def __init__(self, name: str = "genesis_xgboost") -> None:
        super().__init__(name=name, framework="xgboost", model_type="gradient_boosting")

    def load(self, artifact_path: str | Path, repo_root: Path) -> "XGBoostAdapter":
        path = self._guard_artifact(artifact_path)
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - env dependent
            raise GenesisArtifactMissing(
                f"XGBoost not installed; cannot load '{path.name}'."
            ) from exc

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path))
        if not hasattr(self._model, "_estimator_type"):
            self._model._estimator_type = "classifier"
        self._artifact_path = path
        self._provenance = build_provenance(
            path, framework="xgboost", model_type="gradient_boosting",
            repo_root=repo_root,
        )
        self._artifact_hash = self._provenance["artifact_hash_sha256"]
        logger.info(
            "Loaded XGBoost Genesis '%s' from %s (%s)",
            self.name, path.name, self._artifact_hash[:12],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._validate_features(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        p = self._model.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.24).astype(int)


class RandomForestAdapter(ModelAdapter):
    """Wraps the trained RandomForest Genesis model (ensemble).

    Only the *exact approved* RandomForest artifact is accepted; any
    foundational "best model" / replacement is rejected by the adapter
    guards.
    """

    def __init__(self, name: str = "genesis_randomforest") -> None:
        super().__init__(name=name, framework="sklearn", model_type="random_forest")

    def load(self, artifact_path: str | Path, repo_root: Path) -> "RandomForestAdapter":
        path = self._guard_artifact(artifact_path)
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - env dependent
            raise GenesisArtifactMissing(
                f"joblib not installed; cannot load '{path.name}'."
            ) from exc

        self._model = joblib.load(str(path))
        if not hasattr(self._model, "predict_proba"):
            raise GenesisArtifactMissing(
                f"Loaded '{path.name}' is not a probability classifier."
            )
        self._artifact_path = path
        self._provenance = build_provenance(
            path, framework="sklearn", model_type="random_forest",
            repo_root=repo_root,
        )
        self._artifact_hash = self._provenance["artifact_hash_sha256"]
        logger.info(
            "Loaded RandomForest Genesis '%s' from %s (%s)",
            self.name, path.name, self._artifact_hash[:12],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._validate_features(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        p = self._model.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.24).astype(int)
