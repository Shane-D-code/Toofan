"""TOOFAN ModelAdapter base abstraction.

A ModelAdapter wraps a single underlying ML model (its artifact) and exposes
a uniform ``predict_proba``/``predict`` interface.  It records framework,
model type and provenance, enforces the feature schema, and performs
SHA-256 hashing on load.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .errors import GenesisArtifactMissing, GenesisFeatureError
from .features import GENESIS_FEATURES
from .provenance import build_provenance

logger = logging.getLogger(__name__)

# Approved Genesis frameworks (hard allow-list -> prevents accidental loads
# of CatBoost / ExtraTrees / gradient boosting / neural nets).
APPROVED_GENESIS_MODELS = {"lightgbm", "xgboost", "randomforest"}
APPROVED_GENESIS_FRAMEWORKS = {"lightgbm", "xgboost", "sklearn"}

# Artifact basenames that are known REJECTIONS for Genesis. The adapter
# guards so it can never accidentally load these generic artifacts.
DISALLOWED_ARTIFACT_TOKENS = [
    "catboost",
    "extratree",
    "extra_tree",
    "gradientboosting",
    "histgradientboosting",
    "best_model",
    "bestmodel",
    "logistic",
    "svm",
    ".pkl",
    ".keras",
    ".pt",
    ".npy",
]


class ModelAdapter(ABC):
    """Base class enforcing the approved Genesis model contract."""

    def __init__(self, name: str, framework: str, model_type: str) -> None:
        if framework not in APPROVED_GENESIS_FRAMEWORKS:
            raise GenesisArtifactMissing(
                f"Refusing to load Genesis '{name}' with unsupported "
                f"framework '{framework}'. Allowed: {sorted(APPROVED_GENESIS_FRAMEWORKS)}."
            )
        self.name = name
        self.framework = framework
        self.model_type = model_type
        self._model = None
        self._provenance: Optional[Dict[str, Any]] = None
        self._artifact_path: Optional[Path] = None
        self._artifact_hash: Optional[str] = None
        self._feature_schema = list(GENESIS_FEATURES)

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def load(self, artifact_path: str | Path, repo_root: Path) -> "ModelAdapter":
        """Load the underlying model artifact. Returns self."""

    @abstractmethod
    def predict_proba(self, X: "np.ndarray") -> np.ndarray:
        """Return probability of the positive (genesis, class 1) label."""

    def predict(self, X: "np.ndarray") -> np.ndarray:
        """Return a hard binary prediction using the documented 0.24 threshold."""
        p = self.predict_proba(X)
        return (p >= 0.24).astype(int)

    # -- guards ------------------------------------------------------------
    def _guard_artifact(self, artifact_path: str | Path) -> Path:
        path = Path(artifact_path)
        if not path.exists():
            raise GenesisArtifactMissing(
                f"Genesis artifact missing: {path}  ({self.name})."
            )
        lower = path.name.lower()
        for token in DISALLOWED_ARTIFACT_TOKENS:
            if token in lower:
                raise GenesisArtifactMissing(
                    f"Refusing to load disallowed Genesis artifact '{path.name}' "
                    f"(token '{token}'). Approved: LightGBM, XGBoost, RandomForest "
                    "(exact OPTIMIZED .joblib artifacts only)."
                )
        return path

    def _validate_features(self, X: "np.ndarray") -> None:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(GENESIS_FEATURES):
            raise GenesisFeatureError(
                f"Genesis feature count mismatch for '{self.name}'. "
                f"Expected {len(GENESIS_FEATURES)}, got {X.shape[1]}."
            )
        if not np.all(np.isfinite(X)):
            # The imputer runs in the inference pipeline before this point; a
            # NaN here signals insufficient raw input that could not be
            # imputed, so we refuse to fabricate a prediction.
            raise GenesisFeatureError(
                f"Genesis input contains non-finite values for '{self.name}'."
            )

    # -- accessors ---------------------------------------------------------
    @property
    def provenance(self) -> Dict[str, Any]:
        return dict(self._provenance or {})

    @property
    def artifact_path(self) -> Optional[Path]:
        return self._artifact_path

    @property
    def artifact_hash(self) -> Optional[str]:
        return self._artifact_hash

    @property
    def feature_schema(self) -> List[str]:
        return list(self._feature_schema)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
