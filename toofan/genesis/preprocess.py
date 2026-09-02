"""Genesis preprocessing pipeline.

Reproduces the inference preprocessing used by the trained Genesis models.
If the OPTIMIZED artifact contains its own pipeline, that pipeline should be
used directly (the adapters do that).  Here we apply:

    raw feature dict
        -> ordered feature vector (34 features, canonical order)
        -> imputation (existing imputer artifact, if present)
        -> validated ndarray for model inference

We NEVER double-impute, never double-transform, never reorder incorrectly,
and never silently fill missing features with arbitrary values.  If the raw
input lacks a required feature, we raise ``GenesisInsufficientInput`` and
return an explicit DATA_UNAVAILABLE status at the caller level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .errors import GenesisInsufficientInput
from .features import GENESIS_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Validated feature vector ready for model inference."""

    X: np.ndarray
    imputed: bool = False
    missing_fields: List[str] = None

    @property
    def shape(self):
        return self.X.shape


class GenesisPreprocessor:
    """Builds the 34-feature vector and applies the existing imputer."""

    def __init__(self, imputer_path: Optional[str | Path] = None) -> None:
        self._imputer = None
        self._imputer_path = Path(imputer_path) if imputer_path else None

    @property
    def imputer_available(self) -> bool:
        return self._imputer is not None

    def load_imputer(self, imputer_path: str | Path) -> None:
        """Load the Genesis imputer artifact (must exist)."""
        path = Path(imputer_path)
        if not path.exists():
            from .errors import GenesisArtifactMissing

            raise GenesisArtifactMissing(f"Genesis imputer missing: {path}")
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise GenesisArtifactMissing("joblib not installed.") from exc

        self._imputer = joblib.load(str(path))
        self._imputer_path = path
        logger.info("Loaded Genesis imputer from %s", path.name)

    def build_feature_vector(self, features: Dict[str, float]) -> PreprocessingResult:
        """Convert an ordered feature dict into a validated 34-vector.

        Args:
            features: dict keyed by canonical feature name -> numeric value.

        Returns:
            PreprocessingResult containing a 1-row ndarray.

        Raises:
            GenesisInsufficientInput: if any required feature is missing.
        """
        missing = [f for f in GENESIS_FEATURES if f not in features]
        if missing:
            raise GenesisInsufficientInput(
                f"Genesis input missing required field(s): {missing}. "
                "Returning DATA_UNAVAILABLE — will not fabricate a prediction."
            )

        vector = np.asarray([float(features[f]) for f in GENESIS_FEATURES], dtype=float)

        imputed = False
        if self._imputer is not None:
            vector = self._imputer.transform(vector.reshape(1, -1))[0]
            imputed = True

        return PreprocessingResult(
            X=vector.reshape(1, -1),
            imputed=imputed,
            missing_fields=missing,
        )
