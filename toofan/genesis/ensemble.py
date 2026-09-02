"""Calibrated soft-voting Genesis ensemble.

Only three models participate:

    LightGBM     weight 0.40
    XGBoost      weight 0.35
    RandomForest weight 0.25

p_ensemble = 0.40*p_lightgbm + 0.35*p_xgboost + 0.25*p_randomforest

- No majority voting of hard labels.
- No averaging of class labels.
- No equal weights.
- CatBoost / ExtraTrees / gradient boosting are never added.

Calibration: the weights (0.40/0.35/0.25) are a *soft-voting* weighting, NOT
a learned calibration.  Unless a validated calibration artifact exists in the
repository, raw == calibrated and ``calibrated=False``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .errors import GenesisModelUnavailable
from .model_adapters import (
    LightGBMAdapter,
    RandomForestAdapter,
    XGBoostAdapter,
)

logger = logging.getLogger(__name__)

ENSEMBLE_NAME = "genesis_soft_voting_ensemble"

# Exact documented ensemble weights.
ENSEMBLE_WEIGHTS = {
    "lightgbm": 0.40,
    "xgboost": 0.35,
    "randomforest": 0.25,
}


@dataclass
class EnsembleResult:
    """Result of a single ensemble inference."""

    ensemble_probability: float
    lightgbm_probability: float
    xgboost_probability: float
    randomforest_probability: float
    available_components: dict
    calibrated: bool = False


class GenesisSoftVotingEnsemble:
    """Calibrated soft-voting ensemble over the three approved models."""

    def __init__(self, lightgbm: Optional[LightGBMAdapter] = None,
                 xgboost: Optional[XGBoostAdapter] = None,
                 randomforest: Optional[RandomForestAdapter] = None) -> None:
        self.lightgbm = lightgbm
        self.xgboost = xgboost
        self.randomforest = randomforest
        self.calibrated = False
        self.calibrator = None

    def _require_components(self) -> None:
        missing = [
            name
            for name, adapter in (
                ("lightgbm", self.lightgbm),
                ("xgboost", self.xgboost),
                ("randomforest", self.randomforest),
            )
            if adapter is None or not adapter.is_loaded
        ]
        if missing:
            raise GenesisModelUnavailable(
                f"Ensemble requires ALL three approved components; missing: "
                f"{missing}. Ensemble = UNAVAILABLE."
            )

    def predict(self, X: np.ndarray) -> EnsembleResult:
        """Perform weighted-soft-voting ensemble inference on a single row.

        Returns per-component probabilities (class 1) and the weighted
        ensemble probability.
        """
        self._require_components()

        p_lgb = float(self.lightgbm.predict_proba(np.atleast_2d(X))[0])
        p_xgb = float(self.xgboost.predict_proba(np.atleast_2d(X))[0])
        p_rf = float(self.randomforest.predict_proba(np.atleast_2d(X))[0])

        p_ens = (
            ENSEMBLE_WEIGHTS["lightgbm"] * p_lgb
            + ENSEMBLE_WEIGHTS["xgboost"] * p_xgb
            + ENSEMBLE_WEIGHTS["randomforest"] * p_rf
        )

        result = EnsembleResult(
            ensemble_probability=p_ens,
            lightgbm_probability=p_lgb,
            xgboost_probability=p_xgb,
            randomforest_probability=p_rf,
            available_components={
                "lightgbm": True,
                "xgboost": True,
                "randomforest": True,
            },
            calibrated=self.calibrated,
        )
        logger.info(
            "Ensemble: p_lgb=%.4f p_xgb=%.4f p_rf=%.4f -> p_ens=%.4f "
            "(weights %s)",
            p_lgb, p_xgb, p_rf, p_ens,
            ENSEMBLE_WEIGHTS,
        )
        return result
