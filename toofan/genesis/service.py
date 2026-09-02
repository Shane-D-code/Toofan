"""Genesis inference service.

This is the entry point for Genesis prediction, supporting two explicit
modes:

    MODE A: "production"  -> LightGBM only
    MODE B: "ensemble"    -> LightGBM + XGBoost + RandomForest soft-voting

It consumes a ``ForecastContext`` plus a 34-feature dict, produces a
standardized ``GenesisPrediction``, and never fabricates predictions.

Availability logic:
    - production requires the LightGBM artifact
    - ensemble requires ALL three approved artifacts
    - if a required artifact is missing, the corresponding path returns a
      GenesisPrediction with status "UNAVAILABLE" (never a substitute model)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..core.config import REPO_ROOT
from ..core.schemas import GenesisPrediction, ForecastContext, to_float_probability, utc_now_iso
from .ensemble import ENSEMBLE_WEIGHTS, GenesisSoftVotingEnsemble
from .errors import (
    GenesisFeatureError,
    GenesisInsufficientInput,
    GenesisModelUnavailable,
)
from .factory import ModelFactory
from .features import GENESIS_FEATURES
from .model_adapters import (
    LightGBMAdapter,
    RandomForestAdapter,
    XGBoostAdapter,
)
from .preprocess import GenesisPreprocessor
from .risk import risk_level_from_class

logger = logging.getLogger(__name__)

PRODUCTION_KEY = "genesis_lightgbm"
XGBOOST_KEY = "genesis_xgboost"
RANDOMFOREST_KEY = "genesis_randomforest"


class GenesisService:
    """Top-level Genesis integration used by the orchestrator."""

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        model_factory: Optional[ModelFactory] = None,
        artifacts_dir: Optional[str | Path] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        self._cfg = cfg or {}
        genesis_cfg = self._cfg.get("genesis", {})
        self._repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir else (
            Path(genesis_cfg.get("artifacts_dir", self._repo_root / "artifacts" / "genesis"))
            if artifacts_dir is None and genesis_cfg.get("artifacts_dir")
            else (self._repo_root / "artifacts" / "genesis")
        )

        self._mode = genesis_cfg.get("mode", "production")
        if self._mode not in ("production", "ensemble"):
            raise ValueError(
                f"Invalid genesis.mode '{self._mode}'. Valid: production | ensemble."
            )

        self._threshold = float(genesis_cfg.get("threshold", 0.24))
        self._factory = model_factory or ModelFactory(
            artifacts_dir=self._artifacts_dir, repo_root=self._repo_root
        )

        artifacts = genesis_cfg.get("artifacts", {})
        self._default_files = {
            "lightgbm": artifacts.get("lightgbm", "tc_genesis_lightgbm_300_OPTIMIZED.joblib"),
            "xgboost": artifacts.get("xgboost", "tc_genesis_xgboost_300_OPTIMIZED.joblib"),
            "randomforest": artifacts.get("randomforest", "tc_genesis_randomforest_300_OPTIMIZED.joblib"),
            "imputer": artifacts.get("imputer", "tc_genesis_300_imputer.joblib"),
        }

        self._preprocessor = GenesisPreprocessor()
        self._ensemble = None
        self._available = self._probe_artifacts()

    # -- artifact probing --------------------------------------------------
    def _default_artifact_path(self, key: str) -> Path:
        return self._artifacts_dir / self._default_files[key]

    def _probe_artifacts(self) -> Dict[str, str]:
        """Report file-level availability for the three approved artifacts."""
        status = {}
        for key in ("lightgbm", "xgboost", "randomforest"):
            path = self._default_artifact_path(key)
            status[key] = "AVAILABLE" if path.exists() else "MISSING"
        return status

    # -- lazy loading ------------------------------------------------------
    def _load_lightgbm(self) -> LightGBMAdapter:
        try:
            return self._factory.load(PRODUCTION_KEY, self._default_artifact_path("lightgbm"))
        except GenesisModelUnavailable:
            raise

    def _load_xgboost(self) -> XGBoostAdapter:
        try:
            return self._factory.load(XGBOOST_KEY, self._default_artifact_path("xgboost"))
        except GenesisModelUnavailable:
            raise

    def _load_randomforest(self) -> RandomForestAdapter:
        try:
            return self._factory.load(RANDOMFOREST_KEY, self._default_artifact_path("randomforest"))
        except GenesisModelUnavailable:
            raise

    # -- imputer -----------------------------------------------------------
    def _ensure_imputer(self) -> None:
        if self._preprocessor.imputer_available:
            return
        path = self._default_artifact_path("imputer")
        if path.exists():
            self._preprocessor.load_imputer(path)
        else:
            # Imputer is optional: if absent, the build_feature_vector path
            # simply skips imputation (no double/implicit transform).
            logger.info("No imputer artifact present; skipping imputation.")

    # -- availability reporting --------------------------------------------
    def availability(self) -> Dict[str, str]:
        status = dict(self._available)
        status["production"] = (
            "AVAILABLE" if status.get("lightgbm") == "AVAILABLE" else "UNAVAILABLE"
        )
        all_three = all(status[k] == "AVAILABLE" for k in ("lightgbm", "xgboost", "randomforest"))
        status["ensemble"] = "AVAILABLE" if all_three else "UNAVAILABLE"
        return status

    # -- prediction ----------------------------------------------------------
    def predict(
        self,
        context: ForecastContext,
        features: Dict[str, float],
        mode: Optional[str] = None,
    ) -> GenesisPrediction:
        """Run Genesis inference.

        Args:
            context: ForecastContext (storm metadata).
            features: dict of the 34 feature values.
            mode: override for a single call ("production" | "ensemble").
                  The service mode is used when None.

        Returns:
            A standardized GenesisPrediction.
        """
        mode = mode or self._mode
        if mode not in ("production", "ensemble"):
            raise ValueError(f"Invalid genesis.mode '{mode}'.")

        self._ensure_imputer()

        try:
            prepped = self._preprocessor.build_feature_vector(features)
        except GenesisInsufficientInput as exc:
            return self._unavailable_prediction(
                context, mode, reason=str(exc)
            )

        X = prepped.X
        gctx = {
            "mode_requested": mode,
            "imputed": prepped.imputed,
            "available": self._available,
        }

        if mode == "production":
            return self._predict_production(context, X, gctx)
        return self._predict_ensemble(context, X, gctx)

    # -- MODE A: production ------------------------------------------------
    def _predict_production(self, context, X, gctx) -> GenesisPrediction:
        if self._available.get("lightgbm") != "AVAILABLE":
            return self._unavailable_prediction(
                context, "production",
                reason="LightGBM artifact missing -> production Genesis UNAVAILABLE "
                       "(no substitute model used).",
            )
        try:
            lgb = self._load_lightgbm()
            prob = to_float_probability(lgb.predict_proba(X)[0])
        except (GenesisModelUnavailable, GenesisFeatureError) as exc:
            return self._unavailable_prediction(context, "production", reason=str(exc))

        pred = self._finalize(
            context=context,
            mode="production",
            model_name="genesis_lightgbm",
            probability=prob,
            raw_probability=prob,
            calibrated_probability=None,
            calibrated=False,
            model=lgb,
        )
        logger.info(
            "Genesis production: model=%s mode=%s p=%.4f threshold=%.4f "
            "pred=%d version=%s hash=%s",
            "genesis_lightgbm", "production", prob, self._threshold, pred.predicted_class,
            pred.model_version, pred.artifact_hash[:12],
        )
        return pred

    # -- MODE B: ensemble ----------------------------------------------------
    def _predict_ensemble(self, context, X, gctx) -> GenesisPrediction:
        avail = self._available
        if not all(avail[k] == "AVAILABLE" for k in ("lightgbm", "xgboost", "randomforest")):
            return self._unavailable_prediction(
                context, "ensemble",
                reason=f"Ensemble requires ALL three approved components; "
                       f"available={avail}. Ensemble = UNAVAILABLE.",
            )
        try:
            lgb = self._load_lightgbm()
            xgb = self._load_xgboost()
            rf = self._load_randomforest()
            ensemble = GenesisSoftVotingEnsemble(lightgbm=lgb, xgboost=xgb, randomforest=rf)
        except GenesisModelUnavailable as exc:
            return self._unavailable_prediction(context, "ensemble", reason=str(exc))

        try:
            res = ensemble.predict(X)
        except (GenesisModelUnavailable, GenesisFeatureError) as exc:
            return self._unavailable_prediction(context, "ensemble", reason=str(exc))

        prob = to_float_probability(res.ensemble_probability)
        pred = self._finalize(
            context=context,
            mode="ensemble",
            model_name="genesis_soft_voting_ensemble",
            probability=prob,
            raw_probability=prob,
            calibrated_probability=None,
            calibrated=False,
            model=lgb,  # representative adapter for provenance of the lead model
            ensemble_result=res,
        )
        logger.info(
            "Genesis ensemble: p_lgb=%.4f p_xgb=%.4f p_rf=%.4f p_ens=%.4f "
            "threshold=%.4f pred=%d weights=%s",
            res.lightgbm_probability, res.xgboost_probability,
            res.randomforest_probability, prob, self._threshold,
            pred.predicted_class, ENSEMBLE_WEIGHTS,
        )
        return pred

    # -- shared finalization --------------------------------------------------
    def _finalize(
        self,
        context: ForecastContext,
        mode: str,
        model_name: str,
        probability: float,
        raw_probability: float,
        calibrated_probability: Optional[float],
        calibrated: bool,
        model,
        ensemble_result=None,
    ) -> GenesisPrediction:
        predicted_class = 1 if probability >= self._threshold else 0
        # confidence is a defensive measure of how far the probability is from
        # the decision boundary (clamped to [0,1]).
        distance = abs(probability - self._threshold)
        confidence = float(min(1.0, distance / max(self._threshold, 1.0 - self._threshold)))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        provenance = model.provenance if model is not None else {}

        pred = GenesisPrediction(
            probability=probability,
            predicted_class=predicted_class,
            risk_level=risk_level_from_class(predicted_class),
            confidence=confidence,
            threshold=self._threshold,
            model_name=model_name,
            model_version=provenance.get("model_version", "1.0"),
            artifact_path=provenance.get("artifact_path", ""),
            artifact_hash=provenance.get("artifact_hash_sha256", ""),
            feature_schema=list(GENESIS_FEATURES),
            provenance=provenance,
            timestamp=utc_now_iso(),
            candidate_latitude=context.candidate_latitude,
            candidate_longitude=context.candidate_longitude,
            status="SUCCESS",
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            calibrated=calibrated,
        )
        if ensemble_result is not None:
            pred.ensemble_probability = ensemble_result.ensemble_probability
            pred.lightgbm_probability = ensemble_result.lightgbm_probability
            pred.xgboost_probability = ensemble_result.xgboost_probability
            pred.randomforest_probability = ensemble_result.randomforest_probability
            pred.artifact_path = ", ".join(
                [str(p) for p in (
                    self._default_artifact_path("lightgbm"),
                    self._default_artifact_path("xgboost"),
                    self._default_artifact_path("randomforest"),
                )]
            )
        return pred

    # -- explicit UNAVAILABLE ----------------------------------------------
    def _unavailable_prediction(
        self, context: ForecastContext, mode: str, reason: str
    ) -> GenesisPrediction:
        logger.warning("Genesis %s UNAVAILABLE: %s", mode, reason)
        return GenesisPrediction(
            probability=0.0,
            predicted_class=0,
            risk_level="NON_GENESIS",
            confidence=0.0,
            threshold=self._threshold,
            model_name="genesis_lightgbm" if mode == "production" else "genesis_soft_voting_ensemble",
            model_version="",
            artifact_path="",
            artifact_hash="",
            feature_schema=list(GENESIS_FEATURES),
            provenance={"reason": reason},
            timestamp=utc_now_iso(),
            candidate_latitude=context.candidate_latitude,
            candidate_longitude=context.candidate_longitude,
            status="UNAVAILABLE",
        )
