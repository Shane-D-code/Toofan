"""TOOFAN ModelFactory — single model-loading registry.

All Genesis models are registered and loaded through this single factory.
There is no second, independent model-loading framework.  The factory is
model-agnostic and simply maps a registration key to a concrete adapter.

Registered Genesis keys:

    genesis_lightgbm              -> LightGBM             (PRIMARY)
    genesis_xgboost               -> XGBoost              (ensemble)
    genesis_randomforest          -> RandomForest         (ensemble)
    genesis_soft_voting_ensemble  -> soft-voting ensemble (composite)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Type

from ..core.config import REPO_ROOT
from .adapter import ModelAdapter
from .ensemble import GenesisSoftVotingEnsemble
from .errors import GenesisArtifactMissing, GenesisModelUnavailable
from .model_adapters import (
    LightGBMAdapter,
    RandomForestAdapter,
    XGBoostAdapter,
)

logger = logging.getLogger(__name__)

LIGHTGBM_KEY = "genesis_lightgbm"
XGBOOST_KEY = "genesis_xgboost"
RANDOMFOREST_KEY = "genesis_randomforest"
ENSEMBLE_KEY = "genesis_soft_voting_ensemble"


class ModelFactory:
    """Registry that instantiates and loads Genesis adapters by key."""

    # Ingredient key -> (adapter class, artifact filename), used for the
    # composite ensemble. Hard-coded to the approved three models.
    ENSEMBLE_INGREDIENTS = {
        LIGHTGBM_KEY: ("lightgbm", LightGBMAdapter),
        XGBOOST_KEY: ("xgboost", XGBoostAdapter),
        RANDOMFOREST_KEY: ("randomforest", RandomForestAdapter),
    }

    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        repo_root: Optional[Path] = None,
        artifact_filenames: Optional[Dict[str, str]] = None,
    ) -> None:
        self._repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        # artifact_filenames maps role ('lightgbm'/'xgboost'/'randomforest')
        # to exact OPTIMIZED filenames; defaults below.
        self._artifact_filenames = artifact_filenames or {
            "lightgbm": "tc_genesis_lightgbm_300_OPTIMIZED.joblib",
            "xgboost": "tc_genesis_xgboost_300_OPTIMIZED.joblib",
            "randomforest": "tc_genesis_randomforest_300_OPTIMIZED.joblib",
        }
        self._registry: Dict[str, Type[ModelAdapter]] = {
            LIGHTGBM_KEY: LightGBMAdapter,
            XGBOOST_KEY: XGBoostAdapter,
            RANDOMFOREST_KEY: RandomForestAdapter,
        }
        # Composite keys that are assembled from the single-model adapters.
        self._composite_registry: Dict[str, str] = {
            ENSEMBLE_KEY: "genesis_soft_voting_ensemble",
        }
        self._adapters: Dict[str, ModelAdapter] = {}
        self._ensembles: Dict[str, GenesisSoftVotingEnsemble] = {}

    # -- registration ------------------------------------------------------
    def register(self, key: str, adapter_class: Type[ModelAdapter]) -> None:
        """Register a single-model adapter class under a key (used by tests)."""
        self._registry[key] = adapter_class

    def register_composite(self, key: str, kind: str) -> None:
        """Register a composite (ensemble) key."""
        self._composite_registry[key] = kind

    # -- resolution --------------------------------------------------------
    def resolve_artifact_path(self, key_or_role: str) -> Path:
        """Resolve the on-disk path for an artifact.

        ``key_or_role`` may be a registration key or a role
        ('lightgbm'/'xgboost'/'randomforest').
        """
        role = {
            LIGHTGBM_KEY: "lightgbm",
            XGBOOST_KEY: "xgboost",
            RANDOMFOREST_KEY: "randomforest",
        }.get(key_or_role, key_or_role)
        base = self._artifacts_dir or (self._repo_root / "artifacts" / "genesis")
        return base / self._artifact_filenames[role]

    # -- loading -----------------------------------------------------------
    def load(self, key: str, artifact_path: str | Path | None = None) -> object:
        """Load (and cache) a Genesis model by key.

        For a single model, loads the adapter from ``artifact_path`` (or the
        default resolved path).  For the composite ensemble key, assembles
        the three approved adapters.  Raises ``GenesisModelUnavailable`` if
        the artifact is missing or the model cannot be constructed.
        """
        if key not in self._registry and key not in self._composite_registry:
            raise GenesisModelUnavailable(
                f"Unknown Genesis model key '{key}'. Registered: "
                f"{sorted(self._registry)} | composite: "
                f"{sorted(self._composite_registry)}."
            )

        if key in self._composite_registry:
            return self._load_ensemble(key)

        if key in self._adapters:
            return self._adapters[key]

        path = Path(artifact_path) if artifact_path else self.resolve_artifact_path(key)
        adapter_cls = self._registry[key]
        try:
            adapter = adapter_cls()
            adapter.load(path, self._repo_root)
        except GenesisArtifactMissing as exc:
            raise GenesisModelUnavailable(str(exc)) from exc

        self._adapters[key] = adapter
        return adapter

    def _load_ensemble(self, key: str) -> GenesisSoftVotingEnsemble:
        if key in self._ensembles:
            return self._ensembles[key]
        missing = []
        loaded = {}
        for ing_key, (role, _cls) in self.ENSEMBLE_INGREDIENTS.items():
            try:
                loaded[role] = self.load(ing_key)
            except GenesisModelUnavailable as exc:
                missing.append(role)
                logger.warning(
                    "Ensemble ingredient '%s' unavailable: %s", role, exc
                )
        if missing:
            raise GenesisModelUnavailable(
                f"Ensemble requires ALL three approved components; missing: "
                f"{missing}. Ensemble = UNAVAILABLE."
            )
        ensemble = GenesisSoftVotingEnsemble(
            lightgbm=loaded["lightgbm"],
            xgboost=loaded["xgboost"],
            randomforest=loaded["randomforest"],
        )
        self._ensembles[key] = ensemble
        return ensemble

    def get(self, key: str) -> Optional[object]:
        """Return a loaded adapter/ensemble, or None if not yet loaded."""
        if key in self._adapters:
            return self._adapters[key]
        if key in self._ensembles:
            return self._ensembles[key]
        return None

    # -- availability ------------------------------------------------------
    def availability(self) -> Dict[str, str]:
        """Report per-model file availability without forcing a load."""
        status: Dict[str, str] = {}
        for role in ("lightgbm", "xgboost", "randomforest"):
            path = self.resolve_artifact_path(role)
            status[role] = "AVAILABLE" if path.exists() else "MISSING"
        all_three = all(status[r] == "AVAILABLE" for r in ("lightgbm", "xgboost", "randomforest"))
        status["production"] = status["lightgbm"]
        status["ensemble"] = "AVAILABLE" if all_three else "UNAVAILABLE"
        return status
