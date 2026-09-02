"""TOOFAN pipeline orchestrator.

The orchestrator coordinates the DAG of independent hazard modules.  It
controls the dependencies — Genesis never calls downstream modules directly.

Conceptual DAG:

    Genesis
       ↓
    CycloneState
       ↓
    Trajectory · Intensity · RI · Recurvature
       ↓
    Rainfall · Wind
       ↓
    Flood · Landslide
       ↓
    HazardRiskEngine

Phase 1 = ingest + Genesis (produces CycloneState with a GenesisPrediction).
Phase 2 = downstream hazard modules, scheduled by the orchestrator based on
          the CycloneState.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .core.config import load_config
from .core.schemas import CycloneState, ForecastContext, GenesisPrediction
from .genesis.factory import ModelFactory
from .genesis.service import GenesisService

logger = logging.getLogger(__name__)


class CyclonePipelineOrchestrator:
    """Coordinates Genesis (Phase 1) and downstream hazard modules (Phase 2)."""

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        artifacts_dir: Optional[str | Path] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        self._cfg = cfg or load_config()
        self._repo_root_e = repo_root
        self._genesis_service = GenesisService(
            cfg=self._cfg,
            artifacts_dir=artifacts_dir,
            repo_root=repo_root,
        )
        # Downstream module hooks (registered by the runtime).
        self._downstream_hooks: Dict[str, callable] = {}

    # -- registration ------------------------------------------------------
    def register_downstream(self, module_name: str, fn: callable) -> None:
        """Register a Phase 2 module hook (trajectory, intensity, rainfall...)."""
        self._downstream_hooks[module_name] = fn

    # -- Phase 1 -------------------------------------------------------------
    def run_genesis(
        self,
        context: ForecastContext,
        features: Dict[str, float],
        mode: Optional[str] = None,
    ) -> CycloneState:
        """Phase 1: produce a CycloneState carrying a GenesisPrediction."""
        pred = self._genesis_service.predict(context, features, mode=mode)
        state = CycloneState(
            storm_id=context.storm_id,
            timestamp=context.timestamp,
            latitude=context.candidate_latitude,
            longitude=context.candidate_longitude,
            genesis=pred,
        )
        return state

    # -- Phase 2 -------------------------------------------------------------
    def run_downstream(self, state: CycloneState) -> Dict[str, object]:
        """Phase 2: invoke registered downstream modules on a CycloneState.

        Order is respected according to the documented DAG.  Genesis does not
        call these directly; the orchestrator schedules them.
        """
        results: Dict[str, object] = {"genesis": state.genesis}

        if not state.has_genesis:
            logger.warning(
                "Genesis is %s for storm %s; Phase 2 proceeds but downstream "
                "modules receive a non-genesis state.",
                state.genesis.status if state.genesis else "absent",
                state.storm_id,
            )

        # DAG ordering phase 2.
        phase2_groups = [
            ("trajectory", "intensity", "ri", "recurvature"),
            ("rainfall", "wind"),
            ("flood", "landslide"),
            ("hazard_risk_engine",),
        ]
        for group in phase2_groups:
            for name in group:
                fn = self._downstream_hooks.get(name)
                if fn is None:
                    results[name] = None
                    logger.debug("No downstream hook registered for '%s'.", name)
                    continue
                try:
                    results[name] = fn(state)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Downstream module '%s' failed: %s", name, exc)
                    results[name] = {"status": "ERROR", "error": str(exc)}
        return results

    # -- conveniences ---------------------------------------------------------
    @property
    def genesis(self) -> GenesisService:
        return self._genesis_service
