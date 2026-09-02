"""TOOFAN Genesis integration — public entry point.

``configure_runtime()`` MUST be called before importing / running any ML
framework.  This entry module does NOT import lightgbm, xgboost, or sklearn
at module import time (they are imported lazily only when an adapter loads).
"""

from __future__ import annotations

from .core.config import configure_runtime, load_config
from .core.schemas import (
    CycloneState,
    ForecastContext,
    GenesisPrediction,
    sha256_of_file,
    utc_now_iso,
)
from .genesis.factory import ModelFactory
from .genesis.service import GenesisService
from .orchestrator import CyclonePipelineOrchestrator

__all__ = [
    "configure_runtime",
    "load_config",
    "CycloneState",
    "ForecastContext",
    "GenesisPrediction",
    "sha256_of_file",
    "utc_now_iso",
    "ModelFactory",
    "GenesisService",
    "CyclonePipelineOrchestrator",
]
