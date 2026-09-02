"""TOOFAN shared data structures.

These are the standardized inter-module contracts consumed by the TOOFAN
pipepline:  ``ForecastContext`` -> ``CycloneState`` -> ``GenesisPrediction``.
Downstream modules (trajectory, intensity, RI, rainfall, wind, flood,
landslide) are driven by the orchestrator, which schedules dependencies;
Genesis does NOT call any downstream module directly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ForecastContext:
    """Inherently at-least-one-observation context for a forecast."""

    storm_id: str
    timestamp: datetime
    candidate_latitude: float
    candidate_longitude: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CycloneState:
    """Canonical cyclone state emitted by Genesis and consumed downstream.

    ``genesis`` is optional because a fully-formed DAG may be evaluated with
    Genesis disabled or unavailable for a given storm.
    """

    storm_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    genesis: Optional["GenesisPrediction"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_genesis(self) -> bool:
        return self.genesis is not None and self.genesis.status == "SUCCESS"


@dataclass
class GenesisPrediction:
    """Standardized TOOFAN Genesis inference result.

    ``probability`` is the Genesis probability of class 1 (24-hour genesis).
    ``raw_probability`` is the un-calibrated model output; when a validated
    calibration artifact is available ``calibrated_probability`` holds the
    calibrated value and ``calibrated`` is True.  When no calibration artifact
    exists, raw == calibrated and ``calibrated`` is False — the model never
    pretends raw probabilities are calibrated.
    """

    probability: float
    predicted_class: int
    risk_level: str
    confidence: float
    threshold: float
    model_name: str
    model_version: str
    artifact_path: str
    artifact_hash: str
    feature_schema: List[str]
    provenance: Dict[str, Any]
    timestamp: str
    candidate_latitude: float = 0.0
    candidate_longitude: float = 0.0
    status: str = "SUCCESS"
    raw_probability: float = 0.0
    calibrated_probability: Optional[float] = None
    calibrated: bool = False
    # Ensemble-only fields (empty for single production model).
    ensemble_probability: Optional[float] = None
    lightgbm_probability: Optional[float] = None
    xgboost_probability: Optional[float] = None
    randomforest_probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "probability": self.probability,
            "predicted_class": self.predicted_class,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "artifact_path": self.artifact_path,
            "artifact_hash": self.artifact_hash,
            "feature_schema": self.feature_schema,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "candidate_latitude": self.candidate_latitude,
            "candidate_longitude": self.candidate_longitude,
            "status": self.status,
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "calibrated": self.calibrated,
            "ensemble_probability": self.ensemble_probability,
            "lightgbm_probability": self.lightgbm_probability,
            "xgboost_probability": self.xgboost_probability,
            "randomforest_probability": self.randomforest_probability,
        }
        return out


def sha256_of_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float_probability(value: Any) -> float:
    """Clamp a raw model score into a clean float in [0, 1]."""
    v = float(np.clip(float(value), 0.0, 1.0))
    return v
