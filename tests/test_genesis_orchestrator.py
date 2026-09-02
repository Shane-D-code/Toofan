"""Tests for orchestrator (Phase 1 / Phase 2) and runtime compatibility."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from toofan.core.config import configure_runtime, load_config
from toofan.core.schemas import CycloneState, ForecastContext
from toofan.genesis.ensemble import ENSEMBLE_WEIGHTS

from toofan.genesis.service import GenesisService


def _ctx(storm_id="S1", lat=9.0, lon=80.0):
    return ForecastContext(
        storm_id=storm_id,
        timestamp=datetime(2024, 7, 15, 6, 0, tzinfo=timezone.utc),
        candidate_latitude=lat,
        candidate_longitude=lon,
    )


# -- Q. native runtime compatibility -----------------------------------------
def test_configure_runtime_sets_omp(tmp_path):
    # Use a config with runtime.enabled true
    cfg = load_config()
    configure_runtime(cfg)
    assert os.environ.get("OMP_NUM_THREADS") == "1"


def test_configure_runtime_does_not_clobber_explicit(tmp_path):
    os.environ["OMP_NUM_THREADS"] = "4"
    cfg = {"runtime": {"enabled": True, "omp_num_threads": 1}}
    configure_runtime(cfg)
    # setdefault -> existing value preserved
    assert os.environ.get("OMP_NUM_THREADS") == "4"
    os.environ.pop("OMP_NUM_THREADS", None)


# -- orchestrator Phase 1 / Phase 2 ------------------------------------------
def test_orchestrator_runs_genesis_downstream_schedule(feature_dict):
    from toofan.orchestrator import CyclonePipelineOrchestrator

    cfg = {"genesis": {"mode": "production", "threshold": 0.24}}
    orch = CyclonePipelineOrchestrator(cfg=cfg, artifacts_dir="__no_such_dir__")

    calls = []

    def fake_trajectory(state):
        calls.append(("trajectory", state.storm_id))
        return {"status": "OK"}

    def fake_intensity(state):
        calls.append(("intensity", state.storm_id))
        return {"status": "OK"}

    def fake_rainfall(state):
        calls.append(("rainfall", state.storm_id))
        return {"status": "OK"}

    def fake_hazard(state):
        calls.append(("hazard_risk_engine", state.storm_id))
        return {"status": "OK"}

    orch.register_downstream("trajectory", fake_trajectory)
    orch.register_downstream("intensity", fake_intensity)
    orch.register_downstream("rainfall", fake_rainfall)
    orch.register_downstream("hazard_risk_engine", fake_hazard)

    state = orch.run_genesis(_ctx(), feature_dict, mode="production")
    assert isinstance(state, CycloneState)
    assert state.genesis is not None
    assert state.genesis.status == "UNAVAILABLE"  # no real artifacts

    results = orch.run_downstream(state)
    # DAG order
    assert [c[0] for c in calls] == [
        "trajectory",
        "intensity",
        "rainfall",
        "hazard_risk_engine",
    ]
    assert results["genesis"] is state.genesis


def test_orchestrator_genesis_does_not_call_downstream():
    """Verify Genesis itself never invokes downstream modules directly."""
    from toofan.genesis.service import GenesisService
    import inspect

    src = inspect.getsource(GenesisService)
    for downstream in ("trajectory", "intensity", "rainfall", "flood", "landslide"):
        assert downstream.lower() not in src.lower()


# -- P. Phase 1 orchestrator integration -------------------------------------
def test_phase1_produces_cyclone_state_with_genesis(feature_dict):
    from toofan.orchestrator import CyclonePipelineOrchestrator

    cfg = {"genesis": {"mode": "production", "threshold": 0.24}}
    orch = CyclonePipelineOrchestrator(cfg=cfg, artifacts_dir="__no_such_dir__")
    state = orch.run_genesis(_ctx(), feature_dict)
    assert isinstance(state, CycloneState)
    assert state.storm_id == "S1"
    assert state.genesis is not None
    assert hasattr(state.genesis, "probability")
