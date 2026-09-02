"""TOOFAN configuration loading and runtime safeguards.

``configure_runtime()`` MUST be called before importing any ML framework
(LightGBM, XGBoost, sklearn).  It pins OpenMP thread counts so that native
libraries (libomp, libiomp, mkl, etc.) do not spawn conflicting thread
pools — a known cause of crashes on Windows when multiple frameworks are
imported in one process.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def configure_runtime(cfg: dict | None = None) -> None:
    """Apply native-runtime compatibility safeguards.

    Must run BEFORE ML framework imports. Uses ``os.environ.setdefault`` so
    an explicit user override is never clobbered.
    """
    runtime = cfg.get("runtime", {}) if cfg else {}
    if runtime.get("enabled", True):
        os.environ.setdefault("OMP_NUM_THREADS", str(runtime.get("omp_num_threads", 1)))
        os.environ.setdefault("MKL_NUM_THREADS", str(runtime.get("omp_num_threads", 1)))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(runtime.get("omp_num_threads", 1)))

        # Avoid conflicting OpenMP runtime on Windows (skip in-process atexit).
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def load_config(path: str | Path | None = None) -> dict:
    """Load pipeline configuration from YAML.

    Args:
        path: Optional path to a YAML config. Defaults to
            ``config/pipeline.yaml`` at the repository root.

    Returns:
        The configuration dictionary.
    """
    path = Path(path) if path else REPO_ROOT / "config" / "pipeline.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg
