"""Configuration loading helpers.

Loads ``config.yaml`` from the repository root and exposes a handful of
determinism helpers. ``get_seed`` returns an ``int`` random seed used to seed
NumPy, Python's ``random`` and each estimator.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

# Repository root is one level above this package (src/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load the pipeline configuration from a YAML file.

    Args:
        path: Optional path to a YAML config. Defaults to ``config.yaml`` at
            the repository root.

    Returns:
        The configuration dictionary.
    """
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def get_config(path: str | Path | None = None) -> dict:
    """Alias for :func:`load_config` (used as the public entry point)."""
    return load_config(path)


def get_seed(cfg: dict) -> int:
    """Return and apply the configured random seed.

    Seeds ``random``, ``numpy`` and sets the OS-level PYTHONHASHSEED hint. Any
    estimator / split that accepts a ``random_state`` should pass this integer
    explicitly (which the rest of the package does) so results are
    reproducible run-to-run.
    """
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed


def ensure_dirs(cfg: dict) -> None:
    """Create the output directories referenced by the configuration."""
    paths = cfg["paths"]
    for key in ("results_dir", "models_dir", "figures_dir"):
        d = REPO_ROOT / paths[key]
        d.mkdir(parents=True, exist_ok=True)
