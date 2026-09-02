"""Genesis model provenance recording.

For every loaded Genesis artifact we record:

    artifact filename
    artifact path (relative to repo root)
    SHA-256 hash
    framework
    model type
    model version
    feature count
    feature schema
    target
    training dataset metadata (if available)

Provenance makes it impossible to confuse these fixed, untouched artifacts
with any newly trained replacement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..core.schemas import sha256_of_file
from .features import GENESIS_FEATURES, GENESIS_FEATURE_COUNT

PRODUCTION_TARGET = "genesis_24h"


def artifact_version_from_name(name: str) -> str:
    """Derive a stable human-readable version from the artifact filename."""
    base = Path(name).name
    if "OPTIMIZED" in base:
        return "1.0-optimized"
    return "1.0"


def build_provenance(
    artifact_path: str | Path,
    framework: str,
    model_type: str,
    repo_root: Path,
    dataset_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a full provenance record for a Genesis artifact.

    Computes SHA-256 on disk and embeds the canonical feature schema/target.
    """
    path = Path(artifact_path)
    name = path.name
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel = path.resolve()

    provenance: Dict[str, Any] = {
        "artifact_filename": name,
        "artifact_path": str(rel),
        "artifact_hash_sha256": sha256_of_file(path),
        "framework": framework,
        "model_type": model_type,
        "model_version": artifact_version_from_name(name),
        "feature_count": GENESIS_FEATURE_COUNT,
        "feature_schema": GENESIS_FEATURES,
        "target": PRODUCTION_TARGET,
        "dataset": {
            "storms": 191,
            "samples": 300,
            "years": "2015-2024",
            "basin": "North Indian Ocean",
            "note": (
                "Prototype: report states synthetic SST/SST-anomaly and "
                "TCHP/OHC700 were used; storm-aware CV was lower than "
                "final held-out test."
            ),
        },
    }
    if dataset_metadata:
        provenance["dataset_metadata"] = dataset_metadata

    return provenance


def digest_provenance(provenance: Dict[str, Any]) -> str:
    """Return a compact, stable digest string for logging/auditing."""
    return json.dumps(provenance, sort_keys=True, default=str)
