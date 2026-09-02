"""TOOFAN Genesis runner (interactive demo path).

Usage:
    python run_genesis.py --mode production|ensemble [--csv features.csv]

The runner calls ``configure_runtime()`` BEFORE any ML framework import (the
GenesisAdapters import lightgbm/xgboost/scikit-learn lazily at load time, so
OpenMP threads are pinned safely before native runtimes spin up).

If the trained OPTIMIZED artifacts are not present, the runner prints the
expected artifact filenames and reports explicit UNAVAILABLE status — it never
fabricates a genesis prediction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

# Must run before importing any ML framework.
from toofan import configure_runtime, load_config

configure_runtime(load_config())

from toofan.core.schemas import ForecastContext  # noqa: E402
from toofan.genesis.features import GENESIS_FEATURES  # noqa: E402
from toofan.genesis.service import GenesisService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_genesis")


def build_features_from_csv(path: str) -> dict:

    import pandas as pd

    df = pd.read_csv(path)
    if df.shape[1] < len(GENESIS_FEATURES):
        logger.error(
            "CSV has %d columns; expected at least %d Genesis features.",
            df.shape[1], len(GENESIS_FEATURES),
        )
        sys.exit(2)
    row = df.iloc[0]
    features = {name: float(row[name]) for name in GENESIS_FEATURES if name in row}
    return features


def main() -> int:
    ap = argparse.ArgumentParser(description="TOOFAN Genesis prediction.")
    ap.add_argument("--mode", choices=["production", "ensemble"], default=None)
    ap.add_argument("--csv", help="Optional CSV with the 34 Genesis features.")
    ap.add_argument("--storm-id", default="CANDIDATE")
    ap.add_argument("--lat", type=float, default=10.0)
    ap.add_argument("--lon", type=float, default=88.0)
    args = ap.parse_args()

    cfg = load_config()
    service = GenesisService(cfg=cfg)

    print("Genesis availability:", json.dumps(service.availability(), indent=2))
    if service.availability().get("production") == "UNAVAILABLE":
        print(
            "\nWARNING: trained Genesis artifacts were not found under "
            f"{service._artifacts_dir}.\n"
            "Expected files:\n"
            "  " + "\n  ".join(service._default_files.values()) + "\n"
            "No prediction will be fabricated."
        )

    context = ForecastContext(
        storm_id=args.storm_id,
        timestamp=datetime.now(timezone.utc),
        candidate_latitude=args.lat,
        candidate_longitude=args.lon,
    )

    if args.csv:
        features = build_features_from_csv(args.csv)
    else:
        # Deterministic placeholder demo input (all-zero features).
        features = {name: 0.0 for name in GENESIS_FEATURES}
        print("\nNOTE: using placeholder (zero) feature vector for demo. "
              "Pass --csv for a real prediction.")

    pred = service.predict(context, features, mode=args.mode)
    print("\n=== GenesisPrediction ===")
    print(json.dumps(pred.to_dict(), indent=2, default=str))
    print(f"\nstatus: {pred.status}")
    return 0 if pred.status == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
