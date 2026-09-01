#!/usr/bin/env python3
"""Recover the satellite IR branch from raw NCEP/CPC 4km IR .nc4 granules.

Usage:

    python recover_satellite.py

Steps:
1. Audit every ``merg_*_4km-pixel.nc4`` granule in Cnnfiles/ and write
   ``satellite_nc4_audit.csv``.
2. Match granules to real IMD observations (documented time tolerance).
3. Extract storm-centred, globally-normalised 128x128 crops and write the
   recovered dataset to ``satellite_cnn_recovered/`` (images/, metadata,
   extraction log, normalisation statistics).
4. Render a QC sample grid so we can visually confirm the cyclone is centred.

No data is fabricated and no granule is invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import REPO_ROOT, get_seed, load_config
from src import data as data_mod
from src import satellite_recovery as sr


def main() -> None:
    cfg = load_config()
    seed = get_seed(cfg)

    nc4_dir = REPO_ROOT / cfg["satellite"]["nc4_dir"]
    results_dir = REPO_ROOT / cfg["paths"]["results_dir"]

    # 1) Audit the raw granules.
    audit = sr.write_nc4_audit(nc4_dir, results_dir / "satellite_nc4_audit.csv")
    print(audit[["file", "granule_datetime_utc", "mean", "nan_fraction"]].to_string(index=False))

    # 2) Load IMD for matching (only rows with a known RI target).
    imd = data_mod.load_imd(cfg)

    # 3) Build the recovered dataset.
    rec = sr.build_recovered_dataset(nc4_dir, imd, cfg)

    # 4) QC sample grid.
    figs_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    sample_dir = figs_dir / "satellite_samples"
    sr.plot_sample_grid(rec["metadata"], sample_dir / "recovered_samples_grid.png", cols=4)

    print("\nRecovery complete. Review the QC grid before proceeding.")


if __name__ == "__main__":
    main()
