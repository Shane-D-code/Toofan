"""Build the canonical multimodal RI dataset (Phase 4 of the SIH master plan).

The pipeline draws three modalities:

- IMD       : historical intensity / position (max_wind_kt, pressure, deltas).
- ERA5      : atmospheric environment at the storm centre.
- Satellite : storm-centred IR crop (embeddings/probabilities produced by the
              Colab CNN; the raw crops live under satellite_cnn_recovered/).

``build_multimodal`` joins them on ``(storm_id, datetime_utc)`` (inner join so
only rows with a known target survive) and emits explicit availability flags
``has_imd`` / ``has_era5`` / ``has_satellite``. Not every modality is required
in every row; missing cells are left as NaN. It also attaches the IMD feature
columns requested in the master plan (deltas, wind, pressure) and the ERA5
levels + derived feature columns via ``features.add_era5_derived``.

Output: ``ri_multimodal_dataset.csv`` at the repository root (and a copy under
the dataset/staging dir if configured).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT
from . import features as feat_mod
from . import data as data_mod


# Columns the IMD branch must expose (per the master plan).
_IMD_EXPORT = [
    "storm_id", "datetime_utc", "latitude", "longitude", "RI_24h",
    "max_wind_kt", "central_pressure_hpa", "pressure_drop_hpa",
    "wind_6h_change", "wind_minus_6h_kt", "delta_v_minus_6h_kt",
    "wind_minus_12h_kt", "delta_v_minus_12h_kt",
    "wind_minus_24h_kt", "delta_v_minus_24h_kt",
]

# Satellite metadata columns to expose.
_SAT_EXPORT = [
    "storm_id", "datetime_utc", "image_file", "satellite_datetime",
    "delta_minutes", "nan_fraction",
]


def load_imd_for_multimodal(cfg: dict) -> pd.DataFrame:
    """Load IMD and rename to the canonical export schema."""
    imd = data_mod.load_imd(cfg)
    if "wind_6h_change" not in imd.columns:
        # Derive from wind minus 6h if needed (wind - wind_minus_6h).
        imd["wind_6h_change"] = imd["max_wind_kt"] - imd["wind_minus_6h_kt"]
    imd["storm_id"] = imd["storm_id"].astype(str)
    return imd


def build_multimodal(
    cfg: dict,
    sat_meta: pd.DataFrame | None = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Join IMD + ERA5 + Satellite into one canonical multimodal table.

    Args:
        cfg: pipeline configuration.
        sat_meta: optional satellite metadata DataFrame (with image_path). If
            None, it is loaded from cfg on the cluster.
        out_path: where to write ``ri_multimodal_dataset.csv``. Defaults to the
            repository root.

    Returns:
        The multimodal DataFrame.
    """
    imd = load_imd_for_multimodal(cfg)
    era5 = data_mod.load_era5(cfg)
    era5 = feat_mod.add_era5_derived(era5)

    # Make ERA5 a clean feature frame keyed by (storm_id, datetime_utc).
    era5_feats = feat_mod.era5_feature_columns()
    era5_frame = era5[["storm_id", "datetime_utc"] + era5_feats].copy()
    era5_frame["storm_id"] = era5_frame["storm_id"].astype(str)
    era5_frame["datetime_utc"] = pd.to_datetime(era5_frame["datetime_utc"])

    # IMD master frame (all rows, to keep has_imd semantics even without ERA5).
    imd_frame = imd[_IMD_EXPORT].copy()
    imd_frame["storm_id"] = imd_frame["storm_id"].astype(str)
    imd_frame["datetime_utc"] = pd.to_datetime(imd_frame["datetime_utc"])
    imd_frame["has_imd"] = 1

    # Satellite frame.
    if sat_meta is None:
        sat_meta = data_mod.load_satellite_metadata(cfg)
    sat_frame = None
    if sat_meta is not None and len(sat_meta):
        sat_frame = sat_meta[_SAT_EXPORT].copy()
        sat_frame["storm_id"] = sat_frame["storm_id"].astype(str)
        sat_frame["datetime_utc"] = pd.to_datetime(sat_frame["datetime_utc"])
        sat_frame["satellite_datetime"] = pd.to_datetime(sat_frame["satellite_datetime"])
        # LEAKAGE SAFEGUARD (Phase 5): a satellite image taken AFTER the IMD
        # observation time cannot be used as a predictor at that observation
        # (it contains information from inside the RI window). We KEEP such
        # rows in the recovered metadata for transparency but do NOT flag them
        # as usable satellite predictors. A small clock tolerance (5 min) is
        # allowed so exact-hour matches are retained.
        sat_frame = sat_frame[sat_frame["satellite_datetime"] <=
                              sat_frame["datetime_utc"] + pd.Timedelta(minutes=5)]
        sat_frame["has_satellite"] = 1

    # Combine: start from IMD master, left-join satellite.
    df = imd_frame.copy()
    df["has_era5"] = 0
    if sat_frame is not None:
        df = df.merge(sat_frame, on=["storm_id", "datetime_utc"],
                      how="left", suffixes=("", "_sat"))
        df["has_satellite"] = df["has_satellite"].fillna(0).astype(int)
    else:
        df["has_satellite"] = 0

    # Left-join ERA5 features.
    df = df.merge(era5_frame, on=["storm_id", "datetime_utc"], how="left")
    df["has_era5"] = df[feat_mod.era5_feature_columns()[0]].notna().astype(int)

    # Collision cleanup (satellite delta_minutes vs era5_delta_minutes).
    for c in df.columns:
        if c.endswith("_sat") and c[:-4] in df.columns:
            df.drop(columns=[c], inplace=True)

    df = df.sort_values(["storm_id", "datetime_utc"]).reset_index(drop=True)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"[multimodal] Wrote {len(df)} rows -> {p}")
        summary = {
            "rows": int(len(df)),
            "storms": int(df["storm_id"].nunique()),
            "RI": int((df["RI_24h"] == 1).sum()),
            "has_imd": int(df["has_imd"].sum()),
            "has_era5": int(df["has_era5"].sum()),
            "has_satellite": int(df["has_satellite"].sum()),
            "all_three": int(((df["has_imd"] == 1) & (df["has_era5"] == 1)
                              & (df["has_satellite"] == 1)).sum()),
        }
        print(f"[multimodal] {summary}")
    return df
