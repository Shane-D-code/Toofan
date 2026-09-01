"""Feature engineering.

The IMD branch uses its existing predictors directly. This module focuses on
the ERA5 branch: it builds a clean feature set and adds a *small* number of
physically meaningful derived predictors (layer-mean humidity, wind-speed
magnitudes at each level, upper-level divergence proxies, simple
thermodynamic / wind differences between levels). We deliberately avoid
generating hundreds of synthetic features, which would overfit the small
dataset.

Land interaction features are also provided.  A cyclone near land behaves
very differently from an oceanic cyclone.  The paper explicitly performs an
ocean-only analysis by excluding samples within 300 km of coastline.  We
add ``distance_to_land``, ``over_land`` and ``distance_to_coast`` as
optional features that can be used for sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ERA5_RAW_COLS, IMD_FEATURE_COLS

# Pressure levels present in the ERA5 table.
_LEVELS = [850, 700, 500, 200]


def era5_feature_columns() -> list[str]:
    """Return the names of the ERA5 predictors.

    Includes the raw columns plus the derived predictors produced by
    :func:`add_era5_derived`. Used to know which columns to feed the model.
    """
    base = list(ERA5_RAW_COLS)
    derived = era5_derived_columns()
    return base + [c for c in derived if c not in base]


def era5_feature_columns_with_temporal(lags_h=(6, 12, 24)) -> list[str]:
    """ERA5 predictors including the temporal delta columns."""
    base = era5_feature_columns()
    temporal = temporal_feature_columns(era5_derived_columns(), lags_h)
    return base + [c for c in temporal if c not in base]


def era5_derived_columns() -> list[str]:
    """Names of derived (engineered) ERA5 predictors.

    Kept deliberately small and physically meaningful (Phase 7 of the master
    plan): wind magnitudes, vertical U/V differences, humidity structure,
    temperature structure, dynamic (divergence) structure and shear. We do
    NOT generate hundreds of arbitrary features.
    """
    cols = []
    # Layer-mean relative humidity (850-500 hPa layer).
    cols.append("rh_mean_850_500")
    # Wind-speed magnitude at each level.
    for level in _LEVELS:
        cols.append(f"wind_mag_{level}")
    # 200-850 divergence contrast proxy (upper-level outflow).
    cols.append("divergence_contrast_200_850")
    # Vertical U and V differences (850-200 hPa).
    cols.append("u_shear_850_200")
    cols.append("v_shear_850_200")
    # Vertical wind difference magnitude (signed 850-200).
    cols.append("shear_direction_deg")
    # Humidity structure (850 minus 500, 850 minus 700, 700 minus 500).
    cols.append("r_850_minus_500")
    cols.append("r_850_minus_700")
    cols.append("r_700_minus_500")
    # Temperature structure (850 minus 500, 850 minus 700, 700 minus 500).
    cols.append("t_850_minus_500")
    cols.append("t_850_minus_700")
    cols.append("t_700_minus_500")
    # Dynamic structure: layer-mean divergence contrast (500-850) and 200-500.
    cols.append("divergence_contrast_500_850")
    cols.append("divergence_contrast_200_500")
    return cols


def add_era5_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically meaningful derived predictors to an ERA5 table.

    The operation is purely additive; the input DataFrame is not modified in
    place (a copy is returned).
    """
    out = df.copy()

    # Layer-mean relative humidity across the 850-500 hPa layer.
    out["rh_mean_850_500"] = (
        out["r_850"] + out["r_700"] + out["r_500"]
    ) / 3.0

    # Horizontal wind-speed magnitude at each level: sqrt(u^2 + v^2).
    for level in _LEVELS:
        out[f"wind_mag_{level}"] = np.sqrt(
            out[f"u_{level}"] ** 2 + out[f"v_{level}"] ** 2
        )

    # Upper-level divergence minus lower-level divergence: a proxy for
    # upper-level outflow / deep vertical mass coupling.
    out["divergence_contrast_200_850"] = out["d_200"] - out["d_850"]
    out["divergence_contrast_500_850"] = out["d_500"] - out["d_850"]
    out["divergence_contrast_200_500"] = out["d_200"] - out["d_500"]

    # Signed U and V components of the 850-200 hPa wind shear.
    out["u_shear_850_200"] = out["u_200"] - out["u_850"]
    out["v_shear_850_200"] = out["v_200"] - out["v_850"]
    out["shear_direction_deg"] = np.degrees(
        np.arctan2(out["v_shear_850_200"], out["u_shear_850_200"])
    )

    # Humidity structure (vertical gradients of relative humidity).
    out["r_850_minus_500"] = out["r_850"] - out["r_500"]
    out["r_850_minus_700"] = out["r_850"] - out["r_700"]
    out["r_700_minus_500"] = out["r_700"] - out["r_500"]

    # Temperature structure (vertical gradients of temperature).
    out["t_850_minus_500"] = out["t_850"] - out["t_500"]
    out["t_850_minus_700"] = out["t_850"] - out["t_700"]
    out["t_700_minus_500"] = out["t_700"] - out["t_500"]

    return out


def imd_feature_columns() -> list[str]:
    """Names of IMD predictor columns used by the model."""
    return list(IMD_FEATURE_COLS)


def combined_feature_columns() -> list[str]:
    """Names of predictors for the combined IMD + ERA5 model."""
    return imd_feature_columns() + era5_feature_columns()


def add_temporal_features(df: pd.DataFrame, id_col: str = "storm_id",
                          time_col: str = "datetime_utc",
                          lags_h: list[int] | None = None) -> pd.DataFrame:
    """Add historical temporal changes of ERA5 fields (Phase 8).

    For each storm and observation at time ``t`` we compute the change against
    the earlier observation at ``t - lag`` **within the same storm only**. Only
    historical information is used (deltas are computed from past rows, never
    future rows). Returns a new DataFrame with columns
    ``delta_<lag>h_<feat>`` for each requested feature.

    Args:
        df: ERA5 (derived) feature DataFrame sorted by time within storm.
        id_col: group (storm) column.
        time_col: timestamp column.
        lags_h: lag periods in hours to compare (default [6, 12, 24]).
    """
    if lags_h is None:
        lags_h = [6, 12, 24]

    to_diff = [c for c in df.columns if c.startswith(("d_", "r_", "t_", "u_",
                                                      "v_", "shear_", "wind_mag_",
                                                      "rh_mean_", "divergence_",
                                                      "r_850", "r_700"))]
    # Keep only numeric columns actually present.
    to_diff = [c for c in to_diff if c in df.columns]

    out = df.sort_values([id_col, time_col]).copy()
    out[time_col] = pd.to_datetime(out[time_col])

    frames_to_concat = [out[list(out.columns)]]
    for lag in lags_h:
        delta = pd.Timedelta(hours=lag)
        # Shift within each storm group to get the value at t-lag.
        shifted = out.groupby(id_col)[to_diff].shift(1)
        lagged_times = out.groupby(id_col)[time_col].shift(1)
        shifted = shifted.reset_index(drop=True)
        lagged_times = lagged_times.reset_index(drop=True)
        out_times = out[time_col].reset_index(drop=True)

        # Only accept the change if the lagged row is actually within `lag`
        # hours of the current row (not an arbitrary earlier row).
        valid = (out_times - lagged_times) <= delta
        vals = (out[to_diff].reset_index(drop=True) - shifted).where(valid)
        cols = {f"delta_{lag}h_{c}": vals[c].to_numpy() for c in to_diff}
        frames_to_concat.append(pd.DataFrame(cols, index=out.index))

    out = pd.concat(frames_to_concat, axis=1)
    return out


TEMPORAL_FEATURE_SUFFIXES = ["delta_6h_", "delta_12h_", "delta_24h_"]


def temporal_feature_columns(expected_derived: list[str], lags_h=(6, 12, 24)):
    """Return the names of the temporal delta feature columns."""
    cols = []
    for lag in lags_h:
        for c in expected_derived:
            cols.append(f"delta_{lag}h_{c}")
    return cols


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return a clean feature matrix, target series and usable column list.

    Handles inf/-inf and any fully-missing columns gracefully. Drops rows that
    are entirely missing after cleaning. The target ``RI_24h`` is cast to int.

    Args:
        df: DataFrame containing the predictor columns and ``RI_24h``.
        feature_cols: The candidate predictor column names.

    Returns:
        ``(X, y, usable_cols)`` where ``X``/``y`` are aligned on the row index
        of ``df`` and ``usable_cols`` are the columns with finite data.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely missing (no predictive information).
    usable_cols = [c for c in feature_cols if X[c].notna().any()]
    X = X[usable_cols]

    # Drop rows where all predictors are missing.
    mask = X.notna().any(axis=1)
    X = X[mask]
    y = df.loc[X.index, "RI_24h"].astype(int)

    return X, y, usable_cols


# ---------------------------------------------------------------------------
# Land interaction features (item 4 from reviewer checklist)
# ---------------------------------------------------------------------------

# Approximate coastline points for the Bay of Bengal basin.
# This is a simplified polygon; for production use a full GSHHS coastline.
_BAY_OF_BENGAL_COAST = np.array([
    (8.0, 77.0), (10.0, 80.0), (13.0, 80.5), (16.0, 82.0),
    (19.0, 84.0), (21.0, 87.0), (22.5, 89.0), (23.5, 91.0),
    (21.0, 92.0), (18.0, 94.0), (16.0, 96.0), (14.0, 98.0),
    (12.0, 99.0), (10.0, 98.5), (8.0, 97.0), (6.0, 96.0),
    (2.0, 103.0), (-2.0, 105.0),  # Sumatra
    (22.0, 88.0), (23.0, 90.0), (24.0, 92.0),  # Bangladesh
    (21.5, 85.5), (20.0, 83.0), (17.0, 81.5),  # East India
], dtype=np.float64)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points (km)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_land_interaction_features(
    df: pd.DataFrame,
    coast_points: np.ndarray | None = None,
) -> pd.DataFrame:
    """Add distance-to-land, over-land, and distance-to-coast features.

    Uses a simplified Bay of Bengal coastline polygon.  For each observation
    the distance to the nearest coastline point is computed (cheap Haversine).

    Features added:

    - ``distance_to_land_km``: minimum distance from the storm centre to any
      coastline point (km).
    - ``over_land``: binary flag (1 if within 50 km of coast, 0 otherwise).
    - ``distance_to_coast_km``: same as distance_to_land_km (alias).

    Args:
        df: DataFrame with ``latitude`` and ``longitude`` columns.
        coast_points: Optional (N, 2) array of [lat, lon] coastline points.
            Defaults to the simplified Bay of Bengal coastline.

    Returns:
        DataFrame with three new columns appended.
    """
    if coast_points is None:
        coast_points = _BAY_OF_BENGAL_COAST

    out = df.copy()
    lats = out["latitude"].to_numpy()
    lons = out["longitude"].to_numpy()

    distances = np.full(len(out), np.inf, dtype=np.float64)
    for i in range(len(out)):
        dists = np.array([
            _haversine_km(lats[i], lons[i], cp[0], cp[1])
            for cp in coast_points
        ])
        distances[i] = float(dists.min())

    out["distance_to_land_km"] = distances
    out["distance_to_coast_km"] = distances  # alias
    out["over_land"] = (distances <= 50.0).astype(int)

    return out


def land_sensitivity_split(
    df: pd.DataFrame,
    distance_threshold_km: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into ocean-only and all (or coastal) subsets.

    The paper excludes samples within 300 km of coastline for an ocean-only
    analysis.  This function provides that split for a sensitivity experiment.

    Args:
        df: DataFrame with ``distance_to_land_km`` column.
        distance_threshold_km: Distance threshold (km) for the ocean-only set.

    Returns:
        ``(df_ocean, df_all)`` where ``df_ocean`` has only rows with
        ``distance_to_land_km > threshold``.
    """
    if "distance_to_land_km" not in df.columns:
        df = add_land_interaction_features(df)
    ocean = df[df["distance_to_land_km"] > distance_threshold_km].copy()
    return ocean, df
