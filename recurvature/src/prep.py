"""Load and clean IBTrACS best-track data.

IBTrACS represents missing values as blank/space strings, not NaN, and has
a units row right under the header — both are handled here. USA-agency
(JTWC/NHC) columns are preferred and fall back to WMO-agency columns when
USA data isn't available for a storm.
"""

import pandas as pd

NUMERIC_COLS = [
    "LAT", "LON", "WMO_WIND", "WMO_PRES", "DIST2LAND", "USA_LAT", "USA_LON",
    "USA_WIND", "USA_PRES", "STORM_SPEED", "STORM_DIR",
]

KEEP_COLS = [
    "SID", "SEASON", "NAME", "ISO_TIME", "lat", "lon", "wind", "pres",
    "STORM_SPEED", "STORM_DIR", "DIST2LAND",
]


def load_clean(path: str, min_season: int = 1980, min_points: int = 12) -> pd.DataFrame:
    """Read the raw IBTrACS CSV and return a cleaned, storm-sorted dataframe."""
    df = pd.read_csv(path, skiprows=[1], low_memory=False)  # row 1 = units, skip it
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")

    df = df[df["SEASON"] >= min_season].copy()
    df = df[df["TRACK_TYPE"] == "main"].copy()

    df["lat"] = df["USA_LAT"].fillna(df["LAT"])
    df["lon"] = df["USA_LON"].fillna(df["LON"])
    df["wind"] = df["USA_WIND"].fillna(df["WMO_WIND"])
    df["pres"] = df["USA_PRES"].fillna(df["WMO_PRES"])

    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    df = df[KEEP_COLS].dropna(subset=["lat", "lon", "STORM_DIR", "STORM_SPEED"])

    counts = df.groupby("SID").size()
    good_sids = counts[counts >= min_points].index
    df = df[df["SID"].isin(good_sids)].reset_index(drop=True)
    return df
