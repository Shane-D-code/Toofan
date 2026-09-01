"""Feature engineering and recurvature-label construction.

Direction and month are encoded as sin/cos so a heading of 359 degrees and
1 degree read as close together, not far apart. "Heading momentum"
features (how much the storm has already turned in the last 3h / 9h) are
included since a track already mid-turn is more likely to keep turning.
"""

import numpy as np
import pandas as pd

FUTURE_STEPS = 8        # 8 * 3h = 24h ahead
TURN_THRESHOLD = 45.0   # degrees of heading change counted as "recurving"
PAST_WINDOW = 8         # timesteps of history fed to the sequence models

FEATURE_COLS = [
    "lat", "lon", "wind", "pres", "STORM_SPEED", "dir_sin", "dir_cos",
    "month_sin", "month_cos", "DIST2LAND", "dir_change_3h", "dir_change_9h",
]


def circ_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """Smallest signed difference a-b in degrees, result in (-180, 180]."""
    return (a - b + 180) % 360 - 180


def build_features(
    df: pd.DataFrame,
    future_steps: int = FUTURE_STEPS,
    turn_threshold: float = TURN_THRESHOLD,
) -> pd.DataFrame:
    df = df.copy()

    df["wind"] = df.groupby("SID")["wind"].apply(
        lambda s: s.interpolate().ffill().bfill()
    ).reset_index(level=0, drop=True)
    df["pres"] = df.groupby("SID")["pres"].apply(
        lambda s: s.interpolate().ffill().bfill()
    ).reset_index(level=0, drop=True)
    df["wind"] = df["wind"].fillna(df["wind"].median())
    df["pres"] = df["pres"].fillna(df["pres"].median())

    df["month"] = df["ISO_TIME"].dt.month
    df["dir_sin"] = np.sin(np.deg2rad(df["STORM_DIR"]))
    df["dir_cos"] = np.cos(np.deg2rad(df["STORM_DIR"]))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    grp = df.groupby("SID")["STORM_DIR"]
    df["dir_change_3h"] = grp.diff(1).apply(lambda x: (x + 180) % 360 - 180)
    raw_9h = df["STORM_DIR"] - grp.shift(3)
    df["dir_change_9h"] = ((raw_9h + 180) % 360) - 180

    df["future_dir"] = df.groupby("SID")["STORM_DIR"].shift(-future_steps)
    df["heading_swing"] = circ_diff(df["future_dir"], df["STORM_DIR"]).abs()
    df["recurve_label"] = (df["heading_swing"] >= turn_threshold).astype(float)
    df.loc[df["future_dir"].isna(), "recurve_label"] = np.nan

    return df
