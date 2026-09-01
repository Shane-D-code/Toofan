"""
build_ri_dataset.py
====================
Rebuilds the IMD best-track Rapid Intensification (RI) labeled dataset,
matching the pipeline already developed in TC2.ipynb (steps: dedupe primary
track -> 24h forward wind change -> RI_24h label -> -6/-12/-24h lag features).

RI definition (standard, used for SIH pipeline):
    RI_24h = 1 if wind(t+24h) - wind(t) >= RI_THRESHOLD_KT (default 30 kt)

This script is self-contained and only needs the `imdtrack` package
(pip install imdtrack), which ships IMD best-track data offline -- no
network access to IMD/RSMC servers is required.
"""

import pandas as pd
import numpy as np
import imdtrack as imd

RI_THRESHOLD_KT = 30
FORECAST_HOURS = 24
LAG_HOURS = (6, 12, 24)


def load_master_track() -> pd.DataFrame:
    bt = imd.load()
    df = bt.observations.copy()

    df = df.rename(columns={
        "time": "datetime_utc",
        "lat": "latitude",
        "lon": "longitude",
        "wind": "max_wind_kt",
        "pressure": "central_pressure_hpa",
        "pressure_drop": "pressure_drop_hpa",
        "ci_no": "dvorak_ci",
        "oci": "outer_closed_isobar_hpa",
        "oci_diameter": "oci_diameter_km",
    })

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    # Primary track: one fix per (storm_id, datetime_utc), keep last step
    primary = (
        df.sort_values(["storm_id", "datetime_utc", "step"])
        .drop_duplicates(subset=["storm_id", "datetime_utc"], keep="last")
        .reset_index(drop=True)
    )
    return primary


def add_ri_labels(track: pd.DataFrame) -> pd.DataFrame:
    df = track.sort_values(["storm_id", "datetime_utc"]).reset_index(drop=True)

    # ---- forward 24h wind change -> RI label ----
    future = df[["storm_id", "datetime_utc", "max_wind_kt"]].rename(
        columns={"datetime_utc": "target_time_24h", "max_wind_kt": "wind_24h_kt"}
    )
    df["target_time_24h"] = df["datetime_utc"] + pd.Timedelta(hours=FORECAST_HOURS)
    df = df.merge(
        future.rename(columns={"target_time_24h": "datetime_utc_match"}),
        left_on=["storm_id", "target_time_24h"],
        right_on=["storm_id", "datetime_utc_match"],
        how="left",
    ).drop(columns=["datetime_utc_match"])

    df["delta_v_24h_kt"] = df["wind_24h_kt"] - df["max_wind_kt"]
    df["RI_24h"] = (df["delta_v_24h_kt"] >= RI_THRESHOLD_KT).astype("float")
    df.loc[df["wind_24h_kt"].isna(), "RI_24h"] = np.nan

    # ---- backward lag features (persistence / recent trend) ----
    past = df[["storm_id", "datetime_utc", "max_wind_kt"]].rename(
        columns={"max_wind_kt": "past_wind_kt"}
    )
    for h in LAG_HOURS:
        col_time = f"target_time_minus_{h}h"
        col_wind = f"wind_minus_{h}h_kt"
        col_delta = f"delta_v_minus_{h}h_kt"

        df[col_time] = df["datetime_utc"] - pd.Timedelta(hours=h)
        lookup = past.rename(
            columns={"datetime_utc": col_time, "past_wind_kt": col_wind}
        )
        df = df.merge(lookup, on=["storm_id", col_time], how="left")
        df[col_delta] = df["max_wind_kt"] - df[col_wind]

    df["wind_6h_change"] = df["delta_v_minus_6h_kt"]
    df["forecast_horizon_hours"] = FORECAST_HOURS
    df["ri_threshold_kt"] = RI_THRESHOLD_KT
    return df


def build(basin_filter: str = None) -> pd.DataFrame:
    track = load_master_track()
    ri = add_ri_labels(track)
    if basin_filter:
        ri = ri[ri["basin"] == basin_filter].copy()
    return ri


if __name__ == "__main__":
    ri_all = build()
    ri_bob = build(basin_filter="BOB")

    print("All-basin dataset:", ri_all.shape, "| storms:", ri_all["storm_id"].nunique())
    print("BoB dataset:      ", ri_bob.shape, "| storms:", ri_bob["storm_id"].nunique())

    valid = ri_bob["RI_24h"].notna()
    print("\nBoB valid-target rows:", valid.sum())
    print(ri_bob.loc[valid, "RI_24h"].value_counts())
    print("RI positive rate: {:.2f}%".format(ri_bob.loc[valid, "RI_24h"].mean() * 100))

    ri_all.to_csv("/home/claude/tc_ri_cnn/outputs/IMD_master_RI_all_basins.csv", index=False)
    ri_bob.to_csv("/home/claude/tc_ri_cnn/outputs/IMD_BoB_RI_training_base.csv", index=False)
    print("\nSaved CSVs to outputs/")
