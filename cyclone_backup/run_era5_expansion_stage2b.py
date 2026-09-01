#!/usr/bin/env python3
"""ERA5 expansion — STAGE 2b: extract features from downloaded NetCDF + build.

Reads the already-downloaded NetCDF files in ERA5_expanded/, performs fast
numpy bilinear interpolation at each satellite storm centre, appends to the
canonical ERA5 feature table, and builds the three-way common table.

This stage is idempotent: it never re-downloads. Re-running overwrites the
CSV outputs.

Outputs:
    results/era5_download_validation.csv
    results/RI_ERA5_features_expanded.csv
    results/satellite_imd_era5_common_expanded.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import REPO_ROOT, load_config
from src import data as data_mod

RESULTS = REPO_ROOT / "results"
RAW_DIR = REPO_ROOT / "ERA5_expanded"

VAR_MAP = {"divergence": "d", "relative_humidity": "r",
           "temperature": "t", "u_component_of_wind": "u",
           "v_component_of_wind": "v"}
LEVELS = [850, 700, 500, 200]
PROJECT_COLS = [f"{p}_{l}" for p in "drtuv" for l in LEVELS] + \
               ["shear_850_200"]


def bilinear(lat: float, lon: float, lat2d: np.ndarray, lon2d: np.ndarray,
             field2d: np.ndarray) -> float:
    """Bilinear interpolation on a regular lat/lon grid (latitude may descend)."""
    lat = float(lat); lon = float(lon)
    if lon > lon2d.max():
        lon = lon - 360.0
    if lat < lat2d.min() or lat > lat2d.max() or lon < lon2d.min() or lon > lon2d.max():
        return np.nan
    # Latitude index
    nl = lat2d.shape[0]; nl_ = lon2d.shape[0]
    if lat2d[1] > lat2d[0]:  # ascending
        j = np.searchsorted(lat2d, lat) - 1
    else:  # descending
        j = nl - 1 - (np.searchsorted(-lat2d, -lat) - 1) - 1
    j = min(max(j, 0), nl - 2)
    # Longitude index (ascending)
    i = np.searchsorted(lon2d, lon) - 1
    i = min(max(i, 0), nl_ - 2)

    lat0, lat1 = lat2d[j], lat2d[j + 1]
    lon0, lon1 = lon2d[i], lon2d[i + 1]
    f00 = field2d[j, i]; f10 = field2d[j, i + 1]
    f01 = field2d[j + 1, i]; f11 = field2d[j + 1, i + 1]

    if lat1 == lat0 or lon1 == lon0:
        return np.nan
    wy = (lat - lat0) / (lat1 - lat0)
    wx = (lon - lon0) / (lon1 - lon0)
    return (f00 * (1 - wx) * (1 - wy) + f10 * wx * (1 - wy)
            + f01 * (1 - wx) * wy + f11 * wx * wy)


def extract_file(nc_path: Path, obs: pd.DataFrame) -> pd.DataFrame:
    """Extract features for all obs within a NetCDF file. Fast, one open."""
    ds = xr.open_dataset(str(nc_path))
    try:
        lat2d = ds["latitude"].values
        lon2d = ds["longitude"].values
        lvl_vals = ds["pressure_level"].values
        var_data = {name: ds[name].values  # (valid_time, level, lat, lon)
                    for name in "drtuv" if name in ds}
        rows = []
        for _, row in obs.iterrows():
            rec = {"storm_id": row["storm_id"],
                   "datetime_utc": row["datetime_utc"],
                   "latitude": row["latitude"],
                   "longitude": row["longitude"]}
            for name, arr in var_data.items():
                prefix = VAR_MAP_of(name)
                # pick matching time (nearest by hour); reduces dim
                times = ds["valid_time"].values
                tt = pd.to_datetime(row["datetime_utc"])
                tidx = int(np.argmin(np.abs(pd.to_datetime(times) - tt)))
                for li, lev in enumerate(LEVELS):
                    # find level index (levels 850,700,500,200)
                    li_map = np.where(lvl_vals == lev)[0]
                    if li_map.size == 0:
                        rec[f"{prefix}_{lev}"] = np.nan
                        continue
                    li_ = int(li_map[0])
                    field2d = arr[tidx, li_]  # (lat, lon)
                    rec[f"{prefix}_{lev}"] = bilinear(
                        row["latitude"], row["longitude"],
                        lat2d, lon2d, field2d)
            u850, v850 = rec.get("u_850", np.nan), rec.get("v_850", np.nan)
            u200, v200 = rec.get("u_200", np.nan), rec.get("v_200", np.nan)
            rec["shear_850_200"] = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2) \
                if not (np.isnan(u850) or np.isnan(u200) or
                        np.isnan(v850) or np.isnan(v200)) else np.nan
            rows.append(rec)
        return pd.DataFrame(rows)
    finally:
        ds.close()


def VAR_MAP_of(cds_var: str) -> str:
    for k, v in VAR_MAP.items():
        if cds_var == k or v == cds_var:
            return v
    return cds_var


def main() -> None:
    cfg = load_config()
    print("=" * 72)
    print("ERA5 EXPANSION — STAGE 2b: EXTRACT (from downloads) + BUILD")
    print("=" * 72)

    imd = data_mod.load_imd(cfg)
    era5 = data_mod.load_era5(cfg)
    sat = data_mod.load_satellite_metadata(cfg)

    coverage_before = pd.read_csv(RESULTS / "satellite_era5_coverage_before.csv")
    coverage_before["datetime_utc"] = pd.to_datetime(
        coverage_before["datetime_utc"])
    missing = coverage_before[coverage_before["has_era5"] == 0].copy()
    missing["date"] = missing["datetime_utc"].dt.date.astype(str)

    n_before_era5_sat = int(coverage_before["has_era5"].sum())

    # ---- validate + extract ------------------------------------------------
    print("\n--- VALIDATION ---")
    val_rows = []
    extracted = []
    for nc_path in sorted(RAW_DIR.glob("era5_pressure_levels_*.nc")):
        date = nc_path.name.replace("era5_pressure_levels_", "").replace(".nc", "")
        obs_at = missing[missing["date"] == date]
        try:
            ds = xr.open_dataset(str(nc_path))
            n_time = len(ds["valid_time"])
            n_level = len(ds["pressure_level"])
            n_lat = len(ds["latitude"])
            n_lon = len(ds["longitude"])
            has_all_nan = any(bool(ds[v].isnull().all().item())
                             for v in "drtuv" if v in ds)
            valid = n_time > 0 and n_level == 4 and not has_all_nan
            val_rows.append({"filename": nc_path.name, "valid": bool(valid),
                             "n_time": n_time, "n_level": n_level,
                             "n_lat": n_lat, "n_lon": n_lon,
                             "has_all_nan_var": bool(has_all_nan), "error": ""})
            ds.close()
            if valid and len(obs_at) > 0:
                feats = extract_file(nc_path, obs_at)
                extracted.append(feats)
                n_valid = int(feats["d_850"].notna().sum())
                print(f"  {date}: {len(feats)} obs, {n_valid} valid d_850")
        except Exception as exc:
            val_rows.append({"filename": nc_path.name, "valid": False,
                             "error": str(exc)})
            print(f"  {date}: ERROR {exc}")

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(RESULTS / "era5_download_validation.csv", index=False)
    print(f"\n  {int(val_df['valid'].sum())}/{len(val_df)} files valid")
    print(f"  saved -> results/era5_download_validation.csv")

    new_era5 = pd.concat(extracted, ignore_index=True) if extracted \
        else pd.DataFrame()

    # ---- build expanded table ----------------------------------------------
    print("\n--- BUILD EXPANDED ERA5 TABLE ---")
    if len(new_era5) > 0:
        new_era5["era5_datetime"] = new_era5["datetime_utc"]
        new_era5["era5_delta_minutes"] = 0.0
        new_era5 = new_era5.merge(
            imd[["storm_id", "datetime_utc", "RI_24h"]],
            on=["storm_id", "datetime_utc"], how="left", suffixes=("", "_imd"))
        if "RI_24h_imd" in new_era5.columns:
            new_era5["RI_24h"] = new_era5["RI_24h_imd"].fillna(
                new_era5["RI_24h"])
            new_era5 = new_era5.drop(columns=["RI_24h_imd"])
        out_cols = (["storm_id", "datetime_utc", "latitude", "longitude",
                     "RI_24h", "era5_datetime", "era5_delta_minutes"]
                    + PROJECT_COLS)
        for c in out_cols:
            if c not in new_era5.columns:
                new_era5[c] = np.nan
        new_era5 = new_era5[out_cols].sort_values(
            ["storm_id", "datetime_utc"])

    expanded = pd.concat([era5, new_era5], ignore_index=True)
    expanded = expanded.drop_duplicates(subset=["storm_id", "datetime_utc"],
                                        keep="last")
    expanded["storm_id"] = expanded["storm_id"].astype(str)
    expanded["datetime_utc"] = pd.to_datetime(expanded["datetime_utc"])
    expanded = expanded.sort_values(["storm_id", "datetime_utc"]).reset_index(
        drop=True)
    expanded.to_csv(RESULTS / "RI_ERA5_features_expanded.csv", index=False)
    print(f"  existing rows: {len(era5)} (unmodified)")
    print(f"  new rows     : {len(new_era5)}")
    print(f"  expanded     : {len(expanded)} / "
          f"{expanded['storm_id'].nunique()} storms")
    print(f"  saved -> results/RI_ERA5_features_expanded.csv")

    # Sanity: verify new rows have plausible values (e.g. rh 0-100, t range).
    if len(new_era5) > 0:
        print("\n  New-row value sanity:")
        print(f"    r_850 range  : [{new_era5['r_850'].min():.1f}, "
              f"{new_era5['r_850'].max():.1f}]")
        print(f"    t_850 range  : [{new_era5['t_850'].min():.1f}, "
              f"{new_era5['t_850'].max():.1f}]")
        print(f"    shear_850_200 range: [{new_era5['shear_850_200'].min():.1f}, "
              f"{new_era5['shear_850_200'].max():.1f}]")

    # ---- three-way common table --------------------------------------------
    print("\n--- THREE-WAY (IMD + ERA5 + Satellite) COMMON TABLE ---")
    sat_key = sat[["storm_id", "datetime_utc", "latitude", "longitude",
                   "RI_24h"]].copy()
    sat_key["datetime_utc"] = pd.to_datetime(sat_key["datetime_utc"])
    sat_key["satellite_datetime"] = pd.to_datetime(
        sat["satellite_datetime"] if "satellite_datetime" in sat.columns
        else sat["datetime_utc"])

    sat_key = sat_key.merge(
        imd[["storm_id", "datetime_utc"]].assign(has_imd=1),
        on=["storm_id", "datetime_utc"], how="left")
    sat_key["has_imd"] = sat_key["has_imd"].fillna(0).astype(int)
    sat_key = sat_key.merge(
        expanded[["storm_id", "datetime_utc"]].assign(has_era5=1),
        on=["storm_id", "datetime_utc"], how="left")
    sat_key["has_era5"] = sat_key["has_era5"].fillna(0).astype(int)

    common = sat_key[(sat_key["has_imd"] == 1) &
                     (sat_key["has_era5"] == 1)].copy()

    era5_feats = expanded[["storm_id", "datetime_utc", "latitude", "longitude"]
                          + [c for c in PROJECT_COLS
                             if c in expanded.columns]].reset_index(drop=True)
    era5_feats = era5_feats.drop_duplicates(subset=["storm_id", "datetime_utc"])
    common = common.merge(era5_feats, on=["storm_id", "datetime_utc"],
                          how="left", suffixes=("", "_era"))

    imd_feats = imd[["storm_id", "datetime_utc"] +
                    [c for c in data_mod.IMD_FEATURE_COLS if c in imd.columns]]
    common = common.merge(imd_feats, on=["storm_id", "datetime_utc"],
                          how="left")

    common.to_csv(RESULTS / "satellite_imd_era5_common_expanded.csv",
                  index=False)
    n_common = len(common)
    n_st = common["storm_id"].nunique()
    n_ri = int((common["RI_24h"] == 1).sum())
    print(f"  common obs={n_common}  storms={n_st}  RI={n_ri}  "
          f"non-RI={n_common - n_ri}")
    print(f"  saved -> results/satellite_imd_era5_common_expanded.csv")

    # ---- final report -------------------------------------------------------
    n_after_era5_sat = int(sat_key["has_era5"].sum())
    n_after_all3 = n_common
    storms_after = n_st
    ri_after = n_ri
    n_before_all3 = int(((coverage_before["has_imd"] == 1) &
                         (coverage_before["has_era5"] == 1)).sum())

    print("\n" + "=" * 72)
    print("ERA5 EXPANSION RESULT")
    print("=" * 72)
    print(f"\nBefore:")
    print(f"  Satellite images       : {len(sat)}")
    print(f"  Satellite storms       : {sat['storm_id'].nunique()}")
    print(f"  Satellite + ERA5       : {n_before_era5_sat} obs")
    print(f"  Satellite + IMD + ERA5 : {n_before_all3} obs")
    print(f"\nAfter:")
    print(f"  Satellite images       : {len(sat)}")
    print(f"  Satellite storms       : {sat['storm_id'].nunique()}")
    print(f"  Satellite + ERA5       : {n_after_era5_sat} obs / "
          f"{sat_key.loc[sat_key['has_era5']==1,'storm_id'].nunique()} storms")
    print(f"  Satellite + IMD + ERA5 : {n_after_all3} obs / {storms_after} storms / "
          f"{ri_after} RI / {n_after_all3 - ri_after} non-RI")
    print(f"\nImprovement:")
    print(f"  +{n_after_era5_sat - n_before_era5_sat} common satellite/ERA5 obs")
    print(f"  +{storms_after - coverage_before.loc[coverage_before['has_era5']==1,'storm_id'].nunique()} common storms")
    print(f"  +{ri_after - int(((coverage_before['has_imd']==1)&(coverage_before['has_era5']==1)&(coverage_before['RI_24h']==1)).sum())} RI cases")
    print("=" * 72)


if __name__ == "__main__":
    main()