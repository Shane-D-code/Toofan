#!/usr/bin/env python3
"""ERA5 expansion for satellite-overlap — STAGE 2: download + extract + build.

Downloads ERA5 pressure-level data from Copernicus CDS for satellite
observations that currently lack ERA5 coverage, extracts features at storm
centre via bilinear interpolation, appends them to the canonical ERA5 feature
table, and builds the three-way (IMD + ERA5 + Satellite) common table.

Outputs:
    ERA5_expanded/                  — raw NetCDF downloads (one per date)
    results/era5_expanded_manifest.csv
    results/RI_ERA5_features_expanded.csv
    results/satellite_imd_era5_common_expanded.csv
    results/era5_download_validation.csv
    results/era5_audit_summary.json  (updated with after stats)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import REPO_ROOT, load_config
from src import data as data_mod
from src import features as feat_mod

RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
RAW_DIR = REPO_ROOT / "ERA5_expanded"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ERA5 CDS request parameters (matches existing extraction).
ERA5_VARIABLES = ["divergence", "relative_humidity", "temperature",
                  "u_component_of_wind", "v_component_of_wind"]
ERA5_LEVELS = ["850", "700", "500", "200"]
ERA5_DATASET = "reanalysis-era5-pressure-levels"
# Pressure-level data is 0.25° grid.  A 2° × 2° box around the storm is
# more than enough for bilinear interpolation and keeps each download tiny.
BOX_PAD_DEG = 1.0

# Mapping from CDS variable names → project column names.
VAR_MAP = {
    "divergence": "d",
    "relative_humidity": "r",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}

PROJECT_COLS = ["d_850", "d_700", "d_500", "d_200",
                "r_850", "r_700", "r_500", "r_200",
                "t_850", "t_700", "t_500", "t_200",
                "u_850", "u_700", "u_500", "u_200",
                "v_850", "v_700", "v_500", "v_200",
                "shear_850_200"]


def build_download_plan(coverage_csv: Path) -> pd.DataFrame:
    """Read the coverage-before CSV and return missing obs grouped by date."""
    cov = pd.read_csv(coverage_csv)
    cov["datetime_utc"] = pd.to_datetime(cov["datetime_utc"])
    missing = cov[cov["has_era5"] == 0].copy()
    missing["date"] = missing["datetime_utc"].dt.date.astype(str)
    missing["hour_str"] = missing["datetime_utc"].dt.strftime("%H:00")
    missing["date_hour"] = missing["date"] + " " + missing["hour_str"]
    return missing


def download_era5_batch(date: str, hours: list[str],
                        lat_min: float, lat_max: float,
                        lon_min: float, lon_max: float,
                        out_path: Path) -> dict:
    """Download one ERA5 pressure-level file for a given date + hours.

    CDS area format: [N, W, S, E]  (north latitude, west longitude, etc.)
    """
    import cdsapi
    c = cdsapi.Client()

    # CDS uses North, West, South, East.  For BoB, longitudes are 60-100 E.
    request = {
        "product_type": "reanalysis",
        "variable": ERA5_VARIABLES,
        "pressure_level": ERA5_LEVELS,
        "year": date[:4],
        "month": date[5:7],
        "day": date[8:10],
        "time": hours,
        "area": [lat_max, lon_min, lat_min, lon_max],
        "format": "netcdf",
    }
    t0 = time.time()
    try:
        result = c.retrieve(ERA5_DATASET, request, str(out_path))
        elapsed = time.time() - t0
        return {"status": "ok", "path": str(result),
                "elapsed_s": round(elapsed, 1)}
    except Exception as exc:
        elapsed = time.time() - t0
        return {"status": "error", "error": str(exc),
                "elapsed_s": round(elapsed, 1)}


def bilinear_extract(nc_path: Path, lat: float, lon: float,
                     var_cds: str, level: int) -> float:
    """Bilinear interpolation of a single ERA5 variable at (lat, lon, level).

    ERA5 pressure-level NetCDF from CDS has dimensions
    (time, level, latitude, longitude).  Latitude is decreasing.
    """
    ds = xr.open_dataset(str(nc_path))
    try:
        var_name = [v for v in ds.data_vars if v.lower().startswith(var_cds[:3])]
        if not var_name:
            return np.nan
        var_name = var_name[0]

        # Select the closest level (pressure levels are 850, 700, 500, 200).
        if "level" in ds.dims:
            ds_level = ds[var_name].sel(level=level)
        else:
            ds_level = ds[var_name]

        # Select first (only) time step.
        if "time" in ds_level.dims:
            ds_2d = ds_level.isel(time=0)
        else:
            ds_2d = ds_level

        val = ds_2d.interp(latitude=lat, longitude=lon,
                           method="linear").values.item()
        return float(val)
    except Exception:
        return np.nan
    finally:
        ds.close()


def extract_features_from_nc(nc_path: Path, storm_rows: pd.DataFrame) -> pd.DataFrame:
    """Extract ERA5 features for a set of storm observations from one NetCDF.

    storm_rows must have columns: storm_id, datetime_utc, latitude, longitude.
    Returns a DataFrame with the project's ERA5 raw columns + shear.
    """
    results = []
    for _, row in storm_rows.iterrows():
        vals = {}
        for var_cds, prefix in VAR_MAP.items():
            for level_str in ERA5_LEVELS:
                level = int(level_str)
                col = f"{prefix}_{level_str}"
                vals[col] = bilinear_extract(nc_path, row["latitude"],
                                             row["longitude"],
                                             var_cds, level)
        # Derived: shear_850_200 = wind shear magnitude between 850 and 200 hPa.
        u850 = vals.get("u_850", np.nan)
        v850 = vals.get("v_850", np.nan)
        u200 = vals.get("u_200", np.nan)
        v200 = vals.get("v_200", np.nan)
        vals["shear_850_200"] = np.sqrt((u200 - u850) ** 2 +
                                        (v200 - v850) ** 2)
        vals["storm_id"] = row["storm_id"]
        vals["datetime_utc"] = row["datetime_utc"]
        results.append(vals)

    return pd.DataFrame(results)


def main() -> None:
    cfg = load_config()

    print("=" * 72)
    print("ERA5 EXPANSION — STAGE 2: DOWNLOAD + EXTRACT + BUILD")
    print("=" * 72)

    # ---- load current data --------------------------------------------------
    imd = data_mod.load_imd(cfg)
    era5 = data_mod.load_era5(cfg)
    sat = data_mod.load_satellite_metadata(cfg)

    # ---- load coverage-before CSV -------------------------------------------
    coverage_before = pd.read_csv(RESULTS / "satellite_era5_coverage_before.csv")
    coverage_before["datetime_utc"] = pd.to_datetime(
        coverage_before["datetime_utc"])

    missing = coverage_before[coverage_before["has_era5"] == 0].copy()
    missing["date"] = missing["datetime_utc"].dt.date.astype(str)
    missing["hour_str"] = missing["datetime_utc"].dt.strftime("%H:00")
    missing["date_hour"] = missing["date"] + " " + missing["hour_str"]

    n_before_era5_sat = int(coverage_before["has_era5"].sum())
    n_before_all3 = int(((coverage_before["has_imd"] == 1) &
                         (coverage_before["has_era5"] == 1)).sum())
    print(f"\nBEFORE: satellite+ERA5={n_before_era5_sat}  "
          f"satellite+IMD+ERA5={n_before_all3}")

    # ---- group by date for CDS requests ------------------------------------
    date_groups = missing.groupby("date").agg(
        hours=("hour_str", lambda x: sorted(set(x))),
        lat_min=("latitude", lambda x: x.min() - BOX_PAD_DEG),
        lat_max=("latitude", lambda x: x.max() + BOX_PAD_DEG),
        lon_min=("longitude", lambda x: x.min() - BOX_PAD_DEG),
        lon_max=("longitude", lambda x: x.max() + BOX_PAD_DEG),
        rows=("storm_id", lambda x: len(x)),
    ).reset_index()

    print(f"\n{len(date_groups)} unique dates to download "
          f"({len(missing)} obs total)")

    # ---- download -----------------------------------------------------------
    manifest_rows = []
    for _, grp in date_groups.iterrows():
        date = grp["date"]
        nc_file = RAW_DIR / f"era5_pressure_levels_{date}.nc"
        if nc_file.exists() and nc_file.stat().st_size > 0:
            print(f"  {date}: already present, skipping download")
            manifest_rows.append({
                "filename": nc_file.name, "date": date,
                "hours": ",".join(grp["hours"]),
                "area": f"[{grp['lat_max']:.1f},{grp['lon_min']:.1f},"
                        f"{grp['lat_min']:.1f},{grp['lon_max']:.1f}]",
                "n_obs": int(grp["rows"]), "download_status": "skipped-existing",
                "size_bytes": nc_file.stat().st_size, "error": ""})
            continue
        print(f"\n  Downloading {date} ...", end=" ", flush=True)
        info = download_era5_batch(
            date=date, hours=grp["hours"],
            lat_min=grp["lat_min"], lat_max=grp["lat_max"],
            lon_min=grp["lon_min"], lon_max=grp["lon_max"],
            out_path=nc_file)
        status = info["status"]
        print(f"{status} ({info['elapsed_s']}s)")
        manifest_rows.append({
            "filename": nc_file.name,
            "date": date,
            "hours": ",".join(grp["hours"]),
            "area": f"[{grp['lat_max']:.1f},{grp['lon_min']:.1f},"
                    f"{grp['lat_min']:.1f},{grp['lon_max']:.1f}]",
            "n_obs": int(grp["rows"]),
            "download_status": status,
            "size_bytes": nc_file.stat().st_size if nc_file.exists() else 0,
            "error": info.get("error", ""),
        })
        time.sleep(1)  # polite delay between requests

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(RESULTS / "era5_expanded_manifest.csv", index=False)
    print(f"\n  saved -> results/era5_expanded_manifest.csv "
          f"({len(manifest)} rows)")
    n_ok = (manifest["download_status"] == "ok").sum()
    n_err = (manifest["download_status"] == "error").sum()
    print(f"  downloads: {n_ok} ok, {n_err} errors")

    # ---- hand-off: validate + extract + build common table -------------------
    # Stage 2b performs validation, feature extraction (fast numpy bilinear)
    # and the three-way table build. Keeping that in one place means this
    # download step never re-runs the heavy extraction and stays reproducible.
    print("\n--- hand-off to stage 2b (validate + extract + build) ---")
    import run_era5_expansion_stage2b as _b
    _b.main()

    print("\n" + "=" * 72)
    print("STAGE 2 complete (download done). Common table written by stage 2b.")
    print("=" * 72)


if __name__ == "__main__":
    main()
