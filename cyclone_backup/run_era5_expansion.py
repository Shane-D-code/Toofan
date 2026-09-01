#!/usr/bin/env python3
"""ERA5 expansion for satellite-overlap — STAGE 1: audit + coverage gap.

Implements the ERA5-expansion spec's audit phase (steps 1-2):
  1. Audit current IMD / ERA5 / satellite datasets and the storm split.
  2. For every satellite observation, check whether an ERA5 row covers it
     (exact-project matching) and record the time difference.

Outputs (this stage, all new files):
    results/era5_audit_summary.json      — current dataset sizes + overlap
    results/satellite_era5_coverage_before.csv — per-satellite-obs coverage

No downloads, no model changes, no data mutation. The download plan (stage 2)
is derived from the coverage CSV printed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import REPO_ROOT, load_config
from src import data as data_mod
from src import features as feat_mod

RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SATELLITE_TIME_TOL_MIN = 120  # project's satellite-vs-IMD matching tolerance


def _print_stats(name: str, df: pd.DataFrame) -> None:
    print(f"  {name:<12} obs={len(df):>5}  storms={df['storm_id'].nunique():>4}  "
          f"RI={int((df['RI_24h'] == 1).sum()):>4}  "
          f"non-RI={int((df['RI_24h'] == 0).sum()):>5}")


def main() -> None:
    cfg = load_config()

    print("=" * 72)
    print("ERA5 EXPANSION — STAGE 1: AUDIT + COVERAGE GAP")
    print("=" * 72)

    imd = data_mod.load_imd(cfg)
    era5 = data_mod.load_era5(cfg)
    sat = data_mod.load_satellite_metadata(cfg)
    combined = data_mod.build_combined_imd_era5(cfg)

    print("\n--- 1. CURRENT DATASETS ---")
    _print_stats("IMD", imd)
    _print_stats("ERA5", era5)
    _print_stats("Satellite", sat)
    print(f"  Combined(IMD+ERA5) obs={len(combined)}  "
          f"storms={combined['storm_id'].nunique()}")

    split = data_mod.split_by_storms(combined, cfg)
    split.verify()
    print("\n  Storm-safe split (from combined frame, seed 42):")
    print(f"    train {len(split.train_storms)} storms / "
          f"test {len(split.test_storms)} storms / val {len(split.val_storms)}")
    for name, storms in (("test", split.test_storms), ("val", split.val_storms)):
        s_s = set(storms)
        print(f"    {name} storms with ERA5 table coverage: "
              f"{len(s_s & set(era5['storm_id']))}")

    # ---- 2. Satellite vs IMD / ERA5 overlap --------------------------------
    print("\n--- 2. SATELLITE OVERLAP (project matching) ---")
    sat_t = sat[["storm_id", "datetime_utc", "latitude", "longitude",
                 "RI_24h"]].copy()
    sat_t["satellite_datetime"] = pd.to_datetime(
        sat["satellite_datetime"] if "satellite_datetime" in sat.columns
        else sat["datetime_utc"])
    sat_t["delta_minutes"] = sat["delta_minutes"] \
        if "delta_minutes" in sat.columns else 0.0

    era5_t = era5[["storm_id", "datetime_utc", "era5_datetime",
                   "era5_delta_minutes"]].copy()
    era5_t["datetime_utc"] = pd.to_datetime(era5_t["datetime_utc"])
    era5_t["era5_datetime"] = pd.to_datetime(era5_t["era5_datetime"])

    # Exact join on (storm_id, datetime_utc) == project's canonical matching.
    m = sat_t.merge(era5_t, on=["storm_id", "datetime_utc"], how="left")
    m["has_era5"] = m["era5_datetime"].notna().astype(int)
    m["time_difference_minutes"] = (
        (m["satellite_datetime"] - m["era5_datetime"]).dt.total_seconds() / 60.0
    ).where(m["has_era5"] == 1)

    # Which satellite rows are also in the canonical IMD table?
    imd_keys = imd[["storm_id", "datetime_utc"]].copy()
    imd_keys["_in_imd"] = 1
    m = m.merge(imd_keys, on=["storm_id", "datetime_utc"], how="left")
    m["has_imd"] = m["_in_imd"].fillna(0).astype(int)

    n_sat = len(sat_t)
    n_sat_era5 = int(m["has_era5"].sum())
    n_sat_imd = int(m["has_imd"].sum())
    n_all3 = int(((m["has_imd"] == 1) & (m["has_era5"] == 1)).sum())
    storms_all3 = m.loc[(m["has_imd"] == 1) & (m["has_era5"] == 1),
                        "storm_id"].nunique()
    ri_all3 = int((m["RI_24h"] == 1) & (m["has_imd"] == 1) &
                  (m["has_era5"] == 1).sum() if False else
                 ((m["RI_24h"] == 1) & (m["has_imd"] == 1) &
                  (m["has_era5"] == 1)).sum())

    print(f"  Satellite observations total              : {n_sat}")
    print(f"  Satellite storms total                   : {sat_t['storm_id'].nunique()}")
    print(f"  Satellite + IMD                          : {n_sat_imd} obs")
    print(f"  Satellite + ERA5 (current)               : {n_sat_era5} obs")
    print(f"  Satellite + IMD + ERA5 (current)         : {n_all3} obs / "
          f"{storms_all3} storms / {ri_all3} RI")
    print(f"  RI / non-RI among satellite dataset       : "
          f"{int((m['RI_24h'] == 1).sum())} / {int((m['RI_24h'] == 0).sum())}")

    # Per-satellite-storm breakdown of missing coverage.
    print("\n  Satellite storms lacking an ERA5 row (the gap):")
    gap_s = (m[m["has_era5"] == 0].groupby("storm_id")
             .agg(n_obs=("datetime_utc", "size"),
                  n_ri=("RI_24h", lambda s: int((s == 1).sum())),
                  first_dt=("datetime_utc", "min"),
                  last_dt=("datetime_utc", "max")))
    print(gap_s.to_string())
    print(f"  -> {len(gap_s)} satellite storms need ERA5; "
          f"{int(gap_s['n_ri'].sum())} of their obs are RI cases.")
    print(f"  over-all time window of satellite obs: "
          f"{m['datetime_utc'].min()} .. {m['datetime_utc'].max()}")

    # ---- coverage CSV -------------------------------------------------------
    out_cols = ["storm_id", "satellite_datetime", "datetime_utc", "latitude",
                "longitude", "RI_24h", "has_imd", "has_era5",
                "era5_datetime", "time_difference_minutes"]
    coverage = m[out_cols].sort_values(["storm_id", "datetime_utc"])
    coverage.to_csv(RESULTS / "satellite_era5_coverage_before.csv",
                    index=False)
    print(f"\n  saved -> results/satellite_era5_coverage_before.csv "
          f"({len(coverage)} rows)")

    # ---- summary JSON -------------------------------------------------------
    summary = {
        "stage": "1-audit",
        "current": {
            "imd": {"obs": int(len(imd)), "storms": int(imd["storm_id"].nunique()),
                    "ri": int((imd["RI_24h"] == 1).sum())},
            "era5": {"obs": int(len(era5)), "storms": int(era5["storm_id"].nunique()),
                     "ri": int((era5["RI_24h"] == 1).sum()),
                     "first": str(era5["datetime_utc"].min()),
                     "last": str(era5["datetime_utc"].max())},
            "satellite": {"obs": int(len(sat)),
                          "storms": int(sat["storm_id"].nunique()),
                          "ri": int((sat["RI_24h"] == 1).sum()),
                          "first": str(sat["datetime_utc"].min()),
                          "last": str(sat["datetime_utc"].max())},
        },
        "overlap": {
            "sat_imd": int(n_sat_imd),
            "sat_era5": int(n_sat_era5),
            "sat_imd_era5": {
                "obs": int(n_all3), "storms": int(storms_all3),
                "ri": int(ri_all3),
                "non_ri": int(n_all3 - ri_all3)},
            "sat_storms_need_era5": int(len(gap_s)),
            "sat_obs_need_era5": int((m["has_era5"] == 0).sum()),
            "ri_obs_need_era5": int(((m["has_era5"] == 0) & (m["RI_24h"] == 1)).sum()),
        },
        "notes": [
            "Only exact (storm_id, datetime_utc) joins are used, matching the "
            "project's canonical IMD+ERA5 matching logic.",
            "ERA5 features currently end at the last extracted reanalysis "
            "date, so satellite storms after that date have zero coverage.",
        ],
    }
    with open(RESULTS / "era5_audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  saved -> results/era5_audit_summary.json")

    print("\n" + "=" * 72)
    print("AUDIT DONE. Next: derive minimal CDS download plan from the "
          "coverage-before CSV and (if creds exist) download.")
    print("=" * 72)


if __name__ == "__main__":
    main()