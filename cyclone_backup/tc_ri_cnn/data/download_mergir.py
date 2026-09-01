"""
download_mergir.py
===================
Downloads the FULL MERG-IR (GPM_MERGIR, NCEP/CPC 4km Global IR) archive
needed to train the RI-CNN properly, for every IMD best-track storm fix
that has a valid RI_24h label (+ the -6/-12/-24h lag times each fix needs).

>>> THIS SCRIPT MUST BE RUN SOMEWHERE WITH INTERNET ACCESS TO NASA <<<
>>> GES DISC (e.g. Google Colab, your laptop). It will NOT run inside <<<
>>> a network-restricted sandbox.                                    <<<

Requirements
------------
    pip install earthaccess xarray netCDF4

You need a free NASA Earthdata account: https://urs.earthdata.nasa.gov/
On first run, `earthaccess.login()` will prompt for your username/password
(or read them from a `.netrc` file / EARTHDATA_USERNAME + EARTHDATA_PASSWORD
env vars).

Dataset: GPM_MERGIR V1 (https://doi.org/10.5067/P4HZB9N27EKU), half-hourly,
4km, 60N-60S global IR brightness temperature -- exactly the product the
11 sample merg_*_4km-pixel.nc4 files you already have came from.

Strategy
--------
Rather than downloading years of global half-hourly data (huge), this
script only pulls the specific hours needed:
  - every timestamp of every labeled BoB (or all-basin) RI fix
  - the -6h, -12h, -24h lag timestamps for each fix (for multi-frame /
    persistence CNN inputs, matching the tabular lag features already
    used in the XGBoost model)
This keeps the download to a few thousand hourly granules instead of
~370,000 (40+ years x 48 half-hours/day).
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_ri_dataset import build  # noqa: E402


def required_hours(basin_filter: str, lags=(0, 6, 12, 24)) -> pd.DatetimeIndex:
    ri = build(basin_filter=basin_filter)
    valid = ri[ri["RI_24h"].notna()].copy()
    valid["datetime_utc"] = pd.to_datetime(valid["datetime_utc"])

    all_times = set()
    for lag in lags:
        shifted = (valid["datetime_utc"] - pd.Timedelta(hours=lag)).dt.floor("H")
        all_times.update(shifted.tolist())

    return pd.DatetimeIndex(sorted(all_times))


def download(basin_filter: str, out_dir: str, lags=(0, 6, 12, 24)):
    import earthaccess  # imported here so the rest of the repo doesn't need it

    os.makedirs(out_dir, exist_ok=True)
    hours = required_hours(basin_filter, lags)
    print(f"Need MERG-IR granules for {len(hours)} distinct hours "
          f"({hours.min()} to {hours.max()})")

    earthaccess.login()  # uses .netrc or EARTHDATA_USERNAME/PASSWORD env vars

    downloaded, skipped = 0, 0
    for ts in hours:
        fname_guess = f"merg_{ts.strftime('%Y%m%d%H')}_4km-pixel.nc4"
        target_path = os.path.join(out_dir, fname_guess)
        if os.path.exists(target_path):
            skipped += 1
            continue

        results = earthaccess.search_data(
            short_name="GPM_MERGIR",
            version="1",
            temporal=(ts.strftime("%Y-%m-%dT%H:%M:%S"),
                      (ts + pd.Timedelta(minutes=59)).strftime("%Y-%m-%dT%H:%M:%S")),
        )
        if not results:
            print(f"  [warn] no granule found for {ts}")
            continue

        earthaccess.download(results, out_dir)
        downloaded += 1
        if downloaded % 25 == 0:
            print(f"  ... {downloaded} granules downloaded so far")

    print(f"Done. Downloaded {downloaded} new granules, skipped {skipped} already present.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", default="BOB", help="BOB, ARB, or None for all basins")
    parser.add_argument("--out_dir", default="./mergir_archive")
    parser.add_argument("--lags", default="0,6,12,24",
                         help="comma-separated lag hours to fetch per RI fix")
    args = parser.parse_args()

    basin = None if args.basin.lower() == "none" else args.basin
    lags = tuple(int(x) for x in args.lags.split(","))
    download(basin, args.out_dir, lags)
