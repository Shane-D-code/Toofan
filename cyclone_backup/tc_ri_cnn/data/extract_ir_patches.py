"""
extract_ir_patches.py
======================
Extracts fixed-size satellite-IR (MERG-IR brightness temperature) patches
centered on IMD best-track storm-center fixes, for use as CNN input.

MERG-IR (NCEP/CPC 4km Global IR, half-hourly, 60N-60S) file naming:
    merg_YYYYMMDDHH_4km-pixel.nc4   (contains 2 half-hourly time steps: HH:00, HH:30)

Patch design
------------
- PATCH_KM   : physical half-width of the patch (default 400 km side, i.e.
               +/-200 km from the storm center -> ~100x100 px at 4 km).
- Pixel spacing is ~0.0364 deg (~4.04 km) on this grid.
- Missing/out-of-range pixels are filled with 280 K (a warm/clear-sky filler,
  chosen so it does not look like a cold convective core to the CNN) and a
  binary "valid mask" channel is returned alongside so the network can learn
  to ignore filled regions.
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import xarray as xr

PATCH_KM = 400          # patch side length in km
KM_PER_DEG = 111.0
PIXEL_DEG = 0.036384    # approx native grid spacing (deg)
PATCH_PX = int(round((PATCH_KM / KM_PER_DEG) / PIXEL_DEG))  # ~pixels per side
FILL_TB = 280.0


def index_mergir_files(folder: str) -> pd.DataFrame:
    """Scan a folder of merg_YYYYMMDDHH_4km-pixel.nc4 files and build a
    lookup table of (file_path, half-hour timestamp) pairs."""
    rows = []
    for fp in glob.glob(os.path.join(folder, "merg_*_4km-pixel.nc4")):
        m = re.search(r"merg_(\d{10})_4km-pixel\.nc4", os.path.basename(fp))
        if not m:
            continue
        ds = xr.open_dataset(fp)
        for t in ds.time.values:
            rows.append({"file": fp, "time": pd.Timestamp(t).round("min")})
        ds.close()
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def extract_patch(ds: xr.Dataset, time_val, lat0: float, lon0: float,
                   patch_px: int = PATCH_PX) -> np.ndarray:
    """Extract a (patch_px, patch_px) Tb array centered as close as possible
    to (lat0, lon0) at the given time. Returns array with NaNs for
    out-of-domain pixels (caller decides fill policy)."""
    tb = ds["Tb"].sel(time=time_val, method="nearest")

    lat_idx = int(np.argmin(np.abs(ds.lat.values - lat0)))
    lon_idx = int(np.argmin(np.abs(ds.lon.values - lon0)))
    half = patch_px // 2

    lat_lo, lat_hi = lat_idx - half, lat_idx + half
    lon_lo, lon_hi = lon_idx - half, lon_idx + half

    n_lat, n_lon = tb.shape
    out = np.full((patch_px, patch_px), np.nan, dtype=np.float32)

    src_lat_lo, src_lat_hi = max(lat_lo, 0), min(lat_hi, n_lat)
    src_lon_lo, src_lon_hi = max(lon_lo, 0), min(lon_hi, n_lon)

    dst_lat_lo = src_lat_lo - lat_lo
    dst_lon_lo = src_lon_lo - lon_lo

    block = tb.values[src_lat_lo:src_lat_hi, src_lon_lo:src_lon_hi]
    out[dst_lat_lo:dst_lat_lo + block.shape[0],
        dst_lon_lo:dst_lon_lo + block.shape[1]] = block

    return out


def build_labeled_patch_dataset(ri_df: pd.DataFrame, nc_folder: str,
                                 time_tolerance_min: int = 90) -> dict:
    """For every row of ri_df with a valid RI_24h label, find the nearest
    MERG-IR half-hour snapshot within `time_tolerance_min` minutes and
    extract a centered Tb patch.

    Returns dict with keys: X (N,H,W) float32 Tb, mask (N,H,W) valid-pixel
    mask, y (N,) RI_24h labels, meta (DataFrame) matched metadata.
    """
    file_index = index_mergir_files(nc_folder)
    if file_index.empty:
        raise FileNotFoundError(f"No merg_*_4km-pixel.nc4 files found in {nc_folder}")

    open_cache = {}
    X, M, Y, meta_rows = [], [], [], []

    valid = ri_df[ri_df["RI_24h"].notna()].copy()
    valid["datetime_utc"] = pd.to_datetime(valid["datetime_utc"])

    for _, row in valid.iterrows():
        deltas = (file_index["time"] - row["datetime_utc"]).abs()
        j = deltas.idxmin()
        if deltas.loc[j] > pd.Timedelta(minutes=time_tolerance_min):
            continue

        fp = file_index.loc[j, "file"]
        t_val = file_index.loc[j, "time"]

        if fp not in open_cache:
            open_cache[fp] = xr.open_dataset(fp)
        ds = open_cache[fp]

        patch = extract_patch(ds, np.datetime64(t_val), row["latitude"], row["longitude"])
        mask = (~np.isnan(patch)).astype(np.float32)
        patch_filled = np.where(np.isnan(patch), FILL_TB, patch).astype(np.float32)

        X.append(patch_filled)
        M.append(mask)
        Y.append(row["RI_24h"])
        meta_rows.append({
            "storm_id": row["storm_id"],
            "datetime_utc": row["datetime_utc"],
            "matched_file_time": t_val,
            "time_offset_min": deltas.loc[j].total_seconds() / 60.0,
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "max_wind_kt": row["max_wind_kt"],
            "RI_24h": row["RI_24h"],
        })

    for ds in open_cache.values():
        ds.close()

    return {
        "X": np.stack(X) if X else np.empty((0, PATCH_PX, PATCH_PX), np.float32),
        "mask": np.stack(M) if M else np.empty((0, PATCH_PX, PATCH_PX), np.float32),
        "y": np.array(Y, dtype=np.float32),
        "meta": pd.DataFrame(meta_rows),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from build_ri_dataset import build

    ri_bob = build(basin_filter="BOB")
    result = build_labeled_patch_dataset(ri_bob, "/mnt/user-data/uploads")

    print("Matched patches:", result["X"].shape)
    print("Label distribution:", np.unique(result["y"], return_counts=True))
    print(result["meta"])

    os.makedirs("/home/claude/tc_ri_cnn/outputs", exist_ok=True)
    np.savez_compressed(
        "/home/claude/tc_ri_cnn/outputs/ir_patch_dataset_demo.npz",
        X=result["X"], mask=result["mask"], y=result["y"],
    )
    result["meta"].to_csv("/home/claude/tc_ri_cnn/outputs/ir_patch_dataset_demo_meta.csv", index=False)
    print("Saved demo patch dataset to outputs/")
