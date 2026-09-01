"""Recover the satellite IR branch from raw NCEP/CPC 4km IR .nc4 granules.

The original MVP only had 3 usable ``.npy`` crops on disk (see
PROJECT_AUDIT.md). Freshly downloaded MERG IR granules for the Bay of Bengal
were placed in ``Cnnfiles/``. This module turns those raw ``merg_*_4km-pixel.nc4``
granules into a reproducible, storm-safe satellite dataset:

1. Audit every granule (variable, shape, physical stats, NaN fraction).
2. Match each granule to the nearest IMD observation (storm, timestamp,
   location) within a documented time tolerance.
3. Extract a storm-centred IR brightness-temperature crop at native 4 km
   resolution and resize it to the configured CNN input size.
4. Apply a **global, physically meaningful normalisation** (brightness
   temperature mapped to a fixed 180-310 K window, inverted so that cold
   deep-convective clouds -> high activation). There is NO per-image
   re-scaling, so absolute cloud-top temperature information is preserved
   across all examples.
5. Run quality control and write the normalisation statistics, an extraction
   log and the crop images/metadata.

No data is fabricated. No label is altered. Granules that cannot be matched to
a real IMD observation (or fall outside the time tolerance) are dropped and
logged, not invented.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, get_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamp_from_filename(fname: str):
    """Parse 'YYYYMMDDHH' from a MERG granule filename -> pd.Timestamp (UTC)."""
    m = re.search(r"merg_(\d{4})(\d{2})(\d{2})(\d{2})", fname)
    if not m:
        return None
    y, mo, d, h = m.groups()
    return pd.Timestamp(f"{y}-{mo}-{d} {h}:00:00")


def discover_granules(nc4_dir) -> list[Path]:
    """Return sorted, de-duplicated list of MERG .nc4 granule paths."""
    seen = {}
    for p in sorted(Path(nc4_dir).glob("merg_*_4km-pixel.nc4")):
        # Ignore the ' (1)' duplicate-suffix pattern if the OS does expose both.
        if "(1)" in p.name or " copy" in p.name:
            continue
        ts = _timestamp_from_filename(p.name)
        if ts is not None:
            # Keep only the first file for a given granule timestamp.
            seen.setdefault(ts, p)
    return [seen[k] for k in sorted(seen)]


# ---------------------------------------------------------------------------
# Granule audit
# ---------------------------------------------------------------------------

def inspect_granule(path) -> dict:
    """Return physical statistics for a single granule."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        tb = ds["Tb"]
        arr = np.asarray(tb.isel(time=0).values, dtype=np.float32)
        stats = {
            "file": os.path.basename(path),
            "begin_datetime_utc": str(pd.Timestamp(str(ds["time"].values[0]))),
            "time_steps": int(ds.sizes["time"]),
            "variable": str(tb.name) if hasattr(tb, "name") else "Tb",
            "shape": str(arr.shape),
            "n_lat": int(arr.shape[0]),
            "n_lon": int(arr.shape[1]),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "nan_fraction": float(np.mean(np.isnan(arr))),
            "units": str(tb.attrs.get("units", "K")),
            "lat_min": float(ds["lat"].min()),
            "lat_max": float(ds["lat"].max()),
            "lon_min": float(ds["lon"].min()),
            "lon_max": float(ds["lon"].max()),
            "granule_datetime_utc": str(pd.Timestamp(str(ds["time"].values[0]))),
        }
    return stats


def write_nc4_audit(nc4_dir, out_path) -> pd.DataFrame:
    """Audit every granule and write ``satellite_nc4_audit.csv``."""
    granules = discover_granules(nc4_dir)
    rows = [inspect_granule(p) for p in granules]
    df = pd.DataFrame(rows).sort_values("granule_datetime_utc")
    df.to_csv(out_path, index=False)
    print(f"[recover] NC4 audit -> {out_path} ({len(df)} granules)")
    return df


# ---------------------------------------------------------------------------
# Matching to IMD observations
# ---------------------------------------------------------------------------

def match_granules_to_imd(granules: list[Path], imd: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Match each granule to the nearest IMD observation (storm-safe).

    Matching is by (storm, timestamp, location): for each granule we look up
    the IMD observation closest in time (within ``max_time_diff_min``), which
    uniquely identifies the storm and its centre. Records the time offset.

    Returns a DataFrame keyed by ``(storm_id, datetime_utc)`` of the IMD
    observation (unique per observation).
    """
    tol = pd.Timedelta(minutes=int(cfg["satellite"]["max_time_diff_min"]))
    keep_dup = bool(cfg["satellite"].get("keep_duplicates", False))

    imd = imd.copy()
    imd["datetime_utc"] = pd.to_datetime(imd["datetime_utc"])
    records = []
    for p in granules:
        gtime = pd.Timestamp(_timestamp_from_filename(p.name))
        diff = (imd["datetime_utc"] - gtime).abs()
        mask = diff <= tol
        if mask.sum() == 0:
            records.append({
                "file": os.path.basename(p.name),
                "granule_datetime_utc": gtime,
                "storm_id": None,
                "datetime_utc": None,
                "matched": False,
                "time_diff_minutes": None,
                "reason": "no IMD observation within time tolerance",
            })
            continue
        idx = diff[mask].idxmin()
        r = imd.loc[idx]
        dmin = float((r["datetime_utc"] - gtime).total_seconds() / 60.0)
        records.append({
            "file": os.path.basename(p.name),
            "granule_datetime_utc": gtime,
            "storm_id": str(r["storm_id"]),
            "datetime_utc": r["datetime_utc"],
            "matched": True,
            "time_diff_minutes": abs(round(dmin, 1)),
            "imd_lat": float(r["latitude"]),
            "imd_lon": float(r["longitude"]),
            "RI_24h": int(r["RI_24h"]),
            "reason": "ok",
        })

    mdf = pd.DataFrame(records)

    # Drop the duplicate target observations (keep the nearest granule per
    # IMD observation) unless the user explicitly wants every granule kept.
    used = mdf[mdf["matched"]].copy()
    if not keep_dup:
        # Keep the granule closest in time to each IMD observation.
        used = used.sort_values("time_diff_minutes")
        used = used.drop_duplicates(subset=["storm_id", "datetime_utc"], keep="first")
    return mdf[mdf["matched"] == False].reset_index(drop=True), used.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Crop + normalisation
# ---------------------------------------------------------------------------

def _global_normalization(arr: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Map brightness temperature (K) to [0, 1] using a GLOBAL window.

    ``norm = (clip_max - Tb) / (clip_max - clip_min)`` so that colder clouds
    (deep convection) map to HIGH values. Using fixed physical bounds (not
    per-image min/max) preserves absolute intensity information across all
    examples. Values outside the window are clipped; NaN is mapped to 0.5
    (neutral) and NaN fraction is tracked separately.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (clip_max - arr) / (clip_max - clip_min)
    out = np.clip(out, 0.0, 1.0)
    out = np.where(np.isnan(out), 0.5, out)
    return out.astype(np.float32)


def extract_crop(tb2d: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                 center_lat: float, center_lon: float,
                 half_deg: float, target_size: int, clip_min: float, clip_max: float):
    """Extract a storm-centred crop, global-normalise, and resize.

    Returns ``(crop, nan_fraction)`` where ``crop`` has shape
    ``(target_size, target_size, 1)`` in global-normalised units.
    """
    import cv2

    lat_min, lat_max = center_lat - half_deg, center_lat + half_deg
    lon_min, lon_max = center_lon - half_deg, center_lon + half_deg
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon >= lon_min) & (lon <= lon_max)
    crop = tb2d[np.ix_(lat_sel, lon_sel)]

    nan_frac = float(np.mean(np.isnan(crop))) if crop.size else 1.0

    norm = _global_normalization(crop, clip_min, clip_max)
    if crop.shape != (target_size, target_size):
        norm = cv2.resize(norm, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    crop_3 = norm.reshape((target_size, target_size, 1))
    return crop_3, nan_frac


def build_recovered_dataset(nc4_dir, imd, cfg) -> dict:
    """Run the full recovery and write the recovered dataset.

    Returns a dict with the recovered metadata, extraction log and paths.
    """
    from .config import REPO_ROOT
    cfg_sat = cfg["satellite"]
    half_deg = float(cfg_sat["crop_half_deg"])
    target = int(cfg_sat["img_size"])
    clip_min = float(cfg_sat["tb_clip_min"])
    clip_max = float(cfg_sat["tb_clip_max"])

    out_dir = REPO_ROOT / cfg_sat["recovered_dir"]
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    import xarray as xr

    granules = discover_granules(nc4_dir)
    dropped, matched = match_granules_to_imd(granules, imd, cfg)

    log_rows = []
    meta_rows = []
    for _, r in matched.iterrows():
        path = REPO_ROOT / nc4_dir / r["file"]
        gtime = pd.Timestamp(r["granule_datetime_utc"])
        # Pick the half-hour scan closest to the nominal granule hour.
        with xr.open_dataset(path) as ds:
            times = pd.to_datetime(ds["time"].values)
            t_idx = int(np.argmin(np.abs(times - gtime)))
            arr = np.asarray(ds["Tb"].isel(time=t_idx).values, dtype=np.float32)
            lat = np.asarray(ds["lat"].values, dtype=np.float64)
            lon = np.asarray(ds["lon"].values, dtype=np.float64)

        crop, nan_frac = extract_crop(
            arr, lat, lon, float(r["imd_lat"]), float(r["imd_lon"]),
            half_deg, target, clip_min, clip_max,
        )
        storm = str(r["storm_id"])
        fname = f"{storm}_{pd.Timestamp(r['datetime_utc']).strftime('%Y%m%d_%H%M')}.npy"
        np.save(img_dir / fname, crop)

        meta_rows.append({
            "storm_id": storm,
            "datetime_utc": pd.Timestamp(r["datetime_utc"]),
            "satellite_datetime": gtime,
            "delta_minutes": r["time_diff_minutes"],
            "latitude": r["imd_lat"],
            "longitude": r["imd_lon"],
            "RI_24h": r["RI_24h"],
            "image_file": fname,
            "image_path": str(img_dir / fname),
            "nan_fraction": nan_frac,
            "granule_file": r["file"],
        })
        log_rows.append({"granule_file": r["file"], "image_file": fname,
                         "status": "recovered"})

    for _, r in dropped.iterrows():
        log_rows.append({"granule_file": r["file"], "image_file": None,
                         "status": "dropped", "reason": r.get("reason", "")})

    meta = pd.DataFrame(meta_rows)
    log = pd.DataFrame(log_rows)

    meta.to_csv(out_dir / "metadata.csv", index=False)
    meta.to_csv(out_dir / "metadata_clean.csv", index=False)
    log.to_csv(out_dir / "extraction_log.csv", index=False)

    # Normalisation statistics
    norm_stats = {
        "method": "global_physical_fixed_window",
        "clip_min_kelvin": clip_min,
        "clip_max_kelvin": clip_max,
        "description": "norm = (clip_max - Tb) / (clip_max - clip_min); cold=1",
        "nan_fill": 0.5,
        "n_images": int(len(meta)),
        "n_storms": int(meta["storm_id"].nunique()) if len(meta) else 0,
        "n_ri": int((meta["RI_24h"] == 1).sum()) if len(meta) else 0,
        "n_non_ri": int((meta["RI_24h"] == 0).sum()) if len(meta) else 0,
        "mean_nan_fraction": float(meta["nan_fraction"].mean()) if len(meta) else 0.0,
    }
    with open(out_dir / "normalization.json", "w", encoding="utf-8") as fh:
        json.dump(norm_stats, fh, indent=2)

    print(f"[recover] Recovered {len(meta)} images across {norm_stats['n_storms']} "
          f"storms ({norm_stats['n_ri']} RI / {norm_stats['n_non_ri']} non-RI).")
    print(f"[recover] Wrote dataset -> {out_dir}")
    return {"metadata": meta, "log": log, "norm": norm_stats, "dir": str(out_dir)}


def plot_sample_grid(meta: pd.DataFrame, out_path, cols: int = 4) -> str:
    """Render a grid of recovered crops to visually verify the cyclone is centred."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(meta)
    if n == 0:
        return None
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in range(len(axes)):
        if ax < n:
            row = meta.iloc[ax]
            img = np.load(str(row["image_path"]))[:, :, 0]
            axes[ax].imshow(img, cmap="gray_r")
            axes[ax].set_title(f"{row['storm_id']}\nRI={int(row['RI_24h'])}", fontsize=8)
        axes[ax].axis("off")
    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[recover] Sample grid -> {out_path}")
    return out_path
