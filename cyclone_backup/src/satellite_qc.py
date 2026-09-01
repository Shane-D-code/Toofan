"""Satellite quality control (Phase 2 of the SIH master plan).

For every recovered image we run a battery of automatic checks and write a
``satellite_qc_report.csv``. Checks:

- correct shape (128, 128, 1)
- no all-NaN image
- no constant image
- reasonable brightness-temperature range (before normalisation we record the
  physical Tb extremes so we can sanity-check the global 180-310 K window)
- sufficient valid pixels (low NaN fraction)
- the crop actually contains the storm region (we verify the storm centre is
  inside the crop geographic box in ``satellite_recovery``; here we flag any
  image with an anomalous centre ring of NaNs using a centre-window test)

We also detect duplicate/renamed images (identical bytes on disk) so the same
observation is not counted twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT


def qc_one_image(path: Path, tb_clip_min: float = 180.0,
                 tb_clip_max: float = 310.0) -> dict:
    """Run QC checks on a single recovered crop.

    The crop is stored in GLOBAL-normalised units ``norm=(310-Tb)/(310-180)``.
    We reverse it to get back physical brightness temperature (K) for range
    checks. NaN bucket is 0.5 in normalised space and is not reversed.
    """
    try:
        arr = np.load(str(path))
    except Exception as exc:
        return {"image_file": path.name, "status": "FAIL",
                "reason": f"load error: {exc}", "shape": None,
                "nan_fraction": None, "valid_fraction": None,
                "constant": None, "tb_min_k": None, "tb_max_k": None,
                "center_valid": None}

    d = {"image_file": path.name, "status": "PASS", "reason": ""}

    # Shape check
    if arr.shape != (128, 128, 1):
        d["status"] = "FAIL"
        d["reason"] += "bad shape; "
        d["shape"] = arr.shape
    else:
        d["shape"] = arr.shape

    img = arr[..., 0].astype(np.float32)

    # NaN handling (normalised space)
    nan_frac = float(np.mean(np.isnan(img))) if img.size else 1.0
    valid_frac = float(np.mean(~np.isnan(img))) if img.size else 0.0
    d["nan_fraction"] = nan_frac
    d["valid_fraction"] = valid_frac

    # All-NaN
    if valid_frac == 0.0:
        d["status"] = "FAIL"
        d["reason"] += "all NaN; "

    # Reverse normalisation to get physical Tb from non-NaN pixels.
    # Stored image is `norm = (310 - Tb) / (310 - 180)`, so:
    #   Tb = clip_max - norm * (clip_max - clip_min)
    physical = tb_clip_max - img * (tb_clip_max - tb_clip_min)
    physical = np.where(np.isnan(img), np.nan, physical)
    if valid_frac > 0:
        d["tb_min_k"] = round(float(np.nanmin(physical)), 1)
        d["tb_max_k"] = round(float(np.nanmax(physical)), 1)
    else:
        d["tb_min_k"] = np.nan
        d["tb_max_k"] = np.nan

    # Constant image (no variance among valid pixels).
    const = False
    if valid_frac > 0:
        valid_vals = physical[~np.isnan(physical)]
        const = bool(np.nanstd(valid_vals) < 1e-4)
    d["constant"] = const
    if const:
        d["status"] = "FAIL"
        d["reason"] += "constant image; "

    # Reasonable brightness temperature range (deep convection ~180-220K,
    # warm cloud/land up to ~310K).
    if valid_frac > 0 and (d["tb_min_k"] < 150 or d["tb_max_k"] > 330):
        d["status"] = "FAIL"
        d["reason"] += "unreasonable Tb range; "

    # Centre-window validity: the storm centre must be within the inner 30%
    # of the image. Flag if a large central patch is NaN (crop missed storm).
    h, w = img.shape
    ch0, ch1 = int(h * 0.35), int(h * 0.65)
    cw0, cw1 = int(w * 0.35), int(w * 0.65)
    centre = img[ch0:ch1, cw0:cw1]
    centre_valid = float(np.mean(~np.isnan(centre))) if centre.size else 0.0
    d["center_valid"] = centre_valid
    if centre_valid < 0.8:
        d["status"] = "FAIL"
        d["reason"] += "center window mostly NaN (crop may have missed storm); "

    d["reason"] = d["reason"].strip().rstrip(";")
    if not d["reason"]:
        d["reason"] = "ok"
    return d


def run_qc(metadata: pd.DataFrame, out_path, tb_clip_min=180.0,
           tb_clip_max=310.0) -> pd.DataFrame:
    """Run QC on every row of recovered metadata and write the report."""
    rows = []
    for _, row in metadata.iterrows():
        p = row.get("image_path")
        if p is None or not Path(str(p)).exists():
            rows.append({"image_file": row.get("image_file"),
                         "storm_id": row.get("storm_id"),
                         "status": "FAIL", "reason": "missing image file"})
            continue
        q = qc_one_image(Path(str(p)), tb_clip_min, tb_clip_max)
        q["storm_id"] = row.get("storm_id")
        q["datetime_utc"] = row.get("datetime_utc")
        q["RI_24h"] = row.get("RI_24h")
        rows.append(q)

    df = pd.DataFrame(rows)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[satqc] QC report -> {out_path} "
              f"({(df['status']=='PASS').sum()}/{len(df)} pass)")
    return df


def detect_duplicate_images(metadata: pd.DataFrame) -> pd.DataFrame:
    """Detect byte-identical images (dup/renamed copies) across the dataset.

    Returns a DataFrame with duplicate-group info. Two images are duplicates
    if their raw file bytes match.
    """
    import hashlib

    rows = []
    hashes = {}
    for _, row in metadata.iterrows():
        p = row.get("image_path")
        if p is None or not Path(str(p)).exists():
            continue
        h = hashlib.md5(Path(str(p)).read_bytes()).hexdigest()
        hashes.setdefault(h, []).append({
            "image_file": row.get("image_file"),
            "storm_id": row.get("storm_id"),
            "datetime_utc": row.get("datetime_utc"),
        })
    for h, members in hashes.items():
        if len(members) > 1:
            rows.append({
                "md5": h,
                "n_duplicates": len(members),
                "members": "; ".join(m["image_file"] for m in members),
            })
    df = pd.DataFrame(rows)
    print(f"[satqc] {len(df)} duplicate image group(s) found.")
    return df
