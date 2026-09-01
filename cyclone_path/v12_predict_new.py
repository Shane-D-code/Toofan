"""
v12_predict_new.py
------------------
Run the best V12 model (checkpoints/v12_best_model.pt) on a BRAND-NEW,
unseen cyclone to predict its track at the 12 horizons (2..24 h).

Unlike v12_infer.py (which only exposes storms that already landed in the
train/val/test split of data/processed.csv), this script lets you feed the raw
observed fixes of any new cyclone.

Input: either
  (a) --csv <file>: a CSV with columns
        SID,ISO_TIME,lat,lon[,wind][,mslp][,rmw]
      ISO_TIME may be parsed as datetime for normal timestamps, or as
      float hours-since-epoch (a plain integer like 1710000000 will be treated
      as unix seconds). lat/lon in degrees. wind (kt), mslp (hPa), rmw (nmi)
      are used to build the sst/shear/rmw proxies exactly like training.
      At least 13 fixes are recommended (12 input window + 1).
  (b) --sid <id> --iso_time/hours --lat --lon [--wind --mslp --rmw]
      by repeated call, OR just rely on (a).

The script reuses the exact feature engineering and (training-set-only)
normalization stats stored in the checkpoint, so the features fed to the model
match training exactly.

Usage:
    python v12_predict_new.py --ckpt checkpoints/v12_best_model.pt \
        --new_csv new_cyclone.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from model import CycloneTransformerV11
from v12_common import load_v12
from dataset import CycloneDataset, split_storm_ids

# Reuse the exact production feature-engineering pipeline so the features match
# training (sst/shear/rmw proxies, kinematics, motion windows) bit-for-bit.
try:
    from prepare_dataset import process as _prepare_process  # noqa: F401
    from prepare_dataset import sst_climatology_proxy, willoughby_rahn_rmw, \
        haversine_km, bearing_deg, local_velocity_kmh, find_lookback_idx, \
        wrapped_lon_delta, add_trajectory_motion_features, apply_per_storm, \
        MOTION_WINDOWS_HOURS, MOTION_WINDOW_TOL_HOURS
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Could not import feature engineering from prepare_dataset.py: " + str(e))


def build_new_storm_frame(new_csv):
    """Read a new storm's raw fixes and return a processed-CSV-style DataFrame
    with SID, ISO_TIME, lat, lon, wind, mslp, rmw (reusing prepare_dataset's
    proxy / kinematic logic for a single storm)."""
    df = pd.read_csv(new_csv)
    for col in ("ISO_TIME", "lat", "lon"):
        if col not in df.columns:
            raise ValueError(f"missing required column {col!r} in {new_csv}")

    # Coerce ISO_TIME to datetime (accepts "YYYY-MM-DD HH:MM:SS" strings,
    # already-datetime, or Unix seconds as floats/ints).
    t_raw = df["ISO_TIME"]
    if pd.api.types.is_numeric_dtype(t_raw):
        df["ISO_TIME"] = pd.to_datetime(t_raw, unit="s")
    else:
        df["ISO_TIME"] = pd.to_datetime(t_raw, errors="coerce")

    # Sort by time, coerce coordinates, fill intensity cols to optional defaults.
    df = df.dropna(subset=["ISO_TIME"]).sort_values("ISO_TIME").reset_index(drop=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    if "wind" not in df.columns:
        df["wind"] = np.nan
    if "mslp" not in df.columns:
        df["mslp"] = np.nan
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["mslp"] = pd.to_numeric(df["mslp"], errors="coerce")

    # rmw: ffill like training, else Willoughby-Rahn estimate.
    if "rmw" not in df.columns:
        df["rmw"] = np.nan
    df["rmw"] = pd.to_numeric(df["rmw"], errors="coerce").ffill()
    est = willoughby_rahn_rmw(df["wind"].fillna(df["wind"].median()).fillna(80.0))
    df["rmw"] = df["rmw"].fillna(est)

    # mslp: ffill where missing (matches training fill_storm).
    df["mslp"] = df["mslp"].ffill()
    df["mslp"] = df["mslp"].fillna(1010.0)

    # wind: so the proxies / speed don't blow up; default to 60 kt if absent.
    df["wind"] = df["wind"].ffill().bfill().fillna(60.0)

    # Build proxies identically to prepare_dataset.process().
    df["month"] = pd.to_datetime(df["ISO_TIME"]).dt.month
    df["sst"] = sst_climatology_proxy(df["lat"].values, df["month"].values)
    poci_default = 1010.0
    pressure_deficit = poci_default - df["mslp"]
    df["shear"] = np.clip(20.0 - 0.10 * pressure_deficit, 0.0, 25.0)

    # Kinematics from consecutive fixes (single storm).
    lat = df["lat"].values
    lon = df["lon"].values
    t = df["ISO_TIME"].values.astype("datetime64[s]").astype(np.int64)
    dt_h = np.diff(t).astype(float) / 3600.0
    dt_h = np.concatenate([[np.nan], dt_h])
    dist = np.concatenate([[np.nan], haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])])
    brng = np.concatenate([[np.nan], bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])])
    df["dt_hours"] = dt_h
    df["speed_kmh"] = dist / dt_h
    df["bearing_deg"] = brng
    df["speed_kmh"] = df["speed_kmh"].fillna(0.0)
    df["bearing_deg"] = df["bearing_deg"].fillna(0.0)
    df["dt_hours"] = df["dt_hours"].fillna(3.0)
    df["bearing_sin"] = np.sin(np.radians(df["bearing_deg"]))
    df["bearing_cos"] = np.cos(np.radians(df["bearing_deg"]))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Trajectory motion features (u/v, speed/u/v 3/6/12h, accel, turns).
    df = add_trajectory_motion_features(df)
    # The checkpoint does NOT use the V6 trend columns, so no need for them.

    # Assign a single SID (stable) and select the columns the model expects.
    if "SID" not in df.columns:
        df["SID"] = "NEW"
    df = df[["SID", "ISO_TIME", "lat", "lon", "wind", "mslp", "rmw", "sst", "shear",
             "speed_kmh", "bearing_sin", "bearing_cos", "month_sin", "month_cos",
             "dt_hours",
             "u_kmh", "v_kmh",
             "speed_3h_kmh", "u_3h_kmh", "v_3h_kmh",
             "speed_6h_kmh", "u_6h_kmh", "v_6h_kmh",
             "speed_12h_kmh", "u_12h_kmh", "v_12h_kmh",
             "accel_kmh2", "turn_sin", "turn_cos"]]
    return df.reset_index(drop=True)


def infer(model, ck, device, frame):
    cfg = ck["cfg"]
    horizons = list(cfg["horizons"])
    cols = list(ck["feature_cols"])
    norm_cols = list(ck["norm_cols"])
    stats = ck["stats"]

    # Normalize with training-set-only stats (identical to CycloneDataset).
    fn = frame.copy()
    for c in norm_cols:
        m, s = stats[c]
        fn[c] = (fn[c].astype(float) - m) / s

    # Build the last (most recent) 12-fix window ending at the latest observation.
    g = fn.sort_values("ISO_TIME").reset_index(drop=True)
    n = len(g)
    input_len = cfg["input_len"]
    if n < input_len:
        raise ValueError(
            f"need at least {input_len} fixes; got {n}. Provide {input_len} "
            "observed fixes so the 12-step input window is complete.")

    feats = g[cols].values.astype(np.float32)          # (n, 27)
    raw_latlon = g[["lat", "lon"]].values.astype(np.float64)
    cur = raw_latlon[-1]                                # latest fix = anchor
    x = torch.tensor(feats[-input_len:], dtype=torch.float32).unsqueeze(0).to(device)  # (1,12,27)
    cur_t = torch.tensor(cur, dtype=x.dtype).unsqueeze(0).to(device)  # (1,2)

    model.eval()
    with torch.no_grad():
        out = model(x, cur_pos=cur_t)
    scale = out["scale"].squeeze(0).cpu().numpy()       # (H,1)
    raw_delta = out["delta"].squeeze(0).cpu().numpy()
    cal_delta = out["delta_calibrated"].squeeze(0).cpu().numpy()
    log_var = out["log_var"].squeeze(0).cpu().numpy()   # (H,2)

    anchor_time = g["ISO_TIME"].iloc[-1]
    cur_lat, cur_lon = float(cur[0]), float(cur[1])
    cos_lat = max(float(np.cos(np.deg2rad(cur_lat))), 0.1)

    rows = []
    rows.append(["Current", anchor_time, cur_lat, cur_lon,
                 cur_lat, cur_lon, 1.0, np.nan])
    for i, h in enumerate(horizons):
        r_lat = cur_lat + raw_delta[i, 1] / 111.32
        r_lon = cur_lon + raw_delta[i, 0] / (111.32 * cos_lat)
        c_lat = cur_lat + cal_delta[i, 1] / 111.32
        c_lon = cur_lon + cal_delta[i, 0] / (111.32 * cos_lat)
        fcst_time = anchor_time + pd.Timedelta(hours=h)
        sigma = float(np.sqrt(np.exp(np.clip(log_var[i, :], -10, 5))).mean())
        rows.append([f"+{h:.0f}h", fcst_time, c_lat, c_lon,
                     r_lat, r_lon, float(scale[i, 0]), sigma])

    return pd.DataFrame(rows, columns=[
        "horizon", "forecast_time", "cal_lat", "cal_lon",
        "raw_lat", "raw_lon", "learned_scale", "sigma_km"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v12_best_model.pt")
    ap.add_argument("--new_csv", required=True,
                    help="CSV of a new storm: SID,ISO_TIME,lat,lon[,wind][,mslp][,rmw]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    model, ck = load_v12(args.ckpt, device)

    frame = build_new_storm_frame(args.new_csv)
    print(f"{len(frame)} fixes for new storm "
          f"{frame['SID'].iloc[0]} (recent={frame['ISO_TIME'].iloc[-1]})")
    df = infer(model, ck, device, frame)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote -> {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
