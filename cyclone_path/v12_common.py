"""
v12_common.py
-------------
Shared helpers for V12 evaluation, inference and plotting: model loading (V12,
V11 and V5), full-test-set row collection (raw vs calibrated), per-horizon
summary, error statistics, and collapse/identifiability diagnostics.

V12 uses the CycloneTransformerV11 model (per-horizon bounded scale in
[0.80, 1.20] zero-initialised to exactly 1.0) -- no new model class is
introduced. The accuracy levers live in the DATA (2h-horizon grid with slerp
interpolated targets on 3-hourly fixes), the LOSS (monotonic displacement /
uncertainty hinges + adjacent-horizon trajectory consistency), and a
formula-driven per-horizon weighting.

Like V11, the model emits both RAW and CALIBRATED displacement:
    dx_cal = scale * dx_raw,  dy_cal = scale * dy_raw   (magnitude-only rescale)
Bearing is NOT changed by the scale, so raw and calibrated bearings are
identical by construction.

Everything here is model / plotting-agnostic and used by v12_eval.py,
v12_infer.py and v12_plot.py.
"""
import numpy as np
import pandas as pd
import torch

from dataset import CycloneDataset, split_storm_ids, reconstruct_positions
from losses import haversine_km
from model import CycloneTransformerV11, CycloneTransformerV5


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360.0) % 360.0


def _hav_km_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def _target_resolution(cfg):
    """Target-resolution stored in the checkpoint, defaulting to 'interpolate'."""
    return cfg.get("target_resolution", "interpolate")


def load_v12(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = CycloneTransformerV11(
        feature_dim=len(ck["feature_cols"]),
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg.get("dim_feedforward", 256),
        dropout=cfg.get("dropout", 0.1), horizons=list(cfg["horizons"]),
        target_mode="km",
        max_magnitude_correction=cfg.get("max_magnitude_correction", 0.20),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def load_v11(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = CycloneTransformerV11(
        feature_dim=len(ck["feature_cols"]),
        d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg.get("dim_feedforward", 256),
        dropout=cfg.get("dropout", 0.1), horizons=list(cfg["horizons"]),
        target_mode="km",
        max_magnitude_correction=cfg.get("max_magnitude_correction", 0.20),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def load_v5(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = CycloneTransformerV5(
        feature_dim=len(ck["feature_cols"]), d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_feedforward=cfg.get("dim_feedforward", 256),
        dropout=cfg.get("dropout", 0.1), horizons=list(cfg["horizons"]),
        target_mode=cfg["target_mode"], decoder=cfg.get("decoder", "direct"),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def build_test_ds(csv_path, ck, device):
    cfg = ck["cfg"]
    train_ids, val_ids, test_ids = split_storm_ids(csv_path, seed=cfg.get("seed", 42))
    ds = CycloneDataset(csv_path, test_ids, cfg["input_len"], list(cfg["horizons"]),
                        cfg["target_tolerance"], stats=ck["stats"],
                        feature_cols=ck["feature_cols"], norm_cols=ck["norm_cols"],
                        target_mode="km", target_resolution=_target_resolution(cfg))
    return ds, test_ids


def collect_rows(model, ds, horizons, device, batch_size=512):
    """Return a DataFrame with per-(sample,horizon) raw vs calibrated columns.

    Columns: storm_id, sample, horizon_h, cur_lat, cur_lon,
             raw_pred_lat/lon, cal_pred_lat/lon, actual_lat/lon,
             raw_distance_km, calibrated_distance_km, actual_distance_km,
             raw_ratio, calibrated_ratio, scale, error_km (calibrated),
             raw_error_km, abs_bearing_error.

    For a plain V5 model (no scale), scale=1 and calibrated==raw.
    """
    rows = []
    h = [float(x) for x in horizons]
    n = len(ds)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idxs = list(range(start, end))
            xs = [ds[i][0] for i in idxs]
            ys = [ds[i][1] for i in idxs]
            pos_list = [ds[i][2] for i in idxs]
            xb = torch.stack(xs).to(device)
            yb = torch.stack(ys).to(device)
            posb = torch.stack(pos_list).to(device)
            out = model(xb, cur_pos=posb)
            has_cal = "delta_calibrated" in out
            delta_cal = out["delta_calibrated"] if has_cal else out["delta"]
            scale = out["scale"][..., 0].cpu().numpy() if "scale" in out \
                else np.ones((xb.shape[0], len(h)), dtype=np.float32)
            raw_pos = reconstruct_positions(out["delta"], posb, h, "km").cpu().numpy()
            cal_pos = reconstruct_positions(delta_cal, posb, h, "km").cpu().numpy()
            cur_np = posb.cpu().numpy()
            clat = cur_np[:, 0:1]
            clon = cur_np[:, 1:2]

            raw_d = _hav_km_np(clat, clon, raw_pos[..., 0], raw_pos[..., 1])      # (B,H)
            cal_d = _hav_km_np(clat, clon, cal_pos[..., 0], cal_pos[..., 1])
            pb = _bearing_np(clat, clon, cal_pos[..., 0], cal_pos[..., 1])        # (B,H)

            for bi, i in enumerate(idxs):
                meta = ds.meta[i]
                act_pos = meta["future_pos"]
                sid = str(meta["sid"])
                a_lat = act_pos[:, 0].astype(np.float64)
                a_lon = act_pos[:, 1].astype(np.float64)
                act_d = _hav_km_np(clat[bi], clon[bi], a_lat, a_lon)              # (H,)
                cal_err = _hav_km_np(cal_pos[bi, :, 0], cal_pos[bi, :, 1], a_lat, a_lon)
                raw_err = _hav_km_np(raw_pos[bi, :, 0], raw_pos[bi, :, 1], a_lat, a_lon)
                ab = _bearing_np(clat[bi], clon[bi], a_lat, a_lon)
                berr = np.abs(((pb[bi] - ab + 180.0) % 360.0) - 180.0)
                act_d_safe = np.where(act_d > 0, act_d, np.nan)
                for hh_i, hh in enumerate(h):
                    rows.append({
                        "storm_id": sid, "sample": i, "horizon_h": float(hh),
                        "cur_lat": float(clat[bi, 0]), "cur_lon": float(clon[bi, 0]),
                        "raw_pred_lat": float(raw_pos[bi, hh_i, 0]),
                        "raw_pred_lon": float(raw_pos[bi, hh_i, 1]),
                        "cal_pred_lat": float(cal_pos[bi, hh_i, 0]),
                        "cal_pred_lon": float(cal_pos[bi, hh_i, 1]),
                        "actual_lat": float(a_lat[hh_i]), "actual_lon": float(a_lon[hh_i]),
                        "raw_distance_km": float(raw_d[bi, hh_i]),
                        "calibrated_distance_km": float(cal_d[bi, hh_i]),
                        "actual_distance_km": float(act_d[hh_i]),
                        "raw_ratio": float(raw_d[bi, hh_i] / act_d_safe[hh_i]),
                        "calibrated_ratio": float(cal_d[bi, hh_i] / act_d_safe[hh_i]),
                        "scale": float(scale[bi, hh_i]),
                        "error_km": float(cal_err[hh_i]),
                        "raw_error_km": float(raw_err[hh_i]),
                        "abs_bearing_error": float(berr[hh_i]),
                    })
    return pd.DataFrame(rows)


def per_horizon_summary(df, horizons):
    """Aggregate per-horizon diagnostics from a collect_rows DataFrame (spec #14)."""
    out = {}
    for h in horizons:
        sub = df[df["horizon_h"] == float(h)]
        raw_m = float(sub["raw_distance_km"].mean())
        cal_m = float(sub["calibrated_distance_km"].mean())
        act_m = float(sub["actual_distance_km"].mean())
        act_safe = act_m if act_m > 0 else float("nan")
        r_clip = sub["raw_ratio"].clip(upper=5.0)
        c_clip = sub["calibrated_ratio"].clip(upper=5.0)
        out[float(h)] = {
            "raw_dist_mean": raw_m,
            "cal_dist_mean": cal_m,
            "act_dist_mean": act_m,
            "raw_ratio_mean": raw_m / act_safe,
            "cal_ratio_mean": cal_m / act_safe,
            "raw_ratio_median": float(r_clip.median()),
            "cal_ratio_median": float(c_clip.median()),
            "scale_mean": float(sub["scale"].mean()),
            "scale_median": float(sub["scale"].median()),
            "scale_p10": float(sub["scale"].quantile(0.10)),
            "scale_p90": float(sub["scale"].quantile(0.90)),
            "scale_min": float(sub["scale"].min()),
            "scale_max": float(sub["scale"].max()),
            "mean_error_km": float(sub["error_km"].mean()),
            "median_error_km": float(sub["error_km"].median()),
            "error_p75_km": float(sub["error_km"].quantile(0.75)),
            "error_p90_km": float(sub["error_km"].quantile(0.90)),
            "error_p95_km": float(sub["error_km"].quantile(0.95)),
            "error_max_km": float(sub["error_km"].max()),
            "raw_mean_error_km": float(sub["raw_error_km"].mean()),
            "mean_abs_bearing_deg": float(sub["abs_bearing_error"].mean()),
            "median_abs_bearing_deg": float(sub["abs_bearing_error"].median()),
        }
    return out


def print_collapse_diagnostics(summ, horizons, max_correction=0.20):
    """spec #15/#16/#17: scale saturation, raw-head collapse, identifiability."""
    lo = 1.0 - max_correction
    hi = 1.0 + max_correction
    print("\n===== IDENTIFIABILITY / COLLAPSE CHECKS =====")
    for h in horizons:
        s = summ[float(h)]
        if abs(s["scale_min"] - lo) < 1e-6 or abs(s["scale_max"] - hi) < 1e-6:
            print(f"[+{h:g}h] WARNING: scale saturates at edge [{lo:.2f}, {hi:.2f}] "
                  f"(min {s['scale_min']:.3f}, max {s['scale_max']:.3f})")
        if s["raw_ratio_mean"] < 0.7 and 0.9 < s["cal_ratio_mean"] < 1.1:
            print(f"[+{h:g}h] WARNING: POTENTIAL RAW-HEAD/CALIBRATION-HEAD "
                  f"COMPENSATION (raw_ratio {s['raw_ratio_mean']:.3f} << 1, "
                  f"cal_ratio {s['cal_ratio_mean']:.3f} ~ 1)")
    rs = np.array([summ[float(h)]["raw_ratio_mean"] for h in horizons])
    ss = np.array([summ[float(h)]["scale_mean"] for h in horizons])
    corr = float("nan")
    status = ""
    if rs.std() > 1e-9 and ss.std() > 1e-9:
        corr = float(np.corrcoef(rs, ss)[0, 1])
        if abs(corr) > 0.9:
            sign = "anti-correlation" if corr < 0 else "correlation"
            status = f"  <-- strong {sign} between raw_ratio and scale: identifiability still at risk"
    print(f"  corr(raw_ratio, scale) across horizons = {corr:+.3f}{status}")


def overall_metrics(df, horizons):
    """Overall + per-horizon mean error dict {overall, '2','4',...,'24'}."""
    per = {str(int(round(float(h)))): float(df[df["horizon_h"] == float(h)]["error_km"].mean())
           for h in horizons}
    per["overall"] = float(df["error_km"].mean())
    return per


def error_stats_table(df, horizons):
    """spec #36: mean/median/P75/P90/P95/max per horizon (calibrated error)."""
    out = {}
    for h in horizons:
        e = df[df["horizon_h"] == float(h)]["error_km"]
        out[float(h)] = {
            "mean": float(e.mean()), "median": float(e.median()),
            "p75": float(e.quantile(0.75)), "p90": float(e.quantile(0.90)),
            "p95": float(e.quantile(0.95)), "max": float(e.max()),
        }
    return out
