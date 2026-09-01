"""
dataset.py
----------
Builds (input_window -> multi-horizon target) samples from the processed
IBTrACS CSV. One class, no extra wrapper layers.

V2 changes (see project spec for full rationale):

  * Targets are built from ACTUAL observation timestamps, not from an
    assumed fixed cadence. `horizons` is now a list of forecast lead times
    IN HOURS (e.g. (3, 6, 12, 24)) rather than "number of rows ahead".
    For a given anchor row, the target for horizon `h` hours is the
    observation whose timestamp is closest to (anchor_time + h hours),
    accepted only if it falls within `target_tolerance_hours` of that
    exact lead time. If no such observation exists, the whole sample is
    dropped (all horizons must be resolvable) so that every returned
    sample has a full, correctly-timed target vector -- this keeps the
    (x, y_delta, pos) contract the rest of the codebase (model/loss/infer)
    already relies on.
  * Longitude displacements are computed with dateline-safe wrapping:
        dlon = (future_lon - current_lon + 180) % 360 - 180
    This is applied everywhere a longitude delta is produced.
  * Sample-window indexing uses the mathematically correct inclusive
    upper bound so the last valid window in each storm is not discarded.
  * Feature/target representation (documented, unchanged from V1):
      - Input features (FEATURE_COLS) are per-timestep raw physical
        quantities; the columns listed in NORM_COLS are standardized
        (zero mean / unit variance) using TRAINING-SET statistics only.
        `lat`/`lon` and the already-bounded cyclic encodings
        (bearing_sin/cos, month_sin/cos) are left unnormalized.
      - Targets (`y_delta`) are RAW DEGREE displacements (dlat, dlon),
        not normalized and not kilometers. The model predicts in this
        same raw-degree space (see model.py), and the loss (losses.py)
        converts to physical km via Haversine only for the trajectory
        loss / evaluation metric, and otherwise keeps the uncertainty
        term in this same degree space to avoid unit-mixing. This
        dataset -> model -> loss -> evaluation -> inference chain must
        stay consistent; do not normalize y_delta without also updating
        model.py, losses.py and infer.py.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Original V1/V2 feature set (leading indices preserved for backward
# compatibility / V2-baseline reproduction).
BASE_FEATURE_COLS = [
    "lat", "lon", "wind", "mslp", "rmw", "sst", "shear",
    "speed_kmh", "bearing_sin", "bearing_cos", "month_sin", "month_cos", "dt_hours",
]
BASE_NORM_COLS = ["wind", "mslp", "rmw", "sst", "shear", "speed_kmh", "dt_hours"]

# V6 environmental-feature interface (sections 13-15 of the V6 spec).
# `ENV_FEATURE_COLS` are the CURRENT proxy environmental columns carried into
# the feature tensor. They are documented as proxies (not real measurements).
# `STEERING_FEATURE_COLS` lists the real reanalysis/satellite fields a future
# version will replace the proxies with (e.g. 850/700/500/200 hPa u/v wind,
# vertical wind shear, RH, geopotential height, SST, ...). It is intentionally
# EMPTY for V6 -- no fabricated data is added. The model/dataset accept an
# optional `environment_features` tensor with a documented shape/units so an
# external environmental source can later be plugged in without restructuring.
ENV_FEATURE_COLS = ["sst", "shear"]
STEERING_FEATURE_COLS = []  # future extension only; not fabricated in V6
# Documented external environmental tensor shape/units (future interface):
#   environment_features : (num_env_timesteps, num_env_channels) float tensor
#       channel order   : [u850, v850, u700, v700, u500, v500, u200, v200,
#                          vws, rhum, hgt, sst] (order per STEERING_FEATURE_COLS)
#       units           : u/v wind [m/s], vws [m/s], rhum [%], hgt [gpm], sst [deg C]
#   Currently optional & unused by V6 default models (proxies fill these roles).

# V3 explicit trajectory-motion features (see prepare_dataset.py docstring for
# definitions, units and the multi-window fallback). Appended AFTER the base
# features so the base columns keep their leading indices.
MOTION_FEATURE_COLS = [
    "u_kmh", "v_kmh",
    "speed_3h_kmh", "u_3h_kmh", "v_3h_kmh",
    "speed_6h_kmh", "u_6h_kmh", "v_6h_kmh",
    "speed_12h_kmh", "u_12h_kmh", "v_12h_kmh",
    "accel_kmh2",
    "turn_sin", "turn_cos",
]
# Motion features that are unbounded/continuous and must be standardized using
# TRAINING-SET statistics only. Cyclic (bounded) encodings stay unnormalized,
# matching the existing handling of bearing_sin/cos.
MOTION_NORM_COLS = [
    "u_kmh", "v_kmh",
    "speed_3h_kmh", "u_3h_kmh", "v_3h_kmh",
    "speed_6h_kmh", "u_6h_kmh", "v_6h_kmh",
    "speed_12h_kmh", "u_12h_kmh", "v_12h_kmh",
    "accel_kmh2",
]

# V6 motion-trend features (see prepare_dataset.add_motion_trend_features).
# These are historical first-difference rates computed only from the past.
# `bearing_change_rate` has a physical unit but is not cyclic-encoded here; it
# is an unbounded continuous rate so it is standardized. The window-availability
# flags are binary (0/1) and are NOT standardized (they are already on a natural
# 0-1 scale / act as indicators).
TREND_FEATURE_COLS = [
    "speed_change", "bearing_change_rate", "u_acceleration", "v_acceleration",
    "motion_3h_available", "motion_6h_available", "motion_12h_available",
]
TREND_NORM_COLS = ["speed_change", "bearing_change_rate", "u_acceleration", "v_acceleration"]

# Full V3 feature set (base + motion). When `motion_features=False`, training
# uses only BASE_FEATURE_COLS (V2 baseline / controlled experiments A-C).
FEATURE_COLS = BASE_FEATURE_COLS + MOTION_FEATURE_COLS
NORM_COLS = BASE_NORM_COLS + MOTION_NORM_COLS

# V6 feature set = V3 (base + motion) + motion-trend features (opt-in). When
# `use_motion_trends=True` the trend columns (continuous rates + availability
# flags) are appended. Bearing/cos and month/cos cyclic encodings, and the
# binary availability flags, are left unnormalized.
FEATURE_COLS_V6 = FEATURE_COLS + TREND_FEATURE_COLS
NORM_COLS_V6 = NORM_COLS + TREND_NORM_COLS

# Documented feature audit (see spec item 17). "derived" features are
# computed from the raw track/intensity record, not independently
# observed; "proxy" features are stand-ins for environmental fields that
# IBTrACS does not provide at all and must not be presented as real
# satellite/reanalysis measurements.
FEATURE_METADATA = {
    "lat":          {"source": "IBTrACS LAT",              "unit": "deg",     "normalized": False, "kind": "observed"},
    "lon":          {"source": "IBTrACS LON",               "unit": "deg",     "normalized": False, "kind": "observed"},
    "wind":         {"source": "IBTrACS WMO_WIND",          "unit": "kt",      "normalized": True,  "kind": "observed"},
    "mslp":         {"source": "IBTrACS WMO_PRES (filled)", "unit": "hPa",     "normalized": True,  "kind": "observed/filled"},
    "rmw":          {"source": "IBTrACS USA_RMW / Willoughby-Rahn (2004) estimate", "unit": "nmi", "normalized": True, "kind": "observed/derived"},
    "sst":          {"source": "lat/month climatology PROXY (not satellite SST)", "unit": "deg C", "normalized": True, "kind": "proxy"},
    "shear":        {"source": "assumed-POCI minus MSLP pressure-deficit PROXY (not reanalysis shear)", "unit": "kt (proxy)", "normalized": True, "kind": "proxy"},
    "speed_kmh":    {"source": "derived from consecutive fixes (Haversine/dt)", "unit": "km/h", "normalized": True,  "kind": "derived"},
    "bearing_sin":  {"source": "derived bearing (previous fix -> current fix)", "unit": "unitless (sin)", "normalized": False, "kind": "derived"},
    "bearing_cos":  {"source": "derived bearing (previous fix -> current fix)", "unit": "unitless (cos)", "normalized": False, "kind": "derived"},
    "month_sin":    {"source": "calendar month of ISO_TIME",  "unit": "unitless (sin)", "normalized": False, "kind": "derived"},
    "month_cos":    {"source": "calendar month of ISO_TIME",  "unit": "unitless (cos)", "normalized": False, "kind": "derived"},
    "dt_hours":     {"source": "actual gap to previous fix",  "unit": "hours",  "normalized": True,  "kind": "derived"},

    # V3 explicit trajectory-motion features (units documented; all calculated
    # from observations available at each timestep -- never from the future).
    "u_kmh":        {"source": "eastward velocity, previous fix -> current", "unit": "km/h", "normalized": True, "kind": "derived"},
    "v_kmh":        {"source": "northward velocity, previous fix -> current", "unit": "km/h", "normalized": True, "kind": "derived"},
    "speed_3h_kmh": {"source": "avg speed over trailing ~3h window",          "unit": "km/h", "normalized": True, "kind": "derived"},
    "u_3h_kmh":     {"source": "avg east velocity over trailing ~3h window",  "unit": "km/h", "normalized": True, "kind": "derived"},
    "v_3h_kmh":     {"source": "avg north velocity over trailing ~3h window", "unit": "km/h", "normalized": True, "kind": "derived"},
    "speed_6h_kmh": {"source": "avg speed over trailing ~6h window",          "unit": "km/h", "normalized": True, "kind": "derived"},
    "u_6h_kmh":     {"source": "avg east velocity over trailing ~6h window",  "unit": "km/h", "normalized": True, "kind": "derived"},
    "v_6h_kmh":     {"source": "avg north velocity over trailing ~6h window", "unit": "km/h", "normalized": True, "kind": "derived"},
    "speed_12h_kmh":{"source": "avg speed over trailing ~12h window",         "unit": "km/h", "normalized": True, "kind": "derived"},
    "u_12h_kmh":    {"source": "avg east velocity over trailing ~12h window", "unit": "km/h", "normalized": True, "kind": "derived"},
    "v_12h_kmh":    {"source": "avg north velocity over trailing ~12h window","unit": "km/h", "normalized": True, "kind": "derived"},
    "accel_kmh2":   {"source": "rate of change of translation speed (delta speed / dt)", "unit": "km/h^2", "normalized": True, "kind": "derived"},
    "turn_sin":     {"source": "cyclic sin of heading change between consecutive segments", "unit": "unitless (sin)", "normalized": False, "kind": "derived"},
    "turn_cos":     {"source": "cyclic cos of heading change between consecutive segments", "unit": "unitless (cos)", "normalized": False, "kind": "derived"},

    # V6 historical motion-trend features (rates of change of the storm's own
    # motion, computed only from observations before the forecast time).
    "speed_change":        {"source": "rate of change of translation speed", "unit": "km/h per h", "normalized": True, "kind": "derived"},
    "bearing_change_rate": {"source": "rate of change of heading",            "unit": "deg per h",  "normalized": True, "kind": "derived"},
    "u_acceleration":      {"source": "rate of change of eastward velocity",  "unit": "km/h^2",     "normalized": True, "kind": "derived"},
    "v_acceleration":      {"source": "rate of change of northward velocity", "unit": "km/h^2",     "normalized": True, "kind": "derived"},
    "motion_3h_available": {"source": "1 if a true ~3h lookback existed, else 0", "unit": "bool (0/1)", "normalized": False, "kind": "derived"},
    "motion_6h_available": {"source": "1 if a true ~6h lookback existed, else 0", "unit": "bool (0/1)", "normalized": False, "kind": "derived"},
    "motion_12h_available":{"source": "1 if a true ~12h lookback existed, else 0","unit": "bool (0/1)", "normalized": False, "kind": "derived"},
}


def wrapped_lon_delta(future_lon, current_lon):
    """Dateline-safe longitude difference, e.g. 179 -> -179 gives +2, not -358."""
    return (future_lon - current_lon + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# V7 spherical helpers (numpy, used for building distance_direction targets and
# for forward/inverse geodesic destination math).
# --------------------------------------------------------------------------- #
EARTH_R = 6371.0088  # km, consistent with losses.EARTH_RADIUS_KM


def _haversine_km_np(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (numpy, degrees)."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Great-circle initial bearing (degrees, 0-360 clockwise from north)."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brg = np.degrees(np.arctan2(x, y))
    return (brg + 360.0) % 360.0


# Physical conversion constant (deg <-> local km), consistent with losses.py.
KM_PER_DEG = 111.32


def _ecef_unit(lat_deg, lon_deg):
    """Lat/lon (degrees) -> unit 3D vector on the unit sphere (ECEF, r=1)."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], axis=-1)


def _ecef_to_latlon(v):
    """Unit 3D vector -> (lat, lon) in degrees."""
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, lon


def slerp_position(lat_lo, lon_lo, lat_hi, lon_hi, f):
    """Great-circle (spherical linear) interpolation between two lat/lon fixes.

    Returns (lat, lon) interpolated a fraction ``f`` (0..1) of the way from
    (lat_lo, lon_lo) to (lat_hi, lon_hi) along the great circle, dateline-safe.
    When the two fixes are (nearly) antipodal or coincident we fall back to a
    straight Euclidean lerp to avoid dividing by ~0 of sin(omega)."""
    f = float(f)
    a = _ecef_unit(lat_lo, lon_lo)
    b = _ecef_unit(lat_hi, lon_hi)
    dot = float(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))
    omega = float(np.arccos(dot))
    if omega < 1e-6 or np.pi - omega < 1e-6:
        v = (1.0 - f) * a + f * b
    else:
        s_lo = np.sin((1.0 - f) * omega)
        s_hi = np.sin(f * omega)
        v = (s_lo * a + s_hi * b) / np.sin(omega)
    lat, lon = _ecef_to_latlon(v)
    return float(lat), float(lon)


def reconstruct_positions(pred, cur_pos, horizons, target_mode="delta"):
    """
    Convert a predicted target tensor (B, H, 2) back into absolute geographic
    (lat, lon) positions (B, H, 2), for any supported `target_mode`.

      "delta"  : pred = (dlat, dlon) degrees            -> pos = cur + delta
      "km"     : pred = (dx_km east, dy_km north)       -> pos = cur + local inverse
      "motion" : pred = per-horizon segment velocity (u_kmh, v_kmh);
                 cumulative km displacement = sum(segment_u * segment_hours)

    Uses the local linearization at the anchor (cur), matching how targets are
    built in the dataset. Longitude wrapping is preserved by the raw add (the
    final haversine distance is wrap-invariant).
    """
    import torch

    # Align the anchor to the prediction's device/dtype so the function works
    # even when callers pass pred and cur_pos on different devices (e.g. a CPU
    # ground-truth target y with a GPU cur_pos during eval/plotting).
    pred = pred.to(dtype=pred.dtype, device=pred.device)
    cur_pos = cur_pos.to(dtype=pred.dtype, device=pred.device).float()
    B = cur_pos.shape[0]
    H = pred.shape[1]
    horizons_t = torch.as_tensor(list(horizons), dtype=pred.dtype, device=pred.device)
    cur_lat = cur_pos[:, 0:1]            # (B, 1)
    cur_lon = cur_pos[:, 1:2]
    cos_lat = torch.cos(torch.deg2rad(cur_lat)).clamp_min(0.1)  # (B, 1)

    if target_mode == "delta":
        dlat = pred[..., 0]
        dlon = pred[..., 1]
    elif target_mode == "km":
        # pred[...,0] = dx_km (east), pred[...,1] = dy_km (north)
        dy_km = pred[..., 1]
        dx_km = pred[..., 0]
        dlat = dy_km / KM_PER_DEG
        dlon = dx_km / (KM_PER_DEG * cos_lat)
    elif target_mode == "motion":
        # pred = (u_kmh, v_kmh) per segment -> cumulative km displacement.
        # segment hours: [h0, h1-h0, h2-h1, ...]
        prev = torch.cat([torch.zeros(1, device=horizons_t.device, dtype=horizons_t.dtype),
                          horizons_t[:-1]])
        seg_hours = (horizons_t - prev).unsqueeze(0)  # (1, H)
        cum_dx = (pred[..., 0] * seg_hours).cumsum(dim=1)  # (B, H) east km
        cum_dy = (pred[..., 1] * seg_hours).cumsum(dim=1)  # (B, H) north km
        dlat = cum_dy / KM_PER_DEG
        dlon = cum_dx / (KM_PER_DEG * cos_lat)
    elif target_mode == "distance_direction":
        # V7 target: pred = (distance_km, direction_sin, direction_cos) per
        # horizon, CUMULATIVE from the current position. Bearing is measured
        # clockwise from north; dx_km (east) = d*sin(bearing),
        # dy_km (north) = d*cos(bearing). We convert to an absolute geographic
        # destination using the PROPER spherical (great-circle) direct-geodesic
        # calculation rather than a local linear approximation, so large 24h
        # movements remain accurate and ±180 deg longitude wrapping is handled
        # automatically.
        d_km = pred[..., 0].clamp_min(0.0)         # (B, H) non-negative distance
        s = pred[..., 1]                            # direction_sin
        c = pred[..., 2]                            # direction_cos
        bearing_rad = torch.atan2(s, c)             # robust to sin/cos magnitude
        dy_km = d_km * torch.cos(bearing_rad)       # north
        dx_km = d_km * torch.sin(bearing_rad)       # east
        R = torch.tensor(EARTH_R, dtype=pred.dtype, device=pred.device)
        lat1 = torch.deg2rad(cur_lat).to(device=pred.device, dtype=pred.dtype)  # (B, 1)
        lon1 = torch.deg2rad(cur_lon).to(device=pred.device, dtype=pred.dtype)
        angular = d_km / R                          # (B, H)
        lat2 = torch.asin(torch.sin(lat1) * torch.cos(angular)
                          + torch.cos(lat1) * torch.sin(angular) * torch.cos(bearing_rad))
        lon2 = lon1 + torch.atan2(
            torch.sin(bearing_rad) * torch.sin(angular) * torch.cos(lat1),
            torch.cos(angular) - torch.sin(lat1) * torch.sin(lat2),
        )
        pred_lat = torch.rad2deg(lat2)
        pred_lon = torch.rad2deg(lon2)
        return torch.stack([pred_lat, pred_lon], dim=-1)   # (B, H, 2)
    else:
        raise ValueError(f"unknown target_mode {target_mode}")

    pred_lat = cur_lat + dlat
    pred_lon = cur_lon + dlon
    return torch.stack([pred_lat, pred_lon], dim=-1)  # (B, H, 2)


def positions_to_km(pos, cur_pos):
    """Invert the local-km reconstruction: absolute (lat, lon) (B,H,2) relative to
    anchor cur_pos (B,2) -> cumulative local displacement (B,H,2) = (dx_km, dy_km).
    Used to supply true cumulative km for teacher forcing in the sequential decoder."""
    import torch
    cur_lat = cur_pos[:, 0:1]
    cur_lon = cur_pos[:, 1:2]
    cos_lat = torch.cos(torch.deg2rad(cur_lat)).clamp_min(0.1)
    dlat = pos[..., 0] - cur_lat
    dlon = wrapped_lon_delta(pos[..., 1], cur_lon)
    dy_km = dlat * KM_PER_DEG
    dx_km = dlon * KM_PER_DEG * cos_lat
    return torch.stack([dx_km, dy_km], dim=-1)  # (B,H,2)




class CycloneDataset(Dataset):
    """
    Each sample:
        x       : (input_len, feature_dim)   past feature window
        y       : (num_horizons, 2)          true target in `target_mode` space
        pos     : (2,)                       current (lat, lon), the anchor point

    `self.meta` (parallel list, same index as `self.samples`) holds per-sample
    metadata: sid, anchor index, resolved future lat/lon/times per horizon, and
    motion-difficulty measures used for hard-example sampling / error analysis.

    `horizons` is a list of forecast lead times in HOURS (e.g. [3, 6, 12, 24],
    or V12's [2, 4, 6, ..., 24]), resolved against the storm's actual
    observation timestamps rather than assumed to be evenly spaced rows.

    `target_resolution` ("nearest", default, or "interpolate") controls how each
    horizon target is resolved against the (3-hourly) fixes. "interpolate"
    (V12) slerps lat/lon along the great circle between the two fixes bracketing
    the lead time, giving correct sub-fix targets on a 2h grid (the raw fixes
    are 3h apart); it never extrapolates past the last fix and drops a
    (sample,horizon) with no fix within tolerance, keeping the all-or-nothing
    per-sample contract.
    """

    def __init__(self, csv_path, storm_ids, input_len=8, horizons=(3, 6, 12, 24),
                 target_tolerance_hours=1.5, dt_hours=3.0, stats=None,
                 feature_cols=None, norm_cols=None, target_mode="delta",
                 coord_encode=False, use_motion_trends=False,
                 target_resolution="nearest",
                 environment_features=None):
        # V6: `use_motion_trends` opts-in the historical motion-trend feature
        # columns (default False -> exact V5 feature set for ablation A/C).
        # `environment_features` is an OPTIONAL external environmental tensor
        # (see module docstring / STEERING_FEATURE_COLS for the future shape and
        # units). It is NOT required for V6 and NOT joined when None -- the
        # existing proxy features stay in the tensor. This is the clean seam
        # a future version uses to inject real reanalysis/satellite data.
        # V5: `target_mode` selects the prediction target representation.
        #   "delta"  -> y = (dlat, dlon) raw degrees (V3 default, backward compatible)
        #   "km"     -> y = (dx_km east, dy_km north) local kilometer displacement
        #   "motion" -> y = per-horizon segment-average velocity (u_kmh east, v_kmh north)
        # The anchor position `pos` and (for reconstruction) the true future
        # latitudes/longitudes per horizon are always retained in `self.meta` so
        # losses/evaluation can convert any representation back to geographic km.

        # V5 `coord_encode`: improve the raw coordinate representation (Experiment F).
        # Longitude is a circular variable stored in a mixed (-179.8..262.9) range,
        # which is a poor raw feature. This maps lon into [-180, 180) and adds
        # sin/cos cylindrical encodings, and replaces the raw `lon`/`lat` channels
        # with them. The encoding is a deterministic function of the coordinate
        # (no training stats needed), so it is identical for train/val/test.
        df = pd.read_csv(csv_path, parse_dates=["ISO_TIME"])
        df = df[df["SID"].isin(storm_ids)].sort_values(["SID", "ISO_TIME"])
        self.input_len = input_len
        self.horizons = list(horizons)          # hours ahead, e.g. [3, 6, 12, 24]
        self.target_tolerance_hours = target_tolerance_hours
        self.dt_hours = dt_hours                # nominal cadence; used only as a search-window hint
        self.target_mode = target_mode

        # V12: `target_resolution` selects HOW each horizon target is resolved
        # against the (3-hourly) observation fixes:
        #   "nearest"      : snap to the observation closest to (anchor + h) if
        #                    it falls within `target_tolerance_hours` (V2-V11 behaviour).
        #   "interpolate"  : if (anchor + h) lies strictly BETWEEN two fixes, slerp
        #                    lat/lon along the great circle between them (linear by
        #                    time fraction); used directly when it lands on a fix.
        #                    Never extrapolates past the last fix; a (sample,horizon)
        #                    with no fix within tolerance is dropped, so the dataset
        #                    stays all-or-nothing per sample (same contract as before).
        #                 This is the main V12 accuracy lever for finer 2h grid.
        self.target_resolution = target_resolution
        assert target_resolution in ("nearest", "interpolate"), \
            f"unknown target_resolution {target_resolution!r}"

        # Feature/normalization column selection (V3). `feature_cols` selects
        # which columns are fed to the model; `norm_cols` selects which of
        # those are standardized. Defaults to the full V3 sets. Callers can
        # pass BASE_FEATURE_COLS/BASE_NORM_COLS + motion_features=False to
        # reproduce the V2 baseline / run controlled experiments A-C.
        # V6: select feature/norm columns. If `feature_cols` is explicitly
        # given (e.g. from a checkpoint), it wins (this keeps ablation B / the
        # motion-trends-off comparison exact). Otherwise default to the V6 set
        # when `use_motion_trends` is on, else the V3 set.
        if feature_cols is not None:
            self.feature_cols = list(feature_cols)
            self.norm_cols = list(norm_cols) if norm_cols is not None else list(NORM_COLS)
        else:
            if use_motion_trends:
                self.feature_cols = list(FEATURE_COLS_V6)
                self.norm_cols = list(NORM_COLS_V6)
            else:
                self.feature_cols = list(FEATURE_COLS)
                self.norm_cols = list(NORM_COLS)
        self.use_motion_trends = use_motion_trends
        self.environment_features = environment_features
        self.leak_check = True  # programmatic future-data leakage guard (V6 #20)

        if coord_encode:
            # Build the encoded coordinate feature set from a clean base and create
            # the encoded columns unconditionally (deterministically) so that
            # train/val/test always produce the identical feature layout even when
            # `feature_cols` is passed across splits. The encoding itself needs no
            # training statistics (pure function of the coordinate).
            base_feats = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
            new_feats = [c for c in base_feats if c not in ("lat", "lon")]
            base_norms = list(norm_cols) if norm_cols is not None else list(NORM_COLS)
            self.norm_cols = [c for c in base_norms if c not in ("lat", "lon")]

            if "lat" in df.columns:
                df["lat_sin"] = np.sin(np.deg2rad(df["lat"].values))
                df["lat_cos"] = np.cos(np.deg2rad(df["lat"].values))
                new_feats += ["lat_sin", "lat_cos"]
            if "lon" in df.columns:
                lon_e = ((df["lon"].values + 180.0) % 360.0) - 180.0
                df["lon_e"] = lon_e
                df["lon_sin"] = np.sin(np.deg2rad(lon_e))
                df["lon_cos"] = np.cos(np.deg2rad(lon_e))
                new_feats += ["lon_e", "lon_sin", "lon_cos"]
            self.feature_cols = new_feats
            for c in ("lon_e", "lon_sin", "lon_cos", "lat_sin", "lat_cos"):
                if c in self.feature_cols:
                    self.norm_cols.append(c)

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"feature columns missing from CSV: {missing}. Rerun "
                "prepare_dataset.py to add the V3 motion + V6 trend columns."
            )
        # V6 environmental-feature interface: verify the proxy environmental
        # columns used by this dataset are present (they are part of the base
        # feature set). Future real environmental data would be injected via
        # `environment_features` and need not be a CSV column.
        env_missing = [c for c in ENV_FEATURE_COLS if c in self.feature_cols and c not in df.columns]
        if env_missing:
            raise ValueError(f"environmental feature columns missing: {env_missing}")

        # normalization stats (mean/std), computed on the given data if not provided.
        # IMPORTANT: callers must pass `stats=train_ds.stats` for val/test splits --
        # never recompute stats on val/test/full data (see module docstring).
        if stats is None:
            self.stats = {c: (float(df[c].mean()), float(df[c].std() + 1e-6)) for c in self.norm_cols}
        else:
            self.stats = stats

        df = df.copy()
        for c in self.norm_cols:
            m, s = self.stats[c]
            df[c] = (df[c] - m) / s

        self.samples = []
        self.meta = []
        # V5: physical conversion constant (deg <-> local km), consistent with
        # the value used in losses.py. East km uses the local cos(lat) scale.
        KM_PER_DEG = 111.32
        for sid, g in df.groupby("SID"):
            g = g.sort_values("ISO_TIME").reset_index(drop=True)
            n = len(g)
            if n < input_len + 1:
                continue

            feats = g[self.feature_cols].values.astype(np.float32)
            raw_latlon = g[["lat", "lon"]].values.astype(np.float64)  # unnormalized
            times = g["ISO_TIME"].values.astype("datetime64[s]").astype(np.int64)  # seconds since epoch

            # Mathematically correct inclusive upper bound: the last valid
            # window has its anchor at index n-1, i.e. start = n - input_len.
            for start in range(0, n - input_len + 1):
                end = start + input_len          # exclusive
                anchor_idx = end - 1
                anchor_time = times[anchor_idx]
                cur_pos = raw_latlon[anchor_idx]
                cos_lat = float(np.cos(np.deg2rad(cur_pos[0])))

                target_cols = 3 if target_mode == "distance_direction" else 2
                y_target = np.zeros((len(self.horizons), target_cols), dtype=np.float32)
                deltas_deg = np.zeros((len(self.horizons), 2), dtype=np.float32)
                km_disp = np.zeros((len(self.horizons), 2), dtype=np.float32)
                future_idx = np.zeros(len(self.horizons), dtype=np.int64)
                future_pos = np.zeros((len(self.horizons), 2), dtype=np.float64)
                future_times = np.zeros(len(self.horizons), dtype=np.int64)
                valid = True
                for hi, h in enumerate(self.horizons):
                    target_time = anchor_time + int(h * 3600)
                    # search only within this storm's remaining (future) rows
                    future_times_ = times[anchor_idx + 1:]
                    if future_times_.size == 0:
                        valid = False
                        break

                    if self.target_resolution == "nearest":
                        # ---- snap to the observation closest to target_time ----
                        j = np.searchsorted(future_times_, target_time)
                        candidates = []
                        if j < future_times_.size:
                            candidates.append(anchor_idx + 1 + j)
                        if j > 0:
                            candidates.append(anchor_idx + 1 + j - 1)
                        best_idx, best_diff = None, None
                        for c in candidates:
                            diff_hours = abs(times[c] - target_time) / 3600.0
                            if best_diff is None or diff_hours < best_diff:
                                best_idx, best_diff = c, diff_hours
                        if best_idx is None or best_diff > self.target_tolerance_hours:
                            valid = False
                            break
                        fp = raw_latlon[best_idx]
                        future_idx[hi] = best_idx
                        future_times[hi] = times[best_idx]
                    else:
                        # ---- interpolate: slerp lat/lon between the two fixes
                        #      that BRACKET (anchor + h); exact fix used directly;
                        #      never extrapolate past the last fix ----
                        j = np.searchsorted(times, target_time)  # first idx with time >= target
                        if j < n and times[j] == target_time and j > anchor_idx:
                            # exact hit on a future observation
                            fp = raw_latlon[j]
                            future_idx[hi] = j
                            future_times[hi] = times[j]
                        else:
                            lo = j - 1                      # last fix with time <  target
                            seg_hi = j                      # first fix with time >= target
                            if lo < anchor_idx or seg_hi >= n:
                                # either before the anchor (impossible for h>0) or
                                # past the last fix (extrapolation) -> drop
                                valid = False
                                break
                            nearest_diff_h = min((target_time - times[lo]) / 3600.0,
                                                 (times[seg_hi] - target_time) / 3600.0)
                            if nearest_diff_h > self.target_tolerance_hours:
                                # no fix within tolerance of this lead time -> drop
                                valid = False
                                break
                            span_h = (times[seg_hi] - times[lo]) / 3600.0
                            f = (target_time - times[lo]) / (3600.0 * span_h)
                            la, lo_ = slerp_position(
                                raw_latlon[lo, 0], raw_latlon[lo, 1],
                                raw_latlon[seg_hi, 0], raw_latlon[seg_hi, 1], f)
                            fp = np.array([la, lo_], dtype=np.float64)
                            future_idx[hi] = seg_hi
                            future_times[hi] = target_time

                    dlat = fp[0] - cur_pos[0]
                    dlon = wrapped_lon_delta(fp[1], cur_pos[1])
                    dy_km = dlat * KM_PER_DEG
                    dx_km = dlon * KM_PER_DEG * max(cos_lat, 0.1)  # guard polar division

                    deltas_deg[hi, 0] = dlat
                    deltas_deg[hi, 1] = dlon
                    km_disp[hi, 0] = dx_km
                    km_disp[hi, 1] = dy_km
                    future_pos[hi] = fp

                    if target_mode == "delta":
                        y_target[hi, 0] = dlat
                        y_target[hi, 1] = dlon
                    elif target_mode == "km":
                        y_target[hi, 0] = dx_km
                        y_target[hi, 1] = dy_km
                    elif target_mode == "distance_direction":
                        # V7: distance_km + direction (sin, cos) of the great-circle
                        # initial bearing from the CURRENT position to the future
                        # position. Cumulative per horizon (current -> +h).
                        _brg = _initial_bearing_deg(cur_pos[0], cur_pos[1], fp[0], fp[1])
                        _dist = _haversine_km_np(cur_pos[0], cur_pos[1], fp[0], fp[1])
                        y_target[hi, 0] = _dist
                        y_target[hi, 1] = np.sin(np.deg2rad(_brg))
                        y_target[hi, 2] = np.cos(np.deg2rad(_brg))
                    elif target_mode == "motion":
                        # segment-average velocity from the previous cumulative
                        # point (anchor for the first segment) to this horizon.
                        h_prev = 0.0 if hi == 0 else float(self.horizons[hi - 1])
                        seg_hours = h - h_prev
                        prev_dx = km_disp[hi - 1, 0] if hi > 0 else 0.0
                        prev_dy = km_disp[hi - 1, 1] if hi > 0 else 0.0
                        y_target[hi, 0] = (km_disp[hi, 0] - prev_dx) / seg_hours
                        y_target[hi, 1] = (km_disp[hi, 1] - prev_dy) / seg_hours
                    else:
                        raise ValueError(f"unknown target_mode {target_mode}")

                if not valid:
                    continue

                x = feats[start:end]
                if not np.isfinite(x).all() or not np.isfinite(y_target).all():
                    # Fail loudly rather than silently training on corrupt rows.
                    raise ValueError(
                        f"Non-finite value encountered in storm {sid} at window start {start}. "
                        "Check upstream preprocessing (prepare_dataset.py)."
                    )

                # V6 data-leakage protection (spec #20): every input feature must
                # come from time <= T (indices [start, end), anchor = end-1), and
                # every target from time > T (future_idx). Verified programmatically.
                if self.leak_check:
                    assert anchor_idx == end - 1, "anchor must be the last input timestep"
                    # all input rows are strictly before any target row
                    if np.any(future_idx <= anchor_idx):
                        raise RuntimeError(
                            f"data-leakage: future target index {int(future_idx.max())} "
                            f"not after anchor {anchor_idx} for storm {sid} at start {start}"
                        )

                # Motion-difficulty metadata (from the anchor row's own features)
                # for hard-example sampling and error analysis (#8).
                anchor_feats = g.iloc[anchor_idx]
                def _f(name):
                    try:
                        return float(anchor_feats.get(name, 0.0))
                    except Exception:
                        return 0.0
                meta = {
                    "sid": str(sid),
                    "anchor_idx": int(anchor_idx),
                    "future_idx": future_idx.copy(),
                    "future_pos": future_pos.copy(),
                    "future_times": future_times.copy(),
                    "deltas_deg": deltas_deg.copy(),
                    "km_disp": km_disp.copy(),
                    "accel_anchor": _f("accel_kmh2"),
                    "turn_anchor": _f("turn_cos"),
                    "speed_anchor": _f("speed_kmh"),
                    "wind_anchor": _f("wind"),
                }
                self.samples.append((x, y_target, cur_pos.astype(np.float32), str(sid)))
                self.meta.append(meta)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y_target, cur_pos, _sid = self.samples[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y_target),
            torch.from_numpy(cur_pos),
        )



def split_storm_ids(csv_path, val_frac=0.15, test_frac=0.15, seed=42):
    """Split by storm SID (not by timestep) to avoid leaking a storm's future into train."""
    df = pd.read_csv(csv_path, usecols=["SID"])
    ids = np.array(df["SID"].unique(), dtype=object)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    ids = ids[perm]
    n = len(ids)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_ids = ids[:n_val]
    test_ids = ids[n_val:n_val + n_test]
    train_ids = ids[n_val + n_test:]

    # Guard against data leakage: storm-disjoint splits are a correctness
    # requirement, not just a nice-to-have.
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    assert not (s_train & s_val), "train/val storm overlap detected"
    assert not (s_train & s_test), "train/test storm overlap detected"
    assert not (s_val & s_test), "val/test storm overlap detected"

    return list(train_ids), list(val_ids), list(test_ids)


if __name__ == "__main__":
    train_ids, val_ids, test_ids = split_storm_ids("data/processed.csv")
    print(len(train_ids), len(val_ids), len(test_ids))
    ds = CycloneDataset("data/processed.csv", train_ids)
    print("num samples:", len(ds))
    x, y, pos = ds[0]
    print(x.shape, y.shape, pos.shape)
