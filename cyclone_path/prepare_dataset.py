"""
prepare_dataset.py
-------------------
Downloads IBTrACS v04r01 (since1980 subset), cleans it, engineers features,
and writes a single processed CSV ready for training.

Usage:
    python prepare_dataset.py --out data/processed.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import requests

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.since1980.list.v04r01.csv"
)

RAW_COLUMNS = ["SID", "ISO_TIME", "NATURE", "LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_RMW"]

VALID_NATURE = {"TS", "TY", "HU", "TC", "SS"}
MIN_TIMESTEPS = 10


def download(raw_path):
    if os.path.exists(raw_path):
        print(f"[skip] {raw_path} already exists")
        return
    print(f"Downloading IBTrACS from {IBTRACS_URL} ...")
    r = requests.get(IBTRACS_URL, stream=True, timeout=120)
    r.raise_for_status()
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print("Download complete.")


def load_raw(raw_path):
    # Row 0 = column names, row 1 = units/description -> skip it.
    df = pd.read_csv(raw_path, skiprows=[1], usecols=RAW_COLUMNS, low_memory=False)
    return df


def sst_climatology_proxy(lat, month):
    """
    Crude lat/month climatology proxy for sea-surface temperature (deg C).
    Warmest near the equator, seasonal wobble by hemisphere.
    Replace with NOAA OISST (matched by time+location) for production use.
    """
    base = 28.0 - 0.30 * np.abs(lat - 5.0)
    # Northern hemisphere warms in NH summer (peak ~Sep), southern in SH summer (peak ~Mar)
    phase = np.where(lat >= 0, 9, 3)
    seasonal = 1.5 * np.cos(2 * np.pi * (month - phase) / 12.0)
    sst = base + seasonal
    return np.clip(sst, 20.0, 31.5)


def willoughby_rahn_rmw(vmax_kt):
    """Willoughby & Rahn (2004) empirical RMW estimate (nautical miles)."""
    return 35.0 * np.exp(-0.01 * vmax_kt)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360) % 360


def wrapped_lon_delta(lon2, lon1):
    """Dateline-safe longitude difference (deg), e.g. 179 -> -179 gives +2."""
    return (lon2 - lon1 + 180.0) % 360.0 - 180.0


def local_velocity_kmh(lat1, lon1, lat2, lon2, dt_h):
    """
    Eastward (u) and northward (v) translation velocity (km/h) between two
    fixes, using the local spherical approximation:
        northward km ~ dlat_deg * 111.32
        eastward  km ~ dlon_deg * 111.32 * cos(mean_lat)
    Longitude difference is dateline-wrapped so there is no discontinuity
    near +/-180. Returns NaN if dt is missing/zero.
    """
    if dt_h is None or not np.isfinite(dt_h) or dt_h <= 0:
        return np.nan, np.nan
    mean_lat = np.radians(0.5 * (lat1 + lat2))
    dlon = wrapped_lon_delta(lon2, lon1)
    dlon_km = dlon * 111.32 * np.cos(mean_lat)
    dlat_km = (lat2 - lat1) * 111.32
    return dlon_km / dt_h, dlat_km / dt_h


def find_lookback_idx(times, i, lookback_s, tol_s=None):
    """
    Find the index j < i of the observation closest to (times[i] - lookback_s),
    within the storm. Returns None if no candidate exists within tolerance.
    """
    target = times[i] - lookback_s
    j = np.searchsorted(times[:i], target)
    candidates = []
    if j > 0:
        candidates.append(j - 1)
    if j < i:
        candidates.append(j)
    if not candidates:
        return None
    best_j, best_d = None, None
    for c in candidates:
        d = abs(times[c] - target)
        if best_d is None or d < best_d:
            best_j, best_d = c, d
    if tol_s is not None and best_d > tol_s:
        return None
    return best_j


MOTION_WINDOWS_HOURS = [3, 6, 12]
MOTION_WINDOW_TOL_HOURS = 1.5


def add_motion_trend_features(df):
    """
    V6: adds carefully calculated HISTORICAL motion-trend features, computed
    ONLY from observations available before the forecast time (never from the
    future). They describe acceleration/turning of the storm's own motion:

      * speed_change         : rate of change of translation speed (km/h per hour)
      * bearing_change_rate  : rate of change of heading (deg per hour)
      * u_acceleration       : rate of change of eastward velocity (km/h^2)
      * v_acceleration       : rate of change of northward velocity (km/h^2)

    These are first differences of quantities already available at each
    timestep, divided by the ACTUAL elapsed time. No high-order derivatives are
    used, so the features stay stable where the data supports them. For rows
    near the start of a storm where a prior fix does not exist, the features
    fall back to a documented neutral 0 ("no prior motion to measure a change
    from") -- they only ever appear as early-context rows in an input window.
    """
    df = df.copy()
    df["speed_change"] = np.nan
    df["bearing_change_rate"] = np.nan
    df["u_acceleration"] = np.nan
    df["v_acceleration"] = np.nan

    for sid, g in df.groupby("SID"):
        idx = g.index
        n = len(g)
        speed = g["speed_kmh"].values.astype(float)
        dt = g["dt_hours"].values.astype(float)
        brng = g["bearing_deg"].values.astype(float)
        u = g["u_kmh"].values.astype(float)
        v = g["v_kmh"].values.astype(float)

        speed_change = np.full(n, np.nan)
        brng_rate = np.full(n, np.nan)
        u_acc = np.full(n, np.nan)
        v_acc = np.full(n, np.nan)
        for i in range(2, n):
            if np.isfinite(dt[i]) and dt[i] > 0:
                if np.isfinite(speed[i]) and np.isfinite(speed[i - 1]):
                    speed_change[i] = (speed[i] - speed[i - 1]) / dt[i]
                if np.isfinite(brng[i]) and np.isfinite(brng[i - 1]):
                    d = (brng[i] - brng[i - 1] + 180.0) % 360.0 - 180.0
                    brng_rate[i] = d / dt[i]
                if np.isfinite(u[i]) and np.isfinite(u[i - 1]):
                    u_acc[i] = (u[i] - u[i - 1]) / dt[i]
                if np.isfinite(v[i]) and np.isfinite(v[i - 1]):
                    v_acc[i] = (v[i] - v[i - 1]) / dt[i]
        df.loc[idx, "speed_change"] = speed_change
        df.loc[idx, "bearing_change_rate"] = brng_rate
        df.loc[idx, "u_acceleration"] = u_acc
        df.loc[idx, "v_acceleration"] = v_acc

    # Neutral fallback for early-context rows (see docstring).
    df["speed_change"] = df["speed_change"].fillna(0.0)
    df["bearing_change_rate"] = df["bearing_change_rate"].fillna(0.0)
    df["u_acceleration"] = df["u_acceleration"].fillna(0.0)
    df["v_acceleration"] = df["v_acceleration"].fillna(0.0)
    return df


def add_trajectory_motion_features(df):
    """
    Adds explicit, physically meaningful trajectory-motion features, computed
    ONLY from observations available at each timestep (never from the future).

    For every row i in a storm (chronologically sorted):
      * u_kmh / v_kmh  : east / north velocity from the immediately previous
                         fix (i-1 -> i), in km/h.
      * speed_W_kmh, u_W_kmh, v_W_kmh for W in {3, 6, 12}: average translation
                         speed / velocity over the trailing ~W-hour window,
                         derived from the observation closest to (time_i - W h)
                         when it falls within MOTION_WINDOW_TOL_HOURS.
      * accel_kmh2     : rate of change of translation speed = (speed[i] -
                         speed[i-1]) / dt, in km/h^2.
      * turn_sin/cos   : cyclic encoding of the change in heading between the
                         previous segment (i-2 -> i-1) and the current one
                         (i-1 -> i). Using sin/cos avoids the 359->0
                         discontinuity.

    Multi-window fallback (documented, see spec item 20):
      If a trailing ~W-hour historical point does not exist within tolerance,
      the window velocity falls back, in order, to:
        1. the next-shorter window's velocity at the same row
           (6h -> 3h; 12h -> 6h -> 3h),
        2. the single-step velocity u_kmh/v_kmh at the same row,
        3. 0 (storm start / no prior observation).
      This is a clearly defined, no-future-data fallback; a missing 3h window
      is never converted into a fake 3h movement from a long gap.

    The window velocity uses the ACTUAL elapsed time (times[j] -> times[i]),
    so an irregular 6-hour gap is correctly reported as ~6h motion, not ~3h.
    """
    df = df.copy()
    df["u_kmh"] = np.nan
    df["v_kmh"] = np.nan
    df["accel_kmh2"] = np.nan
    df["turn_sin"] = np.nan
    df["turn_cos"] = np.nan
    for w in MOTION_WINDOWS_HOURS:
        df[f"speed_{w}h_kmh"] = np.nan
        df[f"u_{w}h_kmh"] = np.nan
        df[f"v_{w}h_kmh"] = np.nan

    for sid, g in df.groupby("SID"):
        idx = g.index
        lat = g["lat"].values.astype(float)
        lon = g["lon"].values.astype(float)
        times = g["ISO_TIME"].values.astype("datetime64[s]").astype(np.int64)
        # single-step speed/bearing dt per row (NaN for first row of the storm)
        speed = g["speed_kmh"].values.astype(float)   # NaN where derived as NaN (first row)
        dt = g["dt_hours"].values.astype(float)

        n = len(g)
        u1 = np.full(n, np.nan)
        v1 = np.full(n, np.nan)
        for i in range(1, n):
            u, v = local_velocity_kmh(lat[i - 1], lon[i - 1], lat[i], lon[i], dt[i])
            u1[i], v1[i] = u, v
        df.loc[idx, "u_kmh"] = u1
        df.loc[idx, "v_kmh"] = v1

        # acceleration: rate of speed change between consecutive single-step speeds
        accel = np.full(n, np.nan)
        for i in range(2, n):
            if np.isfinite(speed[i]) and np.isfinite(speed[i - 1]) and np.isfinite(dt[i]) and dt[i] > 0:
                accel[i] = (speed[i] - speed[i - 1]) / dt[i]
        df.loc[idx, "accel_kmh2"] = accel

        # turning: heading change between consecutive segments, encoded cyclically
        brng = (g["bearing_deg"].values.astype(float) if "bearing_deg" in g else
                np.full(n, np.nan))
        turn_sin = np.full(n, np.nan)
        turn_cos = np.full(n, np.nan)
        for i in range(2, n):
            if np.isfinite(brng[i]) and np.isfinite(brng[i - 1]):
                d = (brng[i] - brng[i - 1] + 180.0) % 360.0 - 180.0
                turn_sin[i] = np.sin(np.radians(d))
                turn_cos[i] = np.cos(np.radians(d))
        df.loc[idx, "turn_sin"] = turn_sin
        df.loc[idx, "turn_cos"] = turn_cos

        # multi-window motion, tiered fallback (see docstring). Windows are
        # computed in ascending order so a longer window can fall back to an
        # already-computed shorter window at the same row. A boolean per row
        # records whether a TRUE ~W-hour lookback was found (motion_W_available)
        # versus a fallback, so the model can learn to distrust fallback values.
        tol_s = MOTION_WINDOW_TOL_HOURS * 3600.0
        avail = {w: np.ones(n) for w in MOTION_WINDOWS_HOURS}  # default: available
        for w in sorted(MOTION_WINDOWS_HOURS):
            sw_speed = np.full(n, np.nan)
            sw_u = np.full(n, np.nan)
            sw_v = np.full(n, np.nan)
            aw = np.zeros(n)  # availability for this window (1 = true lookback)
            # candidate shorter windows already computed (strictly < w)
            shorter = [x for x in MOTION_WINDOWS_HOURS if x < w]
            for i in range(1, n):
                j = find_lookback_idx(times, i, w * 3600.0, tol_s)
                if j is not None:
                    dt_w = (times[i] - times[j]) / 3600.0
                    sw_speed[i] = haversine_km(lat[j], lon[j], lat[i], lon[i]) / dt_w
                    u, v = local_velocity_kmh(lat[j], lon[j], lat[i], lon[i], dt_w)
                    sw_u[i], sw_v[i] = u, v
                    aw[i] = 1.0
                else:
                    # tiered fallback: shorter window -> single step -> 0
                    used = False
                    for s in reversed(shorter):
                        sval = df.loc[idx[i], f"speed_{s}h_kmh"]
                        if np.isfinite(sval):
                            sw_speed[i] = sval
                            sw_u[i] = df.loc[idx[i], f"u_{s}h_kmh"]
                            sw_v[i] = df.loc[idx[i], f"v_{s}h_kmh"]
                            used = True
                            break
                    if not used:
                        if np.isfinite(u1[i]):
                            sw_speed[i] = speed[i]
                            sw_u[i] = u1[i]
                            sw_v[i] = v1[i]
                        else:
                            sw_speed[i] = 0.0
                            sw_u[i] = 0.0
                            sw_v[i] = 0.0
            df.loc[idx, f"speed_{w}h_kmh"] = sw_speed
            df.loc[idx, f"u_{w}h_kmh"] = sw_u
            df.loc[idx, f"v_{w}h_kmh"] = sw_v
            df.loc[idx, f"motion_{w}h_available"] = aw

    # Ensure the availability flags exist as columns for every window.
    for w in MOTION_WINDOWS_HOURS:
        if f"motion_{w}h_available" not in df.columns:
            df[f"motion_{w}h_available"] = 1.0

    # Fill the remaining NaNs (storm start / first one or two fixes, where no
    # previous observation exists) with a clearly-defined NEUTRAL fallback of
    # 0.0 (= "no motion"), matching the existing treatment of speed_kmh and
    # bearing_deg. These rows only ever appear as early context in an input
    # window; the model can learn the meaning of the near-zero values.
    motion_cols = (["u_kmh", "v_kmh", "accel_kmh2", "turn_sin", "turn_cos"]
                   + [f"{p}_{w}h_kmh" for w in MOTION_WINDOWS_HOURS
                      for p in ("speed", "u", "v")])
    df[motion_cols] = df[motion_cols].fillna(0.0)
    return df


def apply_per_storm(df, func):
    """groupby('SID').apply(func), restoring the SID column across pandas versions
    (pandas >= 2.2 drops the grouping column from the per-group frame)."""
    out = df.groupby("SID", group_keys=True).apply(func)
    if "SID" not in out.columns:
        out = out.reset_index(level=0)
    return out.reset_index(drop=True)


def process(df):
    df = df.copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df.dropna(subset=["ISO_TIME", "LAT", "LON"])
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["WMO_WIND"] = pd.to_numeric(df["WMO_WIND"], errors="coerce")
    df["WMO_PRES"] = pd.to_numeric(df["WMO_PRES"], errors="coerce")
    df["USA_RMW"] = pd.to_numeric(df["USA_RMW"], errors="coerce")

    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # Quality filter: named tropical-storm-strength systems only
    df = df[df["NATURE"].isin(VALID_NATURE)]

    # Rename to working schema
    df = df.rename(columns={"WMO_WIND": "wind", "WMO_PRES": "mslp", "USA_RMW": "rmw",
                             "LAT": "lat", "LON": "lon"})

    # --- Missing value handling (per storm) ---
    def fill_storm(g):
        g = g.sort_values("ISO_TIME")
        g["mslp"] = g["mslp"].ffill().clip(870, 1020)
        g["rmw"] = g["rmw"].ffill()
        est = willoughby_rahn_rmw(g["wind"].fillna(g["wind"].median()))
        g["rmw"] = g["rmw"].fillna(est)
        return g

    df = apply_per_storm(df, fill_storm)
    df["wind"] = df.groupby("SID")["wind"].transform(lambda s: s.ffill().bfill())
    df = df.dropna(subset=["wind", "mslp", "rmw"])

    # Minimum length filter
    counts = df.groupby("SID")["SID"].transform("count")
    df = df[counts >= MIN_TIMESTEPS]

    # --- Derived proxies ---
    df["month"] = df["ISO_TIME"].dt.month
    df["sst"] = sst_climatology_proxy(df["lat"].values, df["month"].values)

    # shear proxy: pressure deficit between an assumed outer/POCI pressure (1010 hPa
    # default when unavailable) and MSLP -- deeper storms assumed lower ambient shear.
    poci_default = 1010.0
    pressure_deficit = poci_default - df["mslp"]
    df["shear"] = np.clip(20.0 - 0.10 * pressure_deficit, 0.0, 25.0)

    # --- Kinematics: translation speed & bearing from previous fix ---
    def kinematics(g):
        g = g.sort_values("ISO_TIME")
        lat, lon, t = g["lat"].values, g["lon"].values, g["ISO_TIME"].values
        dt_h = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
        dt_h = np.concatenate([[np.nan], dt_h])
        dist = np.concatenate([[np.nan], haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])])
        brng = np.concatenate([[np.nan], bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])])
        g["dt_hours"] = dt_h
        g["speed_kmh"] = dist / dt_h
        g["bearing_deg"] = brng
        return g

    df = apply_per_storm(df, kinematics)
    df["speed_kmh"] = df["speed_kmh"].fillna(0.0)
    df["bearing_deg"] = df["bearing_deg"].fillna(0.0)
    df["dt_hours"] = df["dt_hours"].fillna(3.0)

    # cyclic encodings (bearing, month) so the model doesn't see a false discontinuity
    df["bearing_sin"] = np.sin(np.radians(df["bearing_deg"]))
    df["bearing_cos"] = np.cos(np.radians(df["bearing_deg"]))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # V3: explicit, physically meaningful trajectory-motion features (see
    # docstring). Computed only from prior observations; the model can attend
    # to both short (3h/6h) and larger-scale (12h) historical motion, plus its
    # own acceleration and turning rate.
    df = add_trajectory_motion_features(df)

    # V6: historical motion-trend features (acceleration / turning rate of the
    # storm's own motion) plus motion-window availability flags.
    df = add_motion_trend_features(df)

    cols = ["SID", "ISO_TIME", "lat", "lon", "wind", "mslp", "rmw", "sst", "shear",
            "speed_kmh", "bearing_sin", "bearing_cos", "month_sin", "month_cos", "dt_hours",
            # V3 motion features (appended so the original 13 keep leading indices)
            "u_kmh", "v_kmh",
            "speed_3h_kmh", "u_3h_kmh", "v_3h_kmh",
            "speed_6h_kmh", "u_6h_kmh", "v_6h_kmh",
            "speed_12h_kmh", "u_12h_kmh", "v_12h_kmh",
            "accel_kmh2", "turn_sin", "turn_cos",
            # V6 motion-trend features + window availability flags
            "speed_change", "bearing_change_rate",
            "u_acceleration", "v_acceleration",
            "motion_3h_available", "motion_6h_available", "motion_12h_available"]
    return df[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/ibtracs.since1980.list.v04r01.csv")
    ap.add_argument("--out", default="data/processed.csv")
    args = ap.parse_args()

    download(args.raw)
    df = load_raw(args.raw)
    df = process(df)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows across {df['SID'].nunique()} storms to {args.out}")


if __name__ == "__main__":
    main()
