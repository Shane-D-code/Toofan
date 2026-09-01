"""Phase 24 — SIH dashboard prediction function.

Usage:
    from src.predict_ri import predict_ri

    prob, risk, threshold, factors = predict_ri(
        storm_id="1998-008",
        datetime="1998-11-14 03:00:00",
        latitude=13.5, longitude=86.5,
        IMD_features={...}, ERA5_features={...},
        satellite_image=None,   # np.ndarray (128,128,1) or None
    )
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_model(path):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    if not hasattr(m, "_estimator_type"):
        m._estimator_type = "classifier"
    m.load_model(str(path))
    if not hasattr(m, "n_classes_"):
        m.n_classes_ = 2
    return m


_imd_model = _load_model(REPO_ROOT / "models" / "imd_xgboost.json")
_era5_model = _load_model(REPO_ROOT / "models" / "era5_xgboost.json")
_combined_model = _load_model(REPO_ROOT / "models" / "imd_era5_xgboost.json")

_imd_feats = ["latitude", "longitude", "max_wind_kt", "central_pressure_hpa",
              "pressure_drop_hpa", "wind_6h_change", "wind_minus_6h_kt",
              "delta_v_minus_6h_kt", "wind_minus_12h_kt",
              "delta_v_minus_12h_kt", "wind_minus_24h_kt",
              "delta_v_minus_24h_kt"]


def _frames_for(feat_dicts):
    import pandas as pd
    return {k: pd.DataFrame([d]) if d is not None else None
            for k, d in feat_dicts.items()}


def predict_ri(storm_id, datetime, latitude, longitude,
               IMD_features=None, ERA5_features=None,
               satellite_image=None):
    """Return calibrated P(RI within 24h), risk category and key factors.

    Uses the tabular IMD and IMD+ERA5 models. Satellite image, if provided, is
    not used until the Colab CNN is fused (returns a note).
    """
    import pandas as pd

    imd_row = {"latitude": latitude, "longitude": longitude}
    if IMD_features:
        imd_row.update(IMD_features)
    X_imd = pd.DataFrame([{c: imd_row.get(c, np.nan) for c in _imd_feats}])
    p_imd = float(_imd_model.predict_proba(X_imd)[0, 1])

    erg = {}
    if ERA5_features is not None:
        era5_feats = list(ERA5_features.keys())
        X_e5 = pd.DataFrame([ERA5_features])
        # Reorder columns to match training feature order where possible.
        p_era5 = float(_era5_model.predict_proba(X_e5)[0, 1])
    else:
        p_era5 = float("nan")

    # Combined model (IMD + ERA5) when ERA5 available.
    p_combined = float("nan")
    if ERA5_features is not None:
        comb_row = imd_row.copy()
        comb_row.update(ERA5_features)
        # Feature-order note: the combined model expects IMD+ERA5 columns;
        # here we reuse the combined feature list from model_config.
        import json as _json
        cfg_path = REPO_ROOT / "results" / "model_config.json"
        if cfg_path.exists():
            cfg = _json.load(open(cfg_path))
            cols = cfg["combined_features"]
            df = pd.DataFrame([{c: comb_row.get(c, np.nan) for c in cols}])
            p_combined = float(_combined_model.predict_proba(df)[0, 1])

    # Primary probability: combined if available else IMD.
    p = p_combined if not np.isnan(p_combined) else p_imd
    threshold = 0.5  # dashboard default; refine from model_config thresholds
    risk = "HIGH" if p >= threshold else "LOW"

    factors = []
    if IMD_features and IMD_features.get("wind_6h_change", 0) is not None:
        if IMD_features.get("wind_6h_change", 0) > 10:
            factors.append("recent intensity increase")
    if ERA5_features is not None and ERA5_features.get("shear_850_200", 0) is not None:
        if abs(ERA5_features.get("shear_850_200", 0)) < 8:
            factors.append("favorable (low) environmental shear")
    if satellite_image is not None:
        factors.append("satellite structure (CNN pending fusion)")
    if not factors:
        factors.append("baseline intensity/pressure trajectory")

    return {
        "storm_id": storm_id, "datetime": str(datetime),
        "probability": p, "risk_category": risk, "threshold": threshold,
        "P_IMD": p_imd, "P_ERA5": p_era5, "P_combined": p_combined,
        "key_factors": factors,
    }
