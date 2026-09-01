#!/usr/bin/env python3
"""IMD temporal feature engineering + Delta-V24 prediction experiment.

Follows the 23-task specification. EXPERIMENTAL improvement over the frozen
baseline — never a modification of it.

Frozen scientific facts (preserved, not overwritten):
  - RI definition: 24 h wind increase >= 30 kt (target RI_24h only).
  - Seed 42. Storm-safe split; identical train/val/test STORM assignment.
  - Strict common test set: 174 obs / 20 storms / 25 RI (same observations as
    run_final_imd_era5_comparison.py).
  - Frozen baselines (results/final_imd_era5_experiment.json):
      IMD        PR-AUC 0.5935  ROC-AUC 0.8572
      ERA5       PR-AUC 0.2969  ROC-AUC 0.7047
      IMD + ERA5 PR-AUC 0.3411  ROC-AUC 0.7462
  - Satellite CNN untouched; ERA5 not retrained.
  - Leakage bar: no predictor may use information after time t. Forbidden set
    {RI_24h, wind_24h_kt, delta_v_24h_kt, target_time_24h, future track} is
    asserted absent from every feature set.

Outputs (new files only; historical results untouched):
    results/imd_baseline_frozen.json
    results/imd_engineered_feature_audit.csv
    results/imd_engineered_dataset.csv
    results/imd_feature_engineering_comparison.csv
    results/imd_feature_engineering_bootstrap.json
    results/imd_engineered_feature_importance.csv
    results/imd_engineered_shap_importance.csv
    results/imd_error_analysis.csv
    results/delta_v24_predictions.csv
    results/delta_v24_regression_metrics.json
    models/imd_baseline_frozen.json
    models/imd_engineered_xgboost.json
    models/delta_v24_xgboost.json
    figures/imd_engineered_{roc_curve,pr_curve,calibration,shap}.png
    figures/delta_v24_{prediction,residuals}.png
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import REPO_ROOT, get_seed, load_config
from src import data as data_mod
from src import features as feat_mod
from src import evaluate as eval_mod

FORBIDDEN_PREDICTORS = {
    "RI_24h", "wind_24h_kt", "delta_v_24h_kt", "target_time_24h",
}

RESULTS = REPO_ROOT / "results"
MODELS = REPO_ROOT / "models"
FIGURES = REPO_ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helpers
# ============================================================================

def snow_hash(rows) -> str:
    return hashlib.sha1("\n".join(sorted(set(rows))).encode()).hexdigest()[:16]


def load_xgb(path: Path):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    if not hasattr(m, "_estimator_type"):
        m._estimator_type = "classifier"
    m.load_model(str(path))
    if not hasattr(m, "n_classes_"):
        m.n_classes_ = 2
    return m


def lag_value(df: pd.DataFrame, value_col: str, lag_h: float, tol_h: float
              ) -> pd.Series:
    """Value of a column ~lag_h hours before each observation (historical only).

    Lookup key is ``t - lag``. Only real observation times <= t can match
    (target < t always), so no future information is used. Row with time
    nearest to ``t - lag`` is accepted within ``tol_h``.
    """
    if value_col == "datetime_utc":
        src = df[["storm_id", "datetime_utc"]].copy()
        src["v"] = src["datetime_utc"]
    else:
        src = df[["storm_id", "datetime_utc", value_col]].rename(
            columns={value_col: "v"})
    target = df[["storm_id", "datetime_utc"]].copy()
    target["_row"] = target.index
    target["_lookup"] = target["datetime_utc"] - pd.Timedelta(hours=lag_h)
    src = src.sort_values(["storm_id", "datetime_utc"])
    target = target.sort_values(["storm_id", "_lookup"])
    merged = pd.merge_asof(
        target, src,
        left_on="_lookup", right_on="datetime_utc",
        by="storm_id", direction="nearest",
        tolerance=pd.Timedelta(hours=tol_h))
    merged = merged.sort_index()
    return pd.Series(merged["v"].to_numpy(), index=df.index)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def storm_block_bootstrap_pr_auc(y, p_a, p_b, groups, n_boot=2000, seed=42):
    """Storm-block bootstrap of PR-AUC difference (engineered - baseline)."""
    from sklearn.metrics import average_precision_score
    rng = np.random.RandomState(seed)
    storms = np.asarray(sorted(set(groups)))
    g = np.asarray(groups)
    y = np.asarray(y)
    diffs = []
    for _ in range(n_boot):
        picked = rng.choice(storms, size=len(storms), replace=True)
        mask = np.isin(g, picked)
        if len(np.unique(y[mask])) < 2:
            continue
        diffs.append(average_precision_score(y[mask], p_b[mask])
                     - average_precision_score(y[mask], p_a[mask]))
    diffs = np.asarray(diffs)
    if len(diffs) < 100:
        return {"n_valid": int(len(diffs)), "ci_2_5": None, "ci_97_5": None,
                "mean": None, "note": "too few valid resamples"}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_valid": int(len(diffs)),
            "ci_2_5": float(lo), "ci_97_5": float(hi), "mean": float(diffs.mean())}


def build_matrices(df, split, feature_set):
    X_tr, y_tr, use = feat_mod.prepare_features(
        df.loc[df["storm_id"].isin(split.train_storms)], feature_set)
    X_va, y_va, _ = feat_mod.prepare_features(
        df.loc[df["storm_id"].isin(split.val_storms)], feature_set)
    X_te, y_te, _ = feat_mod.prepare_features(
        df.loc[df["storm_id"].isin(split.test_storms)], feature_set)
    X_va = X_va.reindex(columns=use)
    X_te = X_te.reindex(columns=use)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, use


# ============================================================================
# 3. Feature engineering
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["storm_id", "datetime_utc"]).reset_index(drop=True)
    wind = out["max_wind_kt"]
    prs = out["central_pressure_hpa"]
    lat = out["latitude"]
    lon = out["longitude"]

    # A. 3 h intensity change (6h/12h/24h already exist as delta_v_minus_*h_kt).
    out["wind_change_3h"] = wind - lag_value(out, "max_wind_kt", 3, 2.5)

    # B. Intensification acceleration.
    out["acceleration_6h"] = (
        out["delta_v_minus_6h_kt"]
        - (out["wind_minus_6h_kt"] - out["wind_minus_12h_kt"]))
    out["acceleration_12h"] = (
        out["delta_v_minus_12h_kt"]
        - (out["wind_minus_12h_kt"] - out["wind_minus_24h_kt"]))

    # C. Pressure tendency (sign: P(t) - P(t-lag); deepening -> negative).
    for lag, tol in [(3, 2.5), (6, 3.5), (12, 5.0), (24, 8.0)]:
        out[f"pressure_change_{lag}h"] = (
            prs - lag_value(out, "central_pressure_hpa", lag, tol))

    # D. Pressure acceleration (change in 6h pressure tendency).
    p6 = lag_value(out, "central_pressure_hpa", 6, 3.5)
    p12 = lag_value(out, "central_pressure_hpa", 12, 5.0)
    out["pressure_acceleration_6h"] = (prs - p6) - (p6 - p12)

    # E. Storm motion over the ~6 h window (haversine; u/v in km/h).
    lat6 = lag_value(out, "latitude", 6, 3.5)
    lon6 = lag_value(out, "longitude", 6, 3.5)
    t6 = lag_value(out, "datetime_utc", 6, 3.5)
    dt_h = ((out["datetime_utc"] - pd.to_datetime(t6, errors="coerce"))
            .dt.total_seconds() / 3600.0)
    spd = np.full(len(out), np.nan)
    uu = np.full(len(out), np.nan)
    vv = np.full(len(out), np.nan)
    has = lat6.notna() & lon6.notna() & dt_h.gt(0)
    for i in out.index[has]:
        d = haversine_km(lat6.loc[i], lon6.loc[i], lat.loc[i], lon.loc[i])
        mlat = np.radians((lat.loc[i] + lat6.loc[i]) / 2.0)
        dx = (lon.loc[i] - lon6.loc[i]) * 111.32 * np.cos(mlat)
        dy = (lat.loc[i] - lat6.loc[i]) * 111.32
        spd[i] = d / dt_h.loc[i]
        uu[i] = dx / dt_h.loc[i]
        vv[i] = dy / dt_h.loc[i]
    out["translation_speed"] = spd
    out["translation_u"] = uu
    out["translation_v"] = vv

    # F. Track evolution.
    out["latitude_change_6h"] = lat - lat6
    out["longitude_change_6h"] = lon - lon6
    lat12 = lag_value(out, "latitude", 12, 5.0)
    lon12 = lag_value(out, "longitude", 12, 5.0)
    out["latitude_change_12h"] = lat - lat12
    out["longitude_change_12h"] = lon - lon12
    return out


# ============================================================================
# 14. Regression
# ============================================================================

def run_regression(eng, split, feat_sets, target, common_test, seed):
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr

    results = {}
    predictions_full = None
    g_te_all = eng.loc[eng["storm_id"].isin(split.test_storms)]

    for name, feats in feat_sets.items():
        g_tr = eng.loc[eng["storm_id"].isin(split.train_storms)]
        g_va = eng.loc[eng["storm_id"].isin(split.val_storms)]
        g_te = g_te_all
        X_tr, y_tr, use = feat_mod.prepare_features(g_tr, feats)
        X_va, y_va, _ = feat_mod.prepare_features(g_va, feats)
        X_te, y_te, _ = feat_mod.prepare_features(g_te, feats)
        y_tr = g_tr.loc[X_tr.index, target].to_numpy()
        y_va = g_va.loc[X_va.index, target].to_numpy()
        y_te = g_te.loc[X_te.index, target].to_numpy()
        X_va = X_va.reindex(columns=use)
        X_te = X_te.reindex(columns=use)

        model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
            max_delta_step=1, random_state=seed,
            objective="reg:squarederror", eval_metric="rmse",
            early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_va = model.predict(X_va)
        p_te = model.predict(X_te)

        m = {
            "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
            "n_test_full": int(len(X_te)),
            "XGB_full_test": {
                "mae": float(mean_absolute_error(y_te, p_te)),
                "rmse": float(np.sqrt(mean_squared_error(y_te, p_te))),
                "r2": float(r2_score(y_te, p_te)),
                "pearson": float(pearsonr(y_te, p_te)[0]),
            },
        }
        results[name] = m
        print(f"  [{name}] XGB  full-test: MAE={m['XGB_full_test']['mae']:.2f} "
              f"RMSE={m['XGB_full_test']['rmse']:.2f} "
              f"R2={m['XGB_full_test']['r2']:.3f} "
              f"corr={m['XGB_full_test']['pearson']:.3f} (n={len(X_te)})")

        if name == "C-physical":
            rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            p_te_rf = rf.predict(X_te)
            results["C-physical"]["RF_full_test"] = {
                "mae": float(mean_absolute_error(y_te, p_te_rf)),
                "rmse": float(np.sqrt(mean_squared_error(y_te, p_te_rf))),
                "r2": float(r2_score(y_te, p_te_rf)),
            }
            print(f"  [C-physical] RF full-test: MAE={results['C-physical']['RF_full_test']['mae']:.2f} "
                  f"RMSE={results['C-physical']['RF_full_test']['rmse']:.2f} "
                  f"R2={results['C-physical']['RF_full_test']['r2']:.3f}")
            # common-test evaluation (same 174 obs as classification)
            common_keys = set(zip(common_test["storm_id"],
                                  common_test["datetime_utc"]))
            g_te_rows = g_te.loc[X_te.index]
            keep = g_te_rows.apply(
                lambda r: (r["storm_id"], r["datetime_utc"]) in common_keys,
                axis=1).to_numpy()
            cm_pos = np.where(keep)[0]
            yc = y_te[cm_pos]
            pc = p_te[cm_pos]
            results["C-physical"]["common_test"] = {
                "n": int(len(cm_pos)),
                "mae": float(mean_absolute_error(yc, pc)),
                "rmse": float(np.sqrt(mean_squared_error(yc, pc))),
                "r2": float(r2_score(yc, pc)),
                "pearson": float(pearsonr(yc, pc)[0]),
            }
            print(f"  [C-physical] common-test: MAE={results['C-physical']['common_test']['mae']:.2f} "
                  f"RMSE={results['C-physical']['common_test']['rmse']:.2f} "
                  f"R2={results['C-physical']['common_test']['r2']:.3f} (n={len(cm_pos)})")
            predictions_full = pd.DataFrame({
                "storm_id": g_te.loc[X_te.index, "storm_id"].to_numpy(),
                "datetime_utc": g_te.loc[X_te.index, "datetime_utc"].to_numpy(),
                "actual_delta_v24_kt": y_te,
                "expected_delta_v24_kt": p_te,
            }).set_index(X_te.index)
            results["C-physical"]["model_file"] = "models/delta_v24_xgboost.json"
            model.get_booster().save_model(str(MODELS / "delta_v24_xgboost.json"))

    results["_meta"] = {
        "target": target,
        "target_never_a_feature": True,
        "forbidden_set": sorted(FORBIDDEN_PREDICTORS),
        "seed": seed,
        "honesty": ("delta_v_24h_kt is the regression target only and is never "
                    "an input feature; same storm-safe split as classification"),
    }
    with open(RESULTS / "delta_v24_regression_metrics.json", "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return results, predictions_full


def write_delta_predictions(pred_full, common_test):
    keys = set(zip(common_test["storm_id"], common_test["datetime_utc"]))
    sel = pred_full.apply(
        lambda r: (r["storm_id"], r["datetime_utc"]) in keys, axis=1)
    sub = pred_full[sel].copy()
    sub = sub.drop_duplicates(subset=["storm_id", "datetime_utc"])
    mapping = sub.set_index(["storm_id", "datetime_utc"])[
        ["actual_delta_v24_kt", "expected_delta_v24_kt"]]
    out = common_test[["storm_id", "datetime_utc", "RI_24h"]].copy()
    out = out.join(mapping, on=["storm_id", "datetime_utc"])
    return out


# ============================================================================
# 17. Calibration
# ============================================================================

def run_calibration(model, model_name, eng, split, feats, common_test,
                    p_test, yar, figure_dir):
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    X_va, y_va, _, _, _, _, _ = build_matrices(eng, split, feats)
    p_va = model.predict_proba(X_va)[:, 1]
    y_va = np.asarray(y_va).astype(int)

    platt = LogisticRegression().fit(p_va.reshape(-1, 1), y_va)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)
    b_platt = brier_score_loss(y_va, platt.predict_proba(p_va.reshape(-1, 1))[:, 1])
    b_iso = brier_score_loss(y_va, iso.predict(p_va))
    best = "isotonic" if b_iso <= b_platt else "platt"

    p_cal = (iso.predict(p_test) if best == "isotonic"
             else platt.predict_proba(p_test.reshape(-1, 1))[:, 1])
    b_raw = brier_score_loss(yar, p_test)
    b_cal = brier_score_loss(yar, p_cal)

    frac_raw, mp_raw = calibration_curve(yar, p_test, n_bins=10, strategy="uniform")
    frac_cal, mp_cal = calibration_curve(yar, p_cal, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mp_raw, frac_raw, marker="o", label="raw (no recalibration)")
    ax.plot(mp_cal, frac_cal, marker="s", label=f"calibrated ({best})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed RI fraction")
    ax.set_title(f"Calibration — {model_name}")
    ax.legend()
    plt.tight_layout()
    path = figure_dir / "imd_engineered_calibration.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Brier: raw={b_raw:.4f} -> calibrated({best})={b_cal:.4f}")
    print(f"  saved -> {path}")
    return {"method": best, "brier_raw": b_raw, "brier_calibrated": b_cal}


# ============================================================================
# Figures
# ============================================================================

def plot_classification_figures(y, probs, tag, out_dir):
    from sklearn.metrics import roc_curve, precision_recall_curve
    plt.figure(figsize=(7, 5))
    for name, p in probs.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], ls="--", color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC — {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "imd_engineered_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, p in probs.items():
        if len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, marker=".", label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "imd_engineered_pr_curve.png", dpi=150)
    plt.close()
    print(f"  saved -> figures/imd_engineered_roc_curve.png, "
          f"figures/imd_engineered_pr_curve.png")


def plot_regression_figures(pred_df, out_dir):
    y = pred_df["actual_delta_v24_kt"].to_numpy()
    p = pred_df["expected_delta_v24_kt"].to_numpy()
    if len(y) == 0:
        print("  [figures] no regression predictions; skipping")
        return
    lims = [min(y.min(), p.min()), max(y.max(), p.max())]
    plt.figure(figsize=(6, 5))
    plt.scatter(y, p, s=20)
    plt.plot(lims, lims, ls="--", color="grey")
    plt.xlabel("Actual ΔV24 (kt)"); plt.ylabel("Predicted ΔV24 (kt)")
    plt.title("ΔV24 predicted vs actual")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_v24_prediction.png", dpi=150)
    plt.close()
    resid = y - p
    plt.figure(figsize=(6, 5))
    plt.scatter(p, resid, s=20)
    plt.axhline(0, ls="--", color="grey")
    plt.xlabel("Predicted ΔV24 (kt)"); plt.ylabel("Residual (actual - predicted)")
    plt.title("ΔV24 residuals")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_v24_residuals.png", dpi=150)
    plt.close()
    print(f"  saved -> figures/delta_v24_prediction.png, "
          f"figures/delta_v24_residuals.png")


def plot_shap(shap_imp, out_dir):
    top = shap_imp.head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top)), top["mean_abs_shap"], color="steelblue")
    plt.yticks(range(len(top)), top["feature"])
    plt.gca().invert_yaxis()
    plt.xlabel("mean |SHAP|")
    plt.title("SHAP mean |value| — best engineered IMD model")
    plt.tight_layout()
    plt.savefig(out_dir / "imd_engineered_shap.png", dpi=150)
    plt.close()
    print("  saved -> figures/imd_engineered_shap.png")


def verdict(pr_base, pr_eng, boot):
    d = pr_eng - pr_base
    lo = boot.get("ci_2_5")
    hi = boot.get("ci_97_5")
    if lo is None:
        return "4. Results are inconclusive (bootstrap CI unavailable)."
    if d > 0.02 and lo > 0:
        return (f"1. Engineered features provide evidence of improvement "
                f"(dPR={d:+.4f}, CI=[{lo:+.4f}, {hi:+.4f}]).")
    if d <= -0.005 and hi < 0:
        return (f"3. Engineered features do not improve the baseline "
                f"(dPR={d:+.4f}, CI=[{lo:+.4f}, {hi:+.4f}]).")
    if d <= -0.005:
        return (f"2b. Engineered features are no better than the baseline "
                f"and possibly worse (dPR={d:+.4f}, CI=[{lo:+.4f}, "
                f"{hi:+.4f}], CI spans 0).")
    return (f"2a. Engineered features provide at most a modest, uncertain "
            f"improvement (dPR={d:+.4f}, CI=[{lo:+.4f}, {hi:+.4f}], "
            f"CI spans 0).")


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = load_config()
    seed = get_seed(cfg)

    print("=" * 78)
    print("IMD FEATURE ENGINEERING + DELTA-V24 EXPERIMENT")
    print("=" * 78)
    print(f"seed={seed}  RI = 24h wind increase >= {cfg['ri']['threshold_kt']} kt")

    # ---- data + fixed split (identical to frozen baseline) ----------------
    imd = data_mod.load_imd(cfg)
    split = data_mod.split_by_storms(imd, cfg)
    split.verify()
    combined = data_mod.build_combined_imd_era5(cfg)

    # TASK 2: audit current IMD features.
    imd_feats = feat_mod.imd_feature_columns()
    print("\n--- TASK 2: AUDIT CURRENT IMD FEATURES ---")
    for i in range(0, len(imd_feats), 4):
        print("   " + " | ".join(f"{c:<28}" for c in imd_feats[i:i + 4]))
    assert not (set(imd_feats) & FORBIDDEN_PREDICTORS)

    # ---- TASK 1: freeze baseline (on the strict common test set) ----------
    from src import evaluate as ev
    # The strict common test set: rows of the combined (IMD+ERA5) frame whose
    # storms fall in the test split, then the matching rows in the CANONICAL
    # imd frame. Identical construction to run_final_imd_era5_comparison.py.
    combined_split = data_mod.Split(
        train=combined[combined["storm_id"].isin(split.train_storms)].copy(),
        val=combined[combined["storm_id"].isin(split.val_storms)].copy(),
        test=combined[combined["storm_id"].isin(split.test_storms)].copy(),
        train_storms=split.train_storms,
        val_storms=split.val_storms,
        test_storms=split.test_storms,
    )
    combined_test = combined_split.test
    common_keys = set(zip(combined_test["storm_id"], combined_test["datetime_utc"]))
    print("\n--- TASK 1: FREEZE BASELINE ---")
    # The strict common test set on the CANONICAL imd frame:
    # 174 obs / 20 storms / 25 RI.
    common_imd = imd[
        imd.apply(lambda r: (r["storm_id"], r["datetime_utc"]) in common_keys,
                  axis=1)].copy()
    assert len(common_imd) == 174, f"expected 174 common-test obs, got {len(common_imd)}"
    assert int((common_imd["RI_24h"] == 1).sum()) == 25
    assert common_imd["storm_id"].nunique() == 20

    frozen_model = load_xgb(MODELS / "imd_xgboost.json")
    X_va, y_va, _ = feat_mod.prepare_features(split.val, imd_feats)
    X_va = X_va.reindex(columns=imd_feats)
    thr_frozen = ev.tune_threshold(
        np.asarray(y_va).astype(int),
        frozen_model.predict_proba(X_va)[:, 1],
        criterion=cfg["evaluate"]["threshold_criterion"],
        grid_step=cfg["evaluate"]["threshold_grid"], seed=seed)
    X_ct, y_ct, _ = feat_mod.prepare_features(common_imd, imd_feats)
    X_ct = X_ct.reindex(columns=imd_feats)
    p_frozen = frozen_model.predict_proba(X_ct)[:, 1]
    m_frozen = ev.classification_metrics(np.asarray(y_ct).astype(int), p_frozen,
                                         thr_frozen)

    frozen = {
        "experiment": "IMD baseline frozen (pre feature engineering)",
        "date": str(pd.Timestamp.now().date()),
        "dataset": str(cfg["paths"]["imd_file"]),
        "observations": int(len(imd)),
        "storms": int(imd["storm_id"].nunique()),
        "RI_count": int((imd["RI_24h"] == 1).sum()),
        "feature_list": imd_feats,
        "random_seed": seed,
        "split": {
            "train": {"storms": int(len(split.train_storms)),
                      "obs": int(len(split.train)),
                      "RI": int((split.train["RI_24h"] == 1).sum())},
            "val": {"storms": int(len(split.val_storms)),
                    "obs": int(len(split.val)),
                    "RI": int((split.val["RI_24h"] == 1).sum())},
            "test": {"storms": int(len(split.test_storms)),
                     "obs": int(len(split.test)),
                     "RI": int((split.test["RI_24h"] == 1).sum())},
            "test_storm_hash": snow_hash(split.test_storms),
            "common_test_storms_hash": snow_hash(common_imd["storm_id"]),
            "common_test_observations": int(len(common_imd)),
            "common_test_RI": int((common_imd["RI_24h"] == 1).sum()),
        },
        "model_params": {k: cfg["imd_model"][k] for k in
                         ("model_type", "n_estimators", "learning_rate",
                          "max_depth", "min_child_weight", "subsample",
                          "colsample_bytree", "early_stopping_rounds",
                          "cv_folds", "metric")},
        "threshold": thr_frozen,
        "evaluation_common_test": ev.flatten_metric_name(m_frozen),
        "frozen_era5": {"pr_auc": 0.2969, "roc_auc": 0.7047},
        "frozen_imd_era5": {"pr_auc": 0.3411, "roc_auc": 0.7462},
    }
    with open(RESULTS / "imd_baseline_frozen.json", "w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=2, default=str)
    shutil.copyfile(MODELS / "imd_xgboost.json", MODELS / "imd_baseline_frozen.json")
    print(f"  frozen baseline PR-AUC={m_frozen['pr_auc']:.4f}  "
          f"ROC={m_frozen['roc_auc']:.4f}  thr={thr_frozen:.3f}")
    print(f"  common-test storms hash: "
          f"{frozen['split']['common_test_storms_hash']}  "
          f"({common_imd['storm_id'].nunique()} storms)")
    print(f"  saved -> results/imd_baseline_frozen.json, "
          f"models/imd_baseline_frozen.json")
    baseline_pr_frozen = m_frozen["pr_auc"]

    # ---- TASK 3: engineer features ----------------------------------------
    print("\n--- TASK 3: ENGINEER PHYSICAL / TRACK FEATURES ---")
    eng = engineer_features(imd)
    common_test = eng[
        eng.apply(lambda r: (r["storm_id"], r["datetime_utc"]) in common_keys,
                  axis=1)].copy()
    assert len(common_test) == 174
    assert common_test["storm_id"].nunique() == 20
    assert int((common_test["RI_24h"] == 1).sum()) == 25
    common_test = common_test.sort_values(
        ["storm_id", "datetime_utc"]).reset_index(drop=True)

    temporal_feats = ["wind_change_3h", "acceleration_6h", "acceleration_12h"]
    physical_feats = (temporal_feats
                      + ["pressure_change_3h", "pressure_change_6h",
                         "pressure_change_12h", "pressure_change_24h",
                         "pressure_acceleration_6h",
                         "translation_speed", "translation_u",
                         "translation_v",
                         "latitude_change_6h", "longitude_change_6h",
                         "latitude_change_12h", "longitude_change_12h"])
    set_a = list(imd_feats)
    set_b = imd_feats + temporal_feats
    set_c = imd_feats + physical_feats
    feature_sets = {"A-baseline": set_a,
                    "B-temporal": set_b,
                    "C-physical": set_c}
    for name, feats in feature_sets.items():
        new = [f for f in feats if f not in set(set_a)]
        print(f"  SET {name}: {len(feats)} predictors "
              f"(new: {len(new)} -> {new})")
    for name, feats in feature_sets.items():
        hit = sorted(set(feats) & FORBIDDEN_PREDICTORS)
        assert not hit, f"[FATAL] forbidden predictors in SET {name}: {hit}"
    print("  LEAKAGE CHECK PASS: no forbidden/future predictors in any set.")

    eng.to_csv(RESULTS / "imd_engineered_dataset.csv", index=False)
    print(f"  saved -> results/imd_engineered_dataset.csv ({len(eng)} rows)")

    # ---- TASK 5: feature quality check ------------------------------------
    print("\n--- TASK 5: FEATURE QUALITY CHECK ---")
    cols = sorted({c for s in feature_sets.values() for c in s})
    rows = []
    rejected = []
    for c in cols:
        x = eng[c].replace([np.inf, -np.inf], np.nan)
        missing = float(x.isna().mean())
        const = x.nunique(dropna=True) <= 1
        rec = {
            "feature": c,
            "set": next(n for n, s in feature_sets.items() if c in s),
            "missing_pct": round(100 * missing, 2),
            "min": round(float(x.min()), 3) if x.notna().any() else np.nan,
            "max": round(float(x.max()), 3) if x.notna().any() else np.nan,
            "mean": round(float(x.mean()), 3) if x.notna().any() else np.nan,
            "std": round(float(x.std()), 3) if x.notna().any() else np.nan,
            "nunique": int(x.nunique(dropna=True)),
        }
        if missing == 1.0 or const or missing > 0.95:
            reasons = []
            if missing == 1.0:
                reasons.append("all-NaN")
            if const:
                reasons.append("constant")
            if 0 < missing <= 0.95:
                reasons.append(">95% missing")
            rec["reject"] = "; ".join(reasons)
            rejected.append(c)
        else:
            rec["reject"] = ""
        rows.append(rec)
    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS / "imd_engineered_feature_audit.csv", index=False)
    print(f"  {len(cols)} candidate features; {len(rejected)} rejected: "
          f"{rejected}" if rejected else f"  {len(cols)} candidate features; none rejected")
    print(audit.to_string(index=False))
    print(f"  saved -> results/imd_engineered_feature_audit.csv")

    # ---- TASK 7: identical split for all sets -----------------------------
    h_a = snow_hash(common_test["storm_id"])
    print(f"  common-test storms hash (all SETs): {h_a} (identical by construction)")

    # ---- TASKS 8-9: train + evaluate --------------------------------------
    print("\n--- TASKS 8-9: TRAIN + EVALUATE CLASSIFIERS (storm-safe, seed 42) ---")
    from src import models as model_mod
    from sklearn.metrics import average_precision_score, roc_auc_score

    models = {}
    thresholds = {}
    test_probs = {}
    val_metrics = {}
    for name, feats in feature_sets.items():
        X_tr, y_tr, X_va, y_va, _X_te, _y_te, use = build_matrices(eng, split, feats)
        g_tr = eng.loc[eng["storm_id"].isin(split.train_storms),
                       "storm_id"].to_numpy()[:len(X_tr)]
        model = model_mod.train_xgboost(
            X_tr, y_tr, g_tr, X_va, y_va, _X_te, _y_te, cfg, "imd", seed)
        thr = ev.tune_threshold(
            np.asarray(y_va).astype(int), model.predict_proba(X_va)[:, 1],
            criterion=cfg["evaluate"]["threshold_criterion"],
            grid_step=cfg["evaluate"]["threshold_grid"], seed=seed)
        models[name] = model
        thresholds[name] = thr
        vm = ev.classification_metrics(np.asarray(y_va).astype(int),
                                       model.predict_proba(X_va)[:, 1], thr)
        val_metrics[name] = {"pr_auc": vm["pr_auc"], "roc_auc": vm["roc_auc"],
                             "f1": vm["f1"], "recall": vm["recall"], "thr": thr}
        print(f"  SET {name:>12}: VAL  PR={vm['pr_auc']:.4f} "
              f"ROC={vm['roc_auc']:.4f} F1={vm['f1']:.3f} thr={thr:.3f}")

    print(f"\n  Strict common test ({len(common_test)} obs / "
          f"{common_test['storm_id'].nunique()} storms / "
          f"{int((common_test['RI_24h']==1).sum())} RI):")
    y_common = np.asarray(common_test["RI_24h"]).astype(int)
    groups = common_test["storm_id"].to_numpy()
    for name, feats in feature_sets.items():
        X_ct2, y_ct2, _ = feat_mod.prepare_features(common_test, feats)
        p = models[name].predict_proba(X_ct2)[:, 1]
        test_probs[name] = p
        m = ev.classification_metrics(y_common, p, thresholds[name])
        print(f"  SET {name:>12}: TEST PR={m['pr_auc']:.4f} "
              f"ROC={m['roc_auc']:.4f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f} Brier={m['brier']:.4f} "
              f"thr={thresholds[name]:.3f}")

    pr_a = average_precision_score(y_common, test_probs["A-baseline"])
    pr_b = average_precision_score(y_common, test_probs["B-temporal"])
    pr_c = average_precision_score(y_common, test_probs["C-physical"])
    roc_a = roc_auc_score(y_common, test_probs["A-baseline"])
    roc_c = roc_auc_score(y_common, test_probs["C-physical"])

    comp_rows = []
    for name, feats in feature_sets.items():
        m = ev.classification_metrics(y_common, test_probs[name],
                                      thresholds[name])
        comp_rows.append({
            "Model": name,
            "N_test": len(common_test),
            "N_test_storms": common_test["storm_id"].nunique(),
            "N_RI": int((common_test["RI_24h"] == 1).sum()),
            "ROC_AUC": round(m["roc_auc"], 4),
            "PR_AUC": round(m["pr_auc"], 4),
            "Precision": round(m["precision"], 3),
            "Recall": round(m["recall"], 3),
            "F1": round(m["f1"], 3),
            "Brier": round(m["brier"], 4),
            "threshold": round(thresholds[name], 3),
        })
    comp = pd.DataFrame(comp_rows)
    comp["Delta_PR_AUC_vs_A"] = (comp["PR_AUC"]
                                 - comp["PR_AUC"].iloc[0])
    comp["Delta_ROC_AUC_vs_A"] = (comp["ROC_AUC"]
                                  - comp["ROC_AUC"].iloc[0])
    comp.to_csv(RESULTS / "imd_feature_engineering_comparison.csv", index=False)
    print(f"\n  saved -> results/imd_feature_engineering_comparison.csv")
    print(comp.to_string(index=False))

    # ---- TASK 10: deltas + storm-block bootstrap --------------------------
    print("\n--- TASK 10: ACTUAL IMPROVEMENT + BOOTSTRAP CI ---")
    boot = {}
    for name in ("B-temporal", "C-physical"):
        b = storm_block_bootstrap_pr_auc(
            y_common, test_probs["A-baseline"], test_probs[name],
            groups, n_boot=2000, seed=seed)
        boot[name] = b
        d = pr_b if name == "B-temporal" else pr_c
        base = pr_a
        print(f"  {name:>12}: dPR={d - base:+.4f}  95% CI="
              f"[{b['ci_2_5']:+.4f}, {b['ci_97_5']:+.4f}] "
              f"({b['n_valid']} resamples)" if b["ci_2_5"] is not None
              else f"  {name:>12}: dPR={d - base:+.4f}  CI unavailable")
    with open(RESULTS / "imd_feature_engineering_bootstrap.json", "w",
              encoding="utf-8") as fh:
        json.dump({"seed": seed, "n_boot": 2000, "deltas": boot,
                   "baseline_pr_auc": pr_a}, fh, indent=2)

    # ---- TASK 11: feature importance (best by VALIDATION) -----------------
    best_val = max(val_metrics, key=lambda n: val_metrics[n]["roc_auc"])
    print(f"\n--- TASK 11: FEATURE IMPORTANCE (best by VALIDATION = {best_val}) ---")
    best_model = models[best_val]
    best_feats = feature_sets[best_val]
    imp = best_model.get_booster().get_score(importance_type="gain")
    col_map = {f"f{i}": c for i, c in enumerate(best_feats)}
    imp_df = (pd.DataFrame(
        [{"feature": col_map.get(k, k), "importance": float(v)}
         for k, v in imp.items()])
        .sort_values("importance", ascending=False).reset_index(drop=True))
    imp_df.to_csv(RESULTS / "imd_engineered_feature_importance.csv", index=False)
    print(imp_df.head(15).to_string(index=False))
    print(f"  saved -> results/imd_engineered_feature_importance.csv")

    shap_imp = None
    try:
        import shap
        X_s, _, _ = feat_mod.prepare_features(common_test, best_feats)
        X_s = X_s[best_feats]
        X_s.columns = best_feats
        best_model.get_booster().feature_names = best_feats
        explainer = shap.TreeExplainer(best_model)
        sv = explainer.shap_values(X_s)
        if isinstance(sv, list):
            sv = sv[1]
        shap_imp = (pd.DataFrame(
            {"feature": best_feats,
             "mean_abs_shap": np.abs(sv).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))
        shap_imp.to_csv(RESULTS / "imd_engineered_shap_importance.csv",
                        index=False)
        print(f"  saved -> results/imd_engineered_shap_importance.csv")
        print(shap_imp.head(15).to_string(index=False))
    except Exception as exc:
        print(f"  [shap] unavailable: {type(exc).__name__}: {exc}")

    # ---- TASK 12: error analysis ------------------------------------------
    print("\n--- TASK 12: ERROR ANALYSIS ---")
    thr_a = thresholds["A-baseline"]
    thr_c = thresholds["C-physical"]
    p_a = test_probs["A-baseline"]
    p_c = test_probs["C-physical"]
    err_rows = []
    for i in range(len(common_test)):
        la = int(p_a[i] >= thr_a)
        lc = int(p_c[i] >= thr_c)
        y = int(common_test.iloc[i]["RI_24h"])
        if y == 1:
            cat = ("A_RI_correct_both" if la == 1 and lc == 1 else
                   "B_RI_missed_by_engineered" if la == 1 and lc == 0 else
                   "B2_RI_caught_by_engineered_only" if la == 0 and lc == 1 else
                   "B_RI_missed_both")
        else:
            cat = ("C_nonRI_correct_both" if la == 0 and lc == 0 else
                   "C2_nonRI_false_flag_engineered_only" if la == 0 and lc == 1 else
                   "C3_nonRI_false_flag_baseline_only" if la == 1 and lc == 0 else
                   "C_nonRI_false_flag_both")
        err_rows.append({
            "storm_id": common_test.iloc[i]["storm_id"],
            "datetime_utc": common_test.iloc[i]["datetime_utc"],
            "RI_24h": y,
            "P_baseline": round(float(p_a[i]), 4),
            "P_physical": round(float(p_c[i]), 4),
            "pred_baseline": la,
            "pred_physical": lc,
            "category": cat,
        })
    err = pd.DataFrame(err_rows)
    err.to_csv(RESULTS / "imd_error_analysis.csv", index=False)
    print(f"  category counts:\n{err['category'].value_counts().to_string()}")
    print(f"  saved -> results/imd_error_analysis.csv")

    # ---- TASKS 13-14: delta-V24 regression --------------------------------
    print("\n--- TASKS 13-14: DELTA-V24 REGRESSION ---")
    assert "delta_v_24h_kt" not in (set(set_a) | set(set_b) | set(set_c)), \
        "delta_v_24h_kt must never be a feature"
    reg_feat_sets = {"A-baseline": set_a, "C-physical": set_c}
    reg_results, pred_full = run_regression(
        eng, split, reg_feat_sets, "delta_v_24h_kt", common_test, seed)

    d24 = write_delta_predictions(pred_full, common_test)
    d24.to_csv(RESULTS / "delta_v24_predictions.csv", index=False)
    print(f"  saved -> results/delta_v24_predictions.csv")

    # ---- TASK 15: combined output ------------------------------------------
    print("\n--- TASK 15: COMBINED OUTPUT (P(RI) + expected DeltaV24) ---")
    combo = common_test[["storm_id", "datetime_utc", "RI_24h"]].copy()
    combo["P_RI"] = p_c
    combo["expected_delta_v24_kt"] = d24["expected_delta_v24_kt"].to_numpy()
    combo["actual_delta_v24_kt"] = d24["actual_delta_v24_kt"].to_numpy()
    print(combo.head(8).to_string(index=False))

    # ---- TASK 17: calibration (validation-only fit) ------------------------
    print("\n--- TASK 17: CALIBRATION ---")
    cal = run_calibration(best_model, best_val, eng, split, best_feats,
                          common_test, test_probs[best_val], y_common, FIGURES)

    # ---- TASK 16 (optional multi-task): not warranted by architecture ------
    print("\n--- TASK 16: optional multi-task --------------------------------------------------")
    print("  XGBoost does not share representations across tasks. A genuine")
    print("  multi-task net would need a new architecture; per spec it is")
    print("  treated as an EXPERIMENT and skipped: the two validated single-")
    print("  task models (classification + regression) are the deliverables.")

    # ---- TASK 18: final model selection via VALIDATION ----------------------
    # (best_val already chosen above from validation ROC-AUC; formalised here)
    print(f"\n--- TASK 18: FINAL MODEL SELECTION (by VALIDATION, not test) ---")
    print(f"  selected: {best_val}  (val ROC-AUC {val_metrics[best_val]['roc_auc']:.4f}, "
          f"val F1 {val_metrics[best_val]['f1']:.3f})")
    best_model.get_booster().save_model(str(MODELS / "imd_engineered_xgboost.json"))
    print(f"  saved -> models/imd_engineered_xgboost.json (best-engineered "
          f"classifier, {best_val})")

    # ---- TASK 19: comparison with frozen ERA5 -------------------------------
    print("\n--- TASK 19: STRUCTURED COMPARISON WITH FROZEN ERA5 ---")
    cmp_rows = [
        {"Model": "Old IMD (frozen artifact)", "PR_AUC": baseline_pr_frozen,
         "ROC_AUC": frozen["evaluation_common_test"]["roc_auc"],
         "Source": "frozen JSON (common test)"},
        {"Model": "New IMD SET A (retrained)", "PR_AUC": round(pr_a, 4),
         "ROC_AUC": round(roc_a, 4), "Source": "this run (common test)"},
        {"Model": "New IMD SET C (physical)", "PR_AUC": round(pr_c, 4),
         "ROC_AUC": round(roc_c, 4), "Source": "this run (common test)"},
        {"Model": "ERA5 (frozen)", "PR_AUC": frozen["frozen_era5"]["pr_auc"],
         "ROC_AUC": frozen["frozen_era5"]["roc_auc"], "Source": "frozen JSON"},
        {"Model": "IMD+ERA5 (frozen)", "PR_AUC": frozen["frozen_imd_era5"]["pr_auc"],
         "ROC_AUC": frozen["frozen_imd_era5"]["roc_auc"], "Source": "frozen JSON"},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    print(cmp_df.to_string(index=False))

    # ---- FIGURES -------------------------------------------------------------
    print("\n--- FIGURES ---")
    plot_classification_figures(y_common, test_probs, "engineered", FIGURES)
    plot_regression_figures(d24, FIGURES)
    if shap_imp is not None:
        plot_shap(shap_imp, FIGURES)

    # ---- TASK 22: FINAL SCIENTIFIC VERDICT ----------------------------------
    print("\n" + "=" * 78)
    print("TASK 22: FINAL SCIENTIFIC VERDICT")
    print("=" * 78)
    print(f"  Baseline PR-AUC      : {pr_a:.4f}")
    print(f"  Engineered PR-AUC    : {pr_c:.4f}")
    print(f"  Delta PR-AUC         : {pr_c - pr_a:+.4f}")
    print(f"  Baseline ROC-AUC     : {roc_a:.4f}")
    print(f"  Engineered ROC-AUC   : {roc_c:.4f}")
    print(f"  Delta ROC-AUC        : {roc_c - roc_a:+.4f}")
    bb = boot["C-physical"]
    if bb["ci_2_5"] is not None:
        print(f"  Delta PR-AUC 95% CI : [{bb['ci_2_5']:+.4f}, {bb['ci_97_5']:+.4f}]")
    print("\n  " + "=" * 70)
    print(f"  FINAL VERDICT: {verdict(pr_a, pr_c, bb)}")
    print("  " + "=" * 70)

    # ---- TASK 23 documentation hook (docs updated by the caller/this script) --
    print("\n" + "=" * 78)
    print("EXPERIMENT COMPLETE — reproduction:")
    print("  python3 run_imd_feature_engineering.py")
    print("=" * 78)


if __name__ == "__main__":
    main()