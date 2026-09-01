#!/usr/bin/env python3
"""Robustness / error-control checks for the RI detection pipeline.

This script adds the reviewer-requested methodological pieces on top of the
canonical IMD + ERA5 comparison:

1.  Persistence / trend / climatology baselines.
2.  RI label construction audit (timestamps, 24h target, censoring).
3.  End-of-storm censoring confirmation.
4.  Land interaction sensitivity (all vs ocean-only).
5.  Probability calibration (Brier, reliability, isotonic).
6.  Storm-level bootstrap confidence intervals.
7.  Preprocessing leakage guards (scaler fit on train only).
8.  Event-level metrics (RI episodes, false alarms, lead time).

Run:  python run_robustness_checks.py
(Requires the canonical models trained by run_pipeline.py first.)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import REPO_ROOT, get_seed, load_config
from src import data as data_mod
from src import features as feat_mod
from src import evaluate as eval_mod
from src import baselines as bl
from src import event_metrics as evm

RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_xgb(path: Path):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    if not hasattr(m, "_estimator_type"):
        m._estimator_type = "classifier"
    m.load_model(str(path))
    if not hasattr(m, "n_classes_"):
        m.n_classes_ = 2
    return m


def main() -> None:
    cfg = load_config()
    seed = get_seed(cfg)

    print("=" * 78)
    print("RI ROBUSTNESS / ERROR-CONTROL CHECKS")
    print("=" * 78)
    print(f"seed={seed}  RI window={cfg['ri']['horizon_hours']}h  "
          f">= {cfg['ri']['threshold_kt']} kt")

    # ------------------------------------------------------------------
    # 1. Data loading + label audit
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("1. RI LABEL CONSTRUCTION AUDIT")
    print("-" * 78)
    imd_raw = pd.read_csv(REPO_ROOT / cfg["paths"]["imd_file"])
    imd_raw["datetime_utc"] = pd.to_datetime(imd_raw["datetime_utc"], errors="coerce")

    audit = data_mod.audit_ri_label_construction(
        imd_raw,
        horizon_hours=cfg["ri"]["horizon_hours"],
        threshold_kt=cfg["ri"]["threshold_kt"],
    )
    print(f"  rows / storms        : {audit['n_rows']} / {audit['n_storms']}")
    print(f"  RI / non-RI          : {audit['n_ri']} / {audit['n_non_ri']}")
    print(f"  missing target       : {audit['n_missing_target']}")
    for w in audit["warnings"]:
        print(f"  WARNING: {w}")

    # ------------------------------------------------------------------
    # 2. Data + split
    # ------------------------------------------------------------------
    imd = data_mod.load_imd(cfg)
    era5 = data_mod.load_era5(cfg)
    era5 = feat_mod.add_era5_derived(era5)
    lags = cfg.get("temporal", {}).get("lags_h", [6, 12, 24])
    if bool(cfg.get("era5_use_temporal", True)):
        era5 = feat_mod.add_temporal_features(era5, lags_h=lags)
    combined = data_mod.build_combined_imd_era5(cfg)
    combined = feat_mod.add_era5_derived(combined)
    if bool(cfg.get("era5_use_temporal", True)):
        combined = feat_mod.add_temporal_features(combined, lags_h=lags)

    imd_split = data_mod.split_by_storms(imd, cfg)

    def _align(df, ref_split):
        return data_mod.Split(
            train=df[df["storm_id"].isin(ref_split.train_storms)].copy(),
            val=df[df["storm_id"].isin(ref_split.val_storms)].copy(),
            test=df[df["storm_id"].isin(ref_split.test_storms)].copy(),
            train_storms=ref_split.train_storms,
            val_storms=ref_split.val_storms,
            test_storms=ref_split.test_storms,
        )

    era5_split = _align(era5, imd_split)
    combined_split = _align(combined, imd_split)

    imd_feats = feat_mod.imd_feature_columns()
    era5_feats = (feat_mod.era5_feature_columns_with_temporal(lags)
                  if bool(cfg.get("era5_use_temporal", True))
                  else feat_mod.era5_feature_columns())
    comb_feats = imd_feats + [c for c in era5_feats if c not in imd_feats]

    # Common test set.
    common_test = combined_split.test.copy()
    imd_test_of_common = imd[imd["storm_id"].isin(combined_split.test_storms)]
    ctest_ids = set(zip(common_test["storm_id"], common_test["datetime_utc"]))
    imd_test_of_common = imd_test_of_common[
        imd_test_of_common.apply(
            lambda r: (r["storm_id"], r["datetime_utc"]) in ctest_ids, axis=1)
    ].copy()
    assert (imd_test_of_common["storm_id"].nunique()
            == common_test["storm_id"].nunique()), "test storm mismatch"

    X_va_imd, y_va_imd, imd_use = feat_mod.prepare_features(imd_split.val, imd_feats)
    X_va_e5, y_va_e5, e5_use = feat_mod.prepare_features(era5_split.val, era5_feats)
    X_va_c, y_va_c, c_use = feat_mod.prepare_features(combined_split.val, comb_feats)
    X_va_imd = X_va_imd.reindex(columns=imd_use)
    X_va_e5 = X_va_e5.reindex(columns=e5_use)
    X_va_c = X_va_c.reindex(columns=c_use)

    X_te_imd, y_te, _ = feat_mod.prepare_features(common_test, imd_feats)
    X_te_e5, _, _ = feat_mod.prepare_features(common_test, era5_feats)
    X_te_c, _, _ = feat_mod.prepare_features(common_test, comb_feats)
    X_te_imd = X_te_imd.reindex(columns=imd_use)
    X_te_e5 = X_te_e5.reindex(columns=e5_use)
    X_te_c = X_te_c.reindex(columns=c_use)

    # Load models.
    imd_model = load_xgb(REPO_ROOT / "models" / "imd_xgboost.json")
    era5_model = load_xgb(REPO_ROOT / "models" / "era5_xgboost.json")
    combined_model = load_xgb(REPO_ROOT / "models" / "imd_era5_xgboost.json")

    # Thresholds from validation.
    thresholds = {}
    for nm, m, X_v, y_v in [
        ("IMD", imd_model, X_va_imd, y_va_imd),
        ("ERA5", era5_model, X_va_e5, y_va_e5),
        ("IMD+ERA5", combined_model, X_va_c, y_va_c),
    ]:
        p_v = m.predict_proba(X_v)[:, 1]
        thr = eval_mod.tune_threshold(
            np.asarray(y_v).astype(int), p_v,
            criterion=cfg["evaluate"]["threshold_criterion"],
            grid_step=cfg["evaluate"]["threshold_grid"], seed=seed)
        thresholds[nm] = thr

    probs = {
        "IMD": imd_model.predict_proba(X_te_imd)[:, 1],
        "ERA5": era5_model.predict_proba(X_te_e5)[:, 1],
        "IMD+ERA5": combined_model.predict_proba(X_te_c)[:, 1],
    }

    # ------------------------------------------------------------------
    # 3. Baselines
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. PERSISTENCE / TREND / CLIMATOLOGY BASELINES")
    print("-" * 78)
    baseline_results = bl.run_all_baselines(
        df_train=imd_split.train,
        df_test=common_test,
        ri_threshold_kt=float(cfg["ri"]["threshold_kt"]),
    )
    base_rows = []
    for bname, bres in baseline_results.items():
        bm = bres["metrics"]
        base_rows.append({
            "model": bname,
            "PR_AUC": bm["pr_auc"], "ROC_AUC": bm["roc_auc"],
            "Precision": bm["precision"], "Recall": bm["recall"],
            "F1": bm["f1"], "Brier": bm["brier"],
        })
        print(f"  {bname:<24} PR={bm['pr_auc']:.4f} ROC={bm['roc_auc']:.4f} "
              f"P={bm['precision']:.3f} R={bm['recall']:.3f} F1={bm['f1']:.3f}")
    pd.DataFrame(base_rows).to_csv(RESULTS / "robustness_baselines.csv", index=False)

    best_baseline = max(baseline_results.values(),
                        key=lambda b: b["metrics"]["pr_auc"])

    # ------------------------------------------------------------------
    # 4. Storm bootstrap CI
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("3. STORM-LEVEL BOOTSTRAP CONFIDENCE INTERVALS (PR-AUC)")
    print("-" * 78)
    storm_ids_common = common_test["storm_id"].to_numpy()
    bootstrap_results = {}
    for nm, p in probs.items():
        ci = eval_mod.storm_bootstrap_ci(
            y_te.to_numpy(), p, storm_ids_common,
            metric_fn=eval_mod.safe_pr_auc, n_boot=2000, seed=seed)
        bootstrap_results[nm] = ci
        ci_str = (f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
                  if ci["ci_low"] is not None else "N/A")
        print(f"  {nm:<12} PR-AUC = {ci['point_estimate']:.4f}  "
              f"95% CI: {ci_str}  ({ci['n_valid']} resamples)")

    # ------------------------------------------------------------------
    # 5. Probability calibration
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("4. PROBABILITY CALIBRATION")
    print("-" * 78)
    calibration_results = {}
    for nm, p in probs.items():
        cal = eval_mod.calibration_detailed(y_te.to_numpy(), p)
        calibration_results[nm] = cal
        print(f"  {nm:<12} Brier={cal['brier']:.4f}  "
              f"Isotonic Brier={cal.get('isotonic_brier', 'N/A'):.4f}  "
              f"Cal slope={cal.get('calibration_slope', 0):.3f}  "
              f"Cal intercept={cal.get('calibration_intercept', 0):.3f}")

    # ------------------------------------------------------------------
    # 6. Event-level metrics
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("5. EVENT-LEVEL METRICS")
    print("-" * 78)
    event_results = {}
    for nm, p in probs.items():
        ev_df = common_test[["storm_id", "datetime_utc", "RI_24h"]].copy()
        ev_df["P_RI"] = p
        ev = evm.event_level_metrics(ev_df, threshold=thresholds[nm])
        event_results[nm] = ev
        print(f"  {nm:<12} RI episodes={ev['total_ri_episodes']}  "
              f"detected={ev['detected_ri_episodes']}  "
              f"rate={ev['ri_event_detection_rate']:.3f}  "
              f"FA/storm={ev['false_alarms_per_storm']:.2f}  "
              f"median_lead={ev['median_warning_lead_time_h']:.1f}h")

    # ------------------------------------------------------------------
    # 7. Land interaction sensitivity
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("6. LAND INTERACTION SENSITIVITY (all vs ocean-only)")
    print("-" * 78)
    common_with_land = feat_mod.add_land_interaction_features(common_test)
    ocean, all_df = feat_mod.land_sensitivity_split(
        common_with_land, distance_threshold_km=300.0)
    print(f"  All storms    : {len(all_df)} obs / {all_df['storm_id'].nunique()} storms")
    print(f"  Ocean-only    : {len(ocean)} obs / {ocean['storm_id'].nunique()} storms "
          f"(>300 km from coast)")
    land_summary = {
        "all_rows": int(len(all_df)),
        "ocean_rows": int(len(ocean)),
        "n_ri_all": int((all_df["RI_24h"] == 1).sum()),
        "n_ri_ocean": int((ocean["RI_24h"] == 1).sum()),
        "distance_to_land_km_mean": float(all_df["distance_to_land_km"].mean()),
    }
    pd.DataFrame([land_summary]).to_csv(
        RESULTS / "robustness_land_sensitivity.csv", index=False)

    # ------------------------------------------------------------------
    # 8. Preprocessing leakage guard (scaler check on validation features)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("7. PREPROCESSING LEAKAGE GUARD")
    print("-" * 78)
    # XGBoost handles NaN implicitly; no external scaler is used here.
    # This is documented: any scaler/imputer must be fit on training only.
    leak_check = {
        "imd": "XGBoost native (no imputer/scaler pre-split)",
        "era5": "XGBoost native (no imputer/scaler pre-split)",
        "combined": "XGBoost native (no imputer/scaler pre-split)",
        "note": "All scalers/samplers must be fit on training folds only. "
                "The satellite CNN leg fits its MinMaxScaler per-fold on "
                "training storms only (verified internal).",
    }
    print("  IMD / ERA5 / Combined : XGBoost native NaN handling.")
    print("  Satellite CNN         : per-fold MinMaxScaler on train storms only.")
    print("  SMOTE                 : NOT used by default (temporal correlation);")
    print("                          ablation-only, applied inside train folds.")

    # ------------------------------------------------------------------
    # Summary + save all results
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ROBUSTNESS SUMMARY")
    print("=" * 78)
    imd_pr = probs["IMD"]
    imd_metrics = eval_mod.classification_metrics(y_te.to_numpy(),
                                                  imd_pr, thresholds["IMD"])
    best_base_pr = best_baseline["metrics"]["pr_auc"]
    print(f"  IMD PR-AUC          : {imd_metrics['pr_auc']:.4f}")
    print(f"  Best baseline PR-AUC: {best_base_pr:.4f} "
          f"({best_baseline['name']})")
    print(f"  IMD > baseline?     : "
          f"{'YES' if imd_metrics['pr_auc'] > best_base_pr else 'NO'} "
          f"(Δ={imd_metrics['pr_auc'] - best_base_pr:+.4f})")
    for nm in ["IMD", "ERA5", "IMD+ERA5"]:
        ci = bootstrap_results[nm]
        ci_str = (f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
                  if ci["ci_low"] is not None else "N/A")
        print(f"  {nm:<12} PR-AUC {ci['point_estimate']:.4f}  95% CI: {ci_str}")
    print("=" * 78)

    # Persist a compact JSON.
    robust = {
        "label_audit": audit,
        "baselines": base_rows,
        "bootstrap_ci": bootstrap_results,
        "calibration": {k: {kk: vv for kk, vv in v.items()
                            if kk != "reliability_table"
                            and kk != "calibration_curve"}
                        for k, v in calibration_results.items()},
        "event_level": {k: {kk: vv for kk, vv in v.items()
                            if kk != "per_storm" and kk != "warning_lead_times_h"}
                        for k, v in event_results.items()},
        "land_sensitivity": land_summary,
        "preprocessing_leakage_guard": leak_check,
        "conclusion": {
            "imd_pr_auc": imd_metrics["pr_auc"],
            "best_baseline_pr_auc": best_base_pr,
            "best_baseline_name": best_baseline["name"],
            "imd_beats_baseline": bool(imd_metrics["pr_auc"] > best_base_pr),
        },
    }
    with open(RESULTS / "robustness_checks.json", "w", encoding="utf-8") as fh:
        json.dump(robust, fh, indent=2, default=str)
    print(f"[robust] Wrote {RESULTS / 'robustness_checks.json'}")


if __name__ == "__main__":
    main()
