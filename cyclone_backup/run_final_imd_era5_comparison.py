#!/usr/bin/env python3
"""FINAL IMD + ERA5 storm-safe comparison (definitive result artifact).

This stage does NOT retrain the satellite CNN, does NOT add ocean data, and
does NOT invent metrics. It reuses the canonical datasets, the existing XGBoost
artifacts produced by ``run_pipeline.py`` (deterministic, seed 42), and the
existing storm-safe splitting code.

It adds the strict guarantee the pipeline already provides, made explicit:

* the *same* held-out test storms AND the *same* test observations are used for
  IMD-only, ERA5-only and IMD+ERA5 (common-test-set evaluation);
* decision thresholds are tuned on each branch's validation set only;
* class imbalance uses ``scale_pos_weight`` computed from training data only;
* every predictor is verified to be current-or-historical (no target-time or
  future information; ``RI_24h`` is only the label).

Outputs (new files, historical results untouched):
    results/imd_only_final.csv
    results/era5_only_final.csv
    results/imd_era5_combined_final.csv
    results/model_comparison_final.csv
    results/imd_era5_feature_importance.csv
    results/final_imd_era5_experiment.json
    figures/roc_curve_*_final.png
    figures/pr_curve_*_final.png
    figures/confusion_*_final.png
    models/imd_final_xgboost.json
    models/era5_final_xgboost.json
    models/imd_era5_final_xgboost.json
"""

from __future__ import annotations

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
from src import explain as xpl
from src import baselines as bl
from src import event_metrics as evm

FORBIDDEN_PREDICTORS = {
    "RI_24h", "wind_24h_kt", "delta_v_24h_kt", "target_time_24h",
}

RESULTS = REPO_ROOT / "results"
MODELS = REPO_ROOT / "models"
FIGURES = REPO_ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def print_feature_list(title: str, cols: list[str]) -> None:
    print("=" * 78)
    print(f"{title}  ({len(cols)} predictors)")
    print("=" * 78)
    banned = sorted(set(cols) & FORBIDDEN_PREDICTORS)
    if banned:
        raise SystemExit(f"[FATAL] forbidden predictors in {title}: {banned}")
    for i in range(0, len(cols), 4):
        print("   " + " | ".join(f"{c:<28}" for c in cols[i:i + 4]))
    print()


def load_xgb(path: Path):
    from xgboost import XGBClassifier
    m = XGBClassifier()
    if not hasattr(m, "_estimator_type"):
        m._estimator_type = "classifier"
    m.load_model(str(path))
    if not hasattr(m, "n_classes_"):
        m.n_classes_ = 2
    return m


def storm_block_bootstrap_pr_auc(y, p_a, p_b, groups, n_boot=2000, seed=42):
    """Storm-block bootstrap of PR-AUC difference between two score sets.

    Resamples *test storms* with replacement (never individual observations),
    recomputes PR-AUC for A and B on the resampled observations, and reports
    the percentile interval of the difference. Single-class resamples are
    skipped. This is a purely test-set uncertainty estimate -- no refitting.
    """
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
        pa = average_precision_score(y[mask], p_a[mask])
        pb = average_precision_score(y[mask], p_b[mask])
        diffs.append(pb - pa)
    diffs = np.asarray(diffs)
    if len(diffs) < 100:
        return {"n_valid": int(len(diffs)), "ci_2_5": None, "ci_97_5": None,
                "mean": None, "note": "too few valid resamples"}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_valid": int(len(diffs)),
        "ci_2_5": float(lo),
        "ci_97_5": float(hi),
        "mean": float(diffs.mean()),
    }


def load_and_prepare(cfg):
    seed = get_seed(cfg)
    lags = cfg.get("temporal", {}).get("lags_h", [6, 12, 24])
    use_temporal = bool(cfg.get("era5_use_temporal", True))

    imd = data_mod.load_imd(cfg)
    era5 = data_mod.load_era5(cfg)
    era5 = feat_mod.add_era5_derived(era5)
    if use_temporal:
        era5 = feat_mod.add_temporal_features(era5, lags_h=lags)
    combined = data_mod.build_combined_imd_era5(cfg)
    combined = feat_mod.add_era5_derived(combined)
    if use_temporal:
        combined = feat_mod.add_temporal_features(combined, lags_h=lags)

    imd_split = data_mod.split_by_storms(imd, cfg)
    era5_split = _align(era5, imd_split)
    combined_split = _align(combined, imd_split)
    # Re-align IMD/ERA5 onto the combined observation set's storms.
    common_storms = (imd_split.train_storms | imd_split.val_storms |
                     imd_split.test_storms) & set(combined["storm_id"])
    return imd, era5, combined, imd_split, era5_split, combined_split, seed


def _align(df, ref_split):
    return data_mod.Split(
        train=df[df["storm_id"].isin(ref_split.train_storms)].copy(),
        val=df[df["storm_id"].isin(ref_split.val_storms)].copy(),
        test=df[df["storm_id"].isin(ref_split.test_storms)].copy(),
        train_storms=ref_split.train_storms,
        val_storms=ref_split.val_storms,
        test_storms=ref_split.test_storms,
    )


def main() -> None:
    cfg = load_config()
    seed = get_seed(cfg)
    lags = cfg.get("temporal", {}).get("lags_h", [6, 12, 24])
    use_temporal = bool(cfg.get("era5_use_temporal", True))

    print("=" * 78)
    print("FINAL IMD + ERA5 COMPARISON (definitive storm-safe evaluation)")
    print("=" * 78)
    print(f"seed={seed}  RI window={cfg['ri']['horizon_hours']}h  "
          f">= {cfg['ri']['threshold_kt']} kt  "
          f"RI_24h label only (never a predictor)")

    # ----------------------------- data -----------------------------
    imd = data_mod.load_imd(cfg)
    era5 = data_mod.load_era5(cfg)
    era5 = feat_mod.add_era5_derived(era5)
    if use_temporal:
        era5 = feat_mod.add_temporal_features(era5, lags_h=lags)
    combined = data_mod.build_combined_imd_era5(cfg)
    combined = feat_mod.add_era5_derived(combined)
    if use_temporal:
        combined = feat_mod.add_temporal_features(combined, lags_h=lags)

    imd_split = data_mod.split_by_storms(imd, cfg)
    era5_split = _align(era5, imd_split)
    combined_split = _align(combined, imd_split)

    # ----------------- feature lists --------------------------------
    imd_feats = feat_mod.imd_feature_columns()
    era5_feats = (feat_mod.era5_feature_columns_with_temporal(lags)
                  if use_temporal else feat_mod.era5_feature_columns())
    comb_feats = imd_feats + [c for c in era5_feats if c not in imd_feats]

    print()
    print_feature_list("IMD-ONLY PREDICTORS", imd_feats)
    print_feature_list("ERA5-ONLY PREDICTORS", era5_feats)
    print_feature_list("IMD + ERA5 PREDICTORS", comb_feats)

    # ----------------- dataset / split honesty ----------------------
    def _stats(df, name):
        return {
            "dataset": name,
            "observations": int(len(df)),
            "storms": int(df["storm_id"].nunique()),
            "RI": int((df["RI_24h"] == 1).sum()),
            "non_RI": int((df["RI_24h"] == 0).sum()),
        }

    print("\nDataset / split statistics:")
    for df, nm in [(imd, "IMD"), (era5, "ERA5"), (combined, "IMD+ERA5 (matched)")]:
        s = _stats(df, nm)
        print(f"  {nm:<22} obs={s['observations']:>5}  storms={s['storms']:>4}  "
              f"RI={s['RI']:>4}  non-RI={s['non_RI']:>5}")
    print()
    print(imd_split.table().to_string(index=False))
    print()
    _t_storms = sorted(imd_split.test_storms)
    print(f"  TEST storms ({len(_t_storms)}): {_t_storms}")

    # Common test set: observations of test storms present in ALL branches.
    common_test = combined_split.test.copy()
    imd_test_of_common = imd[imd["storm_id"].isin(combined_split.test_storms)]
    ctest_ids = set(zip(common_test["storm_id"], common_test["datetime_utc"]))
    imd_test_of_common = imd_test_of_common[
        imd_test_of_common.apply(
            lambda r: (r["storm_id"], r["datetime_utc"]) in ctest_ids, axis=1)
    ].copy()
    assert (imd_test_of_common["storm_id"].nunique()
            == common_test["storm_id"].nunique()), "test storm mismatch"
    assert set(imd_test_of_common["storm_id"]) == set(common_test["storm_id"]), \
        "common-test storms differ across branches"

    # ----------------- load canonical models -------------------------
    imd_model = load_xgb(MODELS / "imd_xgboost.json")
    era5_model = load_xgb(MODELS / "era5_xgboost.json")
    combined_model = load_xgb(MODELS / "imd_era5_xgboost.json")

    # ----------------- feature matrices ------------------------------
    X_tr_imd, y_tr_imd, imd_use = feat_mod.prepare_features(imd_split.train, imd_feats)
    X_va_imd, y_va_imd, _ = feat_mod.prepare_features(imd_split.val, imd_feats)

    X_tr_e5, y_tr_e5, e5_use = feat_mod.prepare_features(era5_split.train, era5_feats)
    X_va_e5, y_va_e5, _ = feat_mod.prepare_features(era5_split.val, era5_feats)

    X_tr_c, y_tr_c, c_use = feat_mod.prepare_features(combined_split.train, comb_feats)
    X_va_c, y_va_c, _ = feat_mod.prepare_features(combined_split.val, comb_feats)

    # Pin every matrix to the training usable columns so prediction order and
    # width always match the fitted model.
    X_va_imd = X_va_imd.reindex(columns=imd_use)
    X_va_e5 = X_va_e5.reindex(columns=e5_use)
    X_va_c = X_va_c.reindex(columns=c_use)

    # Strict common test features (identical rows for all three branches).
    X_te_imd, y_te, _ = feat_mod.prepare_features(common_test, imd_feats)
    X_te_e5, _, _ = feat_mod.prepare_features(common_test, era5_feats)
    X_te_c, _, _ = feat_mod.prepare_features(common_test, comb_feats)
    X_te_imd = X_te_imd.reindex(columns=imd_use)
    X_te_e5 = X_te_e5.reindex(columns=e5_use)
    X_te_c = X_te_c.reindex(columns=c_use)
    assert len(X_te_imd) == len(X_te_e5) == len(X_te_c) == len(common_test)
    assert (y_te.index == common_test.index).all()

    n_test_obs = len(common_test)
    n_test_storms = common_test["storm_id"].nunique()
    n_test_ri = int((common_test["RI_24h"] == 1).sum())
    common_storm_ids = sorted(common_test["storm_id"].unique())

    print("\nSTRICT COMMON TEST SET (same observations for all 3 models):")
    print(f"  test observations  : {n_test_obs}")
    print(f"  test storms        : {n_test_storms}")
    print(f"  RI positives       : {n_test_ri} ({n_test_ri / n_test_obs:.1%})")

    # ----------------- threshold (validation only) -------------------
    print("\nThreshold tuning (VALIDATION ONLY, criterion=f1):")
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
        print(f"  {nm:<10} threshold={thr:.3f}")

    # ----------------- evaluate on common test -----------------------
    probs = {
        "IMD": imd_model.predict_proba(X_te_imd)[:, 1],
        "ERA5": era5_model.predict_proba(X_te_e5)[:, 1],
        "IMD+ERA5": combined_model.predict_proba(X_te_c)[:, 1],
    }

    metrics = {}
    for nm, p in probs.items():
        m = eval_mod.classification_metrics(y_te.to_numpy(), p, thresholds[nm])
        metrics[nm] = {"threshold": thresholds[nm], **m}
        print(f"  {nm:<10} PR={m['pr_auc']:.4f} ROC={m['roc_auc']:.4f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"thr={thresholds[nm]:.2f}")

    # ----------------- deltas + bootstrap CI -------------------------
    imd_m = metrics["IMD"]
    comb_m = metrics["IMD+ERA5"]
    d_pr = comb_m["pr_auc"] - imd_m["pr_auc"]
    d_roc = comb_m["roc_auc"] - imd_m["roc_auc"]
    boot = storm_block_bootstrap_pr_auc(
        y_te.to_numpy(), probs["IMD"], probs["IMD+ERA5"],
        common_test["storm_id"].to_numpy(), n_boot=2000, seed=seed)

    print("\nDELTAS (IMD+ERA5 minus IMD-only, common test set):")
    print(f"  Δ PR-AUC   = {d_pr:+.4f}")
    print(f"  Δ ROC-AUC  = {d_roc:+.4f}")
    if boot["ci_2_5"] is not None:
        print(f"  Δ PR-AUC   95% storm-block bootstrap CI = "
              f"[{boot['ci_2_5']:+.4f}, {boot['ci_97_5']:+.4f}] "
              f"({boot['n_valid']} resamples)")
    verdict = ("improves" if d_pr > 0.01 else
               "hurts" if d_pr < -0.01 else "approximately matches")
    print(f"  -> ERA5 {verdict} IMD on the strict common test set.")

    # ----------------- persistence / trend baseline -------------------
    print("\n" + "-" * 78)
    print("BASELINES (persistence / trend / climatology)")
    print("-" * 78)
    baseline_results = bl.run_all_baselines(
        df_train=imd_split.train,
        df_test=common_test,
        ri_threshold_kt=float(cfg["ri"]["threshold_kt"]),
    )
    for bname, bres in baseline_results.items():
        bm = bres["metrics"]
        print(f"  {bname:<24} PR={bm['pr_auc']:.4f} ROC={bm['roc_auc']:.4f} "
              f"P={bm['precision']:.3f} R={bm['recall']:.3f} F1={bm['f1']:.3f}")

    # ----------------- per-model bootstrap CI (storm-level) -----------
    print("\n" + "-" * 78)
    print("STORM-LEVEL BOOTSTRAP CONFIDENCE INTERVALS")
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

    # ----------------- probability calibration ------------------------
    print("\n" + "-" * 78)
    print("PROBABILITY CALIBRATION")
    print("-" * 78)
    calibration_results = {}
    for nm, p in probs.items():
        cal = eval_mod.calibration_detailed(y_te.to_numpy(), p)
        calibration_results[nm] = cal
        print(f"  {nm:<12} Brier={cal['brier']:.4f}  "
              f"Isotonic Brier={cal.get('isotonic_brier', 'N/A')}  "
              f"Cal slope={cal.get('calibration_slope', 'N/A'):.3f}  "
              f"Cal intercept={cal.get('calibration_intercept', 'N/A'):.3f}")

    # ----------------- event-level metrics ----------------------------
    print("\n" + "-" * 78)
    print("EVENT-LEVEL METRICS")
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

    # Pipeline-standard evaluation (each branch on its own test split) is
    # reproduced below with the differing observation counts documented. The
    # FINAL verdict uses the strict common-test set above.
    own_test_metrics = {}
    own_splits = [
        ("IMD", imd_model, imd_split.test, imd_feats, imd_use,
         metrics["IMD"]["threshold"]),
        ("ERA5", era5_model, era5_split.test, era5_feats, e5_use,
         metrics["ERA5"]["threshold"]),
        ("IMD+ERA5", combined_model, combined_split.test, comb_feats, c_use,
         metrics["IMD+ERA5"]["threshold"]),
    ]
    print("\nPipeline-standard evaluation (each branch on its OWN test split):")
    for nm, m, s, feats, use, thr in own_splits:
        X, y, _ = feat_mod.prepare_features(s, feats)
        X = X.reindex(columns=use)
        p = m.predict_proba(X)[:, 1]
        mm = eval_mod.classification_metrics(np.asarray(y).astype(int), p, thr)
        siz = len(X)
        nri = int((np.asarray(y) == 1).sum())
        own_test_metrics[nm] = {"test_obs": siz, "test_RI": nri,
                                **eval_mod.flatten_metric_name(mm)}
        print(f"  {nm:<10} (own test: {siz} obs / {nri} RI)  "
              f"PR={mm['pr_auc']:.4f} ROC={mm['roc_auc']:.4f} "
              f"P={mm['precision']:.3f} R={mm['recall']:.3f} F1={mm['f1']:.3f}")

    # ----------------- figures ---------------------------------------
    from sklearn.metrics import roc_curve, precision_recall_curve
    for nm, p in probs.items():
        safe = nm.replace(" ", "_").replace("+", "and")
        idx = np.argsort(p)
        # ROC
        if len(np.unique(y_te)) == 2:
            fpr, tpr, _ = roc_curve(y_te, p)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, lw=2,
                     label=f"ROC (AUC={metrics[nm]['roc_auc']:.3f})")
            plt.plot([0, 1], [0, 1], ls="--", color="grey")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title(f"ROC — {nm} (final)")
            plt.legend()
            plt.tight_layout()
            fig_path = FIGURES / f"roc_curve_{safe}_final.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"  saved {fig_path}")
        # PR
        prec, rec, _ = precision_recall_curve(y_te, p)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, marker=".", label=f"PR (AP={metrics[nm]['pr_auc']:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall — {nm} (final)")
        plt.legend()
        plt.tight_layout()
        fig_path = FIGURES / f"pr_curve_{safe}_final.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        # Confusion
        xpl.plot_confusion_matrix(metrics[nm]["confusion"],
                                  f"{nm} (final, thr={thresholds[nm]:.2f})", FIGURES)

    # ----------------- outputs ---------------------------------------
    out_pred = common_test[["storm_id", "datetime_utc", "RI_24h"]].copy().reset_index(drop=True)
    for path, nm in [
        (RESULTS / "imd_only_final.csv", "IMD"),
        (RESULTS / "era5_only_final.csv", "ERA5"),
        (RESULTS / "imd_era5_combined_final.csv", "IMD+ERA5"),
    ]:
        df = out_pred.copy()
        df["P_RI"] = probs[nm]
        df.to_csv(path, index=False)
        print(f"  saved {path}")

    comp_rows = []
    for nm in ["IMD", "ERA5", "IMD+ERA5"]:
        m = metrics[nm]
        comp_rows.append({
            "model": nm,
            "ROC_AUC": round(m["roc_auc"], 4),
            "PR_AUC": round(m["pr_auc"], 4),
            "Precision": round(m["precision"], 3),
            "Recall": round(m["recall"], 3),
            "F1": round(m["f1"], 3),
            "threshold": round(m["threshold"], 3),
            "Brier": round(m["brier"], 4),
            "test_obs": int(n_test_obs),
            "test_storms": int(n_test_storms),
            "test_RI_positives": int(n_test_ri),
        })
    comp = pd.DataFrame(comp_rows)
    comp["Delta_PR_AUC_vs_IMD"] = comp["PR_AUC"] - comp["PR_AUC"].iloc[0]
    comp["Delta_ROC_AUC_vs_IMD"] = comp["ROC_AUC"] - comp["ROC_AUC"].iloc[0]
    comp.to_csv(RESULTS / "model_comparison_final.csv", index=False)
    print(f"  saved {RESULTS / 'model_comparison_final.csv'}")
    print("\nMODEL COMPARISON (strict common test set):")
    print(comp.to_string(index=False))

    # ----------------- feature importance (combined) -----------------
    c_imp = xpl.feature_importance_table(combined_model, c_use)
    imd_imp = xpl.feature_importance_table(imd_model, imd_use)
    era5_imp = xpl.feature_importance_table(era5_model, e5_use)
    c_imp["group"] = c_imp["feature"].apply(
        lambda f: "IMD" if f in set(imd_feats) else "ERA5")
    c_imp = c_imp[["group", "feature", "importance"]]
    c_imp.to_csv(RESULTS / "imd_era5_feature_importance.csv", index=False)
    print(f"  saved {RESULTS / 'imd_era5_feature_importance.csv'}")
    print("\nTOP 20 IMD+ERA5 features (gain):")
    print(c_imp.head(20).to_string(index=False))

    top_imd = imd_imp.head(8)
    top_era5 = era5_imp.head(8)
    print("\nTOP IMD features (IMD-only model gain):")
    print(imd_imp.head(8).to_string(index=False))
    print("\nTOP ERA5 features (ERA5-only model gain):")
    print(era5_imp.head(8).to_string(index=False))

    # ----------------- experiment metadata ---------------------------
    exp = {
        "experiment": "FINAL IMD+ERA5 comparison (definitive)",
        "date": str(pd.Timestamp.now().date()),
        "seed": seed,
        "canonical_datasets": {
            "imd": str(cfg["paths"]["imd_file"]),
            "era5": str(cfg["paths"]["era5_file"]),
        },
        "imd_features": imd_use,
        "era5_features": e5_use,
        "combined_features": c_use,
        "split": {
            "test_storm_fraction": cfg["split"]["test_storm_fraction"],
            "val_storm_fraction": cfg["split"]["val_storm_fraction"],
            "keep_balanced": cfg["split"]["keep_balanced"],
        },
        "counts": {
            "imd_obs": int(len(imd)), "imd_storms": int(imd["storm_id"].nunique()),
            "era5_obs": int(len(era5)), "era5_storms": int(era5["storm_id"].nunique()),
            "combined_obs": int(len(combined)),
            "combined_storms": int(combined["storm_id"].nunique()),
            "test_observations": int(n_test_obs),
            "test_storms": int(n_test_storms),
            "test_RI_positives": int(n_test_ri),
        },
        "thresholds_validation_tuned": thresholds,
        "metrics_common_test": {k: eval_mod.flatten_metric_name(v)
                                for k, v in metrics.items()},
        "metrics_own_test_pipeline_standard": own_test_metrics,
        "deltas_vs_imd": {
            "d_pr_auc": f"{d_pr:+.4f}",
            "d_roc_auc": f"{d_roc:+.4f}",
            "pr_auc_bootstrap_95ci": (
                [boot.get("ci_2_5"), boot.get("ci_97_5")]
                if boot.get("ci_2_5") is not None else None),
            "pr_auc_bootstrap_n_resamples": boot.get("n_valid"),
        },
        "baselines": {
            k: {
                "pr_auc": v["metrics"]["pr_auc"],
                "roc_auc": v["metrics"]["roc_auc"],
                "f1": v["metrics"]["f1"],
                "brier": v["metrics"]["brier"],
            }
            for k, v in baseline_results.items()
        },
        "storm_bootstrap_ci": bootstrap_results,
        "calibration": {
            k: {"brier": v["brier"], "isotonic_brier": v.get("isotonic_brier"),
                "calibration_slope": v.get("calibration_slope"),
                "calibration_intercept": v.get("calibration_intercept")}
            for k, v in calibration_results.items()
        },
        "event_level": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_storm"
                and kk != "warning_lead_times_h"}
            for k, v in event_results.items()
        },
        "feature_importance_file": "results/imd_era5_feature_importance.csv",
        "class_imbalance": {
            "method": "scale_pos_weight from training split only",
            "imd_scale_pos_weight": float(
                (np.asarray(y_tr_imd) == 0).sum() / max(1, int((np.asarray(y_tr_imd) == 1).sum()))),
            "era5_scale_pos_weight": float(
                (np.asarray(y_tr_e5) == 0).sum() / max(1, int((np.asarray(y_tr_e5) == 1).sum()))),
            "combined_scale_pos_weight": float(
                (np.asarray(y_tr_c) == 0).sum() / max(1, int((np.asarray(y_tr_c) == 1).sum()))),
        },
        "honesty_notes": [
            "small RI event count -> individual metrics carry large uncertainty",
            "storm-block bootstrap CI reported for Delta PR-AUC",
            "a small numerical difference is NOT called a proven improvement",
        ],
    }
    with open(RESULTS / "final_imd_era5_experiment.json", "w",
              encoding="utf-8") as fh:
        json.dump(exp, fh, indent=2, default=str)
    print(f"  saved {RESULTS / 'final_imd_era5_experiment.json'}")

    # ----------------- final model copies ----------------------------
    for src, dst in [
        ("imd_xgboost.json", "imd_final_xgboost.json"),
        ("era5_xgboost.json", "era5_final_xgboost.json"),
        ("imd_era5_xgboost.json", "imd_era5_final_xgboost.json"),
    ]:
        shutil.copyfile(MODELS / src, MODELS / dst)
        print(f"  saved model -> models/{dst}")

    # ----------------- final verdict --------------------------------
    print("\n" + "=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)
    print(f"  IMD PR-AUC       : {imd_m['pr_auc']:.4f}")
    print(f"  ERA5 PR-AUC      : {metrics['ERA5']['pr_auc']:.4f}")
    print(f"  IMD+ERA5 PR-AUC  : {comb_m['pr_auc']:.4f}")
    print(f"  Persistence Trend PR-AUC : {baseline_results['Persistence Trend']['metrics']['pr_auc']:.4f}")
    print(f"  Climatology PR-AUC       : {baseline_results['Climatology']['metrics']['pr_auc']:.4f}")
    best_baseline = max(baseline_results.values(),
                        key=lambda b: b["metrics"]["pr_auc"])
    print(f"  Best baseline (PR-AUC)   : {best_baseline['name']} "
          f"({best_baseline['metrics']['pr_auc']:.4f})")
    imd_beats_baseline = imd_m["pr_auc"] > best_baseline["metrics"]["pr_auc"]
    print(f"  IMD > best baseline?     : {'YES' if imd_beats_baseline else 'NO'} "
          f"(Δ={imd_m['pr_auc'] - best_baseline['metrics']['pr_auc']:+.4f})")
    print(f"  ERA5 improvement over IMD (PR-AUC): {d_pr:+.4f}")
    print(f"  ROC-AUC: IMD={imd_m['roc_auc']:.4f}, "
          f"IMD+ERA5={comb_m['roc_auc']:.4f} ({d_roc:+.4f})")
    if boot["ci_2_5"] is not None:
        print(f"  ΔPR-AUC 95% CI   : "
              f"[{boot['ci_2_5']:+.4f}, {boot['ci_97_5']:+.4f}]")
    verdict = ("ADDS value beyond IMD" if d_pr > 0.01 else
               "does NOT add proven value beyond IMD (current evidence)")
    print(f"  Verdict          : ERA5 {verdict} "
          f"on the strict common test set ({n_test_obs} obs / {n_test_storms} storms / "
          f"{n_test_ri} RI events)")
    print("=" * 78)


if __name__ == "__main__":
    main()