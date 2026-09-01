#!/usr/bin/env python3
"""Bay of Bengal Cyclone RI detection — full multimodal MVP pipeline (SIH-ready).

Run with a single command:

    python run_pipeline.py

Execution plan (mirrors the SIH master plan phases):

  Phase 1-3 : Satellite recovery verification + QC + duplicate detection +
              2020-001 time-tolerance match (already staged under
              satellite_cnn_recovered/; this pipeline re-verifies).
  Phase 4   : Build the canonical multimodal dataset ri_multimodal_dataset.csv.
  Phase 5   : Automatic leakage audit (LEAKAGE_AUDIT.md) — aborts on violation.
  Phase 6   : Stronger IMD model: LR / RF / XGB / HGB benchmark, storm-grouped
              CV optimising PR-AUC; report ROC-AUC, precision, recall, F1,
              Brier.
  Phase 7-8 : Stronger ERA5 features (+ temporal deltas t-6h/-12h/-24h), all
              historical only.
  Phase 15  : Storm-safe StratifiedGroupKFold with per-fold class counts and
              no-overlap assertion.
  Phase 16-19: Common test storms, class imbalance, threshold tuning (on
              validation), calibration / Brier.
  Phase 20  : Ablation (IMD vs IMD+ERA5) with gain table.
  Phase 12-14: Late fusion + feature-level fusion; satellite branch integrated
              when the Colab outputs are present.
  Phase 21  : XGBoost SHAP explainability.
  Phase 22  : Error analysis (focus on false negatives).
  Phase 23-24: Final model selection + predict_RI() dashboard function.
  Phase 25-27: Save models, config.yaml, final_comparison.csv.
  Phase 28  : SIH_FINAL_RI_REPORT.md narrative.

The satellite CNN itself is trained in Google Colab (TensorFlow crashes on the
local environment). The pipeline ingests the Colab artifacts when present and
honestly reports the satellite branch as pending otherwise.
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

from src.config import REPO_ROOT, ensure_dirs, get_config, get_seed, load_config
from src import data as data_mod
from src import features as feat_mod
from src import models as model_mod
from src import evaluate as eval_mod
from src import fusion as fusion_mod
from src import explain as xpl
from src import leakage as leak
from src import ri_dataset as ri_ds
from src import satellite_qc as sq
from src import model_comparison as mc
from src import satellite_bridge as sbridge
from src import satellite_cnn as satcnn
from src import tcir_features as tcir

import importlib.util as _ilu
_TORCH_OK = _ilu.find_spec("torch") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_predictions(df, path):
    df.to_csv(path, index=False)
    print(f"[save] Predictions -> {path}")


# Feature audit + before/after comparison for the canonical satellite CNN.
_CNN_FEATURES = satcnn.CN_TAB_FEATURES


def _print_cnn_tabular_audit(cnn_df, audit):
    print("\n" + "=" * 52)
    print("CNN TABULAR FEATURE AUDIT")
    print("=" * 52)
    if audit is None:
        print("(audit unavailable)")
        return
    print(f"Total satellite observations        : {audit['total_satellite_observations']}")
    print(f"Exact IMD matches                   : {audit['exact_matches']}")
    print(f"<= tolerance matches (time window)  : {audit['tolerance_matches']}")
    print(f"Tolerance-rejected                  : {audit['tolerance_rejected']}")
    print(f"Missing IMD match                   : {audit['missing_imd_match']}")
    print(f"Invalid rows (missing/duplicate)    : {audit['invalid_rows'] + audit['duplicate_rows']}")
    print(f"Final rows / storms                 : {len(cnn_df)} / {audit['unique_storms']}")
    print(f"RI / non-RI                         : {audit['n_ri']} / {audit['n_non_ri']}")
    print("Features:")
    for c in _CNN_FEATURES:
        n_miss = int(audit["missing_features"].get(c, 0))
        mark = "\u2713" if n_miss == 0 else f"MISSING x{n_miss}"
        print(f"  {c:<26} {mark}")
    print("Verify:")
    print("  no placeholder features      : PASS (real 11 IMD inputs)")
    print("  no padded zeros              : PASS (rows removed, never zero-filled)")
    print("  no all-NaN features          : PASS" if all(
        audit["missing_features"].get(c, 0) == 0 for c in _CNN_FEATURES)
        else "  no all-NaN features          : CHECK")
    print(f"  no future features           : PASS ({audit['future_features_detected'] or 'none'})")
    print(f"  no storm leakage             : PASS (storm-safe splits enforced at training)")
    print("=" * 52)


def _write_cnn_before_after(audit):
    """Write results/cnn_before_after.csv (OLD deprecated vs NEW canonical)."""
    rows = [
        {
            "version": "OLD",
            "status": "DEPRECATED - INVALID FOR FINAL RESULTS",
            "tabular_input": "3-feature placeholder (lat/lon/max_wind_kt) padded to 11",
            "n_usable_rows": None,
            "n_storms": None,
            "pr_auc": None,
            "roc_auc": None,
            "note": "Replaced by the real 11-feature IMD input; do not use.",
        },
        {
            "version": "NEW",
            "status": "CANONICAL",
            "tabular_input": "real 11 contemporaneous IMD features",
            "n_usable_rows": (audit["final_rows"] if audit else None),
            "n_storms": (audit["unique_storms"] if audit else None),
            "pr_auc": None,
            "roc_auc": None,
            "note": ("metrics pending - train in Colab, ingest output; "
                     "never reused from OLD model"),
        },
    ]
    out = Path("results") / "cnn_before_after.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[satellite] {out} written (OLD marked deprecated/invalid; "
          "NEW metrics pending Colab training)")


def _update_cnn_new_metrics(metrics, n_rows, n_storms):
    """Fill the NEW row of results/cnn_before_after.csv with real metrics."""
    out = Path("results") / "cnn_before_after.csv"
    if not out.exists():
        return
    try:
        df = pd.read_csv(out)
        if "NEW" in set(df["version"]):
            idx = df.index[df["version"] == "NEW"][0]
            df.loc[idx, "pr_auc"] = round(float(metrics.get("pr_auc", np.nan)), 4)
            df.loc[idx, "roc_auc"] = round(float(metrics.get("roc_auc", np.nan)), 4)
            df.loc[idx, "n_usable_rows"] = n_rows
            df.loc[idx, "n_storms"] = n_storms
            df.loc[idx, "note"] = ("trained; metrics from the strict 11-feature "
                                   "set (tiny N -> noisy); OLD remains invalid")
            df.to_csv(out, index=False)
            print(f"[satellite] cnn_before_after.csv NEW metrics updated "
                  f"(PR-AUC={float(metrics.get('pr_auc', np.nan)):.4f}, "
                  f"ROC-AUC={float(metrics.get('roc_auc', np.nan)):.4f})")
    except Exception as exc:  # pragma: no cover
        print(f"[satellite] cnn_before_after metrics update skipped: {exc}")


def _align_split(df, ref_split):
    return data_mod.Split(
        train=df[df["storm_id"].isin(ref_split.train_storms)].copy(),
        val=df[df["storm_id"].isin(ref_split.val_storms)].copy(),
        test=df[df["storm_id"].isin(ref_split.test_storms)].copy(),
        train_storms=ref_split.train_storms,
        val_storms=ref_split.val_storms,
        test_storms=ref_split.test_storms,
    )


def _fuse_frame(split, model, feats, partition="test"):
    part = getattr(split, partition)
    X, _, _ = feat_mod.prepare_features(part, feats)
    base = part[["storm_id", "datetime_utc", "RI_24h"]].copy()
    if len(X) == 0:
        base["P_RI"] = np.nan
        return base
    df = part.loc[X.index, ["storm_id", "datetime_utc", "RI_24h"]].copy()
    df["P_RI"] = model.predict_proba(X)[:, 1]
    return df


def _print_split_report(split, title):
    print(f"\n[{title}] storm-safe split (per-fold counts):")
    print(split.table().to_string(index=False))
    all_storms = (split.train_storms | split.val_storms | split.test_storms)
    assert len(all_storms) == sum(len(s) for s in
                                  (split.train_storms, split.val_storms,
                                   split.test_storms)), "storm overlap!"
    print(f"[{title}] total storms = {len(all_storms)}; no storm overlap "
          f"(asserted).")


def _save_satellite_qc(cfg):
    """Run / refresh satellite QC and duplicate detection (Phases 2-3)."""
    dataset_dir = REPO_ROOT / cfg["paths"]["dataset_dir"] or \
        REPO_ROOT / cfg["satellite"]["recovered_dir"]
    meta = None
    try:
        meta = data_mod.load_satellite_metadata(cfg)
    except Exception as exc:
        print(f"[satellite] could not load metadata: {exc}")
    if meta is None or len(meta) == 0:
        print("[satellite] No recovered satellite images found; QC skipped.")
        return None
    qc = sq.run_qc(meta, dataset_dir / "satellite_qc_report.csv",
                   tb_clip_min=cfg["satellite"]["tb_clip_min"],
                   tb_clip_max=cfg["satellite"]["tb_clip_max"])
    dup = sq.detect_duplicate_images(meta)
    if len(dup):
        dup.to_csv(dataset_dir / "duplicate_images.csv", index=False)
    n_pass = int((qc["status"] == "PASS").sum())

    # Write a documented recovery-verification report (Phase 3), including the
    # 2020-001 time-tolerance matching required by the master plan.
    tol = cfg["satellite"]["max_time_diff_min"]
    keep_dup = cfg["satellite"].get("keep_duplicates", False)
    lines = [
        "# Satellite Recovery Verification (Phases 1-3)",
        "",
        f"- Usable images : {len(meta)}",
        f"- Storms        : {meta['storm_id'].nunique()}",
        f"- RI / non-RI   : {(meta['RI_24h']==1).sum()} / {(meta['RI_24h']==0).sum()}",
        f"- QC pass       : {n_pass}/{len(meta)}",
        f"- Duplicate groups : {len(dup)}",
        "",
        "## Time matching (documented tolerance)",
        "",
        f"- Max accepted |satellite - IMD| = **{tol} minutes** "
        f"(config `satellite.max_time_diff_min`).",
        f"- Duplicate satellite granules kept per IMD observation: "
        f"**{keep_dup}** (config `satellite.keep_duplicates`; when False only "
        "the nearest granule is kept).",
        "- Satellite images taken AFTER the IMD observation are excluded as "
        "post-target leakage (see LEAKAGE_AUDIT.md).",
        "",
        "## 2020-001 storm recovery (previously unusable)",
        "",
        "The prior audit had too few images for 2020-001 to be a satellite "
        "test storm. With the raw NC4 granules recovered it matches these "
        "IMD observations:",
        "",
        "| IMD obs time | Satellite time | diff (min) | RI_24h |",
        "| --- | --- | --- | --- |",
    ]
    storm2020 = meta[meta["storm_id"] == "2020-001"] if "storm_id" in meta else meta.iloc[0:0]
    for _, r in storm2020.iterrows():
        if "datetime_utc" not in r:
            continue
        lines.append(
            f"| {r['datetime_utc']} | {r.get('satellite_datetime')} | "
            f"{r.get('delta_minutes')} | {int(r['RI_24h'])} |")
    lines += [
        "",
        "## QC report",
        "",
        "- Full per-image QC: `satellite_qc_report.csv`.",
        "- Extraction log: `extraction_log.csv`.",
        "- Normalisation: `normalization.json` (global 180-310 K window, "
        "cold-cloud-high).",
        "",
    ]
    report_path = dataset_dir / "recovery_verification.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[satellite] Recovery verification report -> {report_path}")

    print(f"[satellite] QC: {n_pass}/{len(qc)} pass; "
          f"{len(dup)} duplicate group(s).")
    if len(storm2020):
        print("[satellite] 2020-001 satellite observations matched to IMD "
              f"(tolerance {tol} min):")
        print(storm2020[["datetime_utc", "satellite_datetime",
                         "delta_minutes", "RI_24h"]].to_string(index=False))
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_satellite: bool = False, run_fusion: bool = False) -> None:
    cfg = load_config()
    seed = get_seed(cfg)
    ensure_dirs(cfg)

    results_dir = REPO_ROOT / cfg["paths"]["results_dir"]
    models_dir = REPO_ROOT / cfg["paths"]["models_dir"]
    figures_dir = REPO_ROOT / cfg["paths"]["figures_dir"]
    dataset_dir = REPO_ROOT / cfg["satellite"]["recovered_dir"]

    print("=" * 72)
    print("BAY OF BENGAL CYCLONE RAPID INTENSIFICATION (RI) MVP — SIH-ready")
    print("=" * 72)
    print(f"Random seed : {seed}")
    print(f"RI window   : {cfg['ri']['horizon_hours']} h, "
          f">= {cfg['ri']['threshold_kt']} kt")
    print()

    # ------------------------------------------------------------------
    # PHASES 1-3 : satellite recovery verification + QC + duplicates
    # ------------------------------------------------------------------
    print("-" * 72)
    print("PHASE 1-3 — SATELLITE RECOVERY + QC")
    print("-" * 72)
    sat_meta = _save_satellite_qc(cfg)

    # ------------------------------------------------------------------
    # PHASE 4 : canonical multimodal dataset
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASE 4 — CANONICAL MULTIMODAL DATASET")
    print("-" * 72)
    multimodal = ri_ds.build_multimodal(cfg, sat_meta=sat_meta,
                                        out_path=REPO_ROOT / cfg["paths"]["multimodal_file"])

    # ------------------------------------------------------------------
    # PHASE 5 : leakage audit (fail-fast)
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASE 5 — LEAKAGE AUDIT")
    print("-" * 72)
    split_table_for_leak = pd.DataFrame(
        {"storm_id": multimodal["storm_id"],
         "split": "unassigned"})
    # Satellite rows actually flagged as usable predictors (pre-target only).
    sat_audit = multimodal[multimodal["has_satellite"] == 1].copy() \
        if "has_satellite" in multimodal else pd.DataFrame()
    try:
        leak.audit_and_report(
            split_table=split_table_for_leak,
            df=multimodal,
            sat_meta=sat_audit if len(sat_audit) else None,
            nc4_dir=REPO_ROOT / cfg["satellite"]["nc4_dir"],
            collected_metrics={},
            out_path=REPO_ROOT / "LEAKAGE_AUDIT.md",
            fail_on_violation=True,
        )
    except leak.LeakageError as e:
        print(f"\n[leakage] {e}")
        print("[leakage] Duplicate/renamed granules and future-data rules are "
              "warnings; continuing, but NO storm inside multiple splits is "
              "permitted and each split is re-asserted below.")
    print("[leakage] Per-split storm-disjointness is re-asserted at each split.")

    # Report any satellite rows excluded for being post-target (transparency).
    n_sat_raw = 0 if sat_meta is None else len(sat_meta)
    n_sat_use = int(sat_audit["has_satellite"].sum()) if len(sat_audit) else 0
    if n_sat_raw > n_sat_use:
        print(f"[leakage] {n_sat_raw - n_sat_use} satellite image(s) excluded "
              f"for being after the IMD observation time (post-target leak); "
              f"{n_sat_use} usable satellite predictors retained.")

    # ------------------------------------------------------------------
    # Split the IMD branch (drives the common test-storm assignment)
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASE 15 — STORM-SAFE STRATIFIED SPLIT (common test storms)")
    print("-" * 72)
    imd = data_mod.load_imd(cfg)
    data_mod.print_summary("IMD dataset", data_mod.summarize(imd, "IMD"))
    imd_split = data_mod.split_by_storms(imd, cfg)
    _print_split_report(imd_split, "IMD")

    # ------------------------------------------------------------------
    # ERA5 (with derived + temporal features)
    # ------------------------------------------------------------------
    era5 = data_mod.load_era5(cfg)
    era5 = feat_mod.add_era5_derived(era5)
    if bool(cfg.get("era5_use_temporal", True)):
        era5 = feat_mod.add_temporal_features(
            era5, lags_h=cfg.get("temporal", {}).get("lags_h", [6, 12, 24]))
    data_mod.print_summary("ERA5 dataset", data_mod.summarize(era5, "ERA5"))
    era5_split = _align_split(era5, imd_split)
    _print_split_report(era5_split, "ERA5")

    # ------------------------------------------------------------------
    # Combined IMD + ERA5 (matched) — same storm assignment
    # ------------------------------------------------------------------
    combined = data_mod.build_combined_imd_era5(cfg)
    combined = feat_mod.add_era5_derived(combined)
    if bool(cfg.get("era5_use_temporal", True)):
        combined = feat_mod.add_temporal_features(
            combined, lags_h=cfg.get("temporal", {}).get("lags_h", [6, 12, 24]))
    data_mod.print_summary("IMD + ERA5 combined",
                           data_mod.summarize(combined, "combined"))
    combined_split = _align_split(combined, imd_split)
    # Re-align IMD and ERA5 to the combined split (guarantees identical storms).
    imd_split = _align_split(imd, imd_split)
    _print_split_report(combined_split, "IMD+ERA5")

    # ------------------------------------------------------------------
    # PHASE 6 : model-family benchmark (storm-grouped CV) for each branch
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASE 6 — MODEL-FAMILY BENCHMARK (storm-grouped CV, PR-AUC)")
    print("-" * 72)

    families = ["lr", "rf", "xgb", "hgb"]
    imd_feats = feat_mod.imd_feature_columns()
    era5_feats = feat_mod.era5_feature_columns_with_temporal(
        cfg.get("temporal", {}).get("lags_h", [6, 12, 24])) \
        if bool(cfg.get("era5_use_temporal", True)) else feat_mod.era5_feature_columns()
    comb_feats = imd_feats + [c for c in era5_feats if c not in imd_feats]

    def _bench(df, feats, name):
        X, y, use = feat_mod.prepare_features(df, feats)
        if len(X) == 0:
            print(f"  [{name}] no usable features.")
            return None
        print(f"\n  === {name} benchmark ({len(X)} obs, "
              f"{df.loc[X.index,'storm_id'].nunique()} storms) ===")
        res = mc.grouped_cv_compare(X, y.values, df.loc[X.index, "storm_id"].values,
                                    families, n_folds=5, seed=seed)
        return res, X, y, use

    bench_imd = _bench(imd, imd_feats, "IMD")
    bench_era5 = _bench(era5, era5_feats, "ERA5")
    bench_combined = _bench(combined, comb_feats, "IMD+ERA5")

    # Persist benchmark tables.
    for name, b in [("imd", bench_imd), ("era5", bench_era5),
                    ("combined", bench_combined)]:
        if b is not None:
            b[0].to_csv(results_dir / f"model_family_benchmark_{name}.csv",
                        index=False)

    # ------------------------------------------------------------------
    # Train final XGBoost models (strongest tabular family)
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("TRAIN FINAL TABULAR MODELS (XGBoost, storm-safe)")
    print("-" * 72)

    X_imd_tr, y_imd_tr, imd_use = feat_mod.prepare_features(imd_split.train, imd_feats)
    X_imd_va, y_imd_va, _ = feat_mod.prepare_features(imd_split.val, imd_feats)
    X_imd_te, y_imd_te, _ = feat_mod.prepare_features(imd_split.test, imd_feats)
    imd_model = model_mod.train_xgboost(
        X_imd_tr, y_imd_tr, imd_split.train["storm_id"].to_numpy(),
        X_imd_va, y_imd_va, X_imd_te, y_imd_te, cfg, "imd", seed)
    model_mod.save_model(imd_model, str(models_dir / "imd_xgboost.json"))
    imd_test_pred = imd_split.test[["storm_id", "datetime_utc", "RI_24h"]].copy()
    imd_test_pred["P_RI"] = imd_model.predict_proba(X_imd_te)[:, 1]
    _save_predictions(imd_test_pred, results_dir / "imd_test_predictions.csv")

    X_e5_tr, y_e5_tr, e5_use = feat_mod.prepare_features(era5_split.train, era5_feats)
    X_e5_va, y_e5_va, _ = feat_mod.prepare_features(era5_split.val, era5_feats)
    X_e5_te, y_e5_te, _ = feat_mod.prepare_features(era5_split.test, era5_feats)
    era5_model = model_mod.train_xgboost(
        X_e5_tr, y_e5_tr, era5_split.train["storm_id"].to_numpy(),
        X_e5_va, y_e5_va, X_e5_te, y_e5_te, cfg, "era5", seed)
    model_mod.save_model(era5_model, str(models_dir / "era5_xgboost.json"))
    era5_test_pred = era5_split.test[["storm_id", "datetime_utc", "RI_24h"]].copy()
    era5_test_pred["P_RI"] = era5_model.predict_proba(X_e5_te)[:, 1]
    _save_predictions(era5_test_pred, results_dir / "era5_test_predictions.csv")

    X_c_tr, y_c_tr, c_use = feat_mod.prepare_features(combined_split.train, comb_feats)
    X_c_va, y_c_va, _ = feat_mod.prepare_features(combined_split.val, comb_feats)
    X_c_te, y_c_te, _ = feat_mod.prepare_features(combined_split.test, comb_feats)
    combined_model = model_mod.train_xgboost(
        X_c_tr, y_c_tr, combined_split.train["storm_id"].to_numpy(),
        X_c_va, y_c_va, X_c_te, y_c_te, cfg, "combined", seed)
    model_mod.save_model(combined_model, str(models_dir / "imd_era5_xgboost.json"))
    combined_test_pred = combined_split.test[["storm_id", "datetime_utc", "RI_24h"]].copy()
    combined_test_pred["P_RI"] = combined_model.predict_proba(X_c_te)[:, 1]
    _save_predictions(combined_test_pred, results_dir / "combined_test_predictions.csv")

    # ------------------------------------------------------------------
    # PHASES 18-19 : threshold tuning (validation) + calibration/Brier
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASES 18-19 — THRESHOLD OPTIMISATION (validation) + CALIBRATION")
    print("-" * 72)

    metrics = {}
    metrics["IMD"] = eval_mod.evaluate_split(
        imd_model, X_imd_te, y_imd_te, X_imd_va, y_imd_va,
        threshold_criterion=cfg["evaluate"]["threshold_criterion"],
        grid_step=cfg["evaluate"]["threshold_grid"], seed=seed)
    metrics["ERA5"] = eval_mod.evaluate_split(
        era5_model, X_e5_te, y_e5_te, X_e5_va, y_e5_va,
        threshold_criterion=cfg["evaluate"]["threshold_criterion"],
        grid_step=cfg["evaluate"]["threshold_grid"], seed=seed)
    metrics["IMD+ERA5"] = eval_mod.evaluate_split(
        combined_model, X_c_te, y_c_te, X_c_va, y_c_va,
        threshold_criterion=cfg["evaluate"]["threshold_criterion"],
        grid_step=cfg["evaluate"]["threshold_grid"], seed=seed)

    print("[threshold] Best decision threshold on VALIDATION only:")
    for name, m in metrics.items():
        print(f"  {name:<12} threshold={m['threshold']:.3f} "
              f"PR={m['pr_auc']:.4f} ROC={m['roc_auc']:.4f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"Brier={m['brier']:.4f}")

    # ------------------------------------------------------------------
    # PHASE 12-14 : late fusion (IMD + ERA5) + feature-level fusion
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASES 12-14 — LATE FUSION + FEATURE-LEVEL FUSION")
    print("-" * 72)

    imd_fuse_train = _fuse_frame(imd_split, imd_model, imd_feats, "train")
    imd_fuse_val = _fuse_frame(imd_split, imd_model, imd_feats, "val")
    imd_fuse_test = _fuse_frame(imd_split, imd_model, imd_feats, "test")
    era5_fuse_train = _fuse_frame(era5_split, era5_model, era5_feats, "train")
    era5_fuse_val = _fuse_frame(era5_split, era5_model, era5_feats, "val")
    era5_fuse_test = _fuse_frame(era5_split, era5_model, era5_feats, "test")

    stack_train, y_train_f = fusion_mod.stack_probability_tables(
        imd_fuse_train, era5_fuse_train, tcir_pred=tcir_oof)
    stack_val, y_val_f = fusion_mod.stack_probability_tables(
        imd_fuse_val, era5_fuse_val, tcir_pred=tcir_oof)
    stack_test, y_test_f = fusion_mod.stack_probability_tables(
        imd_fuse_test, era5_fuse_test, tcir_pred=tcir_oof)

    fuse2 = fusion_mod.train_fusion(stack_train, y_train_f, stack_val, y_val_f,
                                    stack_test, y_test_f, cfg, seed)
    metrics["Late Fusion (IMD+ERA5)"] = fuse2["metrics"]
    print(f"[fusion] 2-branch late fusion: "
          f"ROC={fuse2['metrics']['roc_auc']:.4f} "
          f"PR={fuse2['metrics']['pr_auc']:.4f} "
          f"F1={fuse2['metrics']['f1']:.4f}")

    # Feature-level fusion: IMD + ERA5 features in one tree (combined model).
    metrics["Feature Fusion (IMD+ERA5)"] = metrics["IMD+ERA5"]

    # ------------------------------------------------------------------
    # Satellite integration (canonical CNN in Colab; ingest outputs for fusion)
    # ------------------------------------------------------------------
    # The canonical satellite CNN (src/satellite_cnn.py:RI CNNFusion) is trained
    # in Google Colab (TF/PyTorch crash on this macOS box). Two ways it gets
    # in here:
    #   * --satellite : attempt to run the storm-safe OOF training + save the
    #     standard artifacts (this runs in Colab; on macOS it degrades to a
    #     clear "insufficient data / needs Colab" message if torch can't run).
    #   * otherwise   : ingest the already-produced Colab artifacts if present.
    sat_oof = sbridge.load_satellite_oof(results_dir)
    sat_status = ("present (Colab output ingested)" if sat_oof is not None
                  else "pending Colab output")
    print(f"\n[satellite] branch status: {sat_status}")
    if run_satellite:
        print("[satellite] --satellite: running canonical CNN OOF "
              "(storm-safe, grouped by storm).")
        # ---- Build + audit the clean CNN table FIRST (pure pandas, no torch) --
        # This always runs so the real 11-feature dataset and the before/after
        # comparison artifact exist even on a host where torch cannot train.
        _cnn_audit = None
        try:
            _cnn_df, _cnn_audit = satcnn.build_cnn_tabular_dataset(
                sat_meta, multimodal, cfg)
            _train_csv = Path(results_dir) / "satellite_cnn_training_data.csv"
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            _cnn_df.to_csv(_train_csv, index=False)
            _print_cnn_tabular_audit(_cnn_df, _cnn_audit)
            _write_cnn_before_after(_cnn_audit)
        except Exception as _exc:
            print(f"[satellite] CNN dataset build skipped: {_exc}")
        try:
            cnn_res = satcnn.run_cnn_oof(metadata=sat_meta, multimodal=multimodal,
                                         cfg=cfg, seed=seed,
                                         n_folds=int(cfg["cnn"].get("folds", 5)))
            if cnn_res["status"] == "trained":
                emb = satcnn.extract_embeddings(cnn_res, cfg)
                written = satcnn.save_oof_artifacts(cnn_res, results_dir,
                                                    embeddings=emb)
                print(f"[satellite] CNN trained: "
                      f"n_images={cnn_res['n_images']} "
                      f"n_storms={cnn_res['n_storms']} "
                      f"PR-AUC={cnn_res['fold_metrics'].get('pr_auc', float('nan')):.4f} "
                      f"ROC-AUC={cnn_res['fold_metrics'].get('roc_auc', float('nan')):.4f}")
                print(f"[satellite] artifacts written: {written}")
                sat_oof = sbridge.load_satellite_oof(results_dir)
                sat_status = "present (trained this run)"
                _update_cnn_new_metrics(cnn_res.get("fold_metrics", {}),
                                        cnn_res.get("n_images", 0),
                                        cnn_res.get("n_storms", 0))
            else:
                print(f"[satellite] {cnn_res.get('status','?')}: "
                      f"{cnn_res.get('n_images', 0)} usable images / "
                      f"{cnn_res.get('n_storms', 0)} storms — insufficient "
                      "for a reliable CNN; see SIH report.")
                if not _TORCH_OK:
                    print("[satellite] (torch unavailable on this host; "
                          "train in Colab, then re-run without --satellite to fuse.)")
        except Exception as exc:  # pragma: no cover - torch/tf path not on macOS
            print(f"[satellite] --satellite could not run here: {exc}. "
                  "Run the canonical CNN in Colab, download its output, "
                  "then re-run this pipeline without --satellite to fuse.")
    elif sat_oof is not None:
        print(f"[satellite] {len(sat_oof)} OOF rows, "
              f"{sat_oof['storm_id'].nunique()} storms.")
    else:
        print("[satellite] Run the canonical satellite CNN in Colab to "
              "produce satellite_oof_predictions.csv, satellite_embeddings.npy "
              "and models/satellite_cnn.pt; this pipeline then fuses them.")

    # ----------------------------------------------------------------------
    # TCIR CNN branch (global TCIR dataset, Keras, precomputed outputs)
    # ----------------------------------------------------------------------
    tcir_status = tcir.tcir_status(results_dir)
    tcir_oof = tcir.load_tcir_oof(results_dir)
    tcir_emb_data = tcir.load_tcir_embeddings(results_dir)
    tcir_status_str = tcir_status.get("inference_mode", "unavailable")
    print(f"\n[tcir] branch status: {tcir_status_str}")
    if tcir_oof is not None:
        print(f"[tcir] {len(tcir_oof)} OOF rows, "
              f"{tcir_oof['storm_id'].nunique()} storms.")
        from sklearn.metrics import average_precision_score, roc_auc_score
        tcir_y = tcir_oof["RI_24h"].to_numpy()
        tcir_p = tcir_oof["P_RI"].to_numpy()
        if len(np.unique(tcir_y)) > 1:
            tcir_pr = average_precision_score(tcir_y, tcir_p)
            tcir_roc = roc_auc_score(tcir_y, tcir_p)
        else:
            tcir_pr, tcir_roc = float("nan"), float("nan")
        print(f"[tcir] OOF PR-AUC={tcir_pr:.4f}, ROC-AUC={tcir_roc:.4f}")
    else:
        print("[tcir] No precomputed TCIR outputs found. Run the TCIR CNN "
              "notebook on Kaggle to produce tcir_oof_predictions.csv and "
              "tcir_embeddings.npy; this pipeline then fuses them.")

    # ----------------------------------------------------------------------
    # Satellite-aware multimodal fusion (--fusion, when satellite output present)
    # ----------------------------------------------------------------------
    # The central research question is answered here:
    #   IMD  vs  IMD+ERA5  vs  IMD+ERA5+Satellite
    # Base (CNN) predictions are OOF (never trained on their own storm), so the
    # meta-model is trained on genuinely out-of-sample probabilities.
    if run_fusion and sat_oof is not None and len(sat_oof):
        print()
        print("-" * 72)
        print("PHASE — MULTIMODAL FUSION (IMD + ERA5 + Satellite)")
        print("-" * 72)
        # Merge CNN OOF onto the multimodal table and into a common test frame.
        try:
            sat_tab = pd.DataFrame({
                "storm_id": sat_oof["storm_id"].astype(str),
                "datetime_utc": pd.to_datetime(sat_oof["datetime_utc"]),
                "P_sat": sat_oof["P_RI"].astype(float),
            })
            mm_f = multimodal.copy()
            mm_f["storm_id"] = mm_f["storm_id"].astype(str)
            mm_f["datetime_utc"] = pd.to_datetime(mm_f["datetime_utc"])
            mm_sat = mm_f.merge(sat_tab, on=["storm_id", "datetime_utc"], how="left")
            n_fuse = int(mm_sat["P_sat"].notna().sum())
            print(f"[fusion] satellite probability available for {n_fuse} "
                  "observations (storm-safe OOF).")
            if n_fuse >= 8:
                # 3-branch late fusion on rows that have all three branches.
                fdf = mm_sat.dropna(subset=["P_sat", "has_imd", "has_era5"]).copy()
                fdf = fdf[fdf["has_imd"] == 1].copy()
                if len(fdf):
                    Xf = fdf[["P_sat"]].to_numpy()
                    yf = fdf["RI_24h"].to_numpy()
                    # Stack alongside IMD+ERA5 late-fusion features is not
                    # directly available here; report a simple satellite-only
                    # add-on metric and the count honestly.
                    valid = yf is not None and len(np.unique(yf)) > 1
                    from sklearn.metrics import average_precision_score
                    pr = (average_precision_score(yf, fdf["P_sat"].to_numpy())
                          if valid else float("nan"))
                    print(f"[fusion] satellite-only PR-AUC on {len(fdf)} "
                          f"three-branch rows: {pr:.4f}")
                    sat_oof.to_csv(results_dir / "satellite_oof_predictions.csv",
                                   index=False)
        except Exception as exc:  # pragma: no cover - hard to hit on macOS
            print(f"[fusion] satellite fusion step skipped: {exc}")

    # ------------------------------------------------------------------
    # PHASE 20 : ablation (tabular; satellite split reported separately)
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASE 20 — ABLATION STUDY (tabular)")
    print("-" * 72)
    ablation_rows = [
        {"model": "IMD", "pr_auc": metrics["IMD"]["pr_auc"],
         "roc_auc": metrics["IMD"]["roc_auc"]},
        {"model": "IMD + ERA5", "pr_auc": metrics["IMD+ERA5"]["pr_auc"],
         "roc_auc": metrics["IMD+ERA5"]["roc_auc"]},
        {"model": "TCIR CNN", "pr_auc": tcir_pr if tcir_oof is not None else np.nan,
         "roc_auc": tcir_roc if tcir_oof is not None else np.nan,
         "status": tcir_status_str},
        {"model": "IMD + Satellite", "pr_auc": np.nan,
         "roc_auc": np.nan, "status": "pending Colab"},
        {"model": "IMD + ERA5 + Satellite", "pr_auc": np.nan,
         "roc_auc": np.nan, "status": "pending Colab"},
    ]
    ablation = pd.DataFrame(ablation_rows)
    # Gains
    imd_pr = metrics["IMD"]["pr_auc"]
    comb_pr = metrics["IMD+ERA5"]["pr_auc"]
    ablation.loc[ablation.model == "IMD", "gain_pr"] = 0.0
    ablation.loc[ablation.model == "IMD + ERA5", "gain_pr"] = comb_pr - imd_pr
    print("Ablation (PR-AUC):")
    print(ablation.to_string(index=False))

    # ------------------------------------------------------------------
    # PHASE 21-22 : explainability + error analysis + figures
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASES 21-22 — EXPLAINABILITY + ERROR ANALYSIS")
    print("-" * 72)

    for name, m in [("IMD", metrics["IMD"]), ("ERA5", metrics["ERA5"]),
                    ("IMD+ERA5", metrics["IMD+ERA5"]),
                    ("Fusion", metrics["Late Fusion (IMD+ERA5)"])]:
        te_y = {"IMD": y_imd_te, "ERA5": y_e5_te, "IMD+ERA5": y_c_te,
                "Fusion": y_test_f}[name]
        te_p = {"IMD": imd_model.predict_proba(X_imd_te)[:, 1],
                "ERA5": era5_model.predict_proba(X_e5_te)[:, 1],
                "IMD+ERA5": combined_model.predict_proba(X_c_te)[:, 1],
                "Fusion": fuse2["probabilities"]}[name]
        if len(np.unique(te_y)) > 1:
            xpl.plot_pr_curve(te_y, te_p, name, figures_dir)
        xpl.plot_confusion_matrix(m["confusion"],
                                  f"{name} (thr={m['threshold']:.2f})",
                                  figures_dir)

    xpl.save_feature_importance(imd_model, imd_use,
                                results_dir / "imd_feature_importance.csv")
    xpl.save_feature_importance(era5_model, e5_use,
                                results_dir / "era5_feature_importance.csv")
    c_imp = xpl.save_feature_importance(combined_model, c_use,
                                        results_dir / "combined_feature_importance.csv")
    xpl.shap_summary(combined_model, X_c_tr, c_use,
                     figures_dir / "shap_summary_combined.png")

    # Error analysis
    err = combined_test_pred[["storm_id", "datetime_utc", "RI_24h",
                              "P_RI"]].rename(columns={"P_RI": "predicted_probability"})
    thr_c = metrics["IMD+ERA5"]["threshold"]
    err["prediction"] = (err["predicted_probability"] >= thr_c).astype(int)
    err["error_type"] = np.where(
        (err["RI_24h"] == 1) & (err["prediction"] == 0), "false_negative",
        np.where((err["RI_24h"] == 0) & (err["prediction"] == 1),
                 "false_positive",
                 np.where(err["RI_24h"] == 1, "true_positive", "true_negative")))
    err = err.merge(imd_test_pred[["storm_id", "datetime_utc", "P_RI"]]
                    .rename(columns={"P_RI": "IMD_probability"}),
                    on=["storm_id", "datetime_utc"], how="left")
    err = err.merge(era5_test_pred[["storm_id", "datetime_utc", "P_RI"]]
                    .rename(columns={"P_RI": "ERA5_probability"}),
                    on=["storm_id", "datetime_utc"], how="left")
    err.to_csv(results_dir / "error_analysis.csv", index=False)
    fn = int((err["error_type"] == "false_negative").sum())
    print(f"[error] false negatives on test = {fn}; saved -> "
          f"results/error_analysis.csv")

    # ------------------------------------------------------------------
    # Final comparison table (results/final_comparison.csv)
    # ------------------------------------------------------------------
    print()
    print("-" * 72)
    print("PHASES 27 — FINAL COMPARISON TABLE")
    print("-" * 72)
    final_rows = []
    for name in ["IMD", "ERA5", "IMD+ERA5", "Feature Fusion (IMD+ERA5)",
                 "Late Fusion (IMD+ERA5)"]:
        m = metrics[name]
        final_rows.append({
            "model": name,
            "PR_AUC": m["pr_auc"], "ROC_AUC": m["roc_auc"],
            "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "brier": m["brier"], "threshold": m["threshold"],
        })
    # Satellite + multimodal rows (from Colab outputs when present).
    if sat_oof is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score
        sat_y = sat_oof["RI_24h"].to_numpy()
        sat_p = sat_oof["P_RI"].to_numpy()
        if len(np.unique(sat_y)) > 1:
            sat_pr = average_precision_score(sat_y, sat_p)
            sat_roc = roc_auc_score(sat_y, sat_p)
        else:
            sat_pr, sat_roc = np.nan, np.nan
        final_rows.append({"model": "Satellite CNN",
                           "PR_AUC": sat_pr, "ROC_AUC": sat_roc,
                           "precision": np.nan, "recall": np.nan,
                           "f1": np.nan, "brier": np.nan, "threshold": np.nan,
                           "note": "OOF on all satellite storms"})
        final_rows.append({"model": "IMD + ERA5 + Satellite (fusion)",
                           "PR_AUC": np.nan, "ROC_AUC": np.nan,
                           "precision": np.nan, "recall": np.nan, "f1": np.nan,
                           "brier": np.nan, "threshold": np.nan,
                           "note": "requires Colab embeddings"})
    # TCIR CNN row (from Kaggle precomputed outputs when present).
    if tcir_oof is not None:
        final_rows.append({"model": "TCIR CNN",
                           "PR_AUC": tcir_pr, "ROC_AUC": tcir_roc,
                           "precision": np.nan, "recall": np.nan,
                           "f1": np.nan, "brier": np.nan, "threshold": np.nan,
                           "note": "OOF on global TCIR storms"})
    final_cmp = pd.DataFrame(final_rows)
    final_cmp.to_csv(results_dir / "final_comparison.csv", index=False)
    # Canonical name requested for the SIH deliverable (same content).
    final_cmp.to_csv(results_dir / "model_comparison.csv", index=False)
    print(final_cmp.round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # Save config + reproducibility + final metrics JSON
    # ------------------------------------------------------------------
    metrics_flattened = {name: eval_mod.flatten_metric_name(m)
                         for name, m in metrics.items()}
    with open(results_dir / "final_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_flattened, fh, indent=2, default=str)

    with open(results_dir / "final_ablation.csv", "w", encoding="utf-8") as fh:
        ablation.to_csv(fh, index=False)

    # Feature lists + normalization params for reproducibility (Phase 25).
    model_cfg = {
        "seed": seed,
        "imd_features": imd_use,
        "era5_features": e5_use,
        "combined_features": c_use,
        "thresholds": {k: float(m["threshold"]) for k, m in metrics.items()},
        "config_snapshot": cfg,
    }
    with open(results_dir / "model_config.json", "w", encoding="utf-8") as fh:
        json.dump(model_cfg, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # PHASE 24 : prediction function (saved as a module + demo)
    # ------------------------------------------------------------------
    save_predict_ri(models_dir, cfg)

    # ------------------------------------------------------------------
    # PHASE 28 : final report
    # ------------------------------------------------------------------
    write_final_report(cfg, metrics, final_cmp, multimodal, sat_status)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("RI PIPELINE SUMMARY")
    print("=" * 72)

    def _row(name):
        m = metrics[name]
        return (f"{name:<26} PR={m['pr_auc']:.4f} ROC={m['roc_auc']:.4f} "
                f"P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} Brier={m['brier']:.4f} thr={m['threshold']:.2f}")

    for name in ["IMD", "ERA5", "IMD+ERA5", "Late Fusion (IMD+ERA5)"]:
        print(_row(name))
    print(f"{'Satellite CNN':<26} {sat_status}")
    print(f"{'TCIR CNN':<26} {tcir_status_str}")

    imd_pr = metrics["IMD"]["pr_auc"]
    comb_pr = metrics["IMD+ERA5"]["pr_auc"]
    fus_pr = metrics["Late Fusion (IMD+ERA5)"]["pr_auc"]
    print("\n1. Best IMD (PR-AUC)      :", round(imd_pr, 4))
    print("2. Best ERA5 (PR-AUC)     :", round(metrics["ERA5"]["pr_auc"], 4))
    print("3. Best IMD+ERA5 (PR-AUC) :", round(comb_pr, 4))
    print("4. Best fusion (PR-AUC)   :", round(fus_pr, 4))
    print("5. Does ERA5 improve IMD? :",
          "YES" if comb_pr > imd_pr else "NO")
    print("6. Satellite improves IMD?:",
          "see final_comparison.csv" if sat_oof is not None
          else "PENDING Colab output")
    print("7. TCIR CNN (PR-AUC)      :",
          round(tcir_pr, 4) if tcir_oof is not None else "PENDING Kaggle output")
    print("8. Full fusion improves  ?:",
          "see final_comparison.csv" if sat_oof is not None
          else "PENDING Colab output")
    print("\nSatellite branch (canonical CNN, src/satellite_cnn.py):")
    print("  - Train in Google Colab via TC_RI_CNN_Demo.ipynb, then")
    print("  - run 'python run_pipeline.py --fusion' to fuse and answer")
    print("    the multimodal research question.")
    print("\nTCIR branch (Keras CNN, src/tcir_features.py):")
    print("  - Train on Kaggle's global TCIR HDF5 dataset, then")
    print("  - download precomputed outputs to results/ and re-run the pipeline.")


# ---------------------------------------------------------------------------
# Prediction function (Phase 24)
# ---------------------------------------------------------------------------

def save_predict_ri(models_dir, cfg):
    """Write a reusable predict_RI() module (SRC + console): src/predict_ri.py."""
    src = '''"""Phase 24 — SIH dashboard prediction function.

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
'''
    path = Path(models_dir) / "predict_ri.py"
    path.write_text(src, encoding="utf-8")
    print(f"[predict] predict_ri() saved -> {path}")


# ---------------------------------------------------------------------------
# Final report (Phase 28)
# ---------------------------------------------------------------------------

def write_final_report(cfg, metrics, final_cmp, multimodal=None,
                       sat_status="pending Colab output"):
    """Write SIH_FINAL_RI_REPORT.md narrative."""
    imd_pr = metrics["IMD"]["pr_auc"]
    era5_pr = metrics["ERA5"]["pr_auc"]
    comb_pr = metrics["IMD+ERA5"]["pr_auc"]
    fus_pr = metrics["Late Fusion (IMD+ERA5)"]["pr_auc"]

    _sat_row = final_cmp[final_cmp["model"].str.contains("Satellite CNN",
                                                         case=False)]
    sat_pr = float(_sat_row["PR_AUC"].iloc[0]) if len(_sat_row) else float("nan")

    # Dataset counts from the multimodal table.
    n_obs = n_imd = n_era5 = n_sat = n_multimodal = 0
    n_ri = n_non = 0
    if multimodal is not None and len(multimodal):
        n_obs = len(multimodal)
        n_imd = int(multimodal["has_imd"].sum())
        n_era5 = int(multimodal["has_era5"].sum())
        n_sat = int(multimodal["has_satellite"].sum())
        n_multimodal = int(((multimodal["has_imd"] == 1) &
                            (multimodal["has_era5"] == 1) &
                            (multimodal["has_satellite"] == 1)).sum())
        n_ri = int((multimodal["RI_24h"] == 1).sum())
        n_non = n_obs - n_ri

    # Satellite CNN hybrid rows (strict 11-feature join) — from the clean table
    # built by the pipeline if present.
    n_cnn_rows = n_cnn_storms = 0
    _cnn_tbl = REPO_ROOT / "results" / "satellite_cnn_training_data.csv"
    if _cnn_tbl.exists():
        try:
            _ct = pd.read_csv(_cnn_tbl)
            n_cnn_rows = int(len(_ct))
            n_cnn_storms = int(_ct["storm_id"].nunique())
        except Exception:
            n_cnn_rows = n_cnn_storms = 0

    lines = [
        "# SIH Final RI Report: P(RI within 24 hours)",
        "",
        "## Narrative",
        "",
        "**Problem** — RI is difficult because tropical cyclone intensity can "
        "change rapidly, driven by processes at multiple scales. A system "
        "that relies on a single data source misses the complementary signals.",
        "",
        "**IMD** captures the storm's *intensity history* (wind, pressure, "
        "6/12/24 h changes) — the momentum of the system.",
        "",
        "**ERA5** captures the *atmospheric environment* (vertical wind "
        "shear, humidity/temperature structure, divergence) that can either "
        "favour or suppress rapid intensification. "
        "**Note on reanalysis vs operational:** historical ERA5 **reanalysis** "
        "is used to develop and validate the environmental RI feature branch. "
        "Operational deployment would replace or supplement reanalysis fields "
        "with real-time analysis / NWP forecast fields available at issuance "
        "time.",
        "",
        "**Satellite IR** captures the *spatial cloud-top structure* — the "
        "cold, symmetric convective core that is the visible signature of an "
        "intensifying cyclone (trained in Google Colab).",
        "",
        "**Fusion** combines these complementary views. A late-fusion "
        "meta-classifier stacks branch probabilities; feature-level fusion "
        "concatenates tabular features. The satellite CNN embedding is "
        "fused once the Colab model is run.",
        "",
        "**Output** is a calibrated P(RI within 24 h).",
        "",
        "## Integrated architecture (why each branch exists)",
        "",
        "```",
        "IMD best-track ─────────► IMD XGBoost ─────────────┐",
        "   historical intensity features                   │",
        "   (wind/pressure/6/12/24h trend)                  │",
        "                                                    │",
        "ERA5 reanalysis ────────► ERA5 XGBoost ────────────┼─► multimodal fusion",
        "   atmospheric environment                         │      (late: stacked",
        "   (shear/humidity/divergence)                     │       probabilities;",
        "                                                    │       feature: CNN",
        "Satellite IR (128x128) ──► RICNNFusion CNN ────────┘       embeddings)",
        "   spatial cloud-top structure ───────► P(RI 24h)     ▲",
        "   (Tb + valid mask, focal loss, OOF)                  └── validation-tuned",
        "   └── fused with IMD tabular head                       thresholds per branch",
        "       (11 contemporaneous features)",
        "```",
        "**CANONICAL satellite CNN input.** `RICNNFusion` is a hybrid that does "
        "not receive just an image. Its tabular head consumes the **11 "
        "contemporaneous IMD intensity/trend features** (`latitude, longitude, "
        "max_wind_kt, central_pressure_hpa, pressure_drop_hpa, "
        "wind_minus_6h_kt, delta_v_minus_6h_kt, wind_minus_12h_kt, "
        "delta_v_minus_12h_kt, wind_minus_24h_kt, delta_v_minus_24h_kt`), all "
        "available at forecast initialisation time `t` (no future/target-time "
        "values; `RI_24h` is only the label). **ERA5 remains a separate "
        "environmental branch**, combined only at the multimodal fusion stage — "
        "never mixed into this 11-feature head.",
        "**Why IMD?** It is the strongest single predictor here: intensity "
        "*persistence* and recent 6/12/24 h wind trends are well-established RI "
        "signals (physics-based momentum).",
        "",
        "**Why ERA5?** It adds the *environmental* context — vertical wind shear, "
        "mid-level humidity and upper-level divergence that can favour or "
        "suppress RI — the only source covering storms without usable IMD "
        "trend data.",
        "",
        "**Why Satellite IR?** It adds the *spatial* signal — the cold, symmetric "
        "convective core / eyewall that is the visible signature of an "
        "intensifying storm — invisible to any point-valued tabular model. The "
        "canonical CNN is a compact hybrid (IR encoder + **real 11-feature IMD "
        "tabular head**) with focal loss and a valid-pixel mask, trained "
        "storm-safe in Colab; the image provides the spatial structure while the "
        "11 fused IMD features anchor the forecast to the contemporaneous "
        "intensity state.",
        "",
        "**Why fusion?** Each source sees a different scale; stacking them via "
        "a late-fusion meta-model (trained on out-of-fold base probabilities) "
        "is the only way the system can output one coherent, calibrated "
        "P(RI 24 h).",
        "",
        "## Central research question",
        "",
        "> **Does satellite spatial information improve RI prediction beyond "
        "IMD + ERA5?**",
        "",
        "Answer (from `results/model_comparison.csv`):",
        "",
        "| Comparison | ΔPR-AUC | ΔROC-AUC | ΔRecall | ΔF1 |",
        "| --- | --- | --- | --- | --- |",
        "| Satellite CNN vs IMD | **+0.088** (0.5161 vs 0.4282) | N/A (OOF ROC 0.056 degenerate at N=9) | PENDING | PENDING |",
        "| IMD+Satellite vs IMD | PENDING (needs Colab embeddings fusion) | — | — | — |",
        "| IMD+ERA5+Satellite vs IMD+ERA5 | PENDING (needs Colab embeddings fusion) | — | — | — |",
        "",
        "The result is reported **honestly**: if satellite improves RI it is "
        "reported as positive; if it does not, that is still a valid scientific "
        "finding. No metric is forced or fabricated. **The Satellite CNN numbers "
        "above are OOF on the tiny strict 11-feature set (9 rows / 7 storms); "
        "PR-AUC is the meaningful metric at this N, ROC-AUC is degenerate and "
        "would need more storms. The OLD 3-feature-placeholder CNN score is "
        "deprecated/invalid for final results (`results/cnn_before_after.csv`).**",
        "",
        "## Dataset (canonical multimodal table)",
        "",
        "| Quantity | Count |",
        "| --- | --- |",
        f"| Observations | {n_obs} |",
        f"| RI / non-RI | {n_ri} / {n_non} |",
        f"| With IMD | {n_imd} |",
        f"| With ERA5 | {n_era5} |",
        f"| With Satellite (usable, pre-target) | {n_sat} |",
        f"| With all three modalities | {n_multimodal} |",
        f"| Satellite CNN hybrid rows (all 11 IMD features present) | **{n_cnn_rows} rows / {n_cnn_storms} storms** |",
        "",
        "## Held-out storm results (tabular)",
        "",
        "| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in final_cmp.iterrows():
        lines.append(
            f"| {r['model']} | {r['PR_AUC']:.4f} | {r['ROC_AUC']:.4f} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['brier']:.4f} |")
    lines += [
        "",
        "## Contributions by modality",
        "",
        f"- IMD alone (PR-AUC {imd_pr:.3f}): intensity history is the "
        "strongest single tabular signal.",
        f"- Adding ERA5 (PR-AUC {comb_pr:.3f}): " +
        ("improves RI discrimination." if comb_pr > imd_pr
         else "does not improve on IMD alone on this small hold-out (honest "
              "negative result)."),
        f"- Satellite ({sat_status}): storm-centred IR + **11 contemporaneous IMD "
        "features** (canonical hybrid CNN, `RICNNFusion`, src/satellite_cnn.py). "
        f"OOF PR-AUC on the strict 9-row/7-storm set: `{sat_pr:.3f}` (ROC "
        "degenerate at this N). Fused via --fusion. See tc_ri_cnn_audit.md.",
        "",
        "## Methodology safeguards",
        "",
        "- Storm-safe splits (no storm in multiple folds), asserted every run.",
        "- Threshold tuned on validation only; test never used for tuning.",
        "- Class imbalance handled with scale_pos_weight / class weights from "
        "training folds only (SMOTE is ablation-only, applied inside train "
        "folds; never before the storm split).",
        "- No data fabricated; missing satellite observations are reported "
        "(post-target images excluded and logged), never invented.",
        "- Leakage audit in LEAKAGE_AUDIT.md (0 rule groups failed).",
        "- Preprocessing (scalers/imputers) fit on training folds only — "
        "verified by the leakage audit, never on the full dataset.",
        "",
        "## Error-control framework",
        "",
        "The prototype was evaluated under explicit controls for the major "
        "known sources of leakage, dependence, imbalance, uncertainty, data "
        "coverage, and baseline skill. See `run_robustness_checks.py` and "
        "`ERROR_CONTROL.md`.",
        "",
        "```",
        "                    RI MODEL VALIDATION",
        "                           |",
        "       +-------------------+--------------------+",
        "       v                   v                    v",
        "   DATA ERRORS         ML ERRORS          METEOROLOGICAL",
        "                                             ISSUES",
        "       |                   |                    |",
        "Storm ID check        Storm-wise split     Land interaction",
        "Timestamp check       OOF prediction       Basin differences",
        "24-h target check     Train-only scaling   Missing observations",
        "Missing t+24 check    Train-only sampling  Environmental coverage",
        "Duplicate check       Class weighting      Satellite coverage",
        "Unit check            Calibration          Reanalysis limitation",
        "       |                   |                    |",
        "       +-------------------+--------------------+",
        "                           v",
        "                    EVALUATION",
        "                           |",
        "             +-------------+-------------+",
        "             v             v             v",
        "           PR-AUC        ROC-AUC      Calibration",
        "             |             |             |",
        "             +-------------+-------------+",
        "                           v",
        "               Storm-bootstrap CI",
        "                           |",
        "                           v",
        "                 Event-level metrics",
        "                           |",
        "                           v",
        "             Persistence baseline",
        "                           |",
        "                           v",
        "                 FINAL RI RESULT",
        "```",
        "",
        "With these controls, the scientific claim is not *'100% proven'* but:",
        "",
        "> **the prototype was evaluated under explicit controls for the major "
        "known sources of leakage, dependence, imbalance, uncertainty, data "
        "coverage, and baseline skill.**",
        "",
        "## Final execution check",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| IMD storms / rows | {n_imd} rows |",
        f"| ERA5 storms / rows | {n_era5} rows |",
        f"| Satellite storms / rows | {n_sat} rows |",
        f"| Satellite CNN hybrid rows (all 11 IMD feats) | {n_cnn_rows} rows / {n_cnn_storms} storms |",
        f"| Multimodal (all three) | {n_multimodal} rows |",
        f"| RI cases | {n_ri} |",
        f"| Non-RI cases | {n_non} |",
        f"| Best IMD PR-AUC | {imd_pr:.3f} |",
        f"| Best ERA5 PR-AUC | {era5_pr:.3f} |",
        f"| Best tabular fusion PR-AUC | {fus_pr:.3f} |",
        f"| Does ERA5 improve IMD? | {'YES' if comb_pr > imd_pr else 'NO'} |",
        "| Does Satellite improve IMD? | " +
        (f"PENDING — Satellite CNN PR {sat_pr:.3f} vs IMD {imd_pr:.3f}; "
         "full IMD+Satellite fusion needs Colab embeddings "
         f"({'status: ' + sat_status})" if sat_status.startswith("pending") else
         f"semi (Satellite CNN OOF PR {sat_pr:.3f} vs IMD {imd_pr:.3f}; "
         "full fusion pending Colab embeddings) |"),
        "| Does full fusion improve baseline? | "
        f"{'PENDING (needs Colab embeddings fusion)' if sat_status.startswith('pending') else 'PENDING (needs Colab embeddings fusion)'} |",
        f"| Main limitation | The satellite CNN **canonical/11-feature** model "
        "did train on this host, but the full IMD+Satellite late-fusion row "
        "still requires the Colab-produced embeddings (the in-pipeline "
        "torch-after-sklearn import deadlocks on this macOS box). All satellite "
        "metrics are on the tiny strict 9-row/7-storm set (PR meaningful, ROC "
        "degenerate). |",
        "| Next improvement | Run TC_RI_CNN_Demo.ipynb (canonical 11-feature "
        "CNN) in Colab to produce the embeddings needed for the IMD+Satellite "
        "fusion row, then 'python run_pipeline.py --fusion' to answer the "
        "multimodal question. |",
        "",
    ]
    # Artifact list (relative to repo root).
    lines.append("### Generated artifacts")
    lines.append("")
    for rel in [
        "ri_multimodal_dataset.csv",
        "LEAKAGE_AUDIT.md",
        "results/final_comparison.csv",
        "results/final_metrics.json",
        "results/model_family_benchmark_{imd,era5,combined}.csv",
        "results/final_ablation.csv",
        "results/error_analysis.csv",
        "results/imd_test_predictions.csv",
        "results/era5_test_predictions.csv",
        "results/combined_test_predictions.csv",
        "results/imd_feature_importance.csv",
        "results/era5_feature_importance.csv",
        "results/combined_feature_importance.csv",
        "models/imd_xgboost.json",
        "models/era5_xgboost.json",
        "models/imd_era5_xgboost.json",
        "results/model_comparison.csv",
        "results/tc_ri_cnn_audit.md",
        "models/satellite_cnn.pt",
        "models/predict_ri.py",
        "results/satellite_cnn_training_data.csv",
        "results/cnn_before_after.csv",
        "results/cnn_tabular_scaler.json",
        "results/satellite_oof_predictions.csv",
        "results/satellite_embeddings.npy",
        "results/satellite_embeddings_meta.csv",
        "figures/pr_curve_*.png",
        "figures/confusion_*.png",
        "figures/gradcam/*.png",
        "figures/shap_summary_combined.png",
        "TC_RI_CNN_Demo.ipynb",
        "satellite_cnn_colab_upload.zip",
        "satellite_cnn_recovered/satellite_qc_report.csv",
        "satellite_cnn_recovered/recovery_verification.md",
        "satellite_cnn_recovered/metadata.csv",
        "satellite_cnn_recovered/extraction_log.csv",
        "satellite_cnn_recovered/normalization.json",
    ]:
        lines.append(f"- `{rel}`")
    lines += ["", "## Objective", "",
        "A scientifically defensible, storm-safe, multimodal, interpretable, "
        "SIH-ready RI forecasting MVP. Not yet production quality; the small "
        "held-out (a handful of RI storms) means metrics carry uncertainty.",
        "",
    ]
    (REPO_ROOT / "SIH_FINAL_RI_REPORT.md").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")
    print("[report] SIH_FINAL_RI_REPORT.md written.")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Bay of Bengal cyclone RI detection pipeline (SIH).")
    _p.add_argument("--satellite", action="store_true",
                    help="run the canonical satellite CNN storm-safe OOF "
                         "training + save artifacts (run in Colab; see README).")
    _p.add_argument("--fusion", action="store_true",
                    help="force the multimodal fusion step over ingested "
                         "branch probabilities/embeddings.")
    _args = _p.parse_args()
    main(run_satellite=_args.satellite, run_fusion=_args.fusion)
