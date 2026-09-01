# RI Model Error-Control Framework

The prototype was evaluated under **explicit controls** for the major known
sources of **leakage, dependence, imbalance, uncertainty, data coverage, and
baseline skill**. This document is the single reference for how each source
of error is controlled.

## Error-control flowchart

```
                    RI MODEL VALIDATION
                           |
       +-------------------+--------------------+
       v                   v                    v
   DATA ERRORS         ML ERRORS          METEOROLOGICAL
                                             ISSUES
       |                   |                    |
Storm ID check        Storm-wise split     Land interaction
Timestamp check       OOF prediction       Basin differences
24-h target check     Train-only scaling   Missing observations
Missing t+24 check    Train-only sampling  Environmental coverage
Duplicate check       Class weighting      Satellite coverage
Unit check            Calibration          Reanalysis limitation
       |                   |                    |
       +-------------------+--------------------+
                           v
                    EVALUATION
                           |
             +-------------+-------------+
             v             v             v
           PR-AUC        ROC-AUC      Calibration
             |             |             |
             +-------------+-------------+
                           v
               Storm-bootstrap CI
                           |
                           v
                 Event-level metrics
                           |
                           v
             Persistence baseline
                           |
                           v
                 FINAL RI RESULT
```

## 1. Data-error controls

| Control | How it is enforced | Where |
| --- | --- | --- |
| Storm ID check | Every storm appears in exactly one split (asserted) | `src/data.split_by_storms`, `src/leakage.check_storm_overlap` |
| Timestamp check | Datetime parsed, sorted per storm; gaps > 1.5x window flagged | `src/data.audit_ri_label_construction` |
| 24-h target check | RI_24h recomputable from wind columns; forbidden predictors asserted absent | `run_final_imd_era5_comparison.FORBIDDEN_PREDICTORS` |
| Missing t+24 (censoring) | Rows with NaN RI_24h excluded, never set to 0 | `src/data.load_imd` |
| Duplicate check | Byte-identical satellite images de-duplicated | `src/satellite_qc.detect_duplicate_images` |
| Unit check | Satellite normalization uses a fixed 180–310 K window; wind in kt | `config.yaml`, `src/satellite_cnn` |

## 2. ML-error controls

| Control | How it is enforced | Where |
| --- | --- | --- |
| Storm-wise split | Split at storm level (not observation) | `src/data.split_by_storms` |
| OOF prediction | GroupKFold / StratifiedGroupKFold grouped by storm | `src/models.grouped_cv_pr_auc`, `src/model_comparison` |
| Train-only scaling | MinMaxScaler fit on training storms only (satellite leg); verified by audit | `src/satellite_cnn`, `src/leakage.check_preprocessing_leakage` |
| Train-only sampling | SMOTE (ablation only) applied inside train folds, never pre-split | `src/models`, `src/leakage` |
| Class weighting | scale_pos_weight / class weights from training folds only | `src/models._scale_pos_weight` |
| Calibration | Brier score, reliability curve, isotonic regression | `src/evaluate.calibration_detailed` |
| Threshold tuning | Decision threshold chosen on validation only, never test | `src/evaluate.tune_threshold`, `evaluate_split` |

## 3. Meteorological-error controls

| Control | How it is enforced | Where |
| --- | --- | --- |
| Land interaction | `distance_to_land_km`, `over_land`, `distance_to_coast_km` features + ocean-only sensitivity | `src/features.add_land_interaction_features`, `run_robustness_checks` |
| Basin differences | BoB focus; storms confined to the Bay of Bengal | dataset scope |
| Missing observations | Rows with missing t+24h excluded (censored), not zero-filled | `src/data.load_imd` |
| Environmental coverage | ERA5 sub-coverage reported; combined table built only where both present | `src/data.build_combined_imd_era5` |
| Satellite coverage | Satellite OOF only on overlapping rows; post-target images excluded | `src/ri_dataset`, `src/leakage.check_satellite_after_target` |
| Reanalysis limitation | ERA5 is **reanalysis** for development/validation; operational deployment would use real-time analysis/NWP fields | README, SIH report |

## 4. Evaluation controls

| Control | How it is enforced | Where |
| --- | --- | --- |
| PR-AUC | Primary metric (imbalance-appropriate), safe implementation | `src/evaluate.safe_pr_auc` |
| ROC-AUC | Secondary metric | `src/evaluate.safe_roc_auc` |
| Calibration | Brier + reliability + isotonic | `src/evaluate.calibration_detailed` |
| Storm-bootstrap CI | Resample **storms**, not rows | `src/evaluate.storm_bootstrap_ci` |
| Event-level metrics | RI episodes detected, false alarms/storm, lead time | `src/event_metrics` |
| Persistence baseline | Trend-persistence + climatology + naive | `src/baselines` |

## 5. Running the checks

```
# Train the canonical models (if not already done)
python run_pipeline.py

# Run the definitive IMD + ERA5 comparison (with baselines, CI, calibration)
python run_final_imd_era5_comparison.py

# Run the full robustness / error-control battery
python run_robustness_checks.py
```

The robustness battery writes `results/robustness_checks.json` with the
label audit, baselines, bootstrap CI, calibration, event-level metrics and
land sensitivity.
