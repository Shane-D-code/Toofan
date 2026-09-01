# SIH Final RI Report: P(RI within 24 hours)

## Narrative

**Problem** — RI is difficult because tropical cyclone intensity can change rapidly, driven by processes at multiple scales. A system that relies on a single data source misses the complementary signals.

**IMD** captures the storm's *intensity history* (wind, pressure, 6/12/24 h changes) — the momentum of the system.

**ERA5** captures the *atmospheric environment* (vertical wind shear, humidity/temperature structure, divergence) that can either favour or suppress rapid intensification. **Note on reanalysis vs operational:** historical ERA5 **reanalysis** is used to develop and validate the environmental branch; operational deployment would replace it with real-time analysis / NWP forecast fields available at issuance time.

**Satellite IR** captures the *spatial cloud-top structure* — the cold, symmetric convective core that is the visible signature of an intensifying cyclone (trained in Google Colab).

**Fusion** combines these complementary views. A late-fusion meta-classifier stacks branch probabilities; feature-level fusion concatenates tabular features. The satellite CNN embedding is fused once the Colab model is run.

**Output** is a calibrated P(RI within 24 h).

## Integrated architecture (why each branch exists)

```
IMD best-track ─────────► IMD XGBoost ─────────────┐
   historical intensity features                   │
   (wind/pressure/6/12/24h trend)                  │
                                                    │
ERA5 reanalysis ────────► ERA5 XGBoost ────────────┼─► multimodal fusion
   atmospheric environment                         │      (late: stacked
   (shear/humidity/divergence)                     │       probabilities;
                                                    │       feature: CNN
Satellite IR (128x128) ──► RICNNFusion CNN ────────┘       embeddings)
   spatial cloud-top structure ───────► P(RI 24h)     ▲
   (Tb + valid mask, focal loss, OOF)                  └── validation-tuned
   └── fused with IMD tabular head                       thresholds per branch
       (11 contemporaneous features)
```
**CANONICAL satellite CNN input.** `RICNNFusion` is a hybrid that does not receive just an image. Its tabular head consumes the **11 contemporaneous IMD intensity/trend features** (`latitude, longitude, max_wind_kt, central_pressure_hpa, pressure_drop_hpa, wind_minus_6h_kt, delta_v_minus_6h_kt, wind_minus_12h_kt, delta_v_minus_12h_kt, wind_minus_24h_kt, delta_v_minus_24h_kt`), all available at forecast initialisation time `t` (no future/target-time values; `RI_24h` is only the label). **ERA5 remains a separate environmental branch**, combined only at the multimodal fusion stage — never mixed into this 11-feature head.
**Why IMD?** It is the strongest single predictor here: intensity *persistence* and recent 6/12/24 h wind trends are well-established RI signals (physics-based momentum).

**Why ERA5?** It adds the *environmental* context — vertical wind shear, mid-level humidity and upper-level divergence that can favour or suppress RI — the only source covering storms without usable IMD trend data.

**Why Satellite IR?** It adds the *spatial* signal — the cold, symmetric convective core / eyewall that is the visible signature of an intensifying storm — invisible to any point-valued tabular model. The canonical CNN is a compact hybrid (IR encoder + **real 11-feature IMD tabular head**) with focal loss and a valid-pixel mask, trained storm-safe in Colab; the image provides the spatial structure while the 11 fused IMD features anchor the forecast to the contemporaneous intensity state.

**Why fusion?** Each source sees a different scale; stacking them via a late-fusion meta-model (trained on out-of-fold base probabilities) is the only way the system can output one coherent, calibrated P(RI 24 h).

## Central research question

> **Does satellite spatial information improve RI prediction beyond IMD + ERA5?**

Answer (from `run_satellite_contribution.py`, `results/satellite_ablation_final.csv`):

| Comparison | ΔPR-AUC | ΔROC-AUC | Notes |
| --- | --- | --- | --- |
| Satellite vs IMD | **−0.428** (0.5161 vs 0.9444) | −0.778 (degenerate at N=9) | N=9; satellite ranks worse on this subset |
| IMD+Satellite vs IMD | **−0.096** (0.8486 vs 0.9444) | −0.222 | Late fusion: probability average |
| IMD+ERA5+Satellite vs IMD+ERA5 | N/A | N/A | Only 1 matching observation |

The result is reported **honestly**: if satellite improves RI it is reported as positive; if it does not, that is still a valid scientific finding. No metric is forced or fabricated. **The Satellite CNN numbers are OOF on the tiny strict 11-feature set (9 rows / 7 storms); PR-AUC is the meaningful metric at this N, ROC-AUC is degenerate. The ERA5 overlap is just 1 observation — three-way fusion is descriptive only.**

### Satellite contribution verdict

On the 9-row satellite subset, **the satellite branch does not improve over IMD alone** (PR-AUC 0.516 vs 0.944). The IMD model achieves a much higher PR-AUC because it was trained on thousands of observations and leverages intensity persistence — the strongest signal. The satellite CNN is poorly calibrated (all predictions >0.74, all 3 non-RI cases misclassified). IMD+Satellite late fusion (probability average) partially recovers performance (PR-AUC 0.85) but still ranks below IMD alone. **This is not a proven improvement — the data is too small (N=9, 7 storms) for any statistical claim.** The satellite branch is validated as proof-of-concept (the CNN runs, produces OOF predictions, Grad-CAM shows physically meaningful cold-cloud attention), but **the satellite data needs to grow substantially before the fusion question can be answered.**

## Dataset (canonical multimodal table)

| Quantity | Count |
| --- | --- |
| Observations | 3211 |
| RI / non-RI | 179 / 3032 |
| With IMD | 3211 |
| With ERA5 | 848 |
| With Satellite (usable, pre-target) | 25 |
| With all three modalities | 4 |
| Satellite CNN hybrid rows (all 11 IMD features present) | **9 rows / 7 storms** |

## Held-out storm results (tabular)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| IMD | 0.4282 | 0.8761 | 0.586 | 0.362 | 0.447 | 0.1319 |
| ERA5 | 0.2969 | 0.7047 | 0.286 | 0.080 | 0.125 | 0.1420 |
| IMD+ERA5 | 0.3411 | 0.7462 | 0.326 | 0.600 | 0.423 | 0.2267 |
| Feature Fusion (IMD+ERA5) | 0.3411 | 0.7462 | 0.326 | 0.600 | 0.423 | 0.2267 |
| Late Fusion (IMD+ERA5) | 0.4004 | 0.8059 | 0.500 | 0.240 | 0.324 | 0.1165 |
| Satellite CNN (OOF, 9 obs) | 0.5161 | 0.0556 | 0.667 | 1.000 | 0.800 | N/A |
| IMD (9-row subset) | 0.9444 | 0.8333 | 1.000 | 0.333 | 0.500 | N/A |
| IMD + Satellite (9-row subset) | 0.8486 | 0.6111 | 1.000 | 0.500 | 0.667 | N/A |
| IMD + ERA5 (1 obs) | N/A | N/A | N/A | N/A | N/A | N/A |

*Note: Satellite rows are on the tiny 9-row / 7-storm / 6-RI subset. IMD baseline on the same subset (0.944) is high because the XGBoost was trained on the full dataset and predicts on a subset including its training storms. The satellite OOF is genuinely out-of-sample. These are NOT directly comparable for generalization claims.*

## Contributions by modality

- IMD alone (PR-AUC 0.428 / strict common test 0.5935): intensity history is the strongest single tabular signal; the only proven defensible model.
- Adding ERA5 (PR-AUC 0.341 / strict common test 0.3411): does not improve on IMD alone on this hold-out (honest negative result). Bootstrap CI [−0.474, +0.039].
- Satellite (PR-AUC 0.516 on the 9-row OOF subset): spatial cloud-top structure proof-of-concept. On the 9-row satellite subset, IMD alone scores 0.944 (trained on full data). Satellite ranks worse (0.516). IMD+Satellite (0.85) does not recover IMD performance. N=9 is too small for significance. ERA5 overlap (1 obs) prevents three-way analysis. See `results/satellite_contribution_experiment.json`.

## Methodology safeguards

- Storm-safe splits (no storm in multiple folds), asserted every run.
- Threshold tuned on validation only; test never used for tuning.
- Class imbalance handled with scale_pos_weight / class weights from training folds only.
- No data fabricated; missing satellite observations are reported (post-target images excluded and logged), never invented.
- Leakage audit in LEAKAGE_AUDIT.md (0 rule groups failed).
- Preprocessing (scalers/imputers) fit on training folds only — verified by the leakage audit, never on the full dataset.

## Error-control framework

The prototype was evaluated under **explicit controls** for the major known
sources of leakage, dependence, imbalance, uncertainty, data coverage and
baseline skill (`ERROR_CONTROL.md`; re-run with `python run_robustness_checks.py`):

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

**Key results of the framework on the strict common test set (174 obs / 20 storms / 25 RI):**

- **Baseline skill:** the IMD model (PR-AUC **0.594**) beats the strongest
  baseline, **persistence-trend** (0.373), by **ΔPR-AUC +0.221** — the AI adds
  real skill over simply continuing the storm's previous intensity trend.
- **Uncertainty (storm-bootstrap CI):** IMD PR-AUC 0.594, 95 % CI
  **[0.333, 0.753]**.
- **Calibration:** IMD Brier **0.145** (isotonic-corrected **0.078**); outputs
  are modestly overconfident, so calibrate P(RI) before operational use.
- **Event-level:** IMD detects 2/11 RI episodes with median warning lead **15 h**
  (IMD+ERA5 detects 5/11 with more false alarms).

With these controls the scientific claim is not *"100 % proven"* but:

> **the prototype was evaluated under explicit controls for the major known
> sources of leakage, dependence, imbalance, uncertainty, data coverage and
> baseline skill.**

**ERA5 reanalysis caveat:** ERA5 is **reanalysis**, used here to develop and
validate the environmental RI branch. Operational deployment would replace or
supplement it with **real-time analysis / NWP forecast fields** available at
issuance time — the reported skill is a historical-environment benchmark, not a
claim about live forecast availability.

## Final execution check

| Item | Value |
| --- | --- |
| IMD storms / rows | 3211 rows |
| ERA5 storms / rows | 848 rows |
| Satellite storms / rows | 25 rows |
| Satellite CNN hybrid rows (all 11 IMD feats) | 9 rows / 7 storms |
| Multimodal (all three) | 4 rows |
| RI cases | 179 |
| Non-RI cases | 3032 |
| Best IMD PR-AUC | 0.428 |
| Best ERA5 PR-AUC | 0.297 |
| Best tabular fusion PR-AUC | 0.400 |
| Does ERA5 improve IMD? | NO (ΔPR-AUC −0.2524; CI [−0.474, +0.039]) |
| Does Satellite improve IMD? | NO on the 9-row subset (0.516 vs 0.944); N=9 too small for any claim |
| Does full fusion improve baseline? | NOT EVALUABLE (only 1 observation has all three modalities) |
| Main limitation | Satellite overlap with ERA5 is 1 of 9 observations — the three-way question cannot be answered with current data. The satellite branch (0.516) ranks below IMD (0.944) on the 9-row subset; N=9 / 7 storms is too small for statistical significance. IMD predictions on that subset come from the full-data model, so the two are not directly comparable. |
| Next improvement | Grow the satellite dataset (train CNN on more images; more storms) and broaden ERA5 reanalysis coverage of satellite storms so the multimodal question becomes testable. Increase satellite image count from 26 to hundreds. |

### Generated artifacts

- `ri_multimodal_dataset.csv`
- `LEAKAGE_AUDIT.md`
- `results/final_comparison.csv`
- `results/final_metrics.json`
- `results/model_family_benchmark_{imd,era5,combined}.csv`
- `results/final_ablation.csv`
- `results/error_analysis.csv`
- `results/imd_test_predictions.csv`
- `results/era5_test_predictions.csv`
- `results/combined_test_predictions.csv`
- `results/imd_feature_importance.csv`
- `results/era5_feature_importance.csv`
- `results/combined_feature_importance.csv`
- `models/imd_xgboost.json`
- `models/era5_xgboost.json`
- `models/imd_era5_xgboost.json`
- `results/model_comparison.csv`
- `results/tc_ri_cnn_audit.md`
- `models/satellite_cnn.pt`
- `models/predict_ri.py`
- `results/satellite_cnn_training_data.csv`
- `results/cnn_before_after.csv`
- `results/cnn_tabular_scaler.json`
- `results/satellite_oof_predictions.csv`
- `results/satellite_embeddings.npy`
- `results/satellite_embeddings_meta.csv`
- `figures/pr_curve_*.png`
- `figures/confusion_*.png`
- `figures/gradcam/*.png`
- `figures/shap_summary_combined.png`
- `TC_RI_CNN_Demo.ipynb`
- `satellite_cnn_colab_upload.zip`
- `satellite_cnn_recovered/satellite_qc_report.csv`
- `satellite_cnn_recovered/recovery_verification.md`
- `satellite_cnn_recovered/metadata.csv`
- `satellite_cnn_recovered/extraction_log.csv`
- `satellite_cnn_recovered/normalization.json`

## FINAL IMD+ERA5 comparison (definitive storm-safe experiment)

Experiment stage added 2026-08-31 (`run_final_imd_era5_comparison.py`). Reuses the canonical
datasets and the three canonical XGBoost models (`models/imd_xgboost.json`,
`models/era5_xgboost.json`, `models/imd_era5_xgboost.json`) produced by `run_pipeline.py`
(seed 42). The satellite CNN is untouched; no ocean data was added.

**Strict common test set** — the same 174 observations / 20 storms / 25 RI events for all three
models (every model scored on exactly the same rows, storm-safe split):

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| IMD | 0.8572 | 0.5935 | 0.667 | 0.240 | 0.353 | 0.72 |
| ERA5 | 0.2969 | 0.7047 | 0.286 | 0.080 | 0.125 | 0.63 |
| IMD + ERA5 | 0.7462 | 0.3411 | 0.326 | 0.600 | 0.423 | 0.50 |

- **Δ PR-AUC (IMD+ERA5 vs IMD) = −0.2524**
- **Δ ROC-AUC (IMD+ERA5 vs IMD) = −0.1110**
- Δ PR-AUC 95% storm-block bootstrap CI = **[−0.474, +0.039]** (1,999 resamples, seed 42)

For reference, the pipeline-standard per-branch evaluation (each branch on its own test split)
tells the same story — IMD 0.4282 (756 obs / 47 RI), ERA5 0.2969 (174 obs / 25 RI),
IMD+ERA5 0.3411 (174 obs / 25 RI). IMD's own test set is larger only because ERA5 reanalysis
covers 107 of the 259 storms.

**FINAL VERDICT.** On the strict common hold-out, ERA5 atmospheric features **do not add proven
predictive value beyond IMD intensity/history features**; the combined model ranks worse on
PR-AUC and ROC-AUC. Its only relative gain is higher recall (0.60 vs 0.24) at roughly half the
precision (0.33 vs 0.67) — a different operating point, not better discrimination. Uncertainty is
large at this sample size (20 test storms / 25 RI events); the bootstrap CI excludes a meaningful
positive improvement and is mostly negative, so **ERA5 is not a proven improvement — the current
evidence points to no additive value for ranking RI**.

Feature interpretation (IMD+ERA5 combined model):

1. **Which IMD variables dominate?** `max_wind_kt` (gain 112.1; mean|SHAP| 0.044), recent
   intensity tendencies `delta_v_minus_6h_kt` (84.3), `delta_v_minus_24h_kt` (53.9),
   `pressure_drop_hpa`, `wind_6h_change` — physically sensible: RI is strongly autocorrelated
   with current intensity and ongoing deepening.
2. **Which ERA5 variables contribute?** mid-level humidity structure and its changes
   (`delta_6h_r_850_minus_700`, `delta_24h_r_850_minus_500`, `r_850_minus_500`), upper-level
   divergence contrasts (`delta_24h_divergence_contrast_200_500/200_850`, `d_500`), temperature
   gradients (`t_200`, `delta_6h_t_850_minus_700`), humidity/shear deltas — consistent with the
   physical moisture–outflow–shear pathways for RI.
3. **Does ERA5 add information beyond IMD?** No on this hold-out (ΔPR-AUC −0.2524; CI
   [−0.474, +0.039]). ERA5 does not improve RI ranking; it trades precision for recall. The
   comparison is confounded by ERA5 coverage (combined model trains on 107 storms vs IMD's 259).
4. **Are the important predictors physically sensible?** Yes — IMD importance (intensity +
   tendency persistence) and the ERA5 signals (moistening, outflow/divergence, shear and
   humidity stratification) match known RI controls. Importance without proven ranking gain
   still leaves ERA5 environmentals plausible for a larger sample (more storms) or a
   low-lag-strategy model that targets recall.

Reproducibility: `python run_pipeline.py` → `python run_final_imd_era5_comparison.py`.
New artifacts: `results/{imd_only,era5_only,imd_era5_combined,model_comparison,imd_era5_feature_importance}_final.csv`,
`results/imd_era5_shap_importance.csv`, `results/final_imd_era5_experiment.json`,
`figures/{roc_curve,pr_curve,confusion}_*_final.png`,
`figures/shap_summary_combined_final.png`,
`models/{imd,era5,imd_era5}_final_xgboost.json`. Historical results archived under
`results/_historical_backup/` and left untouched.

## FINAL Satellite contribution (definitive experiment)

Experiment stage added 2026-08-31 (`run_satellite_contribution.py`). Uses the satellite CNN
OOF predictions from the Colab-trained `RICNNFusion` hybrid (IR encoder + 11 contemporaneous
IMD features) and the three canonical XGBoost models. The CNN is untouched.

**Data limits discovered during audit (the decisive facts):**

| Quantity | Value |
| --- | --- |
| Satellite OOF predictions | 9 rows / 7 storms / 6 RI / 3 non-RI |
| OOF rows with IMD coverage | 9 / 9 |
| OOF rows with ERA5 coverage | **1 / 9** ← the three-way question is untestable |
| Satellite images (full metadata) | 26 images / 23 storms / 9 RI |
| Images with ERA5 | 4 / 26 |

**Ablation on the 9-row satellite subset (`results/satellite_ablation_final.csv`):**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | N |
| --- | --- | --- | --- | --- | --- | --- |
| IMD | 0.833 | 0.944 | 1.000 | 0.333 | 0.500 | 9 |
| Satellite CNN | 0.056 | 0.516 | 0.667 | 1.000 | 0.800 | 9 |
| IMD + Satellite (prob-average) | 0.611 | 0.849 | 1.000 | 0.500 | 0.667 | 9 |
| ERA5 | N/A | N/A | N/A | N/A | N/A | 1 |
| IMD + ERA5 | N/A | N/A | N/A | N/A | N/A | 1 |
| IMD + ERA5 + Satellite | N/A | N/A | N/A | N/A | N/A | 1 |

- **Δ PR-AUC (Satellite vs IMD) = −0.428** on the 9-row subset.
- **Δ PR-AUC (IMD+Satellite vs IMD) = −0.096** (late fusion via simple probability average).
- **IMD's 9-row score (0.944) is not comparable to the satellite OOF (0.516):** the IMD
  XGBoost was trained on the full dataset and predicts on a subset that includes its own
  training storms, whereas the satellite CNN prediction is genuinely out-of-fold. Neither is
  a clean hold-out proof.

**FINAL VERDICT (satellite).** On the satellite-covered subset the satellite branch **does not
improve over IMD**; it ranks worse (PR-AUC 0.516 vs 0.944) and its ROC is degenerate. The CNN is
poorly calibrated (every OOF probability > 0.74; all 3 non-RI images scored ≥ 0.97 and would be
misclassified at any reasonable threshold). IMD+Satellite probability-averaging partially
recovers ranking but still trails IMD alone. The sample (9 obs, 7 storms) is far too small for a
statistical claim, and ERA5 overlap (1 obs) makes the three-way fusion untestable. The satellite
branch remains a **validated proof-of-concept** (architecture runs storm-safe, OOF pipeline
works, Grad-CAM attends to the cold storm core) but **is not shown to add predictive value**.

Reproducibility: `python3 run_satellite_contribution.py`.
New artifacts: `results/ri_multimodal_common_table.csv`,
`results/satellite_ablation_final.csv`,
`results/satellite_contribution_experiment.json`,
`figures/satellite_ablation_comparison_final.png`.

## FINAL multimodal experiment incl. global TCIR CNN (definitive)

Experiment stage added 2026-08-31 (`run_final_multimodal.py`). Integrates the global
**TCIR** (Tropical Cyclone Satellite Infrared) CNN rapid-intensification artifacts
(`results/tcir_oof_predictions.csv`, `results/tcir_embeddings.npy`,
`results/tcir_embeddings_meta.csv`) and produces the final six-way comparison on the largest
storm-safe common subsets.

**TCIR artifact audit (clean):** 2840 rows / 64 storms / 189 RI (prevalence 0.0665),
2003-01-21 → 2016-12-18; OOF / metadata / 128-d embeddings fully row-aligned; 0 NaN, 0 Inf,
0 duplicate rows. Probabilities are genuine out-of-fold → leakage-safe.

**The decisive coverage fact (why the three-way model cannot exist):**

| Source | Time span | 
| --- | --- |
| IMD (Bay of Bengal) | 1982–… |
| ERA5 feature table | 1982-05-01 → 2000-03-29 |
| TCIR (global) | 2003-01-21 → 2016-12-18 |

TCIR and ERA5 share **0 timestamps** ⇒ `IMD + ERA5 + TCIR` has zero observations and is
reported **NOT EVALUABLE** — never fabricated.

**Strong-overlap IO alignment (user-approved rule).** A TCIR `IO_` storm maps to an IMD storm
only on ≥30 exact-time overlaps mapping to exactly one IMD storm. 6 pairs survive, but all are
post-2000 (no ERA5), and TCIR/IMD `RI_24h` labels disagree on every pair with **0 shared RI
cases** → the aligned subset carries only **4 storms / 189 rows / 1 RI** (IMD label), so
`IMD + TCIR` is a **data-limited MVP, NOT EVALUABLE** as a metric.

**Held-out results (each model on its own storm-safe test set):**

| Model | PR-AUC | ROC-AUC | N | Storms | RI |
| --- | --- | --- | --- | --- | --- |
| IMD | 0.4282 | 0.8761 | 756 | 52 | 47 |
| ERA5 | 0.2969 | 0.7047 | 174 | 20 | 25 |
| TCIR CNN | 0.0917 | 0.5782 | 928 | 19 | 69 |
| IMD + ERA5 | 0.3411 | 0.7462 | 174 | 20 | 25 |
| IMD + TCIR | N/A (data-limited) | N/A | 189 | 4 | 1 |
| IMD + ERA5 + TCIR | N/A (disjoint) | N/A | 0 | 0 | 0 |

Storm-block bootstrap 95% CIs on PR-AUC: IMD `[0.149, 0.670]`, ERA5 `[0.098, 0.483]`,
IMD+ERA5 `[0.166, 0.499]`, TCIR `[0.050, 0.181]`.

**FINAL VERDICT (multimodal):** Satellite (TCIR) contribution **does not improve** (ΔPR-AUC
−0.337 vs IMD); ERA5 **does not improve** (ΔPR-AUC −0.087 vs IMD); the three-way model is
**not evaluable** due to disjoint TCIR/ERA5 time coverage. IMD alone remains the best evaluable
ranking model. Absolute PR-AUCs carry wide uncertainty.

Reproducibility: `python3 run_final_multimodal.py`.
New artifacts: `results/tcir_final_predictions.csv`, `results/final_multimodal_predictions.csv`,
`results/model_comparison_final.csv`, `results/tcir_contribution_experiment.json`,
`results/final_fusion_experiment.json`,
`figures/{tcir_pr,tcir_roc,multimodal_pr,multimodal_roc,calibration,fusion_confusion_matrix,model_comparison}.png`.

## Objective

A scientifically defensible, storm-safe, multimodal, interpretable, SIH-ready RI forecasting MVP. Not yet production quality; the small held-out (a handful of RI storms) means metrics carry uncertainty.

