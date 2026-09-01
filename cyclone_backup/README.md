# Tropical Cyclone Rapid Intensification (RI) Detection — Bay of Bengal

A reproducible **multimodal MVP** for detecting Rapid Intensification (RI) of
Bay of Bengal tropical cyclones, built for a Smart India Hackathon (SIH)
prototype. It fuses three data sources: IMD best-track observations, ERA5
atmospheric reanalysis, and satellite infrared (IR) imagery.

> **Honesty first.** This is an MVP. No data is fabricated, no metric is
> invented, and no storm is evaluated in training. The satellite CNN branch is
> trained in Google Colab (TensorFlow crashes locally) and reported as PENDING
> until its output is fused (see Limitations).

---

## 1. Problem statement

Rapid Intensification — a cyclone strengthening by ≥ 30 kt in 24 hours — is
one of the hardest events to forecast, and it drives most cyclone-related
casualties and damage. Early, *reliable* warning of an impending RI window
would let coastal authorities in the Bay of Bengal (a densely populated,
storm-prone basin) act sooner.

This MVP asks a concrete research question:

> **Does fusing historical-observation (IMD), atmospheric-environment (ERA5)
> and satellite spatial (IR) information improve RI detection over any single
> source?**

## 2. RI definition

- **Horizon:** 24 hours ahead of an observation.
- **Trigger:** the 24 h change in maximum sustained wind
  (`delta_v_24h_kt`) is **≥ 30 kt**.
- The binary target column is `RI_24h` (1 = RI event).

Setting lives in `config.yaml` (`ri.horizon_hours`, `ri.threshold_kt`).

## 3. Datasets

| Branch | Canonical file | Samples | Storms | RI / non-RI |
| --- | --- | --- | --- | --- |
| IMD | `models/IMD_BoB_RI_training_base.csv` | 3,211 | 259 | 179 / 3,032 |
| ERA5 | `models/RI_ERA5_features_MVP.csv` | 848 | 107 | 76 / 772 |
| Satellite | `satellite_cnn_recovered/` (recovered NC4 → `.npy`) | **26 recovered / 25 usable** | 23 | 9 / 17 |
| Multimodal | `ri_multimodal_dataset.csv` | 3,211 obs (4 identical-triplet) | 259 | 179 / 3,032 |

The pipeline now builds a single **canonical multimodal table**
(`ri_multimodal_dataset.csv`) that links every IMD observation to its matching
ERA5 row (1:1 on `(storm_id, datetime_utc)`, zero observation loss) and to its
closest **pre-target** satellite image (post-target images excluded as leakage;
see `LEAKAGE_AUDIT.md`).

- **IMD** provides observed intensity / position history (wind, pressure,
  drop, past 6/12/24 h changes).
- **ERA5** provides storm-centre atmospheric profiles: divergence (`d`),
  relative humidity (`r`), temperature (`t`), zonal/meridional wind (`u`,`v`)
  at 850/700/500/200 hPa + vertical wind shear, plus physics-derived and
  temporal-delta features.
- **Satellite IR** was **recovered from the raw NC4 granules**
  (see `satellite_cnn_recovered/recovery_verification.md`), giving 26 storm
  images (`2020-001` recovered from the previous unusable state), of which 25
  are pre-target / usable. The hybrid CNN joins each usable image to the
  canonical IMD table on `(storm_id, datetime_utc)` and requires **all 11
  contemporaneous IMD features**
  (`latitude, longitude, max_wind_kt, central_pressure_hpa, pressure_drop_hpa,
  wind_minus_6h_kt, delta_v_minus_6h_kt, wind_minus_12h_kt,
  delta_v_minus_12h_kt, wind_minus_24h_kt, delta_v_minus_24h_kt`); rows missing
  any feature are removed (never padded/imputed), leaving a clean CNN table of
  **9 rows / 7 storms (6 RI / 3 non-RI)** in
  `results/satellite_cnn_training_data.csv`.

See `PROJECT_AUDIT.md` for the full file-by-file audit.

## 4. Feature engineering

- **IMD:** used as-is (intensity history, pressure, position), plus a derived
  `wind_6h_change` (6 h wind tendency).
- **ERA5:** a *small* set of physically meaningful derived predictors was
  added (see `src/features.py`):
  - 850–500 hPa layer-mean relative humidity,
  - horizontal wind-speed magnitude at each level,
  - 200–850 hPa divergence contrast (upper-level outflow proxy),
  - 850−500 hPa temperature difference,
  - signed U/V components and direction of the 850–200 hPa wind shear.
- **Temporal deltas** (`src/features.py` → `add_temporal_features`): current −
  lagged-state `delta_6h/12h/24h_*` columns computed within each storm
  (114 columns), capturing short-range intensity change on top of the base
  fields. The `era5_delta_minutes` column is intentionally excluded from the
  ML features (it is metadata, not a predictor).

We deliberately avoid hundreds of synthetic features to prevent overfitting
the small sample.

## 5. Storm-safe evaluation (no leakage)

The single most important safeguard: **splits are made at the storm level**,
never at the observation level. The splitter (`src/data.py`) partitions storm
IDs into train / validation / test and asserts that **no storm appears in more
than one split**. To keep the three tabular models comparable, the IMD and ERA5
branches are re-split to share the *exact same held-out test storms* as the
combined model. For the definitive IMD/ERA5 comparison, all three models are
scored on a *strict common test set* — the **same 174 observations / 20 storms /
25 RI events** (`results/model_comparison_final.csv`; see § 9c). IMD alone has
52 test storms; ERA5 reanalysis only covers 107 of 259 storms, so the strict
common set restricts to the storms present in every branch.

The pipeline prints, for every split: number of storms, observations, RI /
non-RI counts, and the full train/validation/test distribution.

### 5b. Error-control / robustness framework

Beyond the storm-safe split, the pipeline now documents **explicit controls**
for leakage, dependence, imbalance, uncertainty, data coverage and baseline
skill (`ERROR_CONTROL.md`). Run `python run_robustness_checks.py` to re-run the
whole battery (`results/robustness_checks.json`):

- **Label audit & censoring** (`src/data.audit_ri_label_construction`) — rows
  with missing `RI_24h` (t+24h not yet available) are **excluded, not set to 0**
  (1791 censored rows in the raw IMD file).
- **Baselines** (`src/baselines.py`) — persistence-trend, naive persistence,
  climatology. The IMD model (PR-AUC 0.594) **beats the best baseline**,
  persistence-trend (0.373), by ΔPR-AUC +0.221, so the AI is not just
  continuing the storm's previous intensity trend.
- **Storm-bootstrap CI** (`src/evaluate.storm_bootstrap_ci`) — resamples
  *storms*, not rows. IMD PR-AUC 0.594 (95 % CI [0.333, 0.753]).
- **Probability calibration** (`src/evaluate.calibration_detailed`) — Brier,
  reliability curve, calibration slope/intercept, isotonic regression. IMD
  Brier 0.145 → 0.078 after isotonic; outputs are slightly overconfident.
- **Event-level metrics** (`src/event_metrics.py`) — RI episodes detected,
  false alarms/storm, warning lead time.
- **Land interaction** (`src/features.add_land_interaction_features`) —
  distance-to-land/coast and over-land features plus an ocean-only (>300 km)
  sensitivity split.
- **Preprocessing leakage guards** (`src/leakage.check_preprocessing_leakage`)
  — scalers fit on training storms only; SMOTE is ablation-only and never
  applied before the storm split.

The **ERA5 reanalysis caveat** also applies throughout: reanalysis is used to
develop and validate the environment branch; operational deployment would
replace it with real-time analysis / NWP forecast fields available at issuance
time.

## 6. Training methodology

- **Models:** XGBoost for all tabular branches; a **single canonical satellite
  CNN** (`src/satellite_cnn.py`, hybrid `RICNNFusion` IR encoder + tabular
  head) trained in Google Colab (PyTorch/TensorFlow crash on the local macOS
  environment).
- **Grouped CV:** inner **grouped-by-storm** cross-validation on the training
  set (`src/model_comparison.py`) reports model-selection PR-AUC for
  LR / RF / XGB / HGB. RF and LR are wrapped in
  `make_pipeline(SimpleImputer(median), ...)`.
- **Class imbalance:** tabular branches use `scale_pos_weight` / class weights
  from training folds only; the satellite CNN uses **focal loss** (α=0.75, γ=2).
- **Tuning (modest & explainable):** learning rate, max depth, min child
  weight, subsample, colsample_bytree. No grid-search explosion.
- **Objective:** optimised primarily for **PR-AUC** (not accuracy), because RI
  is rare (≈ 6–9 % of samples).
- **Decision threshold:** selected **on the validation set only** (maximising
  F1), then applied frozen to the test set. Test labels are never used to
  choose the threshold.
- **Reproducibility:** fixed random seed (`seed: 42`) applied everywhere.

### Late fusion

Instead of retraining end-to-end, branch probabilities are **stacked** and fed
to a lightweight logistic-regression meta-classifier. It is trained on
*training storms only* and evaluated on held-out test storms. The satellite
CNN's **out-of-fold** probabilities (a net that never saw each image's storm)
are the fuel, so the meta-model never sees in-sample predictions.

## 7. Model architecture (summary)

```
IMD best-track   ──► XGBoost ─────► P_imd   ─┐
ERA5 reanalysis  ──► XGBoost ─────► P_era5  ─┼─► meta-classifier (logistic)
                                                 (fused RI probability)
Satellite IR ────► RICNNFusion ───► P_cnn  ──┘  (Tb + valid-mask, focal loss,
(128x128 crop)        (src/satellite_cnn.py)       embeddings for feature fusion)
    └── fused with a real 11-IMD-feature tabular head
        (latitude, longitude, max_wind_kt, central_pressure_hpa,
         pressure_drop_hpa, wind_minus_6h_kt, delta_v_minus_6h_kt,
         wind_minus_12h_kt, delta_v_minus_12h_kt, wind_minus_24h_kt,
         delta_v_minus_24h_kt)  ← contemporaneous at time t; ERA5 stays separate
```

The **single** satellite CNN lives in `src/satellite_cnn.py`; the official demo
(`TC_RI_CNN_Demo.ipynb`) and the pipeline both import it (no second
implementation). The satellite CNN consumes a storm-centred IR image (Tb +
valid-pixel mask) **and the 11 contemporaneous IMD intensity/trend features**
(no future/target-time values; `RI_24h` is only the label). ERA5 remains a
separate environmental branch, combined only at the multimodal fusion stage.

## 8. Results (held-out test storms)

These are the numbers produced by `python run_pipeline.py` using a fixed seed.
Tabular branches are evaluated fully here; the satellite CNN runs in Colab.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| IMD only | 0.4282 | 0.8761 | 0.586 | 0.362 | 0.447 | 0.1319 |
| ERA5 only | 0.2969 | 0.7047 | 0.286 | 0.080 | 0.125 | 0.1420 |
| IMD + ERA5 | 0.3411 | 0.7462 | 0.326 | 0.600 | 0.423 | 0.2267 |
| Late fusion (IMD+ERA5) | 0.4004 | 0.8059 | 0.500 | 0.240 | 0.324 | 0.1165 |
| Satellite CNN (OOF, 9 obs) | 0.5161 | 0.0556 | 0.667 | 1.000 | 0.800 | N/A |

**Read honestly:** the hold-out is small (a handful of RI storms), so PR-AUC
values are noisy and not indicative of production quality. Notably, **IMD alone
is the strongest tabular signal and adding raw ERA5 features does NOT improve on
it here** — an honest negative result, not a claim ERA5 is useless (it is the
only source covering storms without IMD trend data, and it dominates the
IMD+ERA5 model's dynamic range).

## 9. Answer to the research question

On this MVP hold-out, **the IMD branch is the single strongest predictor** and
late-fusion does not beat it. The honest finding: **integrating tabular ERA5
into the combined feature model did not improve over IMD alone**, consistent
with limited EMA/feature overlap and a small sample. The **satellite CNN branch
is the remaining untested modality** — recoverable imagery now exists (25 usable
images / 8 RI) and is ready to fuse via Colab, so the multimodal question is
partially answered.**

## 9b. Multimodal fusion status

With 25 usable satellite images and only 4 identical (IMD+ERA5+satellite)
triplets, a storm-safe training/evaluation of the *full* three-way fusion is
not yet statistically meaningful on this sample. The pipeline therefore:

1. trains the satellite CNN branch separately in Colab (storm-safe
   StratifiedGroupKFold OOF),
2. produces OOF probabilities + embeddings the tabular pipeline will fuse,
3. reports the fused result as **PENDING** until that output is ingested.

> **CNN tabular-join note.** The hybrid CNN requires all **11 contemporaneous
> IMD features** at each image's observation time. Only **9 rows / 7 storms
> (6 RI / 3 non-RI)** of the 25 usable images have them (the rest are
> early-lifecycle fixes lacking the `-6h/-12h/-24h` lag features); those rows
> are removed, never padded/imputed (`results/satellite_cnn_training_data.csv`).
> The hybrid CNN is therefore trained on this small, leakage-free set.

## 9c. FINAL IMD + ERA5 comparison (definitive, storm-safe)

Added 2026-08-31 — `run_final_imd_era5_comparison.py`. Reuses the canonical
datasets and the canonical XGBoost models from `run_pipeline.py` (seed 42);
the satellite CNN is untouched and no ocean data was added. **No predictor is
derived from `RI_24h`** (forbidden set: `RI_24h`, `wind_24h_kt`,
`delta_v_24h_kt`, `target_time_24h`). Class imbalance uses
`scale_pos_weight` from training splits only; thresholds are validated-tuned
only.

Strict common test set — the same 174 observations / 20 storms / 25 RI events
scored by all three models:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| IMD | 0.8572 | 0.5935 | 0.667 | 0.240 | 0.353 | 0.72 |
| ERA5 | 0.2969 | 0.7047 | 0.286 | 0.080 | 0.125 | 0.63 |
| IMD + ERA5 | 0.7462 | 0.3411 | 0.326 | 0.600 | 0.423 | 0.50 |

- **Δ PR-AUC (IMD+ERA5 vs IMD) = −0.2524**, Δ ROC-AUC = −0.1110.
- Δ PR-AUC 95% storm-block bootstrap CI = **[−0.474, +0.039]** (1,999 resamples).
- Pipeline-standard per-branch test (IMD on its own 756 obs / 47 RI set) gives
  IMD 0.4282; ERA5 and IMD+ERA5 are identical rows (174 obs / 25 RI) → 0.2969
  and 0.3411. Same conclusion either way.

**Verdict:** ERA5 does **not** add proven ranking value beyond IMD on this
hold-out. The combined model's only gain is recall (0.60 vs 0.24) at much lower
precision (0.33 vs 0.67). With only 20 test storms / 25 RI events, the CI
excludes a meaningful positive gain; tiny numerical differences must not be
read as improvements. Full feature interpretation is in
`results/imd_era5_feature_importance.csv` (gain) and
`results/imd_era5_shap_importance.csv` (mean|SHAP|):
IMD dominance = `max_wind_kt`, `delta_v_minus_6h_kt`, `delta_v_minus_24h_kt`,
`pressure_drop_hpa`; ERA5 contribution = humidity-structure deltas
(`delta_6h_r_850_minus_700`, `delta_24h_r_850_minus_500`), upper-level
divergence contrasts, temperature/shear deltas — physically sensible but, on
the current 25-event hold-out, not a ranking improvement.

## 9d. FINAL Satellite contribution (definitive, § 2c)

Added 2026-08-31 — `run_satellite_contribution.py`. Uses the Colab-trained
`RICNNFusion` CNN OOF probabilities and the three canonical XGBoost models.

**Data limits (decisive).** Satellite OOF covers 9 rows / 7 storms / 6 RI of
which **all 9 have IMD features but only 1 has ERA5** — so the three-way
(IMD+ERA5+Satellite) question is untestable; only IMD vs Satellite is
evaluable.

Ablation on the 9-row subset (`results/satellite_ablation_final.csv`):

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | N |
| --- | --- | --- | --- | --- | --- | --- |
| IMD | 0.9444 | 0.8333 | 1.000 | 0.333 | 0.500 | 9 |
| Satellite CNN | 0.5161 | 0.0556 | 0.667 | 1.000 | 0.800 | 9 |
| IMD + Satellite | 0.8486 | 0.6111 | 1.000 | 0.500 | 0.667 | 9 |
| ERA5 / IMD+ERA5 / full | N/A | N/A | N/A | N/A | N/A | 1 |

- **Δ PR-AUC (Satellite vs IMD) = −0.428**; Δ (IMD+Satellite vs IMD) = −0.096.
- IMD's 0.944 is from the full-data XGBoost predicting on a subset it trained
  on, so it is **not** comparable to the genuinely out-of-fold satellite OOF.
- **Verdict:** the satellite branch does **not** show added value at this
  sample size; the CNN is poorly calibrated (all 9 OOF scores > 0.74, all 3
  non-RI images ≥ 0.97). Proof-of-concept validated (architecture + OOF
  pipeline work, Grad-CAM attends to the cold core), but the multimodal fusion
   question stays unanswered — it needs more satellite images and wider ERA5
   coverage of satellite storms.

## 9e. ERA5 expansion for satellite overlap (data-quality task, § 2e)

Added 2026-08-31 — `run_era5_expansion.py` (audit) +
`run_era5_expansion_stage2.py` (CDS download) +
`run_era5_expansion_stage2b.py` (extract + build). The goal was to make the
three-way (IMD + ERA5 + Satellite) dataset scientifically usable by closing the
ERA5 coverage gap for the existing satellite observations — a strict data
expansion/quality task, **before** any fusion retraining.

**Why.** The prior satellite audit found only **4 / 26** satellite observations
had ERA5 coverage (`run_satellite_contribution.py`), because canonical ERA5
features only extended to storms where reanalysis had been extracted earlier
(roughly to 2000), while the satellite images span **1998–2025**.

**What was done (modest, satellite-priority download).** 20 Copernicus CDS
`reanalysis-era5-pressure-levels` extractions (divergence `d`, relative
humidity `r`, temperature `t`, u/v wind at 850/700/500/200 hPa — identical to
the canonical ERA5 feature set) were downloaded for the exact dates/times of
the 22 satellite observations that lacked ERA5. Each was a small storm-centred
box; no reanalysis after time `t` is used (`era5_delta_minutes = 0`). Features
were bilinearly interpolated to each storm centre and appended to the canonical
ERA5 format (raw values, **not** normalized), preserving the original
`models/RI_ERA5_features_MVP.csv` untouched.

**Results (`results/era5_audit_summary.json`, `results/RI_ERA5_features_expanded.csv`, `results/satellite_imd_era5_common_expanded.csv`):**

| Overlap | Before | After |
| --- | --- | --- |
| Satellite images | 26 | 26 |
| Satellite storms | 23 | 23 |
| Satellite + ERA5 | 4 obs | **26 obs (+22)** |
| Satellite + IMD + ERA5 | 4 obs / 3 RI | **26 obs / 23 storms / 9 RI** |
| Satellite + IMD + ERA5 non-RI | 1 | **17** |

**Feasibility of three-way fusion.** Every existing satellite observation now
has full IMD + ERA5 + Satellite coverage **with valid ERA5 features (26/26 non-null)**,
across **23 independent storms / 9 RI / 17 non-RI**. This is a large, honest
improvement over the 4 obs / 1 usable storm before. It does **not** reach the
50+ obs aspirational target because only **26 satellite images** exist in the
project — the bottleneck is satellite imagery, not ERA5. The three-way
*feature-level* fusion experiment is now **constructible** on 26
observations, but remains statistically thin; a meaningful fusion evaluation
still needs many more satellite images (see `tc_ri_cnn/README.md`).

**Leakage audit (§ `LEAKAGE_AUDIT.md`): PASS** — exact-time matching only,
no future fields, `RI_24h` is label-only, no global normalization, canonical
ERA5 preserved, no duplicate storm/time rows.


## 9f. FINAL multimodal experiment with the global TCIR CNN (definitive)

Added 2026-08-31 — `run_final_multimodal.py`. Integrates the new **TCIR CNN**
(Tropical Cyclone Satellite Infrared) rapid-intensification artifacts
(`results/tcir_oof_predictions.csv`, `results/tcir_embeddings.npy`,
`results/tcir_embeddings_meta.csv`) into the IMD + ERA5 evaluation and runs the
final six-way model comparison on the largest **storm-safe** common subsets.

**TCIR artifact audit (`results/tcir_contribution_experiment.json`, audit
block):** 2840 rows / 64 storms / 189 RI (prevalence 0.0665), datetime
2003-01-21 → 2016-12-18; OOF, metadata and 128-d embeddings fully row-aligned;
0 NaN / 0 Inf / 0 duplicate `(storm_id, datetime_utc)` rows. **Leakage-safe** —
the probabilities are genuine out-of-fold, and nothing after observation time
is used.

**Decisive coverage finding (blocks the three-way fusion).** TCIR spans
**2003–2016** while the ERA5 feature table covers **1982-05-01 → 2000-03-29
only** → there are **0 observations with all three modalities**
(`temporal_overlap_with_era5 = 0`). The `IMD + ERA5 + TCIR` model is
**mathematically impossible** on these files and is reported as **NOT
EVALUABLE**, never fabricated.

**Strong-overlap IO alignment (user-approved).** A TCIR `IO_` storm is matched
to an IMD storm only when it shares ≥ 30 exact 3-hourly timestamps **and**
maps to exactly one IMD storm. 6 pairs survive (2003-001, 2005-010, 2013-001,
2013-009, 2013-010, 2016-009). **Known label conflict:** for every pair TCIR
and IMD `RI_24h` disagree and there are **0 shared RI cases**; all matched
storms are post-2000 so none have ERA5.

**Hold-out results (storm-safe; each model on its own valid test set):**

| Model | PR-AUC | ROC-AUC | N obs | Storms | RI |
| --- | --- | --- | --- | --- | --- |
| IMD | 0.4282 | 0.8761 | 756 | 52 | 47 |
| ERA5 | 0.2969 | 0.7047 | 174 | 20 | 25 |
| TCIR CNN | 0.0917 | 0.5782 | 928 | 19 | 69 |
| IMD + ERA5 | 0.3411 | 0.7462 | 174 | 20 | 25 |
| IMD + TCIR | **N/A** | **N/A** | 189 | 4 | 1 |
| IMD + ERA5 + TCIR | **N/A** | **N/A** | 0 | 0 | 0 |

**Storm-block bootstrap 95% CIs on PR-AUC (percentile, resampling whole
storms):** IMD `[0.149, 0.670]`, ERA5 `[0.098, 0.483]`, IMD+ERA5
`[0.166, 0.499]`, TCIR `[0.050, 0.181]` → wide uncertainty confirms absolute
values are not conclusive.

**Why IMD+TCIR is NOT EVALUABLE:** only **4 storms / 189 rows / 1 RI event**
survive the ≥30-timestamp match, so no storm-safe train/val/test hold-out can
produce a valid ROC/PR-AUC (single-class test fold). Reported honestly as
data-limited MVP, not given a fabricated number.

**Verdicts:**
- **Satellite (TCIR) contribution: DOES NOT IMPROVE** (ΔPR-AUC vs IMD = −0.337).
- **ERA5 contribution: DOES NOT IMPROVE** (ΔPR-AUC vs IMD = −0.087).
- **Final fusion verdict: NOT EVALUABLE** for the three-way model (disjoint
  time coverage). IMD alone is the best evaluable model.

**Outputs:** `results/tcir_final_predictions.csv`,
`results/final_multimodal_predictions.csv`,
`results/model_comparison_final.csv`,
`results/tcir_contribution_experiment.json`,
`results/final_fusion_experiment.json`; figures `tcir_pr_curve.png`,
`tcir_roc_curve.png`, `multimodal_pr_curve.png`, `multimodal_roc_curve.png`,
`calibration_curve.png`, `fusion_confusion_matrix.png`, `model_comparison.png`.

**Reproduce:** `python run_final_multimodal.py`


## 10. Running the pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt
#    (for the satellite CNN, on a torch host e.g. Google Colab)
pip install torch

# 2. Run the full tabular pipeline (evaluated end-to-end on macOS)
python run_pipeline.py

# 2b. FINAL IMD+ERA5 comparison (definitive, strict common test set; § 9c)
python run_final_imd_era5_comparison.py

# 2c. FINAL Satellite contribution experiment (§ 9d)
python3 run_satellite_contribution.py

# 2d. ERA5 expansion for satellite overlap (§ 9e) — audit + download + extract
# (needs a CDS API key in ~/.cdsapirc; exports are saveable across runs)
python3 run_era5_expansion.py              # audit + coverage gap (no download)
python3 run_era5_expansion_stage2.py       # downloads the needed dates (one-time)
python3 run_era5_expansion_stage2b.py      # extract features + build common table

# 2e. FINAL multimodal experiment incl. global TCIR CNN (§ 9f)
python3 run_final_multimodal.py

# 3. Satellite CNN branch — PyTorch/TensorFlow crash locally, so train in
#    Google Colab using the OFFICIAL demo notebook (imports src/satellite_cnn.py),
#    pointed at the recovered data:
#      - open TC_RI_CNN_Demo.ipynb in Colab
#      - run the training cell (run_cnn_oof) to write:
#          results/satellite_oof_predictions.csv
#          results/satellite_embeddings.npy (+ _meta.csv)
#          models/satellite_cnn.pt
#      - copy those back into this repo
#    Then fuse + answer the research question:
python run_pipeline.py --fusion

# CLI summary
python run_pipeline.py               # full tabular pipeline + ingest CNN output
python run_pipeline.py --satellite   # build clean 11-feature CNN table + audit, then attempt OOF training (run in Colab)
python run_pipeline.py --fusion      # force multimodal fusion over ingested branches
```

Outputs:

```
ri_multimodal_dataset.csv            <- canonical multimodal table
LEAKAGE_AUDIT.md                     <- leak rules + audit result (incl. CNN leak-type matrix)
satellite_cnn_recovered/recovery_verification.md   <- NC4 recovery + 2020-001 match
results/final_comparison.csv         <- metric table
results/model_comparison.csv         <- canonical comparison (same content)
results/tc_ri_cnn_audit.md           <- CNN integration + architecture comparison
results/final_metrics.json           <- per-model metrics + thresholds + confusion
results/model_family_benchmark_{imd,era5,combined}.csv  <- LR/RF/XGB/HGB CV
results/final_ablation.csv           <- ablation
results/error_analysis.csv
results/{imd,era5,combined}_test_predictions.csv
results/*_feature_importance.csv
figures/pr_curve_*.png               <- precision-recall curves
figures/confusion_*.png              <- confusion matrices
figures/gradcam/*.png                <- TP/TN/FP/FN Grad-CAM (from the demo)
figures/shap_summary_combined.png    <- only if `shap` installed
models/{imd,era5,imd_era5}_xgboost.json   <- trained models
results/{imd_only,era5_only,imd_era5_combined,model_comparison,imd_era5_feature_importance}_final.csv  <- FINAL IMD+ERA5 comparison (§ 9c)
results/final_imd_era5_experiment.json   <- final experiment metadata + deltas + bootstrap CI
figures/{roc_curve,pr_curve,confusion}_*_final.png   <- FINAL comparison figures
figures/shap_summary_combined_final.png  <- SHAP summary (when `shap` installed)
models/{imd,era5,imd_era5}_final_xgboost.json  <- FINAL model copies
results/ri_multimodal_common_table.csv   <- satellite/common three-branch table (§ 9d)
results/satellite_ablation_final.csv     <- satellite ablation (§ 9d)
results/satellite_contribution_experiment.json  <- satellite experiment metadata (§ 9d)
figures/satellite_ablation_comparison_final.png  <- satellite ablation figure (§ 9d)
models/satellite_cnn.pt              <- canonical satellite CNN weights
results/satellite_cnn_training_data.csv  <- clean 11-feature CNN table (9 rows/7 storms)
results/cnn_tabular_scaler.json     <- per-fold training-only scaler stats
results/cnn_before_after.csv        <- OLD(placeholder, deprecated) vs NEW(real 11-feature)
models/predict_ri.py                 <- load + predict helper (verified)
TC_RI_CNN_Demo.ipynb                 <- official satellite CNN demo (imports src/)
satellite_cnn_colab_upload.zip       <- recovered data packaged for Colab upload
ETA5_expanded/                       <- 20 downloaded ERA5 pressure-level NetCDF files (§ 9e)
results/satellite_era5_coverage_before.csv   <- per-satellite-obs ERA5 coverage before expansion
results/era5_audit_summary.json       <- audit + after/expansion summary (§ 9e)
results/era5_expanded_manifest.csv     <- download manifest (20 ok / 0 err)
results/era5_download_validation.csv   <- validation (20/20 valid)
results/RI_ERA5_features_expanded.csv  <- expanded ERA5 table (870 rows / 126 storms)
results/satellite_imd_era5_common_expanded.csv  <- three-way common table (26 obs/23 storms/9 RI)
results/tcir_final_predictions.csv     <- TCIR CNN test predictions (§ 9f)
results/final_multimodal_predictions.csv  <- final per-model multimodal summary (§ 9f)
results/model_comparison_final.csv     <- six-model comparison table (§ 9f)
results/tcir_contribution_experiment.json  <- TCIR audit + contribution (§ 9f)
results/final_fusion_experiment.json   <- final fusion experiment + bootstrap CIs (§ 9f)
figures/{tcir,multimodal}_*_curve.png   <- TCIR + multimodal PR/ROC (§ 9f)
figures/calibration_curve.png           <- reliability diagram (§ 9f)
figures/fusion_confusion_matrix.png     <- fusion confusion matrices (§ 9f)
figures/model_comparison.png            <- PR-AUC bar chart (§ 9f)
```

## 11. Reproducible seeds

- Global seed: `config.yaml` → `seed: 42`.
- Used by the splitter (storm shuffle), XGBoost `random_state`, the fusion
  meta-learner and the CNN data generator.
- Re-running `python run_pipeline.py` reproduces the results above.

## 12. Limitations

1. **Satellite CNN branch runs only in Colab** (PyTorch/TensorFlow crash on
   this macOS box). The imagery is fully recovered (26 images / 25 usable
   across 23 storms). After the strict 11-feature join to IMD, the hybrid CNN
   trains on **9 rows / 7 storms** (`results/satellite_cnn_training_data.csv`).
   The satellite OOF predictions have been ingested and evaluated
   (`run_satellite_contribution.py`, § 9d): **satellite does not improve over
   IMD on this subset** (PR-AUC 0.516 vs 0.944) — honestly reported, N=9.
2. **Very small multimodal overlap** — only **1 (IMD+ERA5+satellite)
   triplet**, so a three-way *feature-level* fusion is **not testable**; the
   three-way row of the ablation table is descriptive only.
3. **Small tabular hold-out** (a handful of RI test storms). PR-AUC values
   carry large uncertainty; do not treat absolute numbers as conclusive.
4. **IMD alone > IMD+ERA5 on this hold-out** — honest negative result for the
   raw ERA5 feature stack; the ERA5 branch still contributes where IMD trend
   data is lacking.
5. **No temporal sequence** — each observation is modelled independently
   (temporal-delta features added, but no RNN/sequence model).
6. **Single-basin, historical IMD** — may not transfer to other basins /
   future storm climatology.
7. **ERA5 coverage** only extends to storms where reanalysis was downloaded
   (107 of 259 storms).
8. **Satellite CNN class balance weak** (6 RI / 3 non-RI in the strict
   11-feature training set) and no repeated-resampling variance quantification
   on the single test split. Metrics require a fresh Colab run and are noisy.
9. **Global TCIR CNN (multi-basin, 2003–2016) is not ERA5-coeval** — zero
   temporal overlap with the ERA5 table means the three-way model is
   not constructible (§ 9f). TCIR also cannot run locally (TensorFlow plays
   badly on this macOS box), so its result uses precomputed OOF/embeddings.
10. **IMD+TCIR aligned subset is tiny and label-inconsistent** — only
    4 storms / 1 shared RI event, with TCIR and IMD `RI_24h` labels
    disagreeing on every matched storm; reported as data-limited MVP, not a
    usable fusion metric (§ 9f). Wide storm-block bootstrap CIs (e.g. IMD
    `[0.149, 0.670]`) confirm all absolute PR-AUCs are uncertain.

## 13. Future work

- **Grow the satellite dataset** — the dominant limitation. Train the CNN on
  hundreds of images / more storms, and broaden ERA5 reanalysis coverage of
  satellite storms, so the three-way question (IMD vs IMD+ERA5 vs
  IMD+ERA5+Satellite) becomes statistically testable.
- **Re-tune the late-fusion meta-classifier** once a larger satellite set exists
  (probability-level logistic stacking over OOF base probabilities).
- **Ocean heat content** — the repo already contains CMEMS ocean netCDF files
  (`models/cmems_*.nc`); engineering an ocean-heat-content predictor is a
  natural next step.
- **Temporal satellite sequences** — stack consecutive IR crops rather than a
  single frame.
- **Multi-horizon prediction** — forecast RI probability at 12/24/36/48 h.
- **Uncertainty quantification** (conformal prediction / MC-dropout).
- **Repeated storm split resampling** to quantify the variance of the metrics.

---

**Note:** baseline notebook outputs (`models/TC1.ipynb`, `models/TC2.ipynb`)
and the original baseline artefacts (`models/xgboost_IMD_only_baseline.pkl`,
`models/era5_only_xgboost_mvp.json`, `models/satellite_ir_cnn_mvp*.keras`)
are preserved. The improved pipeline (`run_pipeline.py` + `src/`) reproduces
all current results with a fixed seed and storm-safe splits.
