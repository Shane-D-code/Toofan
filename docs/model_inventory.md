# TOOFAN Model Inventory

This inventory documents every model integrated into the TOOFAN pipeline,
with emphasis on the **Genesis** (24-hour tropical cyclone genesis) module.

> **Genesis scientific status: PROTOTYPE.**
> The Genesis dataset is 300 samples / 191 North Indian Ocean storms / 2015–2024.
> The report states synthetic SST/SST-anomaly and TCHP/OHC700 values were used,
> and storm-aware cross-validation performance was lower than the final held-out
> test performance. Do **not** claim "production validated", "operationally
> proven", or "highly accurate" without new evidence. The *architecture* is
> production-oriented; the *scientific model* remains prototype-level.

---

## Genesis Module

**Task:** 24-hour tropical cyclone genesis prediction (binary classification).

| Attribute | Value |
|-----------|-------|
| Target | `genesis_24h` (0 = non-genesis, 1 = genesis) |
| Samples | 300 (150 genesis / 150 non-genesis) |
| Storms | 191 (North Indian Ocean, 2015–2024) |
| Input features | 34 |
| Decision threshold | **0.24** (optimized, NOT 0.50) |
| Output | Probability of genesis (class 1) |

### Active Models

Only **three** models are active in the Genesis module:

| Role | Model | Weight (ensemble) |
|------|-------|-------------------|
| **PRIMARY / PRODUCTION** | **LightGBM** | 0.40 |
| Ensemble component | XGBoost | 0.35 |
| Ensemble component | RandomForest | 0.25 |

The production path uses LightGBM only. The **calibrated soft-voting
ensemble** (`genesis_soft_voting_ensemble`) is the secondary / report / demo /
comparative prediction path.

```
p_ensemble = 0.40 * p_lightgbm + 0.35 * p_xgboost + 0.25 * p_randomforest
```

The ensemble vote is a **weighted probability vote** — it does NOT majority-vote
hard labels, does NOT average class labels, and does NOT use equal weights.

### Approved Artifacts (exact filenames)

These are the **optimized** artifacts used by the integration:

| Model | Artifact |
|-------|----------|
| LightGBM | `tc_genesis_lightgbm_300_OPTIMIZED.joblib` |
| XGBoost | `tc_genesis_xgboost_300_OPTIMIZED.joblib` |
| RandomForest | `tc_genesis_randomforest_300_OPTIMIZED.joblib` |
| Imputer | `tc_genesis_300_imputer.joblib` |

> The older `tc_genesis_BEST_MODEL_300.joblib` and non-optimized
> `tc_genesis_lightgbm_300.joblib` / `tc_genesis_xgboost_300.joblib` /
> `tc_genesis_randomforest_300.joblib` artifacts are **not** used. The adapter
> actively rejects the generic `BEST_MODEL` artifact.

### Excluded Models (challengers only)

The following are **NOT** active Genesis models and are never loaded:

- CatBoost
- ExtraTrees
- GradientBoosting
- HistGradientBoosting
- Logistic Regression
- SVM / SVC
- Neural networks / LSTM / CNN / TCN

They may appear in historical challenge comparisons only.

### Calibration semantics

The ensemble weights (0.40 / 0.35 / 0.25) are a **soft-voting weighting**, not a
learned calibration. No calibration artifact exists in the repository, so the
integration preserves the raw weighted ensemble probability and reports
`calibrated = false`. The implementation clearly distinguishes
`raw_probability` from `calibrated_probability`.

---

## Other TOOFAN Modules (context)

| Module | Model | Framework | Input | Output |
|--------|-------|-----------|-------|--------|
| Cyclone Path | Transformer V12 | PyTorch | Track history | Future positions |
| RI | XGBoost + CNN | XGBoost / Keras | IMD + ERA5 + satellite | P(RI in 24h) |
| Recurvature | XGBoost (+LSTM/TCN) | XGBoost / TF | Track features | P(recurve 24h) |
| Flood | XGBoost | scikit-learn | IMERG + Copernicus | P(flood) |
| Rainfall | Random Forest (2-stage) | scikit-learn | track + IMERG | rainfall + prob |
| Wind | Neural network | Keras | ERA5 U10/V10 | U10/V10 field |

These modules are standalone and scheduled by the orchestrator; Genesis does
not call them directly.
