# Genesis Integration Audit — TOOFAN

Generated during the Genesis model integration into the TOOFAN pipeline.

---

## 1. Exact artifacts used

The Genesis module loads the following **optimized** artifacts (authoritative
production set). All other Genesis artifacts are **not** used.

| Role | Artifact filename |
|------|-------------------|
| LightGBM (PRIMARY / PRODUCTION) | `tc_genesis_lightgbm_300_OPTIMIZED.joblib` |
| XGBoost (ensemble) | `tc_genesis_xgboost_300_OPTIMIZED.joblib` |
| RandomForest (ensemble) | `tc_genesis_randomforest_300_OPTIMIZED.joblib` |
| Imputer | `tc_genesis_300_imputer.joblib` |

The generic `tc_genesis_BEST_MODEL_300.joblib` and the non-optimized
`tc_genesis_lightgbm_300.joblib` / `tc_genesis_xgboost_300.joblib` /
`tc_genesis_randomforest_300.joblib` artifacts are **deliberately rejected** by
the adapter guard rails (see section 15 / the `DISALLOWED_ARTIFACT_TOKENS` list).

> **Important:** The trained production artifacts are **not present** in this
> repository. The integration therefore cannot record their exact on-disk
> SHA-256 here. Instead, every loader computes and records the SHA-256 of each
> artifact **at load time** (see section 2). The verification run below used
> inline tiny models serialized with the same OPTIMIZED filenames to exercise
> the identical loading / hashing / prediction mechanics.

## 2. SHA-256 hashes

SHA-256 is computed over the raw bytes of each artifact on disk by
`toofan.core.schemas.sha256_of_file` and stored in the adapter's provenance
(`artifact_hash_sha256`) on load.

Verification-run hashes (inline fixtures, same filenames):

| Model | SHA-256 (verification fixture) |
|-------|--------------------------------|
| LightGBM | `b10440e293549067f759de6126c18d53ff83eb1d53965c22455c9b5e08774c40` |
| XGBoost | `0dfe847e3f59657d671ddec983377039dbc8609c8bfd3794ba7093a8384c0407` |
| RandomForest | `b4f59bce2a0804ebf160c750cde74e86a0d3af62b3a98e3410c16d58664b12aa` |

These hashes correspond to the tiny models used to exercise the pipeline and
MUST be replaced by the production artifacts' hashes once the real OPTIMIZED
artifacts are placed in `artifacts/genesis/`. The loader records the true,
current hash automatically.

## 3. Model types

| Model | Framework | Model type |
|-------|-----------|------------|
| LightGBM | lightgbm | gradient boosting (primary) |
| XGBoost | xgboost | gradient boosting (ensemble) |
| RandomForest | sklearn | random forest (ensemble) |

## 4. Feature schema

The Genesis models expect a fixed **34-feature** ordered representation
(`toofan.genesis.features.GENESIS_FEATURES`):

```
candidate_latitude, candidate_longitude, month_sin, month_cos, sst,
sst_anomaly, tchp, ohc700, wind850, wind200, vort850, vort200, div850,
div200, shear_mag, shear_u, shear_v, rh500, rh700, rh850, speed_850,
speed_700, speed_500, omega500, pwat, cape, mslp, wind_surface, sst_grad,
dist2land, latitude_anomaly, longitude_anomaly, tempo_sst_anom, pres_anomaly
```

Reorder / mismatch is rejected with `GenesisFeatureError`; missing fields are
rejected with `GenesisInsufficientInput` (status `UNAVAILABLE`).

## 5. Preprocessing path

```
raw feature dict (34 named values)
  -> ordered vector (canonical GENESIS_FEATURES order)
  -> existing imputer transform (tc_genesis_300_imputer.joblib), if present
  -> validated ndarray -> model.predict_proba
```

No double imputation, no double transformation, no reordering, no arbitrary
filling. If a required feature is unavailable, an explicit `UNAVAILABLE` status
is returned (never fabricated).

## 6. Threshold

`genesis.threshold = 0.24` (documented optimized threshold; **not** 0.50).

It is a configuration value, decoupled from model weights so future validated
updates can be made without retouching the artifacts.

## 7. Ensemble weights

Calibrated soft-voting ensemble `genesis_soft_voting_ensemble`:

```
p_ensemble = 0.40 * p_lightgbm + 0.35 * p_xgboost + 0.25 * p_randomforest
```

Exact weights: LightGBM = 0.40, XGBoost = 0.35, RandomForest = 0.25.
This is a weighted **probability** vote — no hard-label majority voting, no
class-label averaging, no equal weights.

Deterministic mocked test: LightGBM=0.80, XGBoost=0.60, RandomForest=0.40
⇒ ensemble = 0.40(0.80)+0.35(0.60)+0.25(0.40)=0.32+0.21+0.10=**0.63**
(asserted within 1e-9 floating-point tolerance).

## 8. Original-vs-adapter fidelity results

Each adapter prediction was compared against a direct, raw framework call on
identical input (single 34-feature row):

| Model | Fidelity check | Result |
|-------|----------------|--------|
| LightGBM | `raw Booster.predict == adapter.predict_proba` | MATCH (allclose ≤ 1e-6) |
| XGBoost | `raw predict_proba[:,1] == adapter.predict_proba` | MATCH (allclose ≤ 1e-6) |
| RandomForest | `raw predict_proba[:,1] == adapter.predict_proba` | MATCH (allclose ≤ 1e-6) |

The adapter does **not** alter the underlying prediction, preprocessing, or
feature ordering.

## 9. Tests executed

- `tests/test_genesis_adapters.py` — A, B, C, D, E, F, I, J (loading +
  prediction + bounded probability + no-substitution guards)
- `tests/test_genesis_ensemble.py` — G (exact 0.63), H (threshold 0.24),
  weights, missing-component behavior
- `tests/test_genesis_factory_provenance.py` — O (factory registration),
  K (provenance), L (SHA-256)
- `tests/test_genesis_service.py` — M (missing artifact), N (invalid features),
  modes, availability
- `tests/test_genesis_orchestrator.py` — P (Phase 1 / Phase 2), Q (native
  runtime compatibility)
- `tests/test_genesis_fidelity.py` — original-vs-adapter fidelity
- `scripts/verify_genesis_integration.py` — end-to-end verification runner

## 10. Test results

- Full pytest suite: **50 passed, 0 failed**.
- `scripts/verify_genesis_integration.py`: **27/27 PASS, 0 FAIL**.

Verification highlights:

```
[PASS] LightGBM / XGBoost / RandomForest loads
[PASS] … predictions in [0,1]
[PASS] Ensemble EXACT weights 0.40/0.35/0.25
[PASS] Ensemble deterministic 0.63
[PASS] No CatBoost in approved set
[PASS] No ExtraTrees in approved set
[PASS] LightGBM / XGBoost / RandomForest SHA-256 recorded matches disk
[PASS] LightGBM / XGBoost / RandomForest fidelity
[PASS] ModelFactory registers 3, unknown rejected
[PASS] Orchestrator Phase1 produces GenesisPrediction (SUCCESS)
[PASS] Threshold 0.24
[PASS] Missing artifact -> UNAVAILABLE
[PASS] OMP_NUM_THREADS configured
```

## 11. Runtime / native compatibility

`toofan.core.config.configure_runtime()` is called before any ML framework
import (in `run_genesis.py` and `scripts/verify_genesis_integration.py`). It
pins `OMP_NUM_THREADS=1` (and MKL/NUMEXPR) via `os.environ.setdefault`, so an
explicit user override is never clobbered, and it sets
`KMP_DUPLICATE_LIB_OK=TRUE` to avoid conflicting OpenMP runtime errors. The
adapter classes import lightgbm / xgboost / sklearn **lazily at load time**, so
the runtime safeguards always run first.

## 12. Known scientific limitations

- The Genesis system is a **prototype**: 300 samples / 191 storms / 2015–2024.
- The report states **synthetic SST/SST-anomaly and TCHP/OHC700** values were used.
- **Storm-aware CV performance was lower** than the final held-out test performance.
- The documented metrics (XGBoost ROC-AUC 0.707 / PR-AUC 0.680 / F1 0.611;
  LightGBM 0.687 / 0.665 / 0.653; RandomForest 0.617 / 0.610 / 0.586; and 2023–24
  holdout F1 XGB 0.643 / LGB 0.643 / RF 0.712) are **documented model-selection
  results** and were not recomputed or changed by this integration.
- The architecture is production-oriented; the scientific model remains
  prototype-level. Do not claim "production validated", "operationally proven",
  or "highly accurate" without new evidence.
- No calibration artifact exists in the repository, so the raw weighted
  ensemble probability is preserved and reported with `calibrated = false`.

---

## Final summary

```
GENESIS PRODUCTION MODEL:  LightGBM
GENESIS ENSEMBLE:          LightGBM 40% / XGBoost 35% / RandomForest 25%
GENESIS THRESHOLD:         0.24
CATBOOST:                  NOT USED
EXTRATREES:                NOT USED
RETRAINING:                NOT PERFORMED
MODEL WEIGHTS:             UNCHANGED
STATUS:                    INTEGRATED / VERIFIED
```

> Production artifacts were not present in the repo; integration logic is fully
> verified against inline fixtures using the exact OPTIMIZED filenames. Drop the
> real OPTIMIZED artifacts into `artifacts/genesis/` to activate production /
> ensemble inference; hashes are then recorded automatically at load.
