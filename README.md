# Cyclone Track Predictor

This lets you predict the future track of a **new** tropical cyclone using the
pre-trained V12 (best) model. You feed in the cyclone's *observed history* and get
back its predicted position for the next 24 hours (at 2 h steps).

---

## 1. What you need (files)

Keep everything in one folder, in this layout:

```
cyclone_project/
│
├── v12_predict_new.py       ← the runner you call (NEW)
├── model.py                 ← defines the neural network (CycloneTransformerV11)
├── losses.py                ← distance math used by the runner
├── dataset.py               ← data handling helpers
├── v12_common.py            ← loads the model checkpoint
├── prepare_dataset.py       ← builds the 27 input features from raw fixes
└── checkpoints/
    └── v12_best_model.pt    ← the trained best model (weights + config + stats)
```

**Python packages required:** `torch`, `numpy`, `pandas` (+ `requests`).

The simplest option is to copy the entire `cyclone_project` folder (it already
contains everything including an install of these packages).

---

## 2. Exact input

Create a CSV with the cyclone's **observed fixes so far**. You must have **at least
12 fixes** (13 recommended, so the model has enough history). Use **3-hourly** spacing
if possible (matches what the model was trained on).

```
SID,ISO_TIME,lat,lon,wind,mslp,rmw
MYCY,2026-08-01 00:00:00,-15.0,80.0,40,995,
MYCY,2026-08-01 03:00:00,-15.6,80.3,45,992,
MYCY,2026-08-01 06:00:00,-16.1,80.7,50,988,
... (at least 12 rows; last row = the most recent fix) ...
```

### Columns

| Column | Required | Description | Units |
|---|---|---|---|
| `SID` | yes | storm name/id (any string) | — |
| `ISO_TIME` | yes | observation time | `YYYY-MM-DD HH:MM:SS` (or Unix seconds) |
| `lat` | yes | latitude | degrees |
| `lon` | yes | longitude | degrees |
| `wind` | no | max sustained wind | kt |
| `mslp` | no | min sea-level pressure | hPa |
| `rmw` | no | radius of max winds | nmi |

Notes:

- `wind`/`mslp`/`rmw` are used to build the SST/shear/RMW proxy features; if you
  omit them the script uses training-like defaults, but **provide them when you can**
  for a more faithful forecast.
- The model predicts positions for `+2, +4, +6, ..., +24` hours **after the last
  row** in your CSV.

---

## 3. How to run

```bash
python v12_predict_new.py --ckpt checkpoints/v12_best_model.pt --new_csv new_cyclone.csv
```

Optional flags:

```bash
# Save the forecast to a CSV file
python v12_predict_new.py --ckpt checkpoints/v12_best_model.pt --new_csv new_cyclone.csv --out forecast.csv

# Force CPU (default: GPU if available, else CPU)
python v12_predict_new.py --ckpt checkpoints/v12_best_model.pt --new_csv new_cyclone.csv --device cpu
```

---

## 4. Exact output

The script prints a table (and optionally writes it to `--out`). Example:

```
horizon       forecast_time    cal_lat    cal_lon    raw_lat    raw_lon  learned_scale  sigma_km
Current 1980-01-03 21:00:00 -16.900000 175.500000 -16.900000 175.500000       1.000000       NaN
    +2h 1980-01-03 23:00:00 -17.338024 175.639136 -17.265022 175.615947       1.199993  0.006738
    +4h 1980-01-04 01:00:00 -17.798546 175.768336 -17.648789 175.723614       1.199998  0.006738
    +6h 1980-01-04 03:00:00 -18.240245 175.917248 -18.016871 175.847707       1.199999  0.006738
    ...
    +24h 1980-01-04 21:00:00 -21.882498 177.718469 -21.052083 177.348725       1.199999 12.182494
```

### Columns explained

| Column | Meaning |
|---|---|
| `horizon` | `Current` = the last observed fix; `+Nh` = forecast N hours after it |
| `forecast_time` | the valid time of that forecast point |
| `cal_lat`, `cal_lon` | **the final predicted position** (calibrated) — use these |
| `raw_lat`, `raw_lon` | raw model output before the learned scale correction |
| `learned_scale` | per-horizon magnitude calibration factor (bounded 0.80–1.20) |
| `sigma_km` | forecast uncertainty, in km (larger = less certain; grows with lead time) |

---

## 5. What each file does

| File | Purpose |
|---|---|
| `v12_predict_new.py` | **The runner you execute.** Reads your new cyclone CSV, builds the 27 features, runs the model, prints/saves the 12-horizon forecast. |
| `model.py` | Defines `CycloneTransformerV11` — the neural network: a Transformer encoder that reads the last 12 fixes and predicts position at all 12 horizons at once. |
| `losses.py` | Defines `haversine_km` (great-circle distance) and other math used to compute track distances. |
| `dataset.py` | Data helpers (how raw rows become fixed-length input windows). |
| `v12_common.py` | `load_v12()` — loads the checkpoint and reconstructs the exact model + config + normalization stats. |
| `prepare_dataset.py` | Builds the **27 input features** (see below) from your raw lat/lon/wind/mslp/rmw fixes — exactly the same way the model was trained. |
| `checkpoints/v12_best_model.pt` | The trained best model: weights, architecture config, and the training-set normalization statistics needed to run inference. |

---

## 6. The 27 input features (built automatically — you don't make these)

The script converts your raw fixes into the 27 numbers per timestep the model
expects: position (`lat`, `lon`), intensity (`wind`, `mslp`, `rmw`), proxies
(`sst`, `shear`), and derived motion (translation `speed_kmh`, bearing sin/cos,
month sin/cos, `dt_hours`, and the u/v velocity + speed/u/v over 3/6/12 h windows,
acceleration, and turn sin/cos).

**You do not compute these** — the script does it for you, identically to training.

---

## 7. Important limits

- **Valid out to 24 h only.** It is a short-range tracker, not a medium-range model.
- **SST/shear are proxies** (from climatology), not real-time satellite data, so real
  reanalysis steering fields are not captured.
- **Needs ≥ 12 observed fixes** to predict; more history = better forecast.
- It predicts **track only** (position), not intensity.

---

# Genesis — 24-hour Tropical Cyclone Genesis (TOOFAN)

The Genesis module predicts 24-hour tropical cyclone genesis. It is a
**prototype** (300 samples / 191 storms / 2015–2024). See
[`docs/model_inventory.md`](docs/model_inventory.md) and
[`docs/model_audit/genesis_integration.md`](docs/model_audit/genesis_integration.md).

## Approved models (ONLY these three)

| Role | Model | Ensemble weight |
|------|-------|-----------------|
| **Production (primary)** | LightGBM | 0.40 |
| Ensemble | XGBoost | 0.35 |
| Ensemble | RandomForest | 0.25 |

- Threshold: **0.24** (optimized, not 0.50)
- Target: `genesis_24h`
- Input: 34 features
- CatBoost / ExtraTrees / gradient boosting / neural nets are **not** used.
- Artifacts must be placed under `artifacts/genesis/`:
  `tc_genesis_{lightgbm,xgboost,randomforest}_300_OPTIMIZED.joblib`
  (+ optional `tc_genesis_300_imputer.joblib`).

## Run

```bash
python run_genesis.py --mode production          # LightGBM only
python run_genesis.py --mode ensemble             # soft-voting (3 models)
python run_genesis.py --mode production --csv feats.csv --storm-id FANI
```

If artifacts are missing, the runner reports an explicit `UNAVAILABLE` status —
it never fabricates a genesis prediction.

## Test

```bash
python -m pytest tests
python scripts/verify_genesis_integration.py
```

