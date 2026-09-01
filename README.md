# Recurvature Cyclone Model

Predicts whether a cyclone will **recurve** (heading change &ge; 45&deg; within
the next 24h) using IBTrACS North Indian Ocean best-track data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download `ibtracs_NI_list_v04r01.csv` from
[NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
and place it in `data/`. The raw CSV isn't tracked in this repo (see
`.gitignore`) — keep it out of version control since it's a redistributable
public dataset, not project-specific data.

## Train

```bash
python -m src.train --csv data/ibtracs_NI_list_v04r01.csv
```

Outputs:
- `results.csv` — test-set metrics (accuracy, precision, recall, F1, ROC-AUC, Brier)
- `xgb_recurve_model.json` — the trained model
- `feature_importance.json` — feature importances

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--turn-threshold` | `45.0` | Degrees of heading change counted as "recurving" |
| `--future-steps` | `8` | Timesteps ahead to check (8 &times; 3h = 24h) |
| `--min-season` | `1980` | Drop storms before this year |
| `--out` | `results.csv` | Where to write metrics |
| `--model-out` | `xgb_recurve_model.json` | Where to save the model |

## Project layout

```
src/
  prep.py       # load + clean IBTrACS
  features.py   # feature engineering + recurve label
  dataset.py    # storm-level train/val/test split
  train.py      # train XGBoost, evaluate, save
data/           # put ibtracs_NI_list_v04r01.csv here (not tracked)
```

## Method

At every 3-hourly timestep, the storm's heading 24h in the future is
compared to its current heading (circular difference). If that swing is
&ge; 45&deg;, the timestep is labeled a recurve event. XGBoost is trained on
12 engineered features per timestep (position, intensity, kinematics,
heading momentum, seasonality, distance to land), with storms split at
the storm level (not row level) to prevent leakage between train/val/test.

## Results

On a held-out test set of 60 storms:

| Metric | Score |
|---|---|
| ROC-AUC | 0.722 |
| Precision | 0.425 |
| Recall | 0.616 |
| F1 | 0.503 |
| Brier | 0.206 |

LSTM and TCN sequence models were also benchmarked on the same task and
came in slightly behind XGBoost (ROC-AUC 0.69 and 0.68) — with the current
dataset size (~276 training storms), tree-based models outperform deep
sequence models. That ranking is worth re-checking if the dataset grows
(e.g. by adding ERA5 steering-flow features).

---

## Cyclone Track Predictor (V12 Model)

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
