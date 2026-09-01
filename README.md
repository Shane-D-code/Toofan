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
