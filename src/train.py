"""Train and evaluate the XGBoost recurvature model.

Usage:
    python -m src.train --csv data/ibtracs_NI_list_v04r01.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

from .prep import load_clean
from .features import build_features, FEATURE_COLS
from .dataset import storm_split, make_tabular

SEED = 42


def evaluate(name: str, y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the TOOFAN XGBoost recurvature model.")
    parser.add_argument("--csv", required=True, help="Path to ibtracs_NI_list_v04r01.csv")
    parser.add_argument("--min-season", type=int, default=1980)
    parser.add_argument("--turn-threshold", type=float, default=45.0, help="Degrees of heading change counted as recurving")
    parser.add_argument("--future-steps", type=int, default=8, help="Timesteps ahead to check (8 * 3h = 24h)")
    parser.add_argument("--out", default="results.csv", help="Where to write the metrics table")
    parser.add_argument("--model-out", default="xgb_recurve_model.json", help="Where to save the trained model")
    args = parser.parse_args()

    np.random.seed(SEED)

    print("Loading and cleaning data...")
    df = load_clean(args.csv, min_season=args.min_season)

    print("Engineering features and labels...")
    df = build_features(df, future_steps=args.future_steps, turn_threshold=args.turn_threshold)

    train_sids, val_sids, test_sids = storm_split(df, seed=SEED)
    print(f"storms -> train {len(train_sids)}, val {len(val_sids)}, test {len(test_sids)}")

    X_tr, y_tr = make_tabular(df, train_sids)
    X_va, y_va = make_tabular(df, val_sids)
    X_te, y_te = make_tabular(df, test_sids)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        random_state=SEED, early_stopping_rounds=30,
    )

    print("Training XGBoost...")
    model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)

    prob = model.predict_proba(X_te_s)[:, 1]
    result = evaluate("XGBoost", y_te, prob)

    res_df = pd.DataFrame([result]).set_index("model").round(3)
    print("\n=== Test-set results (recurvature within next", args.future_steps * 3, "hours) ===")
    print(res_df)
    res_df.to_csv(args.out)

    model.save_model(args.model_out)
    print(f"\nSaved metrics to {args.out}, model to {args.model_out}")

    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    with open("feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)


if __name__ == "__main__":
    main()
