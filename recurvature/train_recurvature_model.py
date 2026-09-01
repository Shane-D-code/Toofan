#!/usr/bin/env python3
"""
Tropical Cyclone Track Recurvature Prediction (24h Ahead)
North Indian Ocean (NI) Basin — IBTrACS Dataset

Models:
  1. XGBoost / Gradient Boosting (Tabular feature snapshot)
  2. LSTM (24h lookback sequence)
  3. TCN (Temporal Convolutional Network)

Usage:
  python train_recurvature_model.py [--data PATH_TO_CSV] [--no-plots]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.ensemble import HistGradientBoostingClassifier

# Try importing XGBoost with fallback to HistGradientBoosting if libomp is missing
HAS_XGBOOST = False
try:
    import xgboost as xgb
    # Test if libomp is present
    dummy_model = xgb.XGBClassifier(n_estimators=1)
    dummy_model.fit(np.zeros((5, 2)), np.array([0, 1, 0, 1, 0]))
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set reproducible seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters / Settings
FUTURE_STEPS   = 8     # 8 * 3h = 24h ahead prediction horizon
TURN_THRESHOLD = 45.0  # degrees of heading change to count as "recurving"
PAST_WINDOW    = 8     # timesteps of history fed into LSTM/TCN (8*3h = 24h context)
MIN_SEASON     = 1980  # Filter out pre-satellite era
MIN_POINTS     = 12    # Filter out very short-lived storms

NUMERIC_COLS = [
    "LAT", "LON", "WMO_WIND", "WMO_PRES", "DIST2LAND", "USA_LAT", "USA_LON",
    "USA_WIND", "USA_PRES", "STORM_SPEED", "STORM_DIR"
]

FEATURE_COLS = [
    "lat", "lon", "wind", "pres", "STORM_SPEED", "dir_sin", "dir_cos",
    "month_sin", "month_cos", "DIST2LAND", "dir_change_3h", "dir_change_9h"
]

DEFAULT_POSSIBLE_PATHS = [
    "ibtracs_NI_list_v04r01.csv",
    "ibtracs.NI.list.v04r01.csv",
    "data/ibtracs_NI_list_v04r01.csv",
    "recurvature/data/ibtracs_NI_list_v04r01.csv",
    "/Users/khoria/Downloads/toofan_repo/recurvature/data/ibtracs_NI_list_v04r01.csv",
    "/Users/khoria/Downloads/toofan_repo/data/ibtracs_NI_list_v04r01.csv",
    "/Users/khoria/Downloads/ibtracs.NI.list.v04r01.csv",
]


def find_csv_path(custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return custom_path
    for path in DEFAULT_POSSIBLE_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not locate ibtracs_NI_list_v04r01.csv. "
        "Please provide the path using --data <path>."
    )


def circ_diff(a, b):
    """Smallest signed difference (a - b) in degrees, result in (-180, 180]."""
    return (a - b + 180) % 360 - 180


def load_clean_data(path, min_season=MIN_SEASON, min_points=MIN_POINTS):
    print(f"[*] Loading data from: {path}")
    df = pd.read_csv(path, skiprows=[1], low_memory=False)  # Row 1 is units, skip it
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")

    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df = df[df["SEASON"] >= min_season].copy()
    if "TRACK_TYPE" in df.columns:
        df = df[df["TRACK_TYPE"] == "main"].copy()

    df["lat"]  = df["USA_LAT"].fillna(df["LAT"])
    df["lon"]  = df["USA_LON"].fillna(df["LON"])
    df["wind"] = df["USA_WIND"].fillna(df["WMO_WIND"])
    df["pres"] = df["USA_PRES"].fillna(df["WMO_PRES"])

    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    keep = [
        "SID", "SEASON", "NAME", "ISO_TIME", "lat", "lon", "wind", "pres",
        "STORM_SPEED", "STORM_DIR", "DIST2LAND"
    ]
    df = df[keep].dropna(subset=["lat", "lon", "STORM_DIR", "STORM_SPEED"])

    counts = df.groupby("SID").size()
    good_sids = counts[counts >= min_points].index
    df = df[df["SID"].isin(good_sids)].reset_index(drop=True)
    return df


def build_features(df):
    df = df.copy()
    df["wind"] = (
        df.groupby("SID")["wind"]
        .apply(lambda s: s.interpolate().ffill().bfill())
        .reset_index(level=0, drop=True)
    )
    df["pres"] = (
        df.groupby("SID")["pres"]
        .apply(lambda s: s.interpolate().ffill().bfill())
        .reset_index(level=0, drop=True)
    )
    df["wind"] = df["wind"].fillna(df["wind"].median())
    df["pres"] = df["pres"].fillna(df["pres"].median())

    df["month"] = df["ISO_TIME"].dt.month
    df["dir_sin"]   = np.sin(np.deg2rad(df["STORM_DIR"]))
    df["dir_cos"]   = np.cos(np.deg2rad(df["STORM_DIR"]))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    grp = df.groupby("SID")["STORM_DIR"]
    df["dir_change_3h"] = grp.diff(1).apply(lambda x: (x + 180) % 360 - 180)
    raw_9h = df["STORM_DIR"] - grp.shift(3)
    df["dir_change_9h"] = ((raw_9h + 180) % 360) - 180

    df["future_dir"] = df.groupby("SID")["STORM_DIR"].shift(-FUTURE_STEPS)
    df["heading_swing"] = circ_diff(df["future_dir"], df["STORM_DIR"]).abs()
    df["recurve_label"] = (df["heading_swing"] >= TURN_THRESHOLD).astype(float)
    df.loc[df["future_dir"].isna(), "recurve_label"] = np.nan
    return df


def storm_split(df, test_size=0.15, val_size=0.15, seed=SEED):
    sid_label = df.dropna(subset=["recurve_label"]).groupby("SID")["recurve_label"].max()
    sids, strat = sid_label.index.values, sid_label.values
    train_sids, test_sids = train_test_split(sids, test_size=test_size, random_state=seed, stratify=strat)
    strat2 = sid_label.loc[train_sids].values
    train_sids, val_sids = train_test_split(
        train_sids, test_size=val_size / (1 - test_size), random_state=seed, stratify=strat2
    )
    return set(train_sids), set(val_sids), set(test_sids)


def make_tabular(df, sids):
    d = df[df["SID"].isin(sids)].dropna(subset=["recurve_label"]).copy()
    d[["dir_change_3h", "dir_change_9h"]] = d[["dir_change_3h", "dir_change_9h"]].fillna(0)
    X = d[FEATURE_COLS].values.astype(np.float32)
    y = d["recurve_label"].values.astype(np.float32)
    return X, y


def make_sequences(df, sids, window=PAST_WINDOW):
    feat_df = df.copy()
    feat_df[["dir_change_3h", "dir_change_9h"]] = feat_df[["dir_change_3h", "dir_change_9h"]].fillna(0)
    X_seq, y_seq = [], []
    for sid, g in feat_df[feat_df["SID"].isin(sids)].groupby("SID"):
        g = g.sort_values("ISO_TIME").reset_index(drop=True)
        feats = g[FEATURE_COLS].values.astype(np.float32)
        labels = g["recurve_label"].values.astype(np.float32)
        for t in range(len(g)):
            if np.isnan(labels[t]):
                continue
            start = max(0, t - window + 1)
            wf = feats[start: t + 1]
            if len(wf) < window:
                pad = np.repeat(wf[0:1], window - len(wf), axis=0)
                wf = np.concatenate([pad, wf], axis=0)
            X_seq.append(wf)
            y_seq.append(labels[t])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def evaluate(name, y_true, y_prob, threshold=0.5):
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


def tcn_block(x, filters, kernel_size, dilation):
    prev = x
    y = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu")(y)
    y = layers.BatchNormalization()(y)
    if prev.shape[-1] != filters:
        prev = layers.Conv1D(filters, 1, padding="same")(prev)
    return layers.Add()([prev, y])


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Tropical Cyclone Recurvature prediction models.")
    parser.add_argument("--data", type=str, default=None, help="Path to ibtracs_NI_list_v04r01.csv dataset.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting figures.")
    args = parser.parse_args()

    csv_path = find_csv_path(args.data)
    df_raw = load_clean_data(csv_path)
    print(f"[*] Dataset loaded: {df_raw.shape[0]} rows, {df_raw['SID'].nunique()} unique storms.")

    df = build_features(df_raw)
    labeled = df.dropna(subset=["recurve_label"])
    print(f"[*] Labeled rows: {len(labeled)}, Recurvature positive rate: {labeled['recurve_label'].mean():.3f}")
    print(f"[*] Storms with >=1 recurve event: {int(labeled.groupby('SID')['recurve_label'].max().sum())} / {labeled['SID'].nunique()}")

    train_sids, val_sids, test_sids = storm_split(df)
    print(f"[*] Storm Split -> Train: {len(train_sids)}, Val: {len(val_sids)}, Test: {len(test_sids)}")

    # Prepare Tabular Data
    Xtab_tr, ytab_tr = make_tabular(df, train_sids)
    Xtab_va, ytab_va = make_tabular(df, val_sids)
    Xtab_te, ytab_te = make_tabular(df, test_sids)

    scaler = StandardScaler().fit(Xtab_tr)
    Xtab_tr_s = scaler.transform(Xtab_tr)
    Xtab_va_s = scaler.transform(Xtab_va)
    Xtab_te_s = scaler.transform(Xtab_te)

    # Prepare Sequence Data
    Xseq_tr, yseq_tr = make_sequences(df, train_sids)
    Xseq_va, yseq_va = make_sequences(df, val_sids)
    Xseq_te, yseq_te = make_sequences(df, test_sids)

    n_feat = Xseq_tr.shape[-1]
    seq_scaler = StandardScaler().fit(Xseq_tr.reshape(-1, n_feat))

    def scale_seq(X):
        return seq_scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape)

    Xseq_tr_s = scale_seq(Xseq_tr)
    Xseq_va_s = scale_seq(Xseq_va)
    Xseq_te_s = scale_seq(Xseq_te)

    print(f"[*] Data shapes -> Tabular Train: {Xtab_tr_s.shape}, Sequence Train: {Xseq_tr_s.shape}")

    results = []
    probs = {}

    # -------------------------------------------------------------
    # 1. XGBoost or HistGradientBoosting Model
    # -------------------------------------------------------------
    pos_weight = (ytab_tr == 0).sum() / (ytab_tr == 1).sum()
    if HAS_XGBOOST:
        print("\n[+] Training XGBoost Classifier...")
        tree_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=SEED,
            early_stopping_rounds=30,
        )
        tree_model.fit(Xtab_tr_s, ytab_tr, eval_set=[(Xtab_va_s, ytab_va)], verbose=False)
        model_name = "XGBoost"
    else:
        print("\n[+] Training HistGradientBoosting Classifier (XGBoost fallback)...")
        sample_weights = np.where(ytab_tr == 1, pos_weight, 1.0)
        tree_model = HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=4,
            learning_rate=0.05,
            random_state=SEED,
        )
        tree_model.fit(Xtab_tr_s, ytab_tr, sample_weight=sample_weights)
        model_name = "HistGradientBoosting"

    prob_tree = tree_model.predict_proba(Xtab_te_s)[:, 1]
    probs[model_name] = (ytab_te, prob_tree)
    results.append(evaluate(model_name, ytab_te, prob_tree))

    # -------------------------------------------------------------
    # 2. LSTM Model
    # -------------------------------------------------------------
    print("\n[+] Training LSTM Neural Network...")
    class_weight_lstm = {0: 1.0, 1: float((yseq_tr == 0).sum() / (yseq_tr == 1).sum())}

    lstm_model = keras.Sequential([
        layers.Input(shape=(PAST_WINDOW, n_feat)),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(16),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    lstm_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    es_lstm = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)
    lstm_model.fit(
        Xseq_tr_s,
        yseq_tr,
        validation_data=(Xseq_va_s, yseq_va),
        epochs=60,
        batch_size=64,
        class_weight=class_weight_lstm,
        callbacks=[es_lstm],
        verbose=0,
    )

    prob_lstm = lstm_model.predict(Xseq_te_s, verbose=0).ravel()
    probs["LSTM"] = (yseq_te, prob_lstm)
    results.append(evaluate("LSTM", yseq_te, prob_lstm))

    # -------------------------------------------------------------
    # 3. TCN Model
    # -------------------------------------------------------------
    print("\n[+] Training Temporal Convolutional Network (TCN)...")
    inp = layers.Input(shape=(PAST_WINDOW, n_feat))
    x = inp
    for d in [1, 2, 4]:
        x = tcn_block(x, filters=32, kernel_size=3, dilation=d)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    tcn_model = keras.Model(inp, out)
    tcn_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    es_tcn = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)
    tcn_model.fit(
        Xseq_tr_s,
        yseq_tr,
        validation_data=(Xseq_va_s, yseq_va),
        epochs=60,
        batch_size=64,
        class_weight=class_weight_lstm,
        callbacks=[es_tcn],
        verbose=0,
    )

    prob_tcn = tcn_model.predict(Xseq_te_s, verbose=0).ravel()
    probs["TCN"] = (yseq_te, prob_tcn)
    results.append(evaluate("TCN", yseq_te, prob_tcn))

    # -------------------------------------------------------------
    # Results & Metrics Table
    # -------------------------------------------------------------
    res_df = pd.DataFrame(results).set_index("model").round(3)
    print("\n" + "=" * 60)
    print("           RECURVATURE MODEL EVALUATION RESULTS           ")
    print("=" * 60)
    print(res_df)
    print("=" * 60 + "\n")

    # -------------------------------------------------------------
    # Plotting Figures & Saving Outputs
    # -------------------------------------------------------------
    if not args.no_plots:
        os.makedirs("plots", exist_ok=True)

        # 1. Feature Importance (if tree_model has feature_importances_)
        if hasattr(tree_model, "feature_importances_"):
            plt.figure(figsize=(7, 4.5))
            imp = pd.Series(tree_model.feature_importances_, index=FEATURE_COLS).sort_values()
            imp.plot.barh(title=f"{model_name} Feature Importance")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            feat_imp_path = os.path.join("plots", "feature_importance.png")
            plt.savefig(feat_imp_path, dpi=300)
            plt.close()
            print(f"[*] Saved feature importance plot to {feat_imp_path}")

        # 2. Performance Comparison Bar Chart
        plt.figure(figsize=(8, 5))
        res_df[["precision", "recall", "f1", "roc_auc"]].plot.bar(figsize=(8, 5), rot=0)
        plt.title(f"Recurvature (>= {TURN_THRESHOLD}° in {FUTURE_STEPS*3}h) — Model Comparison")
        plt.ylabel("Score")
        plt.legend(loc="lower right")
        plt.tight_layout()
        model_comp_path = os.path.join("plots", "model_comparison.png")
        plt.savefig(model_comp_path, dpi=300)
        plt.close()
        print(f"[*] Saved model comparison plot to {model_comp_path}")

        # 3. ROC and Precision-Recall Curves
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for name, (yt, yp) in probs.items():
            RocCurveDisplay.from_predictions(yt, yp, name=name, ax=axes[0])
            PrecisionRecallDisplay.from_predictions(yt, yp, name=name, ax=axes[1])
        axes[0].set_title("ROC Curve")
        axes[1].set_title("Precision-Recall Curve")
        plt.tight_layout()
        roc_pr_path = os.path.join("plots", "roc_pr_curves.png")
        plt.savefig(roc_pr_path, dpi=300)
        plt.close()
        print(f"[*] Saved ROC/PR curves plot to {roc_pr_path}")

    print("\n[+] Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
