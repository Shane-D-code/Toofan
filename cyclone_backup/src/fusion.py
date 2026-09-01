"""Late-fusion meta-classifier.

Instead of retraining everything end-to-end, we take the **probabilities**
produced by each branch model and stack them as features for a lightweight
meta-classifier. The meta-classifier is trained on *training storms only* and
evaluated on the held-out test storms, so the fusion comparison remains
storm-safe.

Supported branches:
  - IMD  (tabular intensity history)
  - ERA5 (tabular atmospheric environment)
  - Satellite CNN (PyTorch, local Bay of Bengal — 2-channel IR)
  - TCIR CNN (Keras, global TCIR — 4-channel IR + MW)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import evaluate_split, tune_threshold, classification_metrics
from .config import get_seed


def stack_probability_tables(
    imd_pred: pd.DataFrame,
    era5_pred: pd.DataFrame,
    cnn_pred: pd.DataFrame | None = None,
    tcir_pred: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Join branch prediction tables on (storm_id, datetime_utc).

    Uses an inner join so that fusion is only evaluated on observations for
    which all requested branches are present. Returns ``(features, y)``.

    Args:
        imd_pred: IMD prediction table with columns storm_id, datetime_utc,
            RI_24h, P_RI.
        era5_pred: ERA5 prediction table (same schema).
        cnn_pred: optional PyTorch satellite CNN prediction table.
        tcir_pred: optional TCIR Keras CNN prediction table.

    Returns:
        ``(stacked_df, y)`` where ``stacked_df`` has one probability column
        per branch plus the identity columns.
    """
    # Start with IMD as the anchor (every observation has IMD).
    merged = imd_pred[["storm_id", "datetime_utc", "P_RI"]].rename(
        columns={"P_RI": "P_imd"}
    )
    # Inner-join ERA5 (fusion only evaluated where both IMD + ERA5 exist).
    merged = merged.merge(
        era5_pred[["storm_id", "datetime_utc", "P_RI"]].rename(
            columns={"P_RI": "P_era5"}
        ),
        on=["storm_id", "datetime_utc"],
        how="inner",
    )
    used_branches = ["imd", "era5"]

    # Optional: PyTorch satellite CNN branch.
    if cnn_pred is not None:
        merged = merged.merge(
            cnn_pred[["storm_id", "datetime_utc", "P_RI"]].rename(
                columns={"P_RI": "P_cnn"}
            ),
            on=["storm_id", "datetime_utc"],
            how="inner",
        )
        used_branches.append("cnn")

    # Optional: TCIR Keras CNN branch.
    if tcir_pred is not None:
        merged = merged.merge(
            tcir_pred[["storm_id", "datetime_utc", "P_RI"]].rename(
                columns={"P_RI": "P_tcir"}
            ),
            on=["storm_id", "datetime_utc"],
            how="inner",
        )
        used_branches.append("tcir")

    # Attach the fused target from the IMD table (all branches share it).
    target = imd_pred[["storm_id", "datetime_utc", "RI_24h"]]
    merged = merged.merge(target, on=["storm_id", "datetime_utc"], how="left")
    y = merged["RI_24h"].to_numpy().astype(int)
    feature_cols = [c for c in merged.columns if c.startswith("P_")]
    out = merged[["storm_id", "datetime_utc"] + feature_cols + ["RI_24h"]].copy()
    out.attrs["branches"] = used_branches
    return out, y


def train_fusion(
    train_df: pd.DataFrame,
    train_y: np.ndarray,
    val_df: pd.DataFrame,
    val_y: np.ndarray,
    test_df: pd.DataFrame,
    test_y: np.ndarray,
    cfg: dict,
    seed: int,
) -> dict:
    """Train a lightweight late-fusion meta-classifier.

    Uses logistic regression by default (robust to small sample sizes). The
    stacked probability columns are the features. Returns the fitted model and
    its test metrics with a validation-tuned threshold.

    Args:
        train_df: stacked probability table for training storms (columns
            P_imd, P_era5, ... and RI_24h).
        train_y: training labels.
        val_df / val_y: validation stacked table + labels (for thresholding).
        test_df / test_y: test stacked table + labels (held out).
        cfg: configuration.
        seed: random seed.

    Returns:
        Dict with keys 'model', 'metrics', 'probabilities' (test P_RI).
    """
    from sklearn.linear_model import LogisticRegression

    meta_type = cfg["fusion"].get("meta_model", "logistic")
    feature_cols = [c for c in train_df.columns if c.startswith("P_")]

    X_tr = train_df[feature_cols].to_numpy()
    X_va = val_df[feature_cols].to_numpy()
    X_te = test_df[feature_cols].to_numpy()

    if meta_type == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=(train_y == 0).sum() / max((train_y == 1).sum(), 1),
            random_state=seed,
        )
        model.fit(X_tr, train_y)
        p_val = model.predict_proba(X_va)[:, 1]
    else:
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(X_tr, train_y)
        p_val = model.predict_proba(X_va)[:, 1]

    p_test = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    threshold = tune_threshold(val_y, p_val, criterion="f1", grid_step=0.01, seed=seed)
    metrics = classification_metrics(test_y, p_test, threshold)

    return {
        "model": model,
        "metrics": metrics,
        "probabilities": p_test,
        "feature_cols": feature_cols,
        "threshold": threshold,
    }
