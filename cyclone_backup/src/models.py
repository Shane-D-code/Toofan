"""Model training for the IMD, ERA5 and IMD+ERA5 tabular branches.

All three branches use XGBoost. The training procedure is shared and
storm-safe:

1. A storm-split train/validation/test is used.
2. On the training set, inner **grouped-by-storm** cross-validation (using the
   storm ID as the group) is used to tune / verify hyperparameters and to
   pick the number of boosting rounds (early stopping on PR-AUC).
3. The final model is retrained on the full training set and evaluated on the
   held-out test set using a validation-tuned decision threshold.

Class imbalance is handled with ``scale_pos_weight`` (computed from the
training set) plus, optionally, ``max_delta_step`` for stability.

NOTE on SMOTE: SMOTE is NOT used as the default imbalance strategy.
For temporal cyclone observations that are highly correlated, synthetic
interpolation can create physically implausible samples.  The safer choices
are:

- ``class_weight`` / ``scale_pos_weight`` (used here)
- Weighted BCE / focal loss (used by the satellite CNN)

SMOTE is only used as an ablation experiment, applied inside training
folds only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from .config import get_seed


def _scale_pos_weight(y: np.ndarray) -> float:
    """Compute XGBoost ``scale_pos_weight`` from the training labels."""
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n1 == 0:
        return 1.0
    return float(n0 / n1)


def grouped_cv_pr_auc(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    base_params: dict,
    seed: int,
    early_stopping_rounds: int = 50,
    num_boost_round: int = 400,
) -> tuple[float, list[float]]:
    """Grouped-by-storm 5-fold cross-validation reporting mean PR-AUC.

    The positivity of the metric (maximise) and determinism are handled here.
    Returns ``(mean_pr_auc, fold_scores)``.
    """
    group_kfold = GroupKFold(n_splits=n_folds)
    fold_scores = []
    X_np = X if isinstance(X, np.ndarray) else X.to_numpy()

    for train_idx, val_idx in group_kfold.split(X_np, y, groups):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        params = dict(base_params)
        params["scale_pos_weight"] = _scale_pos_weight(y_tr)
        params["random_state"] = seed

        model = XGBClassifier(
            n_estimators=num_boost_round,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        p = model.predict_proba(X_va)[:, 1]
        from sklearn.metrics import average_precision_score

        if len(np.unique(y_va)) < 2:
            fold_scores.append(float("nan"))
        else:
            fold_scores.append(float(average_precision_score(y_va, p)))

    fold_scores = [s for s in fold_scores if not np.isnan(s)]
    if not fold_scores:
        return float("nan"), fold_scores
    return float(np.mean(fold_scores)), fold_scores


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: dict,
    branch: str,
    seed: int,
) -> XGBClassifier:
    """Train an XGBoost classifier for a given tabular branch.

    Args:
        X_train/y_train/groups_train: training features, labels and storm groups.
        X_val/y_val: validation features/labels (used for early stopping and
            final threshold tuning downstream).
        X_test/y_test: held-out test features/labels (used only for the final
            held-out evaluation; **not** used for fitting or thresholding).
        cfg: pipeline configuration.
        branch: one of 'imd', 'era5', 'combined' (selects the hyperparameters
            section of the cfg).
        seed: random seed.

    Returns:
        The fitted XGBClassifier (trained on the full training set, with early
        stopping monitored on the validation set).
    """
    model_cfg = cfg[f"{branch}_model"]
    n_folds = int(model_cfg.get("cv_folds", 5))

    base_params = {
        "learning_rate": float(model_cfg.get("learning_rate", 0.05)),
        "max_depth": int(model_cfg.get("max_depth", 4)),
        "min_child_weight": int(model_cfg.get("min_child_weight", 2)),
        "subsample": float(model_cfg.get("subsample", 0.85)),
        "colsample_bytree": float(model_cfg.get("colsample_bytree", 0.85)),
        "max_delta_step": 1,
    }

    # Inner grouped-by-storm CV for PR-AUC (diagnostic / model-selection).
    cv_pr, fold_scores = grouped_cv_pr_auc(
        X_train,
        np.asarray(y_train).astype(int),
        np.asarray(groups_train),
        n_folds=n_folds,
        base_params=base_params,
        seed=seed,
        early_stopping_rounds=int(model_cfg.get("early_stopping_rounds", 50)),
        num_boost_round=int(model_cfg.get("n_estimators", 400)),
    )
    print(
        f"[{branch}] grouped-CV PR-AUC = {cv_pr:.4f} "
        f"(folds: {[round(s, 4) if not np.isnan(s) else 'nan' for s in fold_scores]})"
    )

    # Final model: fit on the full training set with early stopping on the
    # held-out-from-CV validation split.
    params = dict(base_params)
    params["scale_pos_weight"] = _scale_pos_weight(np.asarray(y_train).astype(int))
    params["random_state"] = seed

    model = XGBClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 400)),
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=int(model_cfg.get("early_stopping_rounds", 50)),
        **params,
    )
    model.fit(
        X_train,
        np.asarray(y_train).astype(int),
        eval_set=[(X_val, np.asarray(y_val).astype(int))],
        verbose=False,
    )
    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(RI=1) from a fitted XGBoost model."""
    return model.predict_proba(X)[:, 1]


def save_model(model, path) -> None:
    """Save an XGBoost model, guarding the classifier-identity quirk.

    Some XGBoost versions drop the ``_estimator_type`` class attribute after
    fitting with an early-stopping callback, which breaks ``save_model``.
    Setting it back (it is a classifier) makes the save succeed.
    """
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    model.save_model(path)
    print(f"[models] Saved -> {path}")
