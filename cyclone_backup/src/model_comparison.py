"""Stronger tabular model comparison (Phase 6 of the SIH master plan).

Benchmarks Logistic Regression, Random Forest, XGBoost and
HistGradientBoosting on a given feature table, using **storm-grouped** inner
cross-validation (GroupKFold / StratifiedGroupKFold). PR-AUC is the primary
selection metric; we also report ROC-AUC, precision, recall, F1 and Brier.

Class imbalance is handled with class weights / ``scale_pos_weight`` computed
only on the training split of each fold (never the whole dataset, never the
validation/test fold).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from .config import get_seed


def _safe(fn, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def make_model(name: str, seed: int, base_class_weight: dict | None = None):
    """Return a fresh (unfitted) model instance for the given family.

    For families that cannot handle NaN natively (LR, RF) we wrap with a
    median imputer fitted on the training fold only (no data leakage).
    """
    imp = SimpleImputer(strategy="median")
    if name == "lr":
        return make_pipeline(
            imp, LogisticRegression(max_iter=3000, class_weight=base_class_weight,
                                    random_state=seed))
    if name == "rf":
        return make_pipeline(
            imp, RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=3,
                class_weight=base_class_weight, random_state=seed, n_jobs=1))
    if name == "xgb":
        return XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
            max_delta_step=1, objective="binary:logistic",
            eval_metric="aucpr", random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=4,
            min_samples_leaf=3, random_state=seed,
            class_weight=base_class_weight)
    raise ValueError(f"Unknown model family: {name}")


def grouped_cv_compare(X, y, groups, model_families, n_folds=5, seed=42,
                       verbose=True) -> pd.DataFrame:
    """Run storm-grouped CV for several model families, report PR-AUC.

    Returns a DataFrame with per-family mean (std) PR-AUC, ROC-AUC and the
    fold scores.
    """
    groups = np.asarray(groups)
    y = np.asarray(y).astype(int)
    X = X.reset_index(drop=True)

    # Guard: need at least one group per fold + both classes across the set.
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    try:
        splits = list(skf.split(X, y, groups))
    except ValueError:
        from sklearn.model_selection import GroupKFold
        splits = list(GroupKFold(n_splits=n_folds).split(X, y, groups))

    rows = []
    for fam in model_families:
        prs, rocs = [], []
        for train_idx, val_idx in splits:
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                continue
            # Class weighting / scale_pos_weight from training fold only.
            if fam == "xgb":
                spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                model = XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=4,
                    min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
                    max_delta_step=1, objective="binary:logistic",
                    eval_metric="aucpr", scale_pos_weight=spw,
                    random_state=seed)
            else:
                cw = {0: 1.0, 1: float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))}
                model = make_model(fam, seed, base_class_weight=cw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
                p = model.predict_proba(X_va)[:, 1]
            prs.append(_safe(average_precision_score, y_va, p))
            rocs.append(_safe(roc_auc_score, y_va, p))
        prs = [p for p in prs if not np.isnan(p)]
        rocs = [r for r in rocs if not np.isnan(r)]
        rows.append({
            "model_family": fam,
            "pr_auc_mean": float(np.mean(prs)) if prs else np.nan,
            "pr_auc_std": float(np.std(prs)) if prs else np.nan,
            "roc_auc_mean": float(np.mean(rocs)) if rocs else np.nan,
            "roc_auc_std": float(np.std(rocs)) if rocs else np.nan,
            "n_folds": len(prs),
        })
        if verbose:
            print(f"  [{fam}] PR-AUC = {rows[-1]['pr_auc_mean']:.4f} "
                  f"±{rows[-1]['pr_auc_std']:.4f}  "
                  f"ROC-AUC = {rows[-1]['roc_auc_mean']:.4f} ±{rows[-1]['roc_auc_std']:.4f}")
    return pd.DataFrame(rows)
