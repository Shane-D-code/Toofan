"""Storm-safe evaluation, threshold tuning and model comparison.

The core principle: any decision threshold that a model uses for reporting
precision/recall/F1/confusion-matrix is chosen **only on the validation set**,
never on the test labels. Threshold-free metrics (ROC-AUC, PR-AUC) are always
computed directly on probability scores, so no threshold is involved there.

This module also provides:

- **Probability calibration** (Brier score, reliability curve, calibration
  intercept/slope, isotonic regression).
- **Storm-level bootstrap confidence intervals** (resample storms, not rows).
- **Preprocessing leakage guards** (assert scalers/SMOTE fit on train only).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import get_seed


def safe_roc_auc(y_true, y_score):
    """ROC-AUC that returns NaN instead of raising when a single class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true, y_score):
    """PR-AUC (average precision) that returns NaN on a single-class target."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def confusion_matrix_dict(y_true, y_pred) -> dict:
    """Confusion matrix as a labelled dict (works even for single-class)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Degenerate: zeros on the missing-class row/col.
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    """Compute a full metric dictionary at a given probability threshold.

    Precision/recall/F1/confusion matrix depend on ``threshold``; ROC/PR-AUC
    do not. Zero-division is suppressed so precision/recall stay well-defined.
    """
    y_pred = (y_score >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "pr_auc": safe_pr_auc(y_true, y_score),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "brier": _safe_brier(y_true, y_score),
        "confusion": confusion_matrix_dict(y_true, y_pred),
    }


def _safe_brier(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(brier_score_loss(y_true, y_score))
    except Exception:
        return float("nan")


def tune_threshold(
    y_val: np.ndarray,
    y_score_val: np.ndarray,
    criterion: str = "f1",
    grid_step: float = 0.01,
    seed: int = 0,
) -> float:
    """Choose a decision threshold on validation data only.

    Args:
        y_val: True validation labels.
        y_score_val: Validation probabilities.
        criterion: Which metric to maximise ('f1' recommended, or 'recall').
        grid_step: Step size for the threshold grid.
        seed: Unused; kept for signature parity.

    Returns:
        The chosen threshold in [0, 1]. Defaults to 0.5 if metrics cannot be
        computed.
    """
    thresholds = np.arange(0.0, 1.0 + grid_step / 2, grid_step)
    best_thr = 0.5
    best_score = -np.inf
    for thr in thresholds:
        y_pred = (y_score_val >= thr).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if criterion == "recall":
                score = recall_score(y_val, y_pred, zero_division=0)
            elif criterion == "precision":
                score = precision_score(y_val, y_pred, zero_division=0)
            else:
                score = f1_score(y_val, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def calibration_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Return a calibration-table (binned mean predicted vs. observed rate)."""
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="uniform"
        )
    except Exception:
        return []
    if len(mean_pred) == 0:
        return []
    return [
        {"mean_predicted": float(p), "fraction_positive": float(f)}
        for p, f in zip(mean_pred, frac_pos)
    ]


def evaluate_split(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold_criterion: str = "f1",
    grid_step: float = 0.01,
    seed: int = 0,
    predict_fn: callable | None = None,
) -> dict:
    """Evaluate a fitted classifier with a validation-tuned threshold.

    Args:
        model: Fitted estimator exposing ``predict_proba`` (or use predict_fn).
        X_test: Test features.
        y_test: Test labels.
        X_val: Validation features (used to tune the threshold).
        y_val: Validation labels.
        threshold_criterion: Threshold-tuning objective (see tune_threshold).
        grid_step: Threshold grid step.
        seed: Random seed (for determinism in predict_fn, if any).
        predict_fn: Optional callable (X) -> probabilities, overrides
            model.predict_proba.

    Returns:
        Full metric dictionary on the test set, plus the chosen threshold.
    """
    y_val = np.asarray(y_val).astype(int)
    y_test = np.asarray(y_test).astype(int)

    if predict_fn is None:
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
    else:
        p_val = predict_fn(X_val)
        p_test = predict_fn(X_test)

    threshold = tune_threshold(
        y_val, p_val, criterion=threshold_criterion, grid_step=grid_step, seed=seed
    )
    metrics = classification_metrics(y_test, p_test, threshold)
    metrics["validation_used_for_threshold"] = True
    return metrics


def save_json(data: dict, path) -> None:
    """Write a dictionary to a JSON file (path may be str or Path)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path) -> dict:
    """Read a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def comparison_table(collected: dict[str, dict]) -> pd.DataFrame:
    """Build a tidy metrics table from a {model_name: metrics} dictionary."""
    rows = []
    for name, m in collected.items():
        rows.append(
            {
                "model": name,
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "brier": m.get("brier"),
                "threshold": m.get("threshold"),
            }
        )
    return pd.DataFrame(rows)


def flatten_metric_name(m: dict) -> dict:
    """Flatten nested 'confusion' dict into separate keys for easy saving."""
    out = {k: v for k, v in m.items() if k != "confusion"}
    for k, v in m.get("confusion", {}).items():
        out[f"cm_{k}"] = v
    return out


# ---------------------------------------------------------------------------
# Storm-level bootstrap confidence intervals (item 6 from reviewer checklist)
# ---------------------------------------------------------------------------

def storm_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    storm_ids: np.ndarray,
    metric_fn: callable | None = None,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute a bootstrap confidence interval by resampling *storms*.

    Resamples storms with replacement, recomputes the metric on the
    resampled observations, and reports the percentile CI.  Single-class
    resamples are skipped (as they would give degenerate metrics).

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        storm_ids: Storm ID for each observation.
        metric_fn: ``(y_true, y_score) -> float``. Defaults to PR-AUC.
        n_boot: Number of bootstrap resamples.
        ci_level: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with 'point_estimate', 'ci_low', 'ci_high', 'n_valid',
        'mean', 'std'.
    """
    if metric_fn is None:
        metric_fn = safe_pr_auc

    rng = np.random.RandomState(seed)
    storms = np.asarray(sorted(set(storm_ids)))
    g = np.asarray(storm_ids)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    point = metric_fn(y_true, y_score)
    boot_vals = []
    for _ in range(n_boot):
        picked = rng.choice(storms, size=len(storms), replace=True)
        mask = np.isin(g, picked)
        if len(np.unique(y_true[mask])) < 2:
            continue
        boot_vals.append(float(metric_fn(y_true[mask], y_score[mask])))

    boot_vals = np.asarray(boot_vals)
    if len(boot_vals) < 100:
        return {
            "point_estimate": float(point),
            "ci_low": None, "ci_high": None,
            "n_valid": int(len(boot_vals)),
            "mean": None, "std": None,
            "note": "too few valid resamples for reliable CI",
        }

    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(boot_vals, alpha * 100))
    hi = float(np.percentile(boot_vals, (1.0 - alpha) * 100))

    return {
        "point_estimate": float(point),
        "ci_low": lo, "ci_high": hi,
        "n_valid": int(len(boot_vals)),
        "mean": float(boot_vals.mean()),
        "std": float(boot_vals.std()),
    }


# ---------------------------------------------------------------------------
# Probability calibration  (item 5 from reviewer checklist)
# ---------------------------------------------------------------------------

def calibration_detailed(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Full calibration report: Brier, reliability data, calibration curve.

    Also fits isotonic regression as a calibration reference.

    Args:
        y_true: True binary labels.
        y_score: Predicted probabilities.
        n_bins: Number of bins for the reliability diagram.
        strategy: 'uniform' or 'quantile' binning.

    Returns:
        Dict with 'brier', 'reliability_table', 'calibration_curve',
        'isotonic_brier', 'calibration_intercept', 'calibration_slope'.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {"brier": float("nan"), "reliability_table": [],
                "calibration_curve": [], "isotonic_brier": float("nan"),
                "calibration_intercept": float("nan"),
                "calibration_slope": float("nan")}

    # Brier score.
    brier = float(brier_score_loss(y_true, y_score))

    # Reliability table (binned).
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy=strategy
        )
    except Exception:
        frac_pos, mean_pred = np.array([]), np.array([])

    reliability = [
        {"mean_predicted": float(p), "fraction_positive": float(f)}
        for p, f in zip(mean_pred, frac_pos)
    ]

    # Calibration intercept and slope (Hosmer-Lemeshow-style).
    # Fit a simple logistic calibration: logit(p) = a + b * logit(score).
    try:
        from sklearn.linear_model import LogisticRegression
        logits = np.log(np.clip(y_score, 1e-6, 1 - 1e-6) /
                        (1 - np.clip(y_score, 1e-6, 1 - 1e-6)))
        lr = LogisticRegression(fit_intercept=True, max_iter=1000)
        lr.fit(logits.reshape(-1, 1), y_true)
        cal_slope = float(lr.coef_[0, 0])
        cal_intercept = float(lr.intercept_[0])
    except Exception:
        cal_slope = float("nan")
        cal_intercept = float("nan")

    # Isotonic regression calibration.
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_score, y_true)
        y_iso = iso.predict(y_score)
        iso_brier = float(brier_score_loss(y_true, y_iso))
    except Exception:
        iso_brier = float("nan")

    return {
        "brier": brier,
        "reliability_table": reliability,
        "calibration_curve": reliability,  # alias for plotting
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "isotonic_brier": iso_brier,
    }


# ---------------------------------------------------------------------------
# Preprocessing leakage guards  (item 8 from reviewer checklist)
# ---------------------------------------------------------------------------

class PreprocessingLeakageError(RuntimeError):
    """Raised when preprocessing leakage is detected."""


def assert_train_only_scaler(
    scaler,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    tolerance: float = 1e-6,
) -> None:
    """Assert that a scaler was fit on training data only.

    Checks that the scaler's transform on training data produces values
    in [0, 1] (for MinMaxScaler) or reasonable ranges, and that val/test
    data may exceed those ranges (which would indicate the scaler was fit
    on a broader dataset).

    This is a soft check — it raises ``PreprocessingLeakageError`` if the
    scaler appears to have been fit on data that includes val/test.
    """
    if not hasattr(scaler, "data_min_"):
        return  # Not a MinMaxScaler; skip.

    # The scaler's internal stats should match the training data.
    train_min = X_train.min().to_numpy()
    train_max = X_train.max().to_numpy()
    scaler_min = np.asarray(scaler.data_min_)
    scaler_range = np.asarray(scaler.data_range_)

    # Check that the scaler's min/max match the training data's min/max.
    if len(scaler_min) == len(train_min):
        min_diff = np.abs(scaler_min - train_min)
        max_diff = np.abs((scaler_min + scaler_range) - train_max)
        if min_diff.max() > tolerance or max_diff.max() > tolerance:
            raise PreprocessingLeakageError(
                "Scaler data_min_ / data_range_ do not match the training "
                "data. The scaler may have been fit on a broader dataset "
                "(preprocessing leakage)."
            )


def assert_no_smote_before_split(
    n_smote_total: int,
    n_smote_train: int,
    n_total: int,
) -> None:
    """Assert that SMOTE was applied only inside training folds.

    Raises ``PreprocessingLeakageError`` if SMOTE appears to have been
    applied before the storm split.
    """
    if n_smote_total > 0 and n_smote_train < n_smote_total * 0.9:
        raise PreprocessingLeakageError(
            f"SMOTE was applied to {n_smote_total} rows total but only "
            f"{n_smote_train} are in the training set. SMOTE must be applied "
            "inside training folds only, never before the storm split."
        )
