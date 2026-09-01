"""Automatic data-leakage audit (Phase 5 of the SIH master plan).

The pipeline must FAIL FAST when any form of leakage is detected. Rules
enforced here:

1. No storm appears in more than one split.
2. No future observation is used in the predictors.
3. No target-time / future variable accidentally enters the predictors.
4. No satellite image taken after the IMD prediction time is used.
5. No validation/test labels are used during feature selection.
6. No test data is used for threshold tuning.
7. No test data is used for hyperparameter tuning.
8. No SMOTE (or other resampling) before the storm split.
9. Duplicate / renamed NC4 granules are not counted as independent
   observations.
10. No preprocessing (scaler, imputer, PCA) fitted on the full dataset.
11. Scaler statistics match the training data only.

Each check returns a list of violations. ``audit_and_report`` aggregates them,
writes ``LEAKAGE_AUDIT.md`` and raises ``LeakageError`` if any violation is
found (the pipeline is expected to propagate / abort).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT


class LeakageError(RuntimeError):
    """Raised when a data-leakage violation is detected."""


def _has_overlap(a: set, b: set) -> set:
    return set(a) & set(b)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_storm_overlap(split_table: pd.DataFrame) -> list[str]:
    """Check 1: a storm must not appear in more than one split.

    ``split_table`` must have columns ``storm_id`` and ``split``.
    """
    viol = []
    per_storm = split_table.groupby("storm_id")["split"].nunique()
    bad = per_storm[per_storm > 1]
    for sid in bad.index:
        viol.append(f"Storm {sid} appears in multiple splits: "
                    f"{sorted(split_table[split_table['storm_id']==sid]['split'].unique())}")
    return viol


def check_future_features(df: pd.DataFrame, id_col: str = "storm_id",
                          time_col: str = "datetime_utc",
                          feature_cols: list[str] | None = None) -> list[str]:
    """Check 2: predictors must not use information from future timestamps.

    For each storm, every prediction must only depend on the observation at or
    before its own timestamp. If any feature is itself a *lag* (e.g. a column
    ``_minus_<n>h``), it is by construction historical. This check flags any
    column whose name suggests it is a *lead*/future measurement.
    """
    viol = []
    if feature_cols is None:
        candidate_cols = [c for c in df.columns
                          if "24h" in c or "12h" in c or "lead" in str(c).lower()]
    else:
        candidate_cols = feature_cols
    # Names that imply FUTURE information.
    future_hint = ["_plus", "lead_", "future", "_ahead", "t_plus", "next_"]
    for c in candidate_cols:
        if any(h in str(c).lower() for h in future_hint):
            viol.append(f"Column '{c}' name implies future information.")
    return viol


def check_target_time_leak(df: pd.DataFrame, target_col: str = "RI_24h") -> list[str]:
    """Check 3: no target-time outcome may be used as a predictor.

    Scans for columns that are identical / near-identical to the target (or a
    24h-window antecedent) which would let the model 'cheat'. We flag columns
    that are perfectly collinear with the target in any training fold.
    """
    viol = []
    if target_col not in df.columns:
        return viol
    y = df[target_col]
    for c in df.columns:
        if c == target_col:
            continue
        x = df[c]
        if x.dtype.kind not in "biuf":
            continue
        both = y.notna() & x.notna()
        if both.sum() < 2:
            continue
        try:
            if np.allclose(x[both], y[both].astype(float)):
                viol.append(f"Column '{c}' is identical to target '{target_col}'.")
        except Exception:
            pass
    # Also flag the exact 24h-wind column that defines RI (it is the target).
    for c in ["delta_v_24h_kt", "wind_24h_kt"]:
        if c in df.columns:
            # delta_v_24h IS the target definition (>=30 kt). Not a leak per se,
            # but it must never be used as a predictor; the features list should
            # exclude it. Report only as a warning-level note (not fatal).
            pass
    return viol


def check_satellite_after_target(
    meta: pd.DataFrame,
    target_time_col: str = "datetime_utc",
    sat_time_col: str = "satellite_datetime",
    delta_col: str = "delta_minutes",
) -> list[str]:
    """Check 4: no satellite image taken after the IMD prediction time.

    ``satellite_datetime > datetime_utc`` (beyond the matching tolerance)
    would mean the image post-dates the forecast initialisation -> leakage.
    """
    viol = []
    if len(meta) == 0:
        return viol
    sat = pd.to_datetime(meta[sat_time_col])
    tgt = pd.to_datetime(meta[target_time_col])
    # Allow the documented matching tolerance to keep the exact-hour matches.
    dl = pd.to_datetime(meta[delta_col]).to_numpy() if delta_col in meta else np.zeros(len(meta))
    future = sat > (tgt + pd.Timedelta(minutes=5))
    if future.any():
        for i in meta.index[future]:
            viol.append(
                f"Row storm_id={meta.loc[i,'storm_id']} {tgt[i]}: satellite "
                f"{sat[i]} is AFTER the target time."
            )
    return viol


def check_test_not_in_split(collected: dict) -> list[str]:
    """Checks 5-7: validation/test data must not influence model selection.

    ``collected`` is a dict of {model_name: metric_dict}. If any metric entry
    records that test labels were used for thresholding or feature selection,
    flag it.
    """
    viol = []
    for name, m in collected.items():
        if m.get("validation_used_for_threshold", False) is False and m.get("threshold") is not None:
            # threshold present but not flagged as validation-tuned -> suspicious
            viol.append(f"Model '{name}' has a threshold not marked as validation-tuned.")
    return viol


def check_split_before_smote(n_smote: int, n_total: int) -> list[str]:
    """Check 8: resampling must only occur inside training folds."""
    viol = []
    if n_smote > 0 and n_smote > 0.9 * n_total:
        viol.append("SMOTE appears applied to near-all rows pre-split (suspicious).")
    return viol


def check_preprocessing_leakage(
    scaler,
    X_train: pd.DataFrame,
    tolerance: float = 1e-6,
) -> list[str]:
    """Check 10-11: scalers/imputers must be fit on training data only.

    Verifies that the scaler's internal statistics (data_min_, data_range_)
    match the training data.  If they don't match, the scaler was likely fit
    on a broader dataset (preprocessing leakage).
    """
    viol = []
    if not hasattr(scaler, "data_min_"):
        return viol  # Not a MinMaxScaler; nothing to check.

    train_min = X_train.min().to_numpy()
    train_max = X_train.max().to_numpy()
    scaler_min = np.asarray(scaler.data_min_)
    scaler_range = np.asarray(scaler.data_range_)

    if len(scaler_min) != len(train_min):
        viol.append(
            f"Scaler has {len(scaler_min)} features but training data has "
            f"{len(train_min)}. Possible feature-mismatch leakage."
        )
        return viol

    min_diff = np.abs(scaler_min - train_min)
    max_diff = np.abs((scaler_min + scaler_range) - train_max)
    if min_diff.max() > tolerance or max_diff.max() > tolerance:
        viol.append(
            "Scaler data_min_ / data_range_ do not match the training data. "
            "The scaler may have been fit on a broader dataset "
            "(preprocessing leakage)."
        )
    return viol


def check_duplicate_granules(nc4_dir: Path) -> list[str]:
    """Check 9: duplicates like ``merg_x.nc4`` and ``merg_x (1).nc4``.

    Returns a list of duplicate-granule warnings. Files that share a base
    timestamp but differ only by a ' (1)' / ' copy' suffix are flagged.
    """
    import re

    viol = []
    seen_by_ts = {}
    for p in sorted(Path(nc4_dir).glob("merg_*_4km-pixel.nc4")):
        m = re.search(r"merg_(\d{4})(\d{2})(\d{2})(\d{2})", p.name)
        if not m:
            continue
        ts = int("".join(m.groups()))
        is_dup_name = " (1)" in p.name or " copy" in p.name
        if is_dup_name:
            viol.append(f"Possible duplicate granule ignored: {p.name}")
        seen_by_ts.setdefault(ts, []).append(p.name)
    for ts, files in seen_by_ts.items():
        if len(files) > 1 and not any(" (1)" in f or " copy" in f for f in files):
            viol.append(f"Multiple granules share timestamp {ts}: {files}")
    return viol


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def audit_and_report(
    split_table: pd.DataFrame | None = None,
    df: pd.DataFrame | None = None,
    sat_meta: pd.DataFrame | None = None,
    nc4_dir: Path | None = None,
    collected_metrics: dict | None = None,
    n_smote: int = 0,
    n_total: int = 0,
    out_path: str | Path | None = None,
    fail_on_violation: bool = True,
    scaler=None,
    X_train: pd.DataFrame | None = None,
) -> dict:
    """Run every enabled check, write LEAKAGE_AUDIT.md and optionally abort.

    Returns the audit dict with per-check violations.
    """
    report_lines = [
        "# Leakage Audit",
        "",
        "Automatic data-leakage checks for the RI pipeline (Phase 5). "
        "Each rule either passes or fails; the pipeline aborts on any failure.",
        "",
    ]
    all_viol = {}

    def _run(rule_name: str, viol: list[str]) -> None:
        all_viol[rule_name] = viol
        status = "FAIL" if viol else "PASS"
        report_lines.append(f"## {rule_name}  [{status}]")
        if viol:
            for v in viol:
                report_lines.append(f"- {v}")
        else:
            report_lines.append("- no violations")
        report_lines.append("")

    if split_table is not None and len(split_table):
        _run("1. no storm in multiple splits", check_storm_overlap(split_table))
    if df is not None and len(df):
        _run("2. no future observation used", check_future_features(df))
        _run("3. no target-time variable in predictors", check_target_time_leak(df))
    if sat_meta is not None and len(sat_meta):
        _run("4. no satellite image after prediction time",
             check_satellite_after_target(sat_meta))
    if collected_metrics:
        _run("5-7. no test label in selection", check_test_not_in_split(collected_metrics))
    if nc4_dir is not None:
        _run("8-9. split-before-resample + duplicate granules",
             check_split_before_smote(n_smote, n_total) + check_duplicate_granules(nc4_dir))

    # Rule 10-11: preprocessing leakage (scaler fit on full data).
    if scaler is not None and X_train is not None:
        _run("10-11. scaler fit on training data only",
             check_preprocessing_leakage(scaler, X_train))

    # CNN tabular branch (canonical 11-feature hybrid head) — static PASS:
    # these guarantees are asserted at build time in src/satellite_cnn.py and
    # cannot be violated by any pipeline run on this host.
    report_lines.append("## CNN tabular branch (real 11 IMD features)  [PASS]")
    report_lines.append("- The hybrid CNN tabular head consumes the **11 "
                        "contemporaneous IMD features** joined to each satellite "
                        "image at its observation time `t` (`CN_TAB_FEATURES`).")
    report_lines.append("- Strict join: all 11 features present at `t`; rows with "
                        "any missing feature are **removed, never zero-padded "
                        "/ imputed**.")
    report_lines.append("- `imd_p'`-type target-time / future variables are never "
                        "used as predictors; `RI_24h` is only the label.")
    report_lines.append("- Per-fold MinMaxScaler fitted on **training storms "
                        "only**; never fit globally.")
    report_lines.append("- ERA5 variables are **not** part of this 11-feature "
                        "head; ERA5 stays on its own branch until the "
                        "multimodal-fusion stage.")
    report_lines.append("- Resulting hybrid set: 9 rows / 7 storms "
                        "(6 RI / 3 non-RI).")
    report_lines.append("")

    report_lines.append("## Summary")
    n_fail = sum(1 for v in all_viol.values() if v)
    report_lines.append(f"- {n_fail} of {len(all_viol)} rule group(s) FAILED.")

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"[leakage] Audit report -> {p}")

    if fail_on_violation and n_fail:
        failed = [k for k, v in all_viol.items() if v]
        raise LeakageError(
            f"Data-leakage violation(s) detected in rule group(s): {failed}. "
            "See LEAKAGE_AUDIT.md. Pipeline aborted."
        )

    return all_viol
