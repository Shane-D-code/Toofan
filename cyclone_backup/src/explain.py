"""Explainability: feature importance, optional SHAP and figure generation.

Because the ``shap`` package may not be installed in every environment, this
module degrades gracefully: it always produces a feature-importance table from
the XGBoost gain importance, and additionally attempts a SHAP summary when
``shap`` is available. Nothing here raises if ``shap`` is missing.
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import REPO_ROOT


def feature_importance_table(
    model, feature_cols: list[str], feature_type: str = "gain"
) -> pd.DataFrame:
    """Return a sorted (feature, importance) table from an XGBoost model."""
    imp = model.get_booster().get_score(importance_type=feature_type)
    # get_score returns names in the form f0, f1, ... -- map back to columns.
    col_map = {f"f{i}": c for i, c in enumerate(feature_cols)}
    rows = []
    for key, val in imp.items():
        rows.append({"feature": col_map.get(key, key), "importance": float(val)})
    df = pd.DataFrame(rows)[["feature", "importance"]]
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def save_feature_importance(model, feature_cols: list[str], path) -> pd.DataFrame:
    """Save and return the feature-importance table."""
    df = feature_importance_table(model, feature_cols)
    df.to_csv(path, index=False)
    print(f"[explain] Saved feature importance -> {path}")
    return df


def shap_summary(model, X_sample: pd.DataFrame, feature_cols: list[str], out_path):
    """Attempt a SHAP summary plot; no-op if ``shap`` is unavailable.

    Args:
        model: Fitted XGBoost model.
        X_sample: Small sample of features used for the SHAP background.
        feature_cols: Feature column names in the order the model expects.
        out_path: Where to save the summary figure.
    """
    try:
        import shap
    except Exception as exc:  # pragma: no cover - environment dependent
        print("[explain] SHAP unavailable; using gain importance only. "
              f"({type(exc).__name__})")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Re-tag feature names so SHAP shows the real column names.
        model.get_booster().feature_names = feature_cols
        X_named = X_sample.copy()
        X_named.columns = feature_cols

        explainer = shap.TreeExplainer(model)
        sample = X_named.sample(
            n=min(200, len(X_named)),
            random_state=0,
        ) if len(X_named) > 0 else X_named
        if len(sample) == 0:
            return None

        shap_values = explainer.shap_values(sample)

        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[explain] SHAP summary saved -> {out_path}")
    return True


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    out_dir,
) -> str:
    """Plot and save a precision-recall curve."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker=".", label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {model_name}")
    plt.legend()
    safe = model_name.replace(" ", "_").replace("+", "and")
    path = f"{out_dir}/pr_curve_{safe}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confusion_matrix(cm: dict, model_name: str, out_dir) -> str:
    """Plot and save a labelled confusion-matrix heatmap."""
    matrix = np.array(
        [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["non-RI", "RI"])
    ax.set_yticklabels(["non-RI", "RI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    safe = model_name.replace(" ", "_").replace("+", "and")
    path = f"{out_dir}/confusion_{safe}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_satellite_examples(
    metadata: pd.DataFrame,
    probabilities: dict,
    out_dir,
    max_examples: int = 4,
):
    """Optional helper to visualise satellite images with predicted RI prob.

    ``probabilities`` maps storm_id/datetime or image_file -> P_RI. This is a
    best-effort utility; it silently returns if no images are provided.
    """
    if metadata is None or len(metadata) == 0:
        print("[explain] No satellite images available for examples figure.")
        return None

    import os
    shown = 0
    fig, axes = plt.subplots(
        1, min(max_examples, len(metadata)), figsize=(4 * min(max_examples, len(metadata)), 4)
    )
    if len(metadata) == 0:
        return None
    axes = np.atleast_1d(axes)
    for _, row in metadata.iterrows():
        if shown >= max_examples:
            break
        img_path = row.get("image_path")
        if img_path is None or not os.path.exists(str(img_path)):
            continue
        img = np.load(str(img_path))
        ax = axes[shown]
        ax.imshow(img[:, :, 0], cmap="gray")
        p = probabilities.get(row["image_file"], float("nan"))
        ax.set_title(f"{row['storm_id']} RI={int(row['RI_24h'])}\nP={p:.2f}")
        ax.axis("off")
        shown += 1
    if shown == 0:
        plt.close(fig)
        print("[explain] No satellite images available for examples figure.")
        return None
    path = f"{out_dir}/satellite_examples.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[explain] Satellite examples saved -> {path}")
    return path
