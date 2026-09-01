"""Satellite IR CNN branch (data-limited MVP).

The satellite branch is treated honestly: with only 3 images on disk across 3
storms (see PROJECT_AUDIT.md), no scientifically meaningful train/validation/
test split is possible. This module:

* Checks whether enough usable images exist to train a CNN.
* If so, loads images, applies the configured small augmentations, and trains
  a light CNN with early stopping, checkpointing, batch normalisation,
  dropout and best-epoch selection.
* If not, it does **not** fabricate a model or metrics. It records the
  limitation and returns an empty/None result that the rest of the pipeline
  handles gracefully.

This keeps the branch reproducible and honest even though the data is tiny.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, get_seed


def available_images(metadata: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata down to rows whose image file exists."""
    if metadata is None or len(metadata) == 0:
        return metadata
    exists = metadata["image_path"].apply(
        lambda p: Path(p) is not None and Path(p).exists()
    )
    return metadata[exists].copy()


def _load_images(metadata: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    """Load all images in metadata into X/N arrays. Returns (X, y, meta)."""
    if metadata is None or len(metadata) == 0:
        return None, None, metadata
    X = []
    y = []
    kept = []
    for _, row in metadata.iterrows():
        p = row.get("image_path")
        if p is None or not Path(p).exists():
            continue
        try:
            img = np.load(str(p))
        except Exception:
            continue
        if img.shape != (128, 128, 1):
            continue
        X.append(img.astype(np.float32))
        y.append(int(row["RI_24h"]))
        kept.append(row)
    if not X:
        return None, None, metadata.iloc[0:0]
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            pd.DataFrame(kept))


def run_cnn_branch(
    cfg: dict,
    metadata: pd.DataFrame,
    seed: int,
) -> dict:
    """Train (or honestly skip) the satellite CNN branch.

    Args:
        cfg: pipeline configuration.
        metadata: satellite metadata (already filtered to existing images).
        seed: random seed.

    Returns:
        A result dict with keys: 'status', 'n_images', 'n_storms',
        'ri/non_ri', 'message', and (if trained) 'model', 'metrics'.
    """
    seed = get_seed(cfg)

    if metadata is None or len(metadata) == 0:
        return {
            "status": "skipped",
            "n_images": 0,
            "n_storms": 0,
            "ri": 0,
            "non_ri": 0,
            "message": "No satellite images available on disk.",
        }

    X, y, valid_meta = _load_images(metadata)
    n = 0 if X is None else len(X)
    n_storms = 0 if valid_meta is None or len(valid_meta) == 0 else valid_meta["storm_id"].nunique()
    n_ri = int((y == 1).sum()) if y is not None else 0
    n_non = n - n_ri

    min_images = int(cfg["cnn"].get("min_images_for_training", 12))
    if n < min_images or n_storms < 3:
        return {
            "status": "skipped",
            "n_images": n,
            "n_storms": n_storms,
            "ri": n_ri,
            "non_ri": n_non,
            "message": (
                f"Only {n} usable satellite image(s) across {n_storms} storm(s) "
                f"on disk (need >= {min_images} images and >= 3 storms). "
                "No CNN trained — a storm-safe train/validation/test split is "
                "not scientifically possible at this size. This branch is "
                "reported as a data-limited MVP."
            ),
        }

    # --- Enough data: train the CNN (kept for completeness) ---------------
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    img_size = int(cfg["cnn"].get("img_size", 128))

    # Note: with so few storms a three-way split is impossible; the CNN is
    # trained on all available images purely as a placeholder and is NOT used
    # for the reported held-out comparison.
    class_weight = {0: 1.0, 1: float(n_non / max(n_ri, 1))}

    data_gen = ImageDataGenerator(
        rotation_range=float(cfg["cnn"]["augmentation"].get("rotation_deg", 15)),
        width_shift_range=float(cfg["cnn"]["augmentation"].get("shift_fraction", 0.1)),
        height_shift_range=float(cfg["cnn"]["augmentation"].get("shift_fraction", 0.1)),
        brightness_range=tuple(cfg["cnn"]["augmentation"].get("brightness_range", [0.9, 1.1])),
        horizontal_flip=bool(cfg["cnn"]["augmentation"].get("horizontal_flip", False)),
        vertical_flip=bool(cfg["cnn"]["augmentation"].get("random_flip_vertical", False)),
    )

    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["cnn"].get("learning_rate", 1e-3))),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="pr_auc", curve="PR")],
    )

    checkpoint_path = REPO_ROOT / cfg["paths"]["models_dir"] / "satellite_ir_cnn_mvp_repro.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="pr_auc", mode="max", patience=int(cfg["cnn"].get("patience", 10)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="pr_auc", save_best_only=True,
        ),
    ]

    X_re = X.reshape((len(X), img_size, img_size, 1))
    model.fit(
        data_gen.flow(X_re, y, batch_size=int(cfg["cnn"].get("batch_size", 4))),
        epochs=int(cfg["cnn"].get("epochs", 60)),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(str(checkpoint_path))

    return {
        "status": "trained",
        "n_images": n,
        "n_storms": n_storms,
        "ri": n_ri,
        "non_ri": n_non,
        "message": "CNN trained on all available images (no reliable held-out split).",
        "model_path": str(checkpoint_path),
        "model": model,
    }


def save_cnn_result(result: dict, results_dir) -> None:
    """Write the CNN branch result (including a skip) to JSON."""
    import json

    path = f"{results_dir}/satellite_cnn_branch_result.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[satellite] Wrote branch result -> {path}")
