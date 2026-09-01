"""TCIR 4-channel CNN inference + embedding extraction.

This module loads the trained TCIR CNN (a Keras model trained on Kaggle's
global TCIR dataset) and provides two capabilities:

1. **Full inference** (when TensorFlow is available): load the .keras model,
   preprocess a 201x201x4 TCIR image, and return either the RI probability
   or the 128-dimensional penultimate embedding.

2. **Precomputed bridge** (when TF is unavailable, e.g. on macOS): load the
   precomputed TCIR predictions + embeddings produced by the Kaggle notebook
   and saved as CSV + .npy files.

The TCIR CNN was trained independently on the global TCIR HDF5 dataset (23,118
cases, 4-channel satellite IR: IR+IR+MW+MW). It is a SEPARATE model from the
existing PyTorch satellite CNN (src/satellite_cnn.py) which uses a 2-channel
(local Bay of Bengal images). Both are valid satellite branches; TCIR adds
global tropical cyclone spatial context.

Storm-safe matching: TCIR cases are matched to IMD + ERA5 via
(storm_id, datetime_utc) with a configurable time tolerance.

Artifact contract (produced by the Kaggle notebook):
  results/tcir_oof_predictions.csv   — storm_id, datetime_utc, RI_24h, P_RI
  results/tcir_embeddings.npy        — (N, 128) float32
  results/tcir_embeddings_meta.csv   — storm_id, datetime_utc, RI_24h
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT

# ---------------------------------------------------------------------------
# Paths to the saved Keras model + normalization statistics
# ---------------------------------------------------------------------------

_TCIR_DIR = Path("models/tcir")
_MODEL_FILE = "TCIR_CNN_RI_FINAL.keras"
_MEAN_FILE = "TCIR_channel_mean.npy"
_STD_FILE = "TCIR_channel_std.npy"
_CONFIG_FILE = "TCIR_CNN_config.json"

# Precomputed bridge artifacts (produced by Kaggle notebook).
_TCIR_OOF_CSV = "results/tcir_oof_predictions.csv"
_TCIR_EMB_NPY = "results/tcir_embeddings.npy"
_TCIR_EMB_META = "results/tcir_embeddings_meta.csv"

# ---------------------------------------------------------------------------
# Lazy-loaded Keras model + normalization (only when TF is available)
# ---------------------------------------------------------------------------

_keras_loaded = False
_tcir_model = None
_channel_mean = None
_channel_std = None
_tcir_config = None
_tf_available = False


def _try_load_tf():
    """Attempt to import TensorFlow; return True if successful."""
    global _tf_available
    try:
        import tensorflow as _tf
        _tf_available = True
        return True
    except Exception:
        _tf_available = False
        return False


def _load_keras_assets():
    """Load the Keras model + normalization stats (once)."""
    global _keras_loaded, _tcir_model, _channel_mean, _channel_std, _tcir_config
    if _keras_loaded:
        return
    if not _tf_available and not _try_load_tf():
        return

    model_path = REPO_ROOT / _TCIR_DIR / _MODEL_FILE
    mean_path = REPO_ROOT / _TCIR_DIR / _MEAN_FILE
    std_path = REPO_ROOT / _TCIR_DIR / _STD_FILE
    config_path = REPO_ROOT / _TCIR_DIR / _CONFIG_FILE

    if not all(p.exists() for p in [model_path, mean_path, std_path]):
        print("[tcir] Keras model files not found in models/tcir/; "
              "full inference unavailable.")
        return

    import tensorflow as tf
    _tcir_model = tf.keras.models.load_model(str(model_path))
    _channel_mean = np.load(str(mean_path))
    _channel_std = np.load(str(std_path))

    if config_path.exists():
        with open(config_path) as f:
            _tcir_config = json.load(f)
    else:
        _tcir_config = {"img_size": 128, "channels": 4}

    _keras_loaded = True
    print(f"[tcir] Loaded Keras model: {model_path}")
    print(f"[tcir] Normalization: mean shape={_channel_mean.shape}, "
          f"std shape={_channel_std.shape}")


# ---------------------------------------------------------------------------
# Preprocessing (must exactly match Kaggle training pipeline)
# ---------------------------------------------------------------------------

def preprocess_tcir_image(image: np.ndarray) -> np.ndarray:
    """Preprocess a single TCIR image for the CNN.

    Steps (matching the Kaggle training pipeline exactly):
    1. Convert to float32
    2. Replace NaN/Inf with 0
    3. Resize from 201x201 to 128x128 (bilinear)
    4. Normalize using training-set channel mean/std
    5. Replace any remaining NaN/Inf with 0

    Args:
        image: Raw TCIR image, shape (201, 201, 4) or (128, 128, 4).

    Returns:
        Preprocessed image, shape (128, 128, 4), float32.
    """
    import tensorflow as tf

    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Resize if needed (201x201 -> 128x128).
    target_size = int(_tcir_config.get("img_size", 128)) if _tcir_config else 128
    if image.shape[0] != target_size or image.shape[1] != target_size:
        image = tf.image.resize(
            image, (target_size, target_size), method="bilinear"
        ).numpy()

    # Same normalization used during training.
    image = (image - _channel_mean) / (_channel_std + 1e-6)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    return image


# ---------------------------------------------------------------------------
# Inference functions (require TensorFlow)
# ---------------------------------------------------------------------------

def predict_tcir_single(image: np.ndarray) -> float:
    """Predict RI probability for a single TCIR image.

    Args:
        image: Raw TCIR image, shape (201, 201, 4) or (128, 128, 4).

    Returns:
        RI probability (float in [0, 1]).
    """
    _load_keras_assets()
    if _tcir_model is None:
        return float("nan")
    processed = preprocess_tcir_image(image)
    batch = np.expand_dims(processed, axis=0)
    prob = _tcir_model.predict(batch, verbose=0)[0, 0]
    return float(prob)


def predict_tcir_batch(images: np.ndarray) -> np.ndarray:
    """Predict RI probabilities for a batch of TCIR images.

    Args:
        images: Array of shape (N, 201, 201, 4) or (N, 128, 128, 4).

    Returns:
        Array of shape (N,) with RI probabilities.
    """
    _load_keras_assets()
    if _tcir_model is None:
        return np.full(len(images), np.nan)
    processed = np.stack([preprocess_tcir_image(img) for img in images])
    probs = _tcir_model.predict(processed, verbose=0)[:, 0]
    return probs.astype(np.float64)


def extract_tcir_embedding_single(image: np.ndarray) -> np.ndarray:
    """Extract the 128-D penultimate embedding for a single TCIR image.

    The embedding comes from the Dense(128, relu) layer immediately before
    the final sigmoid output. This is the spatial representation the fusion
    model should consume.

    Args:
        image: Raw TCIR image, shape (201, 201, 4) or (128, 128, 4).

    Returns:
        1-D array of shape (128,) — the TCIR spatial embedding.
    """
    _load_keras_assets()
    if _tcir_model is None:
        return np.full(128, np.nan)
    embedding_model = _build_embedding_model()
    processed = preprocess_tcir_image(image)
    batch = np.expand_dims(processed, axis=0)
    emb = embedding_model.predict(batch, verbose=0)
    return emb[0].astype(np.float64)


def extract_tcir_embeddings_batch(images: np.ndarray) -> np.ndarray:
    """Extract 128-D embeddings for a batch of TCIR images.

    Args:
        images: Array of shape (N, 201, 201, 4) or (N, 128, 128, 4).

    Returns:
        Array of shape (N, 128) — the TCIR spatial embeddings.
    """
    _load_keras_assets()
    if _tcir_model is None:
        return np.full((len(images), 128), np.nan)
    embedding_model = _build_embedding_model()
    processed = np.stack([preprocess_tcir_image(img) for img in images])
    embs = embedding_model.predict(processed, verbose=0)
    return embs.astype(np.float64)


_embedding_model_cache = None


def _build_embedding_model():
    """Build a Keras model that outputs the Dense(128) embedding.

    The TCIR CNN architecture is:
        Conv blocks -> GlobalAveragePooling2D -> Dense(128, relu) -> Dropout -> Dense(1, sigmoid)

    We truncate at the Dense(128) layer (the second-to-last dense layer).
    """
    global _embedding_model_cache
    if _embedding_model_cache is not None:
        return _embedding_model_cache

    # Find the Dense(128) layer — it is the second-to-last Dense layer.
    dense_128_layer = None
    for layer in reversed(_tcir_model.layers):
        if hasattr(layer, "units") and layer.units == 128:
            dense_128_layer = layer
            break

    if dense_128_layer is None:
        # Fallback: use the output of the second-to-last layer.
        dense_128_layer = _tcir_model.layers[-2]

    _embedding_model_cache = __import__("tensorflow").keras.Model(
        inputs=_tcir_model.input,
        outputs=dense_128_layer.output,
    )
    return _embedding_model_cache


# ---------------------------------------------------------------------------
# Bridge: load precomputed TCIR outputs (no TensorFlow needed)
# ---------------------------------------------------------------------------

def load_tcir_oof(results_dir: str | Path | None = None) -> pd.DataFrame | None:
    """Load precomputed TCIR OOF predictions if present; else return None.

    The Kaggle notebook produces results/tcir_oof_predictions.csv with columns:
        storm_id, datetime_utc, RI_24h, P_RI
    """
    if results_dir is None:
        results_dir = REPO_ROOT / "results"
    p = Path(results_dir) / "tcir_oof_predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["storm_id"] = df["storm_id"].astype(str)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    required = ["storm_id", "datetime_utc", "P_RI", "RI_24h"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[tcir] tcir_oof_predictions.csv missing columns {missing}")
        return None
    return df


def load_tcir_embeddings(results_dir: str | Path | None = None):
    """Load precomputed TCIR embeddings + metadata; else return None.

    Returns:
        (embeddings, meta) tuple or None.
        embeddings: np.ndarray of shape (N, 128)
        meta: pd.DataFrame with storm_id, datetime_utc, RI_24h
    """
    if results_dir is None:
        results_dir = REPO_ROOT / "results"
    npy_path = Path(results_dir) / "tcir_embeddings.npy"
    meta_path = Path(results_dir) / "tcir_embeddings_meta.csv"
    if not npy_path.exists() or not meta_path.exists():
        return None
    emb = np.load(str(npy_path))
    meta = pd.read_csv(meta_path)
    meta["storm_id"] = meta["storm_id"].astype(str)
    meta["datetime_utc"] = pd.to_datetime(meta["datetime_utc"])
    return emb, meta


def build_tcir_probability_table(results_dir: str | Path | None = None) -> pd.DataFrame | None:
    """Return the TCIR branch probability table (identity + P_RI) or None."""
    oof = load_tcir_oof(results_dir)
    if oof is None:
        return None
    return oof[["storm_id", "datetime_utc", "RI_24h", "P_RI"]].rename(
        columns={"P_RI": "P_tcir"}).copy()


def build_tcir_embedding_table(results_dir: str | Path | None = None,
                                multimodal: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Join TCIR embeddings as named columns onto the multimodal table.

    Embedding columns are ``tcir_emb_0`` through ``tcir_emb_127``.
    Returns a new DataFrame (None if no embeddings present or no overlap).
    """
    out = load_tcir_embeddings(results_dir)
    if out is None:
        return None
    emb, meta = out
    cols = {f"tcir_emb_{i}": emb[:, i] for i in range(emb.shape[1])}
    emb_df = pd.DataFrame(cols)
    emb_df["storm_id"] = meta["storm_id"].values
    emb_df["datetime_utc"] = pd.to_datetime(meta["datetime_utc"].values)

    if multimodal is None:
        return None
    mm = multimodal.copy()
    mm["datetime_utc"] = pd.to_datetime(mm["datetime_utc"])
    merged = mm.merge(emb_df, on=["storm_id", "datetime_utc"], how="left")
    return merged


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------

def tcir_status(results_dir: str | Path | None = None) -> dict:
    """Report the TCIR branch status (files present, counts, etc.)."""
    if results_dir is None:
        results_dir = REPO_ROOT / "results"
    results_dir = Path(results_dir)

    model_exists = (REPO_ROOT / _TCIR_DIR / _MODEL_FILE).exists()
    mean_exists = (REPO_ROOT / _TCIR_DIR / _MEAN_FILE).exists()
    std_exists = (REPO_ROOT / _TCIR_DIR / _STD_FILE).exists()
    config_exists = (REPO_ROOT / _TCIR_DIR / _CONFIG_FILE).exists()

    oof = load_tcir_oof(results_dir)
    emb = load_tcir_embeddings(results_dir)

    status = {
        "keras_model": model_exists,
        "normalization": mean_exists and std_exists,
        "config": config_exists,
        "tf_available": _tf_available,
        "oof_predictions": oof is not None,
        "oof_rows": len(oof) if oof is not None else 0,
        "oof_storms": int(oof["storm_id"].nunique()) if oof is not None else 0,
        "embeddings_available": emb is not None,
        "embedding_dim": int(emb[0].shape[1]) if emb is not None else 0,
    }

    if model_exists and _tf_available:
        status["inference_mode"] = "full (TensorFlow available)"
    elif oof is not None:
        status["inference_mode"] = "precomputed bridge (Kaggle output)"
    else:
        status["inference_mode"] = "unavailable (no model or precomputed output)"

    return status
