"""Bridge between the Colab-run satellite CNN and the tabular pipeline.

The satellite CNN (trained in Google Colab — TensorFlow crashes on the local
macOS box) produces three artifacts that this module ingests:

- ``results/satellite_oof_predictions.csv`` : storm_id, datetime_utc, RI_24h,
  P_RI  (out-of-fold probabilities from a storm-safe CNN evaluation)
- ``results/satellite_embeddings.npy`` + ``satellite_embeddings_meta.csv`` :
  penultimate-layer CNN embeddings with metadata (storm_id, datetime_utc,
  RI_24h)
- ``models/satellite_cnn.keras`` : the trained CNN (for the dashboard)

If these files exist on disk they are loaded and joined into the multimodal
table for feature-level / late fusion. If not, the tabular pipeline still runs
fully and reports the satellite branch as *pending Colab output* (never
fabricates metrics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT


def load_satellite_oof(results_dir: str | Path, required_cols=("storm_id", "datetime_utc", "P_RI", "RI_24h")):
    """Load satellite OOF predictions if present; else return None."""
    p = Path(results_dir) / "satellite_oof_predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["storm_id"] = df["storm_id"].astype(str)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[satbridge] satellite_oof_predictions.csv missing cols {missing}")
        return None
    return df


def load_satellite_embeddings(results_dir: str | Path):
    """Load satellite embeddings (.npy + meta csv) if present; else None."""
    npy = Path(results_dir) / "satellite_embeddings.npy"
    meta_path = Path(results_dir) / "satellite_embeddings_meta.csv"
    if not npy.exists() or not meta_path.exists():
        return None
    emb = np.load(str(npy))
    meta = pd.read_csv(meta_path)
    meta["storm_id"] = meta["storm_id"].astype(str)
    meta["datetime_utc"] = pd.to_datetime(meta["datetime_utc"])
    return emb, meta


def build_cnn_probability_table(results_dir: str | Path) -> pd.DataFrame | None:
    """Return the CNN branch probability table (identity + P_RI) or None."""
    oof = load_satellite_oof(results_dir)
    if oof is None:
        return None
    return oof[["storm_id", "datetime_utc", "RI_24h", "P_RI"]].rename(
        columns={"P_RI": "P_cnn"}).copy()


def build_embedding_feature_table(results_dir: str | Path, multimodal: pd.DataFrame):
    """Join CNN embeddings as named columns onto the multimodal table.

    Embedding columns are ``sat_emb_0..sat_emb_{d-1}``. Returns a new
    DataFrame (None if no embeddings present or no overlap).
    """
    out = load_satellite_embeddings(results_dir)
    if out is None:
        return None
    emb, meta = out
    cols = {f"sat_emb_{i}": emb[:, i] for i in range(emb.shape[1])}
    emb_df = pd.DataFrame(cols)
    emb_df["storm_id"] = meta["storm_id"].values
    emb_df["datetime_utc"] = pd.to_datetime(meta["datetime_utc"].values)

    mm = multimodal.copy()
    mm["datetime_utc"] = pd.to_datetime(mm["datetime_utc"])
    merged = mm.merge(emb_df, on=["storm_id", "datetime_utc"], how="left")
    return merged
