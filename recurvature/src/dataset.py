"""Storm-level train/val/test split and tabular dataset builder for XGBoost."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLS


def storm_split(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """Split by storm ID (not row) so no timesteps from the same cyclone
    leak between train / val / test."""
    sid_label = df.dropna(subset=["recurve_label"]).groupby("SID")["recurve_label"].max()
    sids, strat = sid_label.index.values, sid_label.values

    train_sids, test_sids = train_test_split(
        sids, test_size=test_size, random_state=seed, stratify=strat
    )
    strat2 = sid_label.loc[train_sids].values
    train_sids, val_sids = train_test_split(
        train_sids, test_size=val_size / (1 - test_size), random_state=seed, stratify=strat2
    )
    return set(train_sids), set(val_sids), set(test_sids)


def make_tabular(df: pd.DataFrame, sids: set):
    d = df[df["SID"].isin(sids)].dropna(subset=["recurve_label"]).copy()
    d[["dir_change_3h", "dir_change_9h"]] = d[["dir_change_3h", "dir_change_9h"]].fillna(0)
    X = d[FEATURE_COLS].values.astype(np.float32)
    y = d["recurve_label"].values.astype(np.float32)
    return X, y
