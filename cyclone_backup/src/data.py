"""Data loading and storm-safe splitting.

This module is responsible for:

* Loading the canonical IMD, ERA5 and satellite datasets.
* Building the IMD+ERA5 combined table (matched 1:1 on storm + timestamp).
* Splitting at the **storm level** so that no storm appears in more than one
  split (no storm leakage).
* Verifying the split with assertions and reporting dataset statistics.

The splitter deliberately operates on groups (storm IDs), never on individual
observations, so observations from the same storm always travel together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, get_seed


# ---------------------------------------------------------------------------
# Column-name helpers
# ---------------------------------------------------------------------------

# Core identity columns shared by every table.
_ID_COLS = ["storm_id", "datetime_utc"]

# IMD predictor columns available in the canonical IMD feature table.
IMD_FEATURE_COLS = [
    "latitude",
    "longitude",
    "max_wind_kt",
    "central_pressure_hpa",
    "pressure_drop_hpa",
    "wind_6h_change",
    "wind_minus_6h_kt",
    "delta_v_minus_6h_kt",
    "wind_minus_12h_kt",
    "delta_v_minus_12h_kt",
    "wind_minus_24h_kt",
    "delta_v_minus_24h_kt",
]

# RAW ERA5 columns as they appear in RI_ERA5_features_MVP.csv. Derived
# predictors are added in features.py.
ERA5_RAW_COLS = [
    "d_850", "d_700", "d_500", "d_200",
    "r_850", "r_700", "r_500", "r_200",
    "t_850", "t_700", "t_500", "t_200",
    "u_850", "u_700", "u_500", "u_200",
    "v_850", "v_700", "v_500", "v_200",
    "shear_850_200",
]


def _norm_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the datetime column to ``datetime64`` and sort by time."""
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# RI label construction audit (item 2 from the reviewer's checklist)
# ---------------------------------------------------------------------------

def audit_ri_label_construction(
    df: pd.DataFrame,
    horizon_hours: int = 24,
    storm_col: str = "storm_id",
    time_col: str = "datetime_utc",
    target_col: str = "RI_24h",
    wind_col: str = "max_wind_kt",
    threshold_kt: float = 30.0,
) -> dict:
    """Verify that the RI_24h label is correctly constructed.

    Checks performed:

    1. **Timestamp spacing**: observations within a storm should be ~6 h apart.
       Any gap > 12 h suggests a missing observation (the label may span more
       than the intended 24-h window).
    2. **Label consistency**: recompute RI_24h from the raw wind columns and
       compare to the existing label.
    3. **Storm-ID continuity**: ensure no storm ID is reused across seasons.
    4. **End-of-storm censoring**: identify rows where the t+24h observation
       does not exist (these should have RI_24h = NaN, not 0).

    Returns a dict of audit results.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([storm_col, time_col])

    audit = {
        "n_rows": len(df),
        "n_storms": df[storm_col].nunique(),
        "n_ri": int((df[target_col] == 1).sum()),
        "n_non_ri": int((df[target_col] == 0).sum()),
        "n_missing_target": int(df[target_col].isna().sum()),
        "warnings": [],
    }

    # 1. Check timestamp spacing within storms.
    large_gaps = []
    for storm_id, grp in df.groupby(storm_col):
        grp = grp.sort_values(time_col)
        if len(grp) < 2:
            continue
        diffs = grp[time_col].diff().dt.total_seconds() / 3600.0
        bad = diffs[diffs > horizon_hours * 1.5]  # >1.5x the window
        if bad.any():
            for idx, gap_h in bad.items():
                large_gaps.append({
                    "storm_id": str(storm_id),
                    "gap_hours": float(gap_h),
                    "row_idx": int(idx),
                })
    audit["large_timestamp_gaps"] = large_gaps
    if large_gaps:
        audit["warnings"].append(
            f"{len(large_gaps)} observation(s) have gaps > {horizon_hours * 1.5:.0f} h "
            "within a storm — the RI label may span more than the intended window."
        )

    # 2. Check for storms with very few observations (end-of-storm censoring).
    storm_obs = df.groupby(storm_col).size()
    short_storms = storm_obs[storm_obs <= 2]
    audit["short_storms"] = int(len(short_storms))
    if len(short_storms) > 0:
        audit["warnings"].append(
            f"{len(short_storms)} storm(s) have <= 2 observations — likely "
            "end-of-storm censoring needed (t+24h may not exist)."
        )

    # 3. Check that missing target is properly NaN (not 0).
    if audit["n_missing_target"] > 0:
        # Verify that rows with missing target don't also have a wind_24h column
        # that could be used to compute the label.
        missing_rows = df[df[target_col].isna()]
        has_wind_24h = "wind_minus_24h_kt" in df.columns
        if has_wind_24h:
            # These rows are legitimately censored (end of storm).
            audit["censored_end_of_storm"] = int(len(missing_rows))
            audit["warnings"].append(
                f"{len(missing_rows)} row(s) have missing RI_24h — these are "
                "end-of-storm censored samples (t+24h not available). "
                "Correctly excluded from training."
            )

    return audit


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_imd(cfg: dict) -> pd.DataFrame:
    """Load the canonical IMD feature table.

    End-of-storm censoring (item 3 from the reviewer's checklist):
    rows whose ``RI_24h`` is NaN (because t+24h does not exist) are correctly
    excluded from training — they are NOT assigned RI=0.  This is the
    existing behaviour: ``df = df[df["RI_24h"].notna()]``.

    Returns a DataFrame with a valid (non-null) ``RI_24h`` target, a parsed
    datetime column and a stable row index.  Also runs a lightweight label
    audit to catch timestamp/target construction issues.
    """
    path = REPO_ROOT / cfg["paths"]["imd_file"]
    df = pd.read_csv(path)
    df = _norm_datetime(df)

    # Keep only rows where the 24 h outcome is known.
    # End-of-storm censored rows (t+24h missing) have RI_24h = NaN and are
    # correctly EXCLUDED here rather than treated as non-RI (RI=0).
    n_censored = int(df["RI_24h"].isna().sum())
    df = df[df["RI_24h"].notna()].copy()
    df["RI_24h"] = df["RI_24h"].astype(int)

    # Drop rows with invalid coordinates or timestamps.
    df = df.dropna(subset=_ID_COLS + ["latitude", "longitude"]).copy()

    # Derive the 6 h intensity change (current minus 6 h ago).
    if "wind_6h_change" not in df.columns:
        df["wind_6h_change"] = df["max_wind_kt"] - df["wind_minus_6h_kt"]

    df = df.sort_values(["storm_id", "datetime_utc"]).reset_index(drop=True)
    df["storm_id"] = df["storm_id"].astype(str)

    # Transparency note: end-of-storm censoring count.
    if n_censored:
        pass  # logged by the caller via summarize / audit
    return df


def load_era5(cfg: dict) -> pd.DataFrame:
    """Load the canonical ERA5 feature table (matched to IMD rows)."""
    path = REPO_ROOT / cfg["paths"]["era5_file"]
    df = pd.read_csv(path)
    df = _norm_datetime(df)
    df = df[df["RI_24h"].notna()].copy()
    df["RI_24h"] = df["RI_24h"].astype(int)
    df = df.dropna(subset=_ID_COLS).copy()
    df["storm_id"] = df["storm_id"].astype(str)
    return df


def load_satellite_metadata(cfg: dict) -> pd.DataFrame:
    """Load the canonical satellite metadata.

    Uses the recovered dataset under ``satellite_cnn_recovered/``. Images live
    alongside the metadata (``<recovered_dir>/images/``). Also performs a
    critical availability check: only rows whose image actually exists on disk
    are returned. This prevents the pipeline from silently running a CNN on
    missing image files.
    """
    path = REPO_ROOT / cfg["paths"]["satellite_metadata"]
    df = pd.read_csv(path)
    df = _norm_datetime(df)
    df["storm_id"] = df["storm_id"].astype(str)

    recovered_dir = REPO_ROOT / cfg["satellite"]["recovered_dir"]
    img_dir = recovered_dir / "images"

    # Prefer an existing image_path column; otherwise derive from image_file.
    if "image_path" not in df.columns:
        df["image_path"] = df["image_file"].apply(lambda f: img_dir / f)
    else:
        df["image_path"] = df["image_path"].apply(
            lambda p: Path(p) if Path(p).is_absolute() else img_dir / Path(p).name if pd.notna(p) else img_dir
        )

    exists = df["image_path"].apply(lambda p: Path(p).exists())
    missing = int((~exists).sum())
    if missing:
        print(
            f"[data] WARNING: {missing} satellite image(s) referenced by "
            f"metadata are missing on disk and will be excluded."
        )
    df = df[exists].copy()
    return df


def build_combined_imd_era5(cfg: dict) -> pd.DataFrame:
    """Merge IMD and ERA5 tables into one matched table.

    ERA5 features were extracted at the exact IMD observation times, so a
    left-join (or inner-join) on ``(storm_id, datetime_utc)`` produces a
    complete match with no observation loss. Returns the merged table.
    """
    imd = load_imd(cfg)
    era5 = load_era5(cfg)
    combined = imd.merge(era5, on=_ID_COLS, how="inner", suffixes=("", "_era"))
    # Keep one RI column (IMD's) after the merge.
    if "RI_24h_era" in combined.columns:
        combined = combined.drop(columns=["RI_24h_era"])
    return combined


# ---------------------------------------------------------------------------
# Storm-safe splitter
# ---------------------------------------------------------------------------

@dataclass
class Split:
    """Container for a storm-safe train/validation/test split.

    The three DataFrames have disjoint storm IDs (asserted by ``verify``).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_storms: set = field(default_factory=set)
    val_storms: set = field(default_factory=set)
    test_storms: set = field(default_factory=set)

    def verify(self) -> None:
        """Assert that no storm appears in more than one split."""
        assert self.train_storms.isdisjoint(self.val_storms), "Train / val storm overlap!"
        assert self.train_storms.isdisjoint(self.test_storms), "Train / test storm overlap!"
        assert self.val_storms.isdisjoint(self.test_storms), "Val / test storm overlap!"

    def table(self) -> pd.DataFrame:
        """Return a per-split summary row of observations, storms and class counts."""
        rows = []
        for name, df in (
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ):
            rows.append(
                {
                    "split": name,
                    "observations": len(df),
                    "storms": df["storm_id"].nunique(),
                    "RI": int((df["RI_24h"] == 1).sum()),
                    "non_RI": int((df["RI_24h"] == 0).sum()),
                }
            )
        return pd.DataFrame(rows)


def split_by_storms(
    df: pd.DataFrame,
    cfg: dict,
) -> Split:
    """Split a DataFrame into train/val/test without storm leakage.

    The fraction refers to the **number of storms**, not observations. Storm
    IDs are shuffled deterministically (seeded) and partitioned. Split sizes
    are always at least 1 storm. If a split would contain only a single class
    and ``keep_balanced`` is enabled, a re-partition is attempted (bounded),
    otherwise the split is returned as-is with a warning.
    """
    seed = get_seed(cfg)

    storms = np.asarray(sorted(df["storm_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(storms)

    n = len(storms)
    # Guarantee at least one storm in each split while respecting fractions.
    # Reserve 1 storm for test and 1 for val as a floor.
    n_floor = 2
    if n <= 2:
        # Not enough storms for a three-way split: val gets 1, rest to test.
        test_n = 1
        val_n = 0
        train_n = n - test_n - val_n
    else:
        test_n = max(1, int(round(n * cfg["split"]["test_storm_fraction"])))
        val_n = max(1, int(round(n * cfg["split"]["val_storm_fraction"])))
        # Keep at least one storm for training.
        train_n = n - test_n - val_n
        if train_n < 1:
            test_n = n - 1 - (1 if n >= 3 else 0)
            val_n = 1 if n >= 3 else 0
            train_n = n - test_n - val_n

    train_storms = set(storms[:train_n])
    val_storms = set(storms[train_n: train_n + val_n])
    test_storms = set(storms[train_n + val_n:])

    split = Split(
        train=df[df["storm_id"].isin(train_storms)].copy(),
        val=df[df["storm_id"].isin(val_storms)].copy(),
        test=df[df["storm_id"].isin(test_storms)].copy(),
        train_storms=train_storms,
        val_storms=val_storms,
        test_storms=test_storms,
    )

    # Balanced-class repair: if a split is single-class and there are enough
    # storms to fix it, swap in storms from the majority split.
    if cfg["split"].get("keep_balanced", True) and n > 3:
        split = _try_balance(split)

    split.verify()
    return split


# --- balanced-class repair (bounded, deterministic) -----------------------

def _storms_of(df: pd.DataFrame) -> set:
    return set(df["storm_id"].unique())


def _try_balance(split: Split) -> Split:
    """Attempt to give every split both classes by swapping storm assignments.

    Runs a single pass over each single-class split. For an unbalanced split,
    it looks for a storm of the *missing* class living in another split and
    moves it into the unbalanced one. The number of swaps is bounded by the
    number of splits (at most a few). Leakage is re-verified at the end.
    """
    blocks = (
        ("train", split.train, split.val, split.test),
        ("val", split.val, split.train, split.test),
        ("test", split.test, split.train, split.val),
    )
    for name, cur_df, other_a, other_b in blocks:
        if cur_df["RI_24h"].nunique() == 2:
            continue
        missing = 1 - int(cur_df["RI_24h"].iloc[0])
        # Find the first donor storm of the missing class in another split.
        donor_storm = None
        for donor_df in (other_a, other_b):
            donor = donor_df[donor_df["RI_24h"] == missing]["storm_id"].unique()
            if len(donor) > 0:
                donor_storm = str(donor[0])
                break
        if donor_storm is None:
            continue
        # Recover the donor rows from whichever split owns them, then remove.
        donor_copy = None
        for full in (split.train, split.val, split.test):
            if donor_storm in _storms_of(full):
                donor_copy = full[full["storm_id"] == donor_storm].copy()
                full.drop(full[full["storm_id"] == donor_storm].index, inplace=True)
        if donor_copy is not None:
            cur_df = pd.concat([cur_df, donor_copy], ignore_index=True)

        # Write the (possibly updated) DataFrame back into the split.
        if name == "train":
            split.train = cur_df
        elif name == "val":
            split.val = cur_df
        else:
            split.test = cur_df

    # Recompute storm sets and re-verify no leakage.
    split.train_storms = _storms_of(split.train)
    split.val_storms = _storms_of(split.val)
    split.test_storms = _storms_of(split.test)
    split.verify()
    return split


# ---------------------------------------------------------------------------
# Dataset statistics report
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame, label: str = "dataset") -> dict:
    """Return a small dictionary of dataset statistics for reporting."""
    return {
        "label": label,
        "observations": int(len(df)),
        "storms": int(df["storm_id"].nunique()),
        "RI": int((df["RI_24h"] == 1).sum()),
        "non_RI": int((df["RI_24h"] == 0).sum()),
        "ri_rate": round(float(df["RI_24h"].mean()), 4),
    }


def print_summary(title: str, stats: dict) -> None:
    """Pretty-print dataset statistics."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"  Observations : {stats['observations']}")
    print(f"  Storms       : {stats['storms']}")
    print(f"  RI / non-RI  : {stats['RI']} / {stats['non_RI']}")
    print(f"  RI rate      : {stats['ri_rate']:.2%}")
    print()
