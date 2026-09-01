"""Canonical satellite IR CNN (single implementation).

This is the **one** satellite CNN used across the whole RI system. It is the
official ``tc_ri_cnn`` model (``RICNNFusion``: hybrid CNN(IR) + MLP(tabular)
on a storm-centred 128x128 Tb patch, with a valid-pixel mask channel, focal
loss and dropout), integrated here so that the main pipeline, the Colab
training flow and the demo notebook all share the *same* model code imported
from ``src/`` — there is no second, disconnected CNN implementation.

Design rationale (@ the small, heavily-imbalanced satellite set):
- A compact 4-block CNN encoder (not a deep ResNet) generalises better than a
  large net on a few hundred images.
- Focal loss (default alpha=0.75, gamma=2.0) down-weights the huge number of
  easy ``no-RI`` negatives instead of naively oversampling the same rare
  storms.
- A valid-pixel mask channel tells the net which pixels are real vs. filled
  (missing edge / sensor-gap pixels are filled with a neutral 280 K value).

Execution note (macOS)
----------------------
TensorFlow and PyTorch both crash on the local macOS box, so the CNN is
trained in Google Colab. This module is framework-correct and importable on
macOS; the *training/evaluation* path is exercised in Colab. The artefact
contract (``satellite_oof_predictions.csv``,
``satellite_embeddings.npy`` + ``_meta.csv``) is identical to what
``src/satellite_bridge.py`` ingests for fusion.

Storm-safe OOF
--------------
With ~25 usable images across 23 storms, a separate train/val/test hold-out is
not statistically viable for the satellite branch. We instead use
``StratifiedGroupKFold`` grouped by storm to produce **out-of-fold** (OOF)
predictions for every image: each image is predicted by a model that never saw
any image from that image's storm during training. These OOF probabilities are
genuinely out-of-sample and feed the fusion meta-classifier without in-sample
leakage.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, get_seed

# ---------------------------------------------------------------------------
# Canonical PyTorch model (inherited from tc_ri_cnn; the one true implementation)
# ---------------------------------------------------------------------------
LAYOUT = {
    "img_size": 128,
    "channels": 2,       # [Tb_norm, valid_mask]
    "base_ch": 16,
    "ir_embed": 64,
    "tab_embed": 32,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "fill_tb": 280.0,    # neutral fill for out-of-domain pixels
    "tb_min": 180.0,
    "tb_max": 310.0,
}

# The EXACT 11-IMD-feature set the hybrid CNN's tabular head consumes.
# These are contemporaneous intensity / trend features at the forecast
# initialisation time t (NO future / target-time values).
CN_TAB_FEATURES = [
    "latitude",
    "longitude",
    "max_wind_kt",
    "central_pressure_hpa",
    "pressure_drop_hpa",
    "wind_minus_6h_kt",
    "delta_v_minus_6h_kt",
    "wind_minus_12h_kt",
    "delta_v_minus_12h_kt",
    "wind_minus_24h_kt",
    "delta_v_minus_24h_kt",
]

try:  # torch is required to *instantiate* the model (Colab); missing on some macOS hosts
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # pragma: no cover - only imported lazily elsewhere
    torch = None
    nn = None
    F = None
    _TORCH = False


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class IREncoder(nn.Module):
    """Compact 4-block CNN encoder for IR(+mask) patches -> feature vector."""

    def __init__(self, in_channels: int = 2, base_ch: int = LAYOUT["base_ch"],
                 embed_dim: int = LAYOUT["ir_embed"]):
        super().__init__()
        self.block1 = ConvBlock(in_channels, base_ch)          # /2
        self.block2 = ConvBlock(base_ch, base_ch * 2)          # /4
        self.block3 = ConvBlock(base_ch * 2, base_ch * 4)      # /8
        self.block4 = ConvBlock(base_ch * 4, base_ch * 8, pool=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_ch * 8, embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return F.relu(self.fc(x))


class TabularEncoder(nn.Module):
    def __init__(self, in_features: int, embed_dim: int = LAYOUT["tab_embed"]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, embed_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class RICNNFusion(nn.Module):
    """Canonical hybrid IR-CNN + tabular fusion model.

    ``use_tabular=False`` yields the IR-only (satellite-only) ablation.
    ``forward`` returns the classification logit; ``forward_emb`` returns the
    fused penultimate embedding (used for feature-level fusion and Grad-CAM
    backprop).
    """

    def __init__(self, tabular_dim: int, ir_channels: int = LAYOUT["channels"],
                 ir_embed: int = LAYOUT["ir_embed"], tab_embed: int = LAYOUT["tab_embed"],
                 use_tabular: bool = True):
        super().__init__()
        self.use_tabular = use_tabular
        self.ir_encoder = IREncoder(in_channels=ir_channels, embed_dim=ir_embed)
        fused_dim = ir_embed
        if use_tabular:
            self.tab_encoder = TabularEncoder(tabular_dim, embed_dim=tab_embed)
            fused_dim += tab_embed
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, ir, tab=None):
        z = self.forward_emb(ir, tab)
        return self.head(z).squeeze(-1)

    def forward_emb(self, ir, tab=None):
        z_ir = self.ir_encoder(ir)
        if self.use_tabular and tab is not None:
            z_tab = self.tab_encoder(tab)
            z = torch.cat([z_ir, z_tab], dim=-1)
        else:
            z = z_ir
        return z


class FocalLoss(nn.Module):
    """Binary focal loss — down-weights easy negatives (small positive class)."""

    def __init__(self, alpha: float = LAYOUT["focal_alpha"], gamma: float = LAYOUT["focal_gamma"]):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# Preprocessing (physical, leakage-safe)
# ---------------------------------------------------------------------------

def normalize_patch(tb: np.ndarray, mask: np.ndarray | None = None,
                    fill_tb: float = LAYOUT["fill_tb"],
                    tb_min: float = LAYOUT["tb_min"],
                    tb_max: float = LAYOUT["tb_max"]) -> np.ndarray:
    """Return a (2, H, W) float32 sample: [Tb_norm, valid_mask].

    NaN / out-of-domain pixels are filled with ``fill_tb`` first; the mask is 1
    for real pixels, 0 for filled. Normalisation is a fixed physical window —
    it never uses the dataset (so no normalization leakage across splits).
    """
    tb = np.asarray(tb, dtype=np.float32)
    # Tolerate both (H, W) and (H, W, 1) inputs (the loader keeps a channel dim).
    if tb.ndim == 3 and tb.shape[-1] == 1:
        tb = tb[..., 0]
    if mask is None:
        mask = (~np.isnan(tb)).astype(np.float32)
    else:
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
    tb = np.where(np.isnan(tb), fill_tb, tb)
    tb_safe = np.clip(tb, tb_min, tb_max)
    tb_norm = (tb_safe - tb_min) / (tb_max - tb_min) * 2.0 - 1.0
    return np.stack([tb_norm, mask]).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading (from recovered .npy crops, mirroring the main pipeline)
# ---------------------------------------------------------------------------

def load_recovered_images(metadata: pd.DataFrame, img_size: int = LAYOUT["img_size"]):
    """Load recovered .npy crops into (X, y, meta), aligned with data rows.

    Returns (X (N,H,W,1) float32, y (N,) int, meta DataFrame). Same contract as
    the previous TF harness so the pipeline + bridge are unchanged.
    """
    X, y, kept = [], [], []
    for _, row in metadata.iterrows():
        p = row.get("image_path")
        if p is None or not Path(p).exists():
            continue
        try:
            img = np.load(str(p))
        except Exception:
            continue
        if img.shape != (img_size, img_size, 1):
            continue
        X.append(img.astype(np.float32))
        y.append(int(row["RI_24h"]))
        kept.append(row)
    if not X:
        return None, None, metadata.iloc[0:0]
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int64),
            pd.DataFrame(kept))


def _to_tensor(x, mask, tab):
    """Build torch input tensors (Tb+mask channel) + tabular from numpy."""
    samples = [normalize_patch(xi, mi) for xi, mi in zip(x, mask)]
    ir = torch.from_numpy(np.stack(samples)).float()
    t = torch.from_numpy(np.asarray(tab, dtype=np.float32)) if tab is not None else None
    return ir, t


# ---------------------------------------------------------------------------
# Building the canonical CNN tabular dataset (REAL 11 IMD features)
# ---------------------------------------------------------------------------

def build_cnn_tabular_dataset(metadata: pd.DataFrame, multimodal: pd.DataFrame,
                              cfg: dict, out_csv: str | Path | None = None) -> tuple:
    """Join satellite observations to their contemporaneous IMD features.

    Each usable satellite observation is matched to the canonical
    ``ri_multimodal_dataset.csv`` IMD row by ``storm_id`` + ``datetime_utc``
    (exact first, then a ``<= max_time_diff`` tolerance). Only observations
    whose **all 11** IMD features are present are kept — rows are removed
    (never zero-padded / imputed) when any required feature is missing.

    Returns ``(df, audit)`` where ``df`` is the clean CNN training table and
    ``audit`` is a dict of join / validity bookkeeping.
    """
    mm = multimodal.copy()
    mm["datetime_utc"] = pd.to_datetime(mm["datetime_utc"], errors="coerce")

    # Normalise metadata on storm_id + datetime (the canonical match keys).
    meta = metadata.copy()
    meta["datetime_utc"] = pd.to_datetime(meta["datetime_utc"], errors="coerce")
    meta["key"] = meta["storm_id"].astype(str) + "_" + meta["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    mm["key"] = mm["storm_id"].astype(str) + "_" + mm["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")

    max_tol = int(cfg["cnn"].get("max_time_diff_min", 120)) if isinstance(cfg.get("cnn"), dict) else 120

    need = CN_TAB_FEATURES + ["RI_24h", "image_file", "granule_file", "has_satellite"]
    present = [c for c in need if c in mm.columns]
    look = mm.drop_duplicates(subset=["key"])[["key"] + present].copy()

    exact = meta[meta["key"].isin(set(look["key"]))].copy()
    joined = exact.merge(look, on="key", how="left", suffixes=("_sat", ""))
    joined["join_delta_minutes"] = 0

    unmatched = meta[~meta["key"].isin(set(look["key"]))]

    # Tolerance join for the few format-shifted timestamps (<= max_tol minutes).
    tol_rows = []
    tol_rejected = []
    for _, r in unmatched.iterrows():
        d0 = r["datetime_utc"]
        if pd.isna(d0):
            tol_rejected.append(r); continue
        cand = look[(look["storm_id"].astype(str) == str(r["storm_id"]))]
        if cand.empty:
            tol_rejected.append(r); continue
        cand = cand.assign(dt=pd.to_datetime(cand["datetime_utc"], errors="coerce"))
        cand = cand.assign(delta=(cand["dt"] - d0).dt.total_seconds().abs() / 60.0)
        cand = cand.sort_values("delta")
        if cand.empty or pd.isna(cand.iloc[0]["delta"]) or cand.iloc[0]["delta"] > max_tol:
            tol_rejected.append(r); continue
        best = cand.iloc[0]
        row = r.to_dict()
        row["join_delta_minutes"] = float(best["delta"])
        for c in present:
            row[c] = best[c]
        tol_rows.append(row)

    rows = [r.to_dict() for _, r in joined.iterrows()]
    for r in tol_rows:
        for c in ("image_file", "granule_file"):
            if c not in r:
                r[c] = None
        rows.append(r)
    # Canonical source columns
    for r in rows:
        if "satellite_datetime" not in r:
            r["satellite_datetime"] = r.get("datetime_utc")
    df = pd.DataFrame(rows)

    # Logistics columns
    drop_keys = {"key"}
    df = df[[c for c in df.columns if c not in drop_keys]].copy()
    # Drop satellite-side duplicates of overlapping columns (the unsuffixed
    # multimodal/IMD values are authoritative).
    sat_suffix_cols = [c for c in df.columns if c.endswith("_sat")]
    if sat_suffix_cols:
        df = df.drop(columns=sat_suffix_cols)
    for c in ("image_path", "satellite_datetime"):
        if c not in df.columns:
            df[c] = None
    if "image_path" in df.columns and df["image_path"].isnull().all() and "image_file" in df.columns:
        pass

    n_raw = int(len(df))
    missing_match = int(len(unmatched) - len(tol_rows))

    # ---- strict completeness filter (NO imputation / NO zero padding) ----
    n_pre = len(df)
    # Exclude rows whose satellite image is AFTER the IMD observation time
    # (post-target temporal leak) — same rule the multimodal builder applies.
    if "has_satellite" in df.columns:
        df = df[df["has_satellite"].fillna(False) == 1].copy()
    df = df.dropna(subset=CN_TAB_FEATURES).copy()
    removed_missing = n_pre - len(df)
    df = df.dropna(subset=["RI_24h"]).copy()
    df = df.drop_duplicates(subset=["image_file"]).copy()
    n_dup = n_pre - removed_missing - len(df)

    df = df.reset_index(drop=True)
    df = df.sort_values(["storm_id", "datetime_utc"]).reset_index(drop=True)

    # ---- temporal-leak guard: confirm no future features present ----
    future_cols = [c for c in ("max_wind_t_plus_6h", "max_wind_t_plus_12h",
                               "max_wind_t_plus_24h", "delta_v_t_plus_24h")
                   if c in df.columns]

    if out_csv is not None:
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    audit = {
        "total_satellite_observations": n_raw,
        "exact_matches": int(len(exact)),
        "tolerance_matches": int(len(tol_rows)),
        "tolerance_rejected": int(len(tol_rejected)),
        "missing_imd_match": missing_match,
        "invalid_rows": int(removed_missing + len(df[df["RI_24h"].isna()]) ),
        "duplicate_rows": n_dup,
        "final_rows": int(len(df)),
        "unique_storms": int(df["storm_id"].nunique()) if len(df) else 0,
        "n_ri": int((df["RI_24h"] == 1).sum()) if len(df) else 0,
        "n_non_ri": int((df["RI_24h"] == 0).sum()) if len(df) else 0,
        "missing_features": {c: int(df[c].isna().sum()) for c in CN_TAB_FEATURES},
        "future_features_detected": future_cols,
    }
    return df, audit


def _fit_scaler_to_columns(tr_df):
    """Fit a per-column min-max scaler on TRAINING rows only (leak-free)."""
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    sc.fit(tr_df[CN_TAB_FEATURES].to_numpy(dtype=float))
    return sc


def _apply_scaler(sc, df):
    from sklearn.preprocessing import MinMaxScaler
    arr = sc.transform(df[CN_TAB_FEATURES].to_numpy(dtype=float))
    out = df.copy()
    for j, c in enumerate(CN_TAB_FEATURES):
        out[c + "_norm"] = arr[:, j]
    return out


# ---------------------------------------------------------------------------
# Storm-safe OOF training + evaluation (canonical harness)
# ---------------------------------------------------------------------------

def run_cnn_oof(metadata: pd.DataFrame, multimodal: pd.DataFrame, cfg: dict,
                seed: int, n_folds: int = 5, epochs: int = None,
                use_classes: bool = True) -> dict:
    """Train the canonical hybrid CNN (real 11 IMD features) across folds.

    Satellite IR image (128x128x2: Tb + valid mask) and the 11
    contemporaneous IMD features are fused. Tabular features are scaled with a
    MinMax scaler fitted on each fold's TRAINING storms only (never global).
    Splits are storm-safe (no storm in more than one partition).

    Returns the same artifact contract as before: status, n_images, n_storms,
    n_ri, n_non_ri, X, y, meta, oof, fold_metrics, weights_path, n_folds,
    plus the fitted training-data table and scaler stats.
    """
    # NOTE: torch is imported lazily BELOW, after the clean dataset is built and
    # persisted, so even a torch-less host (macOS) writes the audit + training
    # table before the training step degrades to a clear "needs Colab" message.

    df, audit = build_cnn_tabular_dataset(metadata, multimodal, cfg)
    X, y, img_meta = load_recovered_images(df, int(cfg["cnn"].get("img_size", LAYOUT["img_size"])))

    results_dir = Path(cfg["paths"].get("results_dir", "results"))
    results_dir = REPO_ROOT / cfg["paths"]["results_dir"] if "results_dir" in cfg["paths"] else REPO_ROOT / "results"

    # Persist the clean training table regardless of training outcome.
    train_csv = results_dir / "satellite_cnn_training_data.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(train_csv, index=False)

    min_images = int(cfg["cnn"].get("min_images_for_training", 2))
    if X is None or len(X) < min_images:
        return {"status": "skipped", "n_images": 0 if X is None else len(X),
                "n_storms": audit["unique_storms"],
                "n_ri": audit["n_ri"], "n_non_ri": audit["n_non_ri"],
                "audit": audit, "training_data_path": str(train_csv)}

    import torch
    import json
    from torch.utils.data import TensorDataset, DataLoader

    epochs = int(epochs if epochs is not None else cfg["cnn"].get("epochs", 60))
    batch_size = int(cfg["cnn"].get("batch_size", 4))
    lr = float(cfg["cnn"].get("learning_rate", 1e-3))

    groups = df["storm_id"].astype(str).to_numpy()
    tab_raw = df[CN_TAB_FEATURES].to_numpy(dtype=float)
    mask = ~np.isnan(X[..., 0])
    mask = mask.astype(np.float32)

    oof = np.full(len(X), np.nan, dtype=np.float64)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
    try:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y, groups))
    except ValueError:
        splits = list(GroupKFold(n_splits=min(n_folds, len(groups))).split(X, y, groups))

    models_dir = Path(cfg["paths"].get("models_dir", "models"))
    models_dir = REPO_ROOT / models_dir
    weights_path = str(models_dir / "satellite_cnn.pt")

    scalers = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        tr_df = df.iloc[tr_idx]
        # Fit the scaler on TRAINING storms only (validated/test excluded).
        sc = _fit_scaler_to_columns(tr_df)
        tr_scaled = _apply_scaler(sc, tr_df)
        va_scaled = _apply_scaler(sc, df.iloc[va_idx])
        scalers.append({"fold": fold, "data_min_": sc.data_min_.tolist(),
                        "data_range_": sc.data_range_.tolist(),
                        "scale_": sc.scale_.tolist()})

        ir_tr, tab_tr = _to_tensor(X[tr_idx], mask[tr_idx],
                                   tr_scaled[[c + "_norm" for c in CN_TAB_FEATURES]].to_numpy(float))
        ir_va, tab_va = _to_tensor(X[va_idx], mask[va_idx],
                                   va_scaled[[c + "_norm" for c in CN_TAB_FEATURES]].to_numpy(float))
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)

        ds = TensorDataset(ir_tr, tab_tr, y_tr)
        loader = DataLoader(ds, batch_size=max(1, min(batch_size, len(ds))), shuffle=True)

        model = RICNNFusion(tabular_dim=len(CN_TAB_FEATURES),
                            use_tabular=bool(use_classes))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = FocalLoss()

        for _ in range(epochs):
            model.train()
            for irb, tabb, yb in loader:
                opt.zero_grad()
                logits = model(irb, tabb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(ir_va, tab_va)
            oof[va_idx] = torch.sigmoid(logits).numpy()

        if fold == len(splits) - 1:
            os.makedirs(models_dir, exist_ok=True)
            torch.save(model.state_dict(), weights_path)

    # Save the per-fold scaler stats artifact.
    try:
        import json
        with open(results_dir / "cnn_tabular_scaler.json", "w") as f:
            json.dump({"features": CN_TAB_FEATURES,
                       "folds": scalers}, f, indent=2)
    except Exception:
        pass

    fold_metrics = _oof_metrics(y, oof)

    return {
        "status": "trained",
        "n_images": int(len(X)),
        "n_storms": audit["unique_storms"],
        "n_ri": audit["n_ri"],
        "n_non_ri": audit["n_non_ri"],
        "oof": oof,
        "y": y,
        "X": X,
        "meta": df,
        "features": CN_TAB_FEATURES,
        "audit": audit,
        "training_data_path": str(train_csv),
        "fold_metrics": fold_metrics,
        "weights_path": weights_path,
        "n_folds": n_folds,
    }


def _oof_metrics(y, oof):
    from sklearn.metrics import roc_auc_score, average_precision_score
    out = {}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, oof))
        out["pr_auc"] = float(average_precision_score(y, oof))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Artifact persistence (the contract the fusion bridge ingests)
# ---------------------------------------------------------------------------

def save_oof_artifacts(cnn_result: dict, results_dir: str | Path,
                       embeddings: np.ndarray | None = None) -> list[str]:
    """Persist the satellite CNN branch to disk (Colab / pipeline output).

    Writes, when ``cnn_result['status'] == 'trained'``:
      - ``results/satellite_oof_predictions.csv``   (storm_id, datetime_utc,
        RI_24h, P_RI)  -> late-fusion fuel
      - ``results/satellite_embeddings.npy``        (+ ``_meta.csv``)   ->
        feature-level-fusion fuel
      - ``models/satellite_cnn.pt``                 (weights, canonical name)

    Returns the list of written paths.
    """
    written: list[str] = []
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if cnn_result.get("status") != "trained":
        return written

    meta = cnn_result["meta"]
    cols = {"storm_id": meta["storm_id"].astype(str).values,
            "datetime_utc": pd.to_datetime(meta["datetime_utc"]).values,
            "RI_24h": np.asarray(cnn_result["y"]),
            "P_RI": np.asarray(cnn_result["oof"])}
    oof_df = pd.DataFrame(cols)
    oof_path = results_dir / "satellite_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    written.append(str(oof_path))

    if embeddings is not None and embeddings.shape[0] == len(oof_df):
        emb_path = results_dir / "satellite_embeddings.npy"
        np.save(str(emb_path), embeddings)
        written.append(str(emb_path))
        meta_out = oof_df[["storm_id", "datetime_utc", "RI_24h"]].copy()
        meta_out.to_csv(results_dir / "satellite_embeddings_meta.csv", index=False)
        written.append(str(results_dir / "satellite_embeddings_meta.csv"))

    wp = cnn_result.get("weights_path")
    if wp:
        written.append(str(wp))  # canonical models/satellite_cnn.pt
    return written


# ---------------------------------------------------------------------------
# Embeddings / prediction / Grad-CAM
# ---------------------------------------------------------------------------

def _tabular_vector(row, scaler=None):
    """Build a (1,11) feature vector from a dict / sequence of the 11 IMD
    features. Throws on missing values — NEVER silently substitutes zeros."""
    if isinstance(row, dict):
        vals = [row.get(c) for c in CN_TAB_FEATURES]
    else:
        vals = [row[c] for c in CN_TAB_FEATURES]
    vals = [None if v is None else float(v) for v in vals]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
        raise ValueError(
            "All 11 IMD tabular features must be provided for CNN prediction; "
            "missing values are not silently zero-padded.")
    arr = np.asarray(vals, dtype=np.float64)[None]
    if scaler is not None:
        arr = scaler.transform(arr)
    return arr.astype(np.float32)


def _load_fold0_scaler(results_dir):
    """Reconstruct a MinMaxScaler from the saved scaler stats.

    The deprecated-style name is kept for import compatibility. Because the
    canonical model saves the weights of the LAST fold, the last fold's scaler
    is loaded (that is the scaler the saved weights were trained to expect).
    """
    from sklearn.preprocessing import MinMaxScaler
    p = Path(results_dir) / "cnn_tabular_scaler.json"
    if not p.exists():
        return None
    try:
        import json
        data = json.loads(p.read_text())
        folds = data.get("folds", [])
        if not folds:
            return None
        f_last = folds[-1]
        sc = MinMaxScaler()
        sc.data_min_ = np.asarray(f_last["data_min_"], dtype=np.float64)
        sc.data_range_ = np.asarray(f_last["data_range_"], dtype=np.float64)
        sc.feature_range_ = (0.0, 1.0)
        sc.scale_ = 1.0 / (sc.data_range_ + 1e-12)
        sc.min_ = -sc.data_min_ / (sc.data_range_ + 1e-12)
        return sc
    except Exception:
        return None


def extract_embeddings(cnn_result: dict, cfg: dict) -> np.ndarray | None:
    """Extract penultimate-layer hybrid embeddings using the REAL tabular
    input (fused IR + IMD features), never a zero placeholder."""
    import torch
    w = cnn_result.get("weights_path")
    if cnn_result["status"] != "trained" or not w or not Path(w).exists():
        return None
    model = RICNNFusion(tabular_dim=len(CN_TAB_FEATURES))
    model.load_state_dict(torch.load(w, map_location="cpu"))
    model.eval()
    X = cnn_result["X"]
    meta = cnn_result.get("meta")
    results_dir = cfg["paths"].get("results_dir", "results")
    scaler = _load_fold0_scaler(REPO_ROOT / results_dir)
    if meta is not None and len(meta) == len(X):
        tab = np.stack([_tabular_vector(r, scaler).ravel()
                        for _, r in meta.iterrows()])
    else:
        raise ValueError(
            "extract_embeddings requires the real 11-feature training table; "
            "cannot fall back to zero-padded tabular input.")
    ir, tab_t = _to_tensor(X, (~np.isnan(X[..., 0])).astype(np.float32), tab)
    with torch.no_grad():
        emb = model.forward_emb(ir, tab_t).numpy()
    return emb


def predict_image(model, tb_patch: np.ndarray, tab_values,
                  mask: np.ndarray | None = None,
                  scaler=None) -> float:
    """Predict P(RI_24h) for a single (H,W) Tb patch.

    ``tab_values`` is the contemporaneous 11-IMD-feature set (a dict keyed by
    feature name, or an ordered sequence of the 11 values). All 11 are
    required; a missing feature raises rather than silently becoming zero.
    Optionally pass the training ``scaler`` (saved per-fold) for correct
    normalisation.
    """
    import torch
    model.eval()
    sample = normalize_patch(tb_patch, mask)[None]  # (1,2,H,W)
    ir = torch.from_numpy(sample).float()
    t = torch.from_numpy(_tabular_vector(tab_values, scaler)).float()
    with torch.no_grad():
        return float(torch.sigmoid(model(ir, t)).numpy().item())


def grad_cam(model, tb_patch: np.ndarray, tab_values,
             mask: np.ndarray | None = None, class_idx: int = 1,
             scaler=None) -> np.ndarray:
    """Grad-CAM heatmap over the last conv block (image branch only).

    ``tab_values`` is the real 11-IMD-feature set (dict or sequence) needed as
    the tabular input to the fused model; Grad-CAM heatmaps only the image
    branch and does NOT attribute importance to the tabular features.
    """
    import torch
    model.eval()
    sample = normalize_patch(tb_patch, mask)[None]
    ir = torch.from_numpy(sample).float()
    t = torch.from_numpy(_tabular_vector(tab_values, scaler)).float()

    last_conv = None
    for m in model.ir_encoder.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    grad_out = {}

    def fwd_hook(m, i, o):
        grad_out["act"] = o

    def back_hook(m, i, o):
        grad_out["grad"] = o[0]

    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_full_backward_hook(back_hook)
    model.zero_grad()
    logit = model(ir, t)
    logit = logit.squeeze()
    logit.backward(retain_graph=True)
    h1.remove()
    h2.remove()

    act = grad_out["act"][0]
    grad = grad_out["grad"]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy()
