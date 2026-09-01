"""
train.py
========
End-to-end training driver for the hybrid IR-CNN + tabular RI classifier.

Usage:
    python train.py --nc_folder /mnt/user-data/uploads --epochs 30

Notes on data volume
---------------------
This script will happily train on however many labeled IR patches are found
in `nc_folder`. With only the ~11 demo MERG-IR files provided, this is a
PIPELINE CORRECTNESS DEMO, not a real trained model -- deep learning on IR
imagery for RI needs at least several hundred to a few thousand positive
RI cases (i.e. a multi-year MERG-IR archive) to generalize. See
`data/download_mergir.py` for how to build that full archive.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from build_ri_dataset import build  # noqa: E402
from extract_ir_patches import build_labeled_patch_dataset  # noqa: E402
from cnn_model import RICNNFusion, FocalLoss  # noqa: E402

TABULAR_FEATURES = [
    "latitude", "longitude", "max_wind_kt", "central_pressure_hpa",
    "pressure_drop_hpa", "wind_minus_6h_kt", "delta_v_minus_6h_kt",
    "wind_minus_12h_kt", "delta_v_minus_12h_kt",
    "wind_minus_24h_kt", "delta_v_minus_24h_kt",
]


class RIDataset(Dataset):
    def __init__(self, X, mask, tab, y):
        self.X = X
        self.mask = mask
        self.tab = tab
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tb = (self.X[idx] - 240.0) / 30.0  # normalize Tb roughly to [-2, 3]
        ir = np.stack([tb, self.mask[idx]], axis=0).astype(np.float32)
        return (
            torch.from_numpy(ir),
            torch.from_numpy(self.tab[idx].astype(np.float32)),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def ri_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pod = tp / (tp + fn) if (tp + fn) else np.nan       # Probability of Detection
    far = fp / (tp + fp) if (tp + fp) else np.nan       # False Alarm Ratio
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan  # Critical Success Index
    return {"POD": pod, "FAR": far, "CSI": csi, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main(nc_folder: str, epochs: int, batch_size: int, lr: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Rebuild IMD RI tabular dataset (BoB)")
    ri_bob = build(basin_filter="BOB")

    print("STEP 2: Match to available MERG-IR patches")
    data = build_labeled_patch_dataset(ri_bob, nc_folder)
    n = len(data["y"])
    print(f"  -> matched patches: {n}")
    if n < 20:
        print("  WARNING: very few matched samples -- this run is a pipeline "
              "correctness demo only, not a statistically meaningful trained model.")

    meta = ri_bob.merge(
        data["meta"][["storm_id", "datetime_utc"]],
        on=["storm_id", "datetime_utc"], how="inner",
    )
    tab_raw = meta[TABULAR_FEATURES].fillna(meta[TABULAR_FEATURES].median()).values
    scaler = StandardScaler()
    tab = scaler.fit_transform(tab_raw)

    groups = meta["storm_id"].values
    y = data["y"]

    if len(np.unique(groups)) >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(splitter.split(tab, y, groups))
    else:
        # too few storms to hold out a group -> fall back to random split for the demo
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        split = max(1, int(n * 0.7))
        train_idx, test_idx = idx[:split], idx[split:]

    def subset(idx):
        return RIDataset(data["X"][idx], data["mask"][idx], tab[idx], y[idx])

    train_ds, test_ds = subset(train_idx), subset(test_idx)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=max(1, len(test_ds)), shuffle=False)

    print(f"STEP 3: Train ({len(train_ds)} samples) / Test ({len(test_ds)} samples) split")

    model = RICNNFusion(tabular_dim=tab.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    print("STEP 4: Training")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for ir, tb_batch, yb in train_loader:
            optimizer.zero_grad()
            logits = model(ir, tb_batch)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        epoch_loss /= max(1, len(train_ds))
        history.append(epoch_loss)
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs} - focal loss: {epoch_loss:.4f}")

    print("STEP 5: Evaluation")
    model.eval()
    with torch.no_grad():
        all_probs, all_y = [], []
        for ir, tb_batch, yb in test_loader:
            logits = model(ir, tb_batch)
            probs = torch.sigmoid(logits).numpy()
            all_probs.append(probs)
            all_y.append(yb.numpy())
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_true = np.concatenate(all_y) if all_y else np.array([])

    results = {"n_train": len(train_ds), "n_test": len(test_ds)}
    if len(np.unique(y_true)) > 1:
        results["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        results["PR_AUC"] = average_precision_score(y_true, y_prob)
        results.update(ri_metrics(y_true, y_prob, threshold=0.5))
    else:
        print("  (test split has only one class present -- metrics need more data)")

    print("\nRESULTS:", results)

    torch.save(model.state_dict(), os.path.join(out_dir, "ri_cnn_fusion_demo.pt"))
    pd.Series(history, name="focal_loss").to_csv(
        os.path.join(out_dir, "training_history.csv"), index_label="epoch"
    )
    pd.DataFrame([results]).to_csv(os.path.join(out_dir, "eval_results.csv"), index=False)
    print(f"\nSaved model + logs to {out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_folder", default="/mnt/user-data/uploads")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="/home/claude/tc_ri_cnn/outputs")
    args = parser.parse_args()
    main(args.nc_folder, args.epochs, args.batch_size, args.lr, args.out_dir)
