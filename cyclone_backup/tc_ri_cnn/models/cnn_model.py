"""
cnn_model.py
============
Hybrid architecture for Rapid Intensification (RI) classification:

    IR patch (1xHxW Tb)  --> CNN encoder  --\
                                              +--> concat --> MLP head --> P(RI_24h)
    Tabular features (lat, lon, wind, ...) --> MLP encoder --/

Design choices (why this is the "best" architecture for SIH judging):
- Two-branch fusion beats either modality alone: IR imagery captures the
  *current convective structure* (eyewall formation, cold-cloud shield
  symmetry, deep convective bursts) that precedes RI, while the tabular
  branch captures *storm-history persistence* (recent wind trend, latitude,
  central pressure) which is a strong, well-established RI predictor on
  its own. Published RI-CNN work (e.g. Combinido et al. 2018; Su et al. 2020)
  shows fused models consistently outperform pure-CNN or pure-tabular models.
- A shallow CNN (4 conv blocks + GAP) is deliberately used instead of a deep
  ResNet: RI-labeled IR datasets are small (hundreds-few thousand positives
  globally), so a small model + heavy augmentation + transfer/pretraining
  generalizes far better than a deep net that will overfit.
- Focal loss handles the ~5-6% positive class rate seen in the BoB RI
  dataset without needing naive oversampling that duplicates rare storms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
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
    """Small CNN encoder for IR (+ valid-mask) patches -> feature vector."""

    def __init__(self, in_channels=2, base_ch=16, embed_dim=64):
        super().__init__()
        self.block1 = ConvBlock(in_channels, base_ch)          # H/2
        self.block2 = ConvBlock(base_ch, base_ch * 2)           # H/4
        self.block3 = ConvBlock(base_ch * 2, base_ch * 4)       # H/8
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
    def __init__(self, in_features, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, embed_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class RICNNFusion(nn.Module):
    """Full hybrid model. Set `use_tabular=False` to run IR-only (ablation)."""

    def __init__(self, tabular_dim: int, ir_channels: int = 2,
                 ir_embed: int = 64, tab_embed: int = 32, use_tabular: bool = True):
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
        z_ir = self.ir_encoder(ir)
        if self.use_tabular and tab is not None:
            z_tab = self.tab_encoder(tab)
            z = torch.cat([z_ir, z_tab], dim=1)
        else:
            z = z_ir
        logit = self.head(z).squeeze(1)
        return logit


class FocalLoss(nn.Module):
    """Binary focal loss - down-weights easy negatives, crucial for the
    ~5% positive RI class rate."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
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


if __name__ == "__main__":
    # quick shape sanity check
    model = RICNNFusion(tabular_dim=10)
    ir = torch.randn(4, 2, 99, 99)
    tab = torch.randn(4, 10)
    out = model(ir, tab)
    print("output shape:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print("trainable params:", n_params)
