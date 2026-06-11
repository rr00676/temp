"""CNN architecture for Timepix3 cluster classification (Stage I).

Architecture:
  3–4 conv blocks (Conv-BN-ReLU-MaxPool)
  → Global average pool
  → Concat with scalar features (n_pixels, tot_sum, tot_max, eccentricity)
  → 2-layer MLP head
  → Softmax over (n_bins + 1) classes incl. background

Target: ~10⁴–10⁵ parameters, CPU-feasible.

Leakage rules:
- Input channels: log1p(ToT) and relative pixel time only.
- No absolute time, no TDC-derived columns, no chip (x, y).
"""

from __future__ import annotations

from typing import List

# lazy import — torch is optional
def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for Stage I (CNN). "
            "Install via: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


class ConvBlock:
    pass   # defined inside build_model() after torch import


def build_model(n_classes: int, n_scalar: int = 4, patch_size: int = 32):
    """Return an nn.Module for the CNN classifier."""
    torch, nn = _require_torch()

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        def forward(self, x):
            return self.net(x)

    class ClusterCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                _ConvBlock(2, 16),   # 32→16
                _ConvBlock(16, 32),  # 16→8
                _ConvBlock(32, 32),  # 8→4
                _ConvBlock(32, 64),  # 4→2
            )
            self.gap = nn.AdaptiveAvgPool2d(1)

            self.head = nn.Sequential(
                nn.Linear(64 + n_scalar, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, patches, scalars):
            """
            patches : (B, 2, P, P)
            scalars : (B, n_scalar)
            """
            x = self.conv(patches)       # (B, 64, 2, 2)
            x = self.gap(x).squeeze(-1).squeeze(-1)  # (B, 64)
            x = torch.cat([x, scalars], dim=1)       # (B, 64+n_scalar)
            return self.head(x)                       # (B, n_classes) logits

        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    model = ClusterCNN()
    return model


class TemperatureScaler:
    """Post-hoc temperature scaling for calibration.

    Fits a single temperature T on the validation set by minimising NLL,
    then divides logits by T before softmax.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: "torch.Tensor", labels: "torch.Tensor") -> float:
        torch, nn = _require_torch()

        T = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(logits / T, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(T.item())
        return self.temperature

    def scale(self, logits: "torch.Tensor") -> "torch.Tensor":
        torch, _ = _require_torch()
        return logits / max(self.temperature, 1e-3)
