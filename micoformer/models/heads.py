from __future__ import annotations

import torch
import torch.nn as nn


class AbundanceBinHead(nn.Module):

    def __init__(self, d_model: int, num_bins: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_bins)

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        return self.proj(token_repr)


class ClassificationHead(nn.Module):
    """下游分类 head。hidden_dim=0 时为 linear probe，>0 时为两层 MLP。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
