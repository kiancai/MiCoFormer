from __future__ import annotations

import torch
import torch.nn as nn


class AbundanceBinHead(nn.Module):

    def __init__(self, d_model: int, num_bins: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_bins)

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        return self.proj(token_repr)


class AbundanceRegressionHead(nn.Module):
    """V5 连续 abundance 回归 head。

    forward 接受 encoder 输出的 token 序列 [B, L, d_model]，
    输出每个 token 位置的标量预测 [B, L]。
    上层在 mask 位置上与 labels_abund_values 计算 Huber loss。

    2 层 MLP（设计原则:小型信号转换模块至少 2 层 + 非线性）
    """

    def __init__(self, d_model: int, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        # [B, L, d_model] → [B, L, 1] → squeeze → [B, L]
        return self.mlp(token_repr).squeeze(-1)


class MetadataHead(nn.Module):
    """V5 Metadata 多任务 head(EnvCategory 单标签分类)。

    输入:PMA 输出的 sample-level 表征 [B, d_model]
    输出:logits [B, num_classes]
    上层用 class-weighted CrossEntropyLoss 计算 loss。
    """

    def __init__(self, d_model: int, num_classes: int = 6, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, sample_repr: torch.Tensor) -> torch.Tensor:
        return self.mlp(sample_repr)


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
