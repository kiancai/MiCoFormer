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


class PriorCoordHead(nn.Module):
    """X2 范式预测 head:从 token 表征解码到外部 prior 坐标(2026-05-28 夜)。

    通用于 phylo / protein / 未来其他 vocab-level prior:
      - input  : encoder 输出 token 序列 [B, L, d_model]
      - output : 每个 token 位置在 prior 空间的预测坐标 [B, L, pe_dim]

    设计:
      - 2 层 MLP(Linear → GELU → Linear),hidden_dim 默认 128
      - **末层 zero-init**(weight=0, bias=0):训练 step 0 预测全 0
          → 配合 MSE loss,起步梯度方向来自 target 自身,主干自然吸收 prior 信号
          → 跟 phylo_pe 末层 zero-init 对称,防 self-distillation 风格 collapse
      - 不接 LayerNorm:输出空间是 prior 坐标(已 normalized),LN 会破坏自然量级
    """

    def __init__(self, d_model: int, pe_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.pe_dim = pe_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, pe_dim),
        )
        # 末层 zero-init(对称 phylo_pe.proj 末层 zero-init)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        # [B, L, d_model] → [B, L, pe_dim]
        return self.mlp(token_repr)


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


class SampleViewLinearHead(nn.Module):
    """sample-level view 回归 head。

    输入是某个 view 专用 PMA seed 的 sample representation [B, d_model]，
    输出该 view 的完整 sample target 向量。保持线性，避免 head 自己吸收非线性可读性。
    """

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, sample_repr: torch.Tensor) -> torch.Tensor:
        return self.proj(sample_repr)


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
