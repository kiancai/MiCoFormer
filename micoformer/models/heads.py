from __future__ import annotations

import torch
from torch import Tensor, nn


class AbundanceRegressionHead(nn.Module):
    """Predict transformed abundance for each token.
    逐 token 预测变换后的丰度。
    """

    def __init__(self, d_model: int, hidden: int = 64) -> None:
        super().__init__()
        # Map each token vector to one value: D -> hidden -> 1.
        # 将每个 token 向量映射为一个数值：D -> hidden -> 1。
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, L, 1] -> [B, L]; B is batch size, L is sequence length.
        # [B, L, D] -> [B, L, 1] -> [B, L]；B 为批大小，L 为序列长度。
        return self.mlp(token_repr).squeeze(-1)


class ConcatQueryHead(nn.Module):
    """Predict candidate presence and abundance from sample and prior vectors.
    根据样本向量与候选菌属先验，预测候选菌属的存在性与丰度。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Fuse concatenated vectors into shared features: 2D -> D.
        # 将拼接向量融合为两个输出共用的特征：2D -> D。
        self.trunk = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # Independent outputs: a raw presence logit and transformed abundance.
        # 两个独立输出：存在性原始 logit（未转成概率）与变换后的丰度。
        self.presence = nn.Linear(d_model, 1)
        self.continuous = nn.Linear(d_model, 1)

    def forward(self, sample: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
        # Concatenate two [N, D] inputs into [N, 2D]; N counts sample-candidate pairs.
        # 将两个 [N, D] 输入拼成 [N, 2D]；N 为样本与候选菌属的配对数量。
        features = self.trunk(torch.cat((sample, candidate), dim=-1))
        # Return presence logits first, then abundance predictions, both [N].
        # 依次返回存在性 logit、丰度预测值，均为 [N]。
        return (
            self.presence(features).squeeze(-1),
            self.continuous(features).squeeze(-1),
        )
