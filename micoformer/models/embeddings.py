from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from micoformer.config import ModelConfig


PAD_TOKEN_ID = 0

# Learnable genus identity with abundance encoding and a shared mask vector.
# 可训练的菌属身份、丰度编码与共享遮盖向量。
class LearnedTokenStem(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Genus table [vocab_size, D]: IDs [B, L] -> vectors [B, L, D].
        # 菌属向量表 [8116, D]：编号 [B, L] 查表得到向量 [B, L, D]。
        self.genus_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=PAD_TOKEN_ID,
        )
        # Abundance MLP: 1 -> H -> D (default 1 -> 64 -> 768), with GELU and LayerNorm.
        # 丰度编码：[B, L, 1] -> [B, L, 64] -> [B, L, D]，经非线性与归一化处理。
        self.abundance_mlp = nn.Sequential(
            nn.Linear(1, config.abundance_hidden_dim),
            nn.GELU(),
            nn.Linear(config.abundance_hidden_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        # Shared trainable abundance-mask vector [D]; normal initialization (mean=0, std=0.02).
        # 共享的可训练丰度遮盖向量 [D]；用均值 0、标准差 0.02 的正态分布初始化。
        self.abundance_mask_embedding = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.abundance_mask_embedding, mean=0.0, std=0.02)

    # Encode genus IDs and abundance [B, L] into token vectors [B, L, D].
    # 将菌属编号与丰度 [B, L] 编码为 token 向量 [B, L, D]。
    def forward(
        self,
        genus_ids: Tensor,
        abundance: Tensor,
        padding_mask: Tensor,
        abundance_mask: Tensor | None = None,
    ) -> Tensor:
        # Use an all-False mask for unmasked encoding.
        # 无遮盖编码时，使用全 False 的丰度遮盖标记。
        if abundance_mask is None:
            abundance_mask = torch.zeros_like(padding_mask)

        # Encode each scalar: [B, L] -> [B, L, 1] -> [B, L, D].
        # 将每个丰度数字编码为向量：[B, L] -> [B, L, 1] -> [B, L, D]。
        abundance_embedding = self.abundance_mlp(abundance.unsqueeze(-1))
        # Replace masked vectors using mask [B, L, 1] and a shared vector [D].
        # 用标记 [B, L, 1] 将遮盖位置的丰度向量直接替换为共享向量 [D]。
        abundance_embedding = torch.where(
            abundance_mask.unsqueeze(-1),
            self.abundance_mask_embedding,
            abundance_embedding,
        )
        # Add genus and abundance vectors [B, L, D]; retain genus identity when masked.
        # 菌属与丰度向量 [B, L, D] 相加，遮盖位置仍保留菌属身份。
        token_embedding = self.genus_embedding(genus_ids) + abundance_embedding
        # Zero all features at ignored positions; keep output shape [B, L, D].
        # 将忽略位置的全部特征归零，输出形状仍为 [B, L, D]。
        return token_embedding.masked_fill(padding_mask.unsqueeze(-1), 0.0)


# Model-supplied genus priors with independent abundance encoding and mask parameters.
# 使用模型传入的菌属先验，独立创建丰度编码与遮盖参数。
class FixedPriorTokenStem(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.abundance_mlp = nn.Sequential(
            nn.Linear(1, config.abundance_hidden_dim),
            nn.GELU(),
            nn.Linear(config.abundance_hidden_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.abundance_mask_embedding = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.abundance_mask_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        genus_ids: Tensor,
        abundance: Tensor,
        padding_mask: Tensor,
        abundance_mask: Tensor | None = None,
        *,
        prior_table: Tensor,
    ) -> Tensor:
        if abundance_mask is None:
            abundance_mask = torch.zeros_like(padding_mask)

        abundance_embedding = self.abundance_mlp(abundance.unsqueeze(-1))
        abundance_embedding = torch.where(
            abundance_mask.unsqueeze(-1),
            self.abundance_mask_embedding,
            abundance_embedding,
        )
        # Read the model-owned prior table: IDs [B, L] -> vectors [B, L, D].
        # 从模型持有的先验表查向量：编号 [B, L] -> 向量 [B, L, D]。
        genus_embedding = F.embedding(
            genus_ids, prior_table, padding_idx=PAD_TOKEN_ID
        )
        output = genus_embedding + abundance_embedding
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
