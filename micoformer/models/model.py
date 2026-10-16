from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from micoformer.config import ModelConfig
from .encoder import EncoderBackbone, masked_token_mean
from .embeddings import FixedPriorTokenStem, LearnedTokenStem
from .heads import AbundanceRegressionHead, ConcatQueryHead


class PretrainingOutput(NamedTuple):
    mlm_loss: Tensor
    continuous_loss: Tensor
    presence_loss: Tensor
    mlm_count: int
    continuous_count: int
    presence_count: int


class MiCoFormer(nn.Module):
    """Shared encoder with MLM abundance and Query abundance/presence objectives.
    共享编码器，执行 MLM 丰度重建与 Query 丰度、存在性预测。
    """

    def __init__(self, prior_table: Tensor, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Share one non-trainable prior table [vocab_size, D], saved with the model.
        # 共用一份不可训练的先验表 [vocab_size, D]，随模型保存与迁移设备。
        self.register_buffer(
            "prior_table",
            prior_table.detach().float().clone(),
            persistent=True,
        )

        if config.identity_type == "fixed":
            input_stem = FixedPriorTokenStem(config)
        else:
            input_stem = LearnedTokenStem(config)
        self.backbone = EncoderBackbone(config, input_stem)
        self.mlm_head = AbundanceRegressionHead(
            config.d_model, config.mlm_head_hidden_dim
        )
        self.query_head = ConcatQueryHead(config.d_model)

    def forward(
        self,
        genus_ids: Tensor,
        abundance: Tensor,
        padding_mask: Tensor,
        query_mask: Tensor,
        mlm_mask: Tensor,
        negative_genus_ids: Tensor,
    ) -> PretrainingOutput:
        """Compute three mean losses and target counts from batch tensors [B, L].
        根据 batch 张量 [B, L] 计算三项平均损失及对应目标数量。
        """

        # Encode the MLM and Query views separately with shared parameters.
        # MLM 与 Query 采用不同遮盖视图，分别编码并共用参数。
        mlm_loss, mlm_count = self._mlm_loss(genus_ids, abundance, padding_mask, mlm_mask)

        continuous_loss, presence_loss, continuous_count, presence_count = self._query_losses(
            genus_ids, abundance, padding_mask, query_mask, negative_genus_ids
        )

        return PretrainingOutput(
            mlm_loss, continuous_loss, presence_loss,
            mlm_count, continuous_count, presence_count,
        )

    def unmasked_representations(
        self,
        genus_ids: Tensor,
        abundance: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Encode unmasked inputs; return token [B, L, D] and sample [B, D] vectors.
        # 不额外遮盖输入，返回 token 表征 [B, L, D] 与样本表征 [B, D]。
        token_embeddings = self.backbone._encode(
            genus_ids,
            abundance,
            padding_mask,
            abundance_mask=None,
            prior_table=self.prior_table,
        )
        return token_embeddings, masked_token_mean(token_embeddings, padding_mask)


    @staticmethod
    def _prepare_abundance_input(
        abundance: Tensor,
        padding_mask: Tensor,
        query_mask: Tensor,
        mlm_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Recenter already-transformed abundance; return abundance and padding [B, L].
        对已变换的丰度重新中心化，返回丰度输入与忽略标记 [B, L]。
        """
        # Ignore PAD and Query-removed tokens; MLM tokens keep their genus identity.
        # 忽略 PAD 与 Query 移除位置；MLM 位置仍保留菌属身份。
        encoder_padding = padding_mask | query_mask
        visible_mask = ~(encoder_padding | mlm_mask)

        # Visible-only means [B, 1] remove target information carried by the old center.
        # 仅用可见丰度计算均值 [B, 1]，消除旧中心携带的目标信息。
        visible_count = visible_mask.sum(1, keepdim=True)
        visible_sum = (abundance * visible_mask.to(abundance.dtype)).sum(1, keepdim=True)
        visible_mean = visible_sum / visible_count.to(abundance.dtype)

        # Recenter visible values and zero hidden ones; preserve abundance as targets.
        # 可见丰度减去均值，隐藏位置置零；原 abundance 保留为预测目标。
        input_abundance = (abundance - visible_mean).masked_fill(~visible_mask, 0.0)
        return input_abundance, encoder_padding

    def _mlm_loss(
        self,
        genus_ids: Tensor,
        abundance: Tensor,
        padding_mask: Tensor,
        mlm_mask: Tensor,
    ) -> tuple[Tensor, int]:
        """Retain genus identities, predict hidden abundance, and return mean loss/count.
        保留菌属身份，预测隐藏丰度，返回平均损失与目标数量。
        """
        # MLM keeps identities and replaces hidden abundance with a trainable mask vector.
        # MLM 保留身份，将隐藏丰度替换为可训练的遮盖向量。
        query_mask = torch.zeros_like(mlm_mask)
        centered, encoder_padding = self._prepare_abundance_input(
            abundance, padding_mask, query_mask, mlm_mask
        )
        token_embeddings = self.backbone._encode(
            genus_ids,
            centered,
            encoder_padding,
            abundance_mask=mlm_mask,
            prior_table=self.prior_table,
        )
        # Predict one centered-log abundance per token: [B, L, D] -> [B, L].
        # 逐 token 预测中心化对数丰度：[B, L, D] -> [B, L]。
        prediction = self.mlm_head(token_embeddings)
        count = int(mlm_mask.sum().item())
        # Compare masked predictions with pre-corruption targets in float32.
        # 仅对遮盖位置计算损失，目标为遮盖前的中心化对数丰度，使用 float32。
        loss = F.smooth_l1_loss(
            prediction[mlm_mask].float(),
            abundance[mlm_mask].float(),
            beta=1.0,
        )
        return loss, count

    def _query_losses(
        self,
        genus_ids: Tensor,
        abundance: Tensor,
        padding_mask: Tensor,
        query_mask: Tensor,
        negative_genus_ids: Tensor,
    ) -> tuple[Tensor, Tensor, int, int]:
        """Remove Query targets, predict candidates, and compute abundance/presence losses.
        移除 Query 目标，预测正负候选菌属，并计算丰度与存在性损失。
        """
        # Query removes target tokens without additional MLM abundance masking.
        # Query 移除目标菌属，不额外施加 MLM 丰度遮盖。
        mlm_mask = torch.zeros_like(query_mask)
        centered, encoder_padding = self._prepare_abundance_input(
            abundance, padding_mask, query_mask, mlm_mask
        )
        token_embeddings = self.backbone._encode(
            genus_ids,
            centered,
            encoder_padding,
            abundance_mask=mlm_mask,
            prior_table=self.prior_table,
        )
        # Pool visible tokens into sample vectors [B, D].
        # 对可见 token 取均值，得到样本向量 [B, D]。
        sample_h = masked_token_mean(token_embeddings, encoder_padding)
        # Match each Query target to its sample; indices and candidate IDs are [N].
        # 将每个 Query 目标配到所属样本，样本索引与候选编号均为 [N]。
        sample_indices = torch.where(query_mask)[0]
        positive_ids = genus_ids[query_mask]
        # Negative IDs are sampled upstream outside the sample's full observed support.
        # 负例由数据层从原样本完整观测集合之外采样。
        negative_ids = negative_genus_ids[query_mask]
        # Candidates use fixed prior vectors [N, D], including learned input mode.
        # 候选向量 [N, D] 始终取固定先验，包括 learned 输入模式。
        positive_candidate = F.embedding(
            positive_ids, self.prior_table, padding_idx=0
        )
        negative_candidate = F.embedding(
            negative_ids, self.prior_table, padding_idx=0
        )
        # Positive candidates produce presence logits and abundance predictions [N].
        # 正例候选输出存在性 logit 与丰度预测，均为 [N]。
        positive_presence, positive_continuous = self.query_head(
            sample_h[sample_indices], positive_candidate
        )
        # Negative candidates supervise presence only.
        # 负例候选仅用于存在性监督。
        negative_presence, _ = self.query_head(
            sample_h[sample_indices], negative_candidate
        )
        # Regress positives against pre-corruption centered-log abundance.
        # 仅对正例回归，目标为遮盖前的中心化对数丰度。
        positive_target = abundance[query_mask].float()
        positive_count = int(positive_ids.numel())
        continuous_loss = F.smooth_l1_loss(
            positive_continuous.float(),
            positive_target.float(),
            beta=1.0,
        )
        # Concatenate logits [2N]: positives first, then negatives, with labels 1/0.
        # 拼接 logit [2N]：先正例、后负例，对应标签为 1/0。
        logits = torch.cat((positive_presence, negative_presence)).float()
        labels = torch.cat(
            (
                torch.ones_like(positive_presence),
                torch.zeros_like(negative_presence),
            )
        ).float()
        # BCEWithLogits computes the mean presence loss directly from logits.
        # BCEWithLogits 直接根据 logit 计算平均存在性损失。
        presence_loss = F.binary_cross_entropy_with_logits(logits, labels)
        # Return abundance/presence losses followed by their target counts.
        # 依次返回丰度损失、存在性损失及对应的目标数量。
        return continuous_loss, presence_loss, positive_count, int(labels.numel())
