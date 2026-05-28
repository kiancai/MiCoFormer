from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from micoformer.models.encoder import MiCoFormerEncoder
from micoformer.models.heads import (
    AbundanceBinHead,
    AbundanceRegressionHead,
    MetadataHead,
    PriorCoordHead,
)
from micoformer.models.pma import PMA
from micoformer.utils.train_utils import build_lr_scheduler
from micoformer.utils.tree_loss import TreeLossHelper


_VALID_ABUNDANCE_LOSS = {"huber", "bin_ce"}
_VALID_POOLING_MODE = {"pma", "mean_pool"}


class MiCoFormerModule(L.LightningModule):
    """V5 预训练 module:MLM(Huber 连续回归)+ Metadata 多任务(EnvCategory 6 类)。

    联合 loss = L_MLM + λ_meta * L_meta
    """

    def __init__(
        self,
        *,
        genus_vocab_size: int,
        total_abundance_bins: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_taxon_id: int = 0,
        pad_bin_id: int = 0,
        # hierarchical 删除后此参数不再使用,保留以免签名/ckpt-hparam 改动(透传给 encoder)
        rank_vocab_sizes: Dict[str, int],
        # V4 R2
        bias_type: str = "phylo",            # V5 默认 phylo
        phylo_mlp_hidden: int = 64,           # V5 默认 64
        # phylo MLP 末层是否保留 bias 项(仅 bias_type='phylo' 时生效)。
        # 默认 True 保持现有 ckpt 兼容;新预训练推荐 False(见 encoder.MiCoFormerEncoder 注释)。
        phylo_bias_last_layer_bias: bool = True,
        n_vars: int = 0,
        # V5 新增
        abundance_encoding: str = "mlp",
        abundance_loss: str = "huber",         # "huber" | "bin_ce"
        use_phylo_pe: bool = True,
        phylo_pe_hidden: int = 128,
        pe_dim: Optional[int] = None,
        grad_checkpointing: bool = False,       # 激活重算开关(透传给 encoder),默认关
        pooling_mode: str = "pma",             # "pma" | "mean_pool"
        pma_nhead: int = 4,
        pma_k: int = 1,
        use_metadata_task: bool = True,
        metadata_loss_weight: float = 0.3,
        metadata_num_classes: int = 6,
        metadata_class_weights: Optional[List[float]] = None,
        huber_beta: float = 1.0,
        # 优化器
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.02,
        lr_scheduler: str = "cosine",
        plateau_factor: float = 0.5,
        plateau_patience: int = 2,
        plateau_min_lr: float = 1e-6,
        # Tree loss(distance-preservation 辅助损失,见 utils/tree_loss.py)
        # tree_loss_weight=0 时整条 helper 不创建,跟现状完全等价
        tree_loss_weight: float = 0.0,
        tree_n_pairs: int = 256,
        tree_n_triplets: int = 128,
        tree_margin: float = 0.5,
        # X2 多任务范式(2026-05-28 夜,详 decisions.md / roadmap §4.1 d):
        #   mlm_weight       : abundance huber 回归权重(0 关掉 MLM,>0 开)
        #   x2_phylo_weight  : 预测 phylo coord MSE 权重(0 关掉 X2_phylo) - 2026-05-29 实测 mean collapse,**废弃**
        #   x2_protein_weight: 预测 protein coord MSE 权重(等 bacformer 出 protein_pe,phase 1 默认 0)
        #   x2_head_hidden   : PriorCoordHead 中间层维度
        # phase 1 默认:mlm=1, x2_phylo=1, x2_protein=0(蛋白 off)
        # phase 2 默认:mlm=1, x2_phylo=1, x2_protein=1(三任务全开)
        # 向后兼容:三个 weight 都=0 + tree_loss_weight=0 时退化为原 V5 MLM(沿用旧 ckpt 行为)
        mlm_weight: float = 1.0,
        x2_phylo_weight: float = 0.0,
        x2_protein_weight: float = 0.0,
        x2_head_hidden: int = 128,
        # X2 protein 通道开关 + 维度(透传给 encoder.protein_pe)
        use_protein_pe: bool = False,
        protein_pe_hidden: int = 128,
        protein_pe_dim: Optional[int] = None,
        # Phylo Soft-Target CE(2026-05-29,替代 X2 32d MSE 范式 — X2 实测 mean collapse):
        #   phylo_ce_weight: phylo-soft-CE loss 权重(0 关掉,>0 开;典型 1.0)
        #   phylo_ce_tau   : soft target 温度 τ — soft target = softmax(-dist/τ)
        #                    推荐 6.5(≈ log1p(patristic_max=656),让"近亲" vs "远亲" prob ratio
        #                    在 e^2~e^4 量级 sharp 但有梯度);τ→0 退化 vanilla CE,τ→∞ 退化 uniform
        #   需要:n_vars > 0 (encoder.dist_matrix buffer 必须创建);phylo_ce_weight>0 时自动 enforce
        # 设计依据:Wasserstein loss (Frogner 2015) + Tree-Wasserstein (Yamada 2021) 的 1st-order
        #   近似;CE-based 不会 mean collapse,且 phylo 强制进 loss
        phylo_ce_weight: float = 0.0,
        phylo_ce_tau: float = 6.5,
    ) -> None:
        super().__init__()

        if abundance_loss not in _VALID_ABUNDANCE_LOSS:
            raise ValueError(
                f"Unknown abundance_loss: {abundance_loss!r}. Expected {sorted(_VALID_ABUNDANCE_LOSS)}."
            )
        if pooling_mode not in _VALID_POOLING_MODE:
            raise ValueError(
                f"Unknown pooling_mode: {pooling_mode!r}. Expected {sorted(_VALID_POOLING_MODE)}."
            )
        # 互斥校验:mlp↔huber, bin↔bin_ce
        if abundance_encoding == "mlp" and abundance_loss != "huber":
            raise ValueError("abundance_encoding='mlp' must pair with abundance_loss='huber'.")
        if abundance_encoding == "bin" and abundance_loss != "bin_ce":
            raise ValueError("abundance_encoding='bin' must pair with abundance_loss='bin_ce'.")

        # X2 多任务一致性校验(2026-05-28 夜)
        if x2_phylo_weight > 0 and not use_phylo_pe:
            raise ValueError(
                "x2_phylo_weight > 0 requires use_phylo_pe=True "
                "(phylo coord 来自 encoder.phylo_pe.coords buffer,必须先加载 PE)."
            )
        if x2_protein_weight > 0 and not use_protein_pe:
            raise ValueError(
                "x2_protein_weight > 0 requires use_protein_pe=True "
                "(protein coord 来自 encoder.protein_pe.coords buffer,必须先加载 PE)."
            )
        if use_protein_pe and protein_pe_dim is None:
            raise ValueError("use_protein_pe=True requires protein_pe_dim.")
        # 所有 weight 都 = 0 = 没 supervision,触发警告级 error
        _any_loss = (
            (mlm_weight > 0) or (x2_phylo_weight > 0) or (x2_protein_weight > 0)
            or (tree_loss_weight > 0) or (phylo_ce_weight > 0)
        )
        if not _any_loss:
            raise ValueError(
                "All training losses are off (mlm/x2_phylo/x2_protein/tree_loss/phylo_ce all=0). "
                "At least one loss weight must be > 0."
            )

        # Phylo-soft-CE 需要 dist_matrix buffer + n_vars
        if phylo_ce_weight > 0 and (n_vars is None or n_vars <= 0):
            raise ValueError(
                "phylo_ce_weight > 0 requires n_vars > 0 to allocate encoder.dist_matrix buffer "
                "(workflow 必须 inject 真实 dist_matrix 才能算 soft target)."
            )

        # 保存所有 __init__ 参数到 self.hparams,便于 checkpoint 保存和恢复
        self.save_hyperparameters()

        self.encoder = MiCoFormerEncoder(
            genus_vocab_size=genus_vocab_size,
            total_abundance_bins=total_abundance_bins,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pad_taxon_id=pad_taxon_id,
            pad_bin_id=pad_bin_id,
            rank_vocab_sizes=rank_vocab_sizes,
            bias_type=bias_type,
            phylo_mlp_hidden=phylo_mlp_hidden,
            phylo_bias_last_layer_bias=phylo_bias_last_layer_bias,
            # n_vars 需要在 bias_type!='none' 或 phylo_ce_weight>0 时传入(后者 loss 需要 dist_matrix)
            n_vars=n_vars if (bias_type != "none" or phylo_ce_weight > 0) else None,
            abundance_encoding=abundance_encoding,
            use_phylo_pe=use_phylo_pe,
            phylo_pe_hidden=phylo_pe_hidden,
            pe_dim=pe_dim,
            # X2 多任务:蛋白 PE 透传
            use_protein_pe=use_protein_pe,
            protein_pe_hidden=protein_pe_hidden,
            protein_pe_dim=protein_pe_dim,
            grad_checkpointing=grad_checkpointing,
        )

        # ============ MLM head ============
        if abundance_loss == "huber":
            self.mlm_head = AbundanceRegressionHead(d_model=d_model)
        else:
            self.mlm_head = AbundanceBinHead(d_model=d_model, num_bins=total_abundance_bins)
        self.bin_ce = nn.CrossEntropyLoss(reduction="none")  # bin 路径用
        self.huber_beta = huber_beta

        # ============ PMA(V5)或 mean_pool ============
        self.pma: Optional[PMA] = None
        if pooling_mode == "pma":
            self.pma = PMA(d_model=d_model, nhead_pma=pma_nhead, k=pma_k)

        # ============ Metadata head ============
        self.metadata_head: Optional[MetadataHead] = None
        if use_metadata_task:
            self.metadata_head = MetadataHead(
                d_model=d_model, num_classes=metadata_num_classes
            )
            # metadata_class_weights:从 list(在 hparams)重建 tensor buffer
            # persistent=False 避免 hparams + state_dict 双份冗余;resume 时从 hparams 自动重建
            if metadata_class_weights is not None:
                self.register_buffer(
                    "_meta_class_weights",
                    torch.tensor(metadata_class_weights, dtype=torch.float32),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "_meta_class_weights",
                    torch.ones(metadata_num_classes, dtype=torch.float32),
                    persistent=False,
                )

        # ============ X2 多任务 heads(2026-05-28 夜) ============
        # phylo_head: d_model → pe_dim;末层 zero-init(防 self-distillation 风格 collapse)
        # protein_head: d_model → protein_pe_dim(条件创建,等 bacformer)
        # weight=0 时不创建对应 head(避免 DDP find_unused)
        self.phylo_head: Optional[PriorCoordHead] = None
        if x2_phylo_weight > 0:
            if pe_dim is None:
                raise ValueError("x2_phylo_weight > 0 requires pe_dim.")
            self.phylo_head = PriorCoordHead(
                d_model=d_model, pe_dim=pe_dim, hidden=x2_head_hidden
            )
        self.protein_head: Optional[PriorCoordHead] = None
        if x2_protein_weight > 0:
            self.protein_head = PriorCoordHead(
                d_model=d_model, pe_dim=protein_pe_dim, hidden=x2_head_hidden
            )

        # ============ Phylo Soft-Target CE head(2026-05-29) ============
        # vocab_head: d_model → genus_vocab_size(含 PAD/UNK 共 V_real+2 dim)
        # loss 时只取 real genus 列(跳过 PAD/UNK),target 是 softmax(-dist_matrix/τ) soft 分布
        # weight=0 时不创建(避免 DDP find_unused)
        self.vocab_head: Optional[nn.Linear] = None
        if phylo_ce_weight > 0:
            self.vocab_head = nn.Linear(d_model, genus_vocab_size)

        # ============ Tree loss helper(可选,默认 off) ============
        # tree_loss_weight=0:不创建 helper,_shared_step 中也跳过分支,与现状完全等价
        # tree_loss_weight>0:创建 helper,workflow 须在 inject_var_buffers 之后调
        #                    `module.tree_loss_helper.set_phylo_dist(encoder.dist_matrix)`
        # 要求 bias_type='phylo'(其它 dist 类型不是连续 patristic 不能拟合 cosine)
        self.tree_loss_helper: Optional[TreeLossHelper] = None
        if tree_loss_weight > 0:
            if bias_type != "phylo":
                raise ValueError(
                    f"tree_loss_weight>0 requires bias_type='phylo' (continuous patristic "
                    f"distance), got bias_type={bias_type!r}."
                )
            self.tree_loss_helper = TreeLossHelper(
                n_pairs=tree_n_pairs,
                n_triplets=tree_n_triplets,
                margin=tree_margin,
            )

    # ------------------------------------------------------------------
    # forward 与 step 辅助
    # ------------------------------------------------------------------
    def _encode(self, batch: Dict[str, torch.Tensor]):
        """统一封装 encoder 调用,根据 abundance_encoding 等 flag 自适应。

        X2 范式:当 x2_phylo_weight>0 或 x2_protein_weight>0 时,encoder forward
        启用 mask_token_id_replace(mask 位置 token embed → genus_mask_token + PE 输出乘 0)
        防止模型从 token_id 直接 lookup phylo/protein coord 答案。
        """
        x2_active = (self.hparams.x2_phylo_weight > 0) or (self.hparams.x2_protein_weight > 0)
        h = self.encoder(
            token_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            abund_bins=batch.get("abund_bins"),
            abund_values=batch.get("abund_values"),
            mask_positions=batch.get("mask_positions"),
            var_indices=batch.get("var_indices"),
            mask_token_id_replace=x2_active,
        )
        return h

    def _pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """sample-level pooling: PMA(V5) 或 masked mean。"""
        h_token = h
        key_padding_mask = ~attention_mask  # True = PAD
        if self.hparams.pooling_mode == "pma":
            return self.pma(h_token, key_padding_mask=key_padding_mask)
        # mean_pool: 对非 PAD 位置求平均
        mask_f = attention_mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (h_token * mask_f).sum(dim=1) / denom

    # ------------------------------------------------------------------
    # 训练 / 验证步
    # ------------------------------------------------------------------
    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,  # "train" / "val"
    ) -> torch.Tensor:
        h = self._encode(batch)
        h_token = h

        # ============ MLM loss(mlm_weight=0 时跳过整段,但 head 已创建则前向看一眼避免 DDP 不参与) ============
        mask_pos = batch["mask_positions"]
        mlm_w = float(self.hparams.mlm_weight)
        if self.hparams.abundance_loss == "huber":
            pred = self.mlm_head(h_token)               # [B, L]
            target = batch["labels_abund_values"]       # [B, L] float32
            if mask_pos.any():
                loss_mlm = F.smooth_l1_loss(
                    pred[mask_pos], target[mask_pos], beta=self.huber_beta, reduction="mean"
                )
                with torch.no_grad():
                    mae = (pred[mask_pos] - target[mask_pos]).abs().mean()
            else:
                loss_mlm = torch.zeros((), device=h.device, dtype=h.dtype)
                mae = torch.zeros((), device=h.device, dtype=h.dtype)
        else:
            logits = self.mlm_head(h_token)              # [B, L, num_bins]
            labels = batch["labels_abund"]               # [B, L] long
            if mask_pos.any():
                m_logits = logits[mask_pos]
                m_labels = labels[mask_pos]
                loss_mlm = self.bin_ce(m_logits, m_labels).mean()
                with torch.no_grad():
                    acc = (m_logits.argmax(dim=-1) == m_labels).float().mean()
                mae = None
                self.log(
                    f"{stage}/acc_mask", acc,
                    prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                    sync_dist=(stage == "val"),
                )
            else:
                loss_mlm = torch.zeros((), device=h.device, dtype=h.dtype)
                mae = None

        # log MLM(始终记 raw loss 数值,即便 mlm_weight=0 也方便诊断)
        self.log(
            f"{stage}/loss_mlm", loss_mlm,
            prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
            sync_dist=(stage == "val"),
        )
        if self.hparams.abundance_loss == "huber" and mae is not None:
            self.log(
                f"{stage}/mlm_mae", mae,
                prog_bar=False, on_step=(stage == "train"), on_epoch=True,
                sync_dist=(stage == "val"),
            )

        # ============ X2 phylo loss(2026-05-28 夜) ============
        # mask 位置预测 phylo coord: target = encoder.phylo_pe.coords[token_ids] (frozen buffer)
        loss_x2_phylo = torch.zeros((), device=h.device, dtype=h.dtype)
        x2_phylo_w = float(self.hparams.x2_phylo_weight)
        if x2_phylo_w > 0 and self.phylo_head is not None:
            if self.encoder.phylo_pe is None or not self.encoder.phylo_pe._coords_loaded:
                raise RuntimeError(
                    "x2_phylo_weight>0 requires encoder.phylo_pe.coords loaded "
                    "(call inject_var_buffers / phylo_pe.set_coords before forward())."
                )
            pred_phylo = self.phylo_head(h_token)                          # [B, L, pe_dim]
            target_phylo = self.encoder.phylo_pe.coords[batch["token_ids"]]  # [B, L, pe_dim] frozen
            if mask_pos.any():
                loss_x2_phylo = F.mse_loss(
                    pred_phylo[mask_pos], target_phylo[mask_pos], reduction="mean"
                )
            self.log(
                f"{stage}/loss_x2_phylo", loss_x2_phylo,
                prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                sync_dist=(stage == "val"),
            )

        # ============ X2 protein loss(2026-05-28 夜,phase 2) ============
        # 镜像 X2 phylo:target = encoder.protein_pe.coords[token_ids]
        loss_x2_protein = torch.zeros((), device=h.device, dtype=h.dtype)
        x2_protein_w = float(self.hparams.x2_protein_weight)
        if x2_protein_w > 0 and self.protein_head is not None:
            if self.encoder.protein_pe is None or not self.encoder.protein_pe._coords_loaded:
                raise RuntimeError(
                    "x2_protein_weight>0 requires encoder.protein_pe.coords loaded."
                )
            pred_protein = self.protein_head(h_token)                          # [B, L, protein_pe_dim]
            target_protein = self.encoder.protein_pe.coords[batch["token_ids"]]
            if mask_pos.any():
                loss_x2_protein = F.mse_loss(
                    pred_protein[mask_pos], target_protein[mask_pos], reduction="mean"
                )
            self.log(
                f"{stage}/loss_x2_protein", loss_x2_protein,
                prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                sync_dist=(stage == "val"),
            )

        # ============ Phylo Soft-Target CE loss(2026-05-29) ============
        # mask 位置预测 vocab id(8114 真实 genus + PAD/UNK),target = softmax(-dist/tau) soft 分布
        # CE-based 不会 mean collapse + phylo 强制进 loss(错预测远亲菌 cost 大)
        loss_phylo_ce = torch.zeros((), device=h.device, dtype=h.dtype)
        phylo_ce_w = float(self.hparams.phylo_ce_weight)
        if phylo_ce_w > 0 and self.vocab_head is not None:
            if self.encoder.dist_matrix is None or not self.encoder._dist_matrix_loaded:
                raise RuntimeError(
                    "phylo_ce_weight>0 requires encoder.dist_matrix loaded "
                    "(call inject_var_buffers with dist_matrix before forward())."
                )
            tau = float(self.hparams.phylo_ce_tau)
            # vocab id 约定:0=PAD, 1=UNK, 2~V_real+1 = real genus(var_index = vocab_id - 2)
            mask_target_vocab = batch["token_ids"][mask_pos]   # [N] long, mask 位置原始 vocab id
            # 只对 real genus(vocab_id >= 2)算 loss,跳过 UNK(=1)即 var_index=-1
            valid_target = mask_target_vocab >= 2
            if valid_target.any():
                target_var_idx = (mask_target_vocab[valid_target] - 2).long()  # [n_valid]
                # dist_matrix[target_var_idx]: [n_valid, V_real] phylo distance to all real genera
                dist_to_true = self.encoder.dist_matrix[target_var_idx].float()  # [n_valid, V_real]
                # soft target: softmax(-dist/tau) — 近亲 prob 高、远亲 prob 低
                target_dist = F.softmax(-dist_to_true / tau, dim=-1)             # [n_valid, V_real]
                # 模型 logits:vocab_head 输出 V_real+2 dim,只取 :V_real(跳过 PAD/UNK 列)
                # 注意:vocab id [2, V_real+1] 对应 vocab_head 输出的 [2:V_real+2] 列
                logits_full = self.vocab_head(h_token)                            # [B, L, V_real+2]
                mask_logits_full = logits_full[mask_pos]                          # [N, V_real+2]
                logits_real = mask_logits_full[valid_target, 2:]                  # [n_valid, V_real]
                log_probs = F.log_softmax(logits_real, dim=-1)                    # [n_valid, V_real]
                loss_phylo_ce = -(target_dist * log_probs).sum(-1).mean()
            self.log(
                f"{stage}/loss_phylo_ce", loss_phylo_ce,
                prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                sync_dist=(stage == "val"),
            )

        # ============ Metadata loss ============
        total_loss = (
            mlm_w * loss_mlm
            + x2_phylo_w * loss_x2_phylo
            + x2_protein_w * loss_x2_protein
            + phylo_ce_w * loss_phylo_ce
        )
        if self.hparams.use_metadata_task:
            if "env_label" not in batch:
                raise RuntimeError(
                    "use_metadata_task=True but batch does not contain 'env_label'. "
                    "Check that DataModule was configured with use_metadata_task=True "
                    "(it should wrap the dataset with _EnvLabelWrappedSubset)."
                )
            sample_repr = self._pool(h, batch["attention_mask"])  # [B, d_model]
            logits_meta = self.metadata_head(sample_repr)         # [B, C]
            env_label = batch["env_label"]                        # [B]
            loss_meta = F.cross_entropy(
                logits_meta, env_label,
                weight=self._meta_class_weights.to(logits_meta.dtype),
            )
            with torch.no_grad():
                meta_acc = (logits_meta.argmax(dim=-1) == env_label).float().mean()
            self.log(
                f"{stage}/loss_meta", loss_meta,
                prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                sync_dist=(stage == "val"),
            )
            self.log(
                f"{stage}/metadata_acc", meta_acc,
                prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                sync_dist=(stage == "val"),
            )
            total_loss = total_loss + float(self.hparams.metadata_loss_weight) * loss_meta

        # ============ Tree loss(distance-preservation 辅助,可选) ============
        # 仅 train 阶段 + tree_loss_weight>0 + helper 已创建 时启用
        # 同 batch h 直接复用,不重 forward
        tlw = float(self.hparams.tree_loss_weight)
        if stage == "train" and tlw > 0 and self.tree_loss_helper is not None:
            tl = self.tree_loss_helper(
                h=h_token,
                var_indices=batch["var_indices"],
                attention_mask=batch["attention_mask"],
            )
            loss_pair = tl["loss_pair"]
            loss_triplet = tl["loss_triplet"]
            loss_tree = loss_pair + loss_triplet
            total_loss = total_loss + tlw * loss_tree
            # log 诊断数值(epoch-level 汇总在 logger CSV 里)
            self.log(f"{stage}/loss_pair", loss_pair, on_step=True, on_epoch=True)
            self.log(f"{stage}/loss_triplet", loss_triplet, on_step=True, on_epoch=True)
            self.log(f"{stage}/triplet_violation_rate", tl["triplet_violation_rate"], on_step=True, on_epoch=True)
            self.log(f"{stage}/d_ap_mean", tl["d_ap_mean"], on_step=True, on_epoch=True)
            self.log(f"{stage}/d_an_mean", tl["d_an_mean"], on_step=True, on_epoch=True)

        self.log(
            f"{stage}/loss", total_loss,
            prog_bar=True, on_step=(stage == "train"), on_epoch=True,
            sync_dist=(stage == "val"),
        )
        return total_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    # ------------------------------------------------------------------
    # 优化器(同旧版)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        # no_decay 分组规则(2026-05-28 完整修复,见 [[feedback-wd-no-decay-rule]]):
        #   ① 含 'bias' / 'LayerNorm.weight' / 'norm.weight' — 标准 BERT/GPT 做法
        #   ② phylo_pe.* / dist_bias.* 整模块 — zero-init 渐进涌现型先验,WD 会持续压回 0
        #      (实测 tmp/20260528_phylo_task_redesign:WD=0.05 时 phylo 全模块静止,
        #       移除后 PE_proj_w 2000 step 单调涨 +0.76%)
        #   ③ abund_mask_token — 类 BERT [MASK] token 的可学习 special token
        #   ④ pma.query — Set Transformer 可学习 query seed,类 [CLS]
        #   ⑤ 兜底 param.ndim == 1 — LayerNorm 嵌套进 nn.Sequential 时,LN gamma 名字不含
        #      'norm'/'LayerNorm' 而被关键字漏抓(如 V5 abund_mlp.3.weight = LayerNorm gamma)
        decay_params = []
        no_decay_params = []
        no_decay_names = ["bias", "LayerNorm.weight", "norm.weight"]
        no_decay_prefixes = ("encoder.phylo_pe.", "encoder.dist_bias.")
        no_decay_exact = {"encoder.abund_mask_token", "pma.query"}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = (
                any(nd in name for nd in no_decay_names)
                or any(name.startswith(p) for p in no_decay_prefixes)
                or name in no_decay_exact
                or param.ndim == 1
            )
            if is_no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        lr_sched_cfg = build_lr_scheduler(
            optimizer,
            scheduler_type=self.hparams.lr_scheduler,
            warmup_ratio=float(self.hparams.warmup_ratio),
            total_steps=total_steps,
            plateau_factor=float(self.hparams.plateau_factor),
            plateau_patience=int(self.hparams.plateau_patience),
            plateau_min_lr=float(self.hparams.plateau_min_lr),
            plateau_mode="min",
            plateau_monitor="val/loss",
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_sched_cfg}
