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
)
from micoformer.models.pma import PMA
from micoformer.utils.train_utils import build_lr_scheduler


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
        token_embedding_mode: Optional[str] = None,  # 旧 alias
        rank_vocab_sizes: Dict[str, int],
        # V4 R2
        bias_type: str = "phylo",            # V5 默认 phylo
        phylo_mlp_hidden: int = 64,           # V5 默认 64
        n_vars: int = 0,
        # V5 新增
        abundance_encoding: str = "mlp",
        abundance_loss: str = "huber",         # "huber" | "bin_ce"
        use_phylo_pe: bool = True,
        phylo_pe_hidden: int = 128,
        pe_dim: Optional[int] = None,
        use_hierarchical_embed: bool = False,
        use_sample_token: bool = False,
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
            token_embedding_mode=token_embedding_mode,
            rank_vocab_sizes=rank_vocab_sizes,
            bias_type=bias_type,
            phylo_mlp_hidden=phylo_mlp_hidden,
            n_vars=n_vars if bias_type != "none" else None,
            abundance_encoding=abundance_encoding,
            use_phylo_pe=use_phylo_pe,
            phylo_pe_hidden=phylo_pe_hidden,
            pe_dim=pe_dim,
            use_sample_token=use_sample_token,
            use_hierarchical_embed=use_hierarchical_embed,
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

    # ------------------------------------------------------------------
    # forward 与 step 辅助
    # ------------------------------------------------------------------
    def _encode(self, batch: Dict[str, torch.Tensor]):
        """统一封装 encoder 调用,根据 abundance_encoding/use_sample_token 等 flag 自适应。"""
        h, sample_repr = self.encoder(
            token_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            abund_bins=batch.get("abund_bins"),
            abund_values=batch.get("abund_values"),
            mask_positions=batch.get("mask_positions"),
            taxon_path_ids=batch.get("taxon_path_ids"),
            var_indices=batch.get("var_indices"),
        )
        return h, sample_repr

    def _h_taxon(self, h: torch.Tensor) -> torch.Tensor:
        """返回不含 [SAMPLE] 位的 token 部分(对齐 labels 长度)。"""
        if self.hparams.use_sample_token:
            return h[:, 1:, :]
        return h

    def _pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """sample-level pooling: PMA(V5) 或 masked mean。"""
        # 输入 h 必须不含 [SAMPLE](use_sample_token=True 时由 _h_taxon 取 token 部分,
        # use_sample_token=False 时本来就没有 [SAMPLE])
        h_token = self._h_taxon(h)
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
        h, _ = self._encode(batch)
        h_token = self._h_taxon(h)

        # ============ MLM loss ============
        mask_pos = batch["mask_positions"]
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

        # log MLM
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

        # ============ Metadata loss ============
        total_loss = loss_mlm
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
            total_loss = loss_mlm + float(self.hparams.metadata_loss_weight) * loss_meta

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
        decay_params = []
        no_decay_params = []
        no_decay_names = ["bias", "LayerNorm.weight", "norm.weight"]

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay_names):
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
