from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
)

from micoformer.models.encoder import MiCoFormerEncoder
from micoformer.models.heads import ClassificationHead
from micoformer.models.pma import PMA
from micoformer.utils.train_utils import build_lr_scheduler, extract_encoder_artifacts_from_ckpt


_VALID_FINETUNE_POOLING = {"pma", "mean_pool"}


class MiCoFormerClassifier(L.LightningModule):
    """多任务下游分类 Lightning Module。加载预训练 encoder,冻结或微调。

    V5 改动:
      - pooling_mode 限定 'pma' | 'mean_pool'（删除 'sample' / 'sample_and_mean',它们依赖 [SAMPLE] token,V5 默认无 [SAMPLE]）
      - pooling_mode='pma' 时从预训练 ckpt 暖启动 PMA 权重(strict=True;失败回退 strict=False + 警告)
    """

    def __init__(
        self,
        *,
        pretrained_ckpt_path: Optional[str] = None,
        task_configs: List[Dict[str, Any]],
        # 每项形如 {"name": "Phenotype", "num_classes": 2}
        pooling_mode: str = "pma",                # V5 默认 pma
        pma_nhead: Optional[int] = None,           # None → 从 ckpt 继承
        pma_k: Optional[int] = None,               # None → 从 ckpt 继承
        head_hidden_dim: int = 0,
        head_dropout: float = 0.1,
        freeze_encoder: bool = True,
        lr_head: float = 1e-3,
        lr_encoder: float = 1e-5,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.1,
        lr_scheduler: str = "cosine",
        plateau_factor: float = 0.5,
        plateau_patience: int = 2,
        plateau_min_lr: float = 1e-6,
        monitor_metric: str = "val/loss_total",
        # 从预训练 ckpt 提取的 encoder 架构参数（首次创建时自动填充,存入 hparams 以支持
        # 从微调 ckpt 直接恢复而无需预训练 ckpt 仍在原路径）
        _encoder_hparams: Optional[Dict] = None,
        # V5: 从预训练 ckpt 提取的 PMA state_dict(首次创建时自动填充,存入 hparams 用于微调恢复)
        _pma_state_dict: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if pooling_mode not in _VALID_FINETUNE_POOLING:
            raise ValueError(
                f"V5 finetune pooling_mode must be one of {sorted(_VALID_FINETUNE_POOLING)}, "
                f"got {pooling_mode!r}. Old 'sample' / 'sample_and_mean' rely on [SAMPLE] token, "
                "which is removed by default in V5."
            )

        if pretrained_ckpt_path is not None:
            # 首次创建:从 ckpt 提取 encoder 架构参数 + encoder/PMA 权重。
            # helper 自动识别 pretrain(MiCoFormerModule)/ finetune(MiCoFormerClassifier)两种 ckpt,
            # 后者支持 CC LOO 从 broad finetune ckpt 起跳(finetune_plan.md §5.1)。
            _encoder_hparams, _pretrained_encoder_sd, _pma_state_dict = \
                extract_encoder_artifacts_from_ckpt(pretrained_ckpt_path)
        elif _encoder_hparams is not None:
            # 从微调 ckpt 恢复:_encoder_hparams 来自已保存的 hparams,
            # Lightning 将从 state_dict 恢复 encoder + pma 权重
            _pretrained_encoder_sd = None
        else:
            raise RuntimeError(
                "Cannot create MiCoFormerClassifier: provide pretrained_ckpt_path or _encoder_hparams. "
                "Old checkpoints (before P1-1 fix) require the pretrain ckpt to still be accessible."
            )

        # 保存 hparams（忽略 pretrained_ckpt_path 与 _pma_state_dict;
        # _pma_state_dict 含 Tensor,不能 YAML 序列化;微调 ckpt 自身的 state_dict
        # 会包含 self.pma.* 权重,resume 时由 Lightning 直接恢复,无需 hparams 副本）
        self.save_hyperparameters(ignore=["pretrained_ckpt_path", "_pma_state_dict"])

        d_model = _encoder_hparams["d_model"]

        # ============ Encoder ============
        # 从架构参数重建 encoder(无论是首次创建还是从微调 ckpt 恢复)
        _bias_type = _encoder_hparams.get("bias_type", "none")
        self.encoder = MiCoFormerEncoder(
            genus_vocab_size=_encoder_hparams["genus_vocab_size"],
            total_abundance_bins=_encoder_hparams["total_abundance_bins"],
            d_model=d_model,
            nhead=_encoder_hparams["nhead"],
            num_layers=_encoder_hparams["num_layers"],
            dim_feedforward=_encoder_hparams["dim_feedforward"],
            dropout=_encoder_hparams.get("dropout", 0.1),
            pad_taxon_id=_encoder_hparams.get("pad_taxon_id", 0),
            pad_bin_id=_encoder_hparams.get("pad_bin_id", 0),
            token_embedding_mode=_encoder_hparams.get("token_embedding_mode"),
            rank_vocab_sizes=dict(_encoder_hparams["rank_vocab_sizes"]),
            bias_type=_bias_type,
            phylo_mlp_hidden=_encoder_hparams.get("phylo_mlp_hidden", 64),
            n_vars=_encoder_hparams.get("n_vars", 0) if _bias_type != "none" else None,
            # V5 flags(从预训练继承)
            abundance_encoding=_encoder_hparams.get("abundance_encoding", "mlp"),
            use_phylo_pe=_encoder_hparams.get("use_phylo_pe", True),
            phylo_pe_hidden=_encoder_hparams.get("phylo_pe_hidden", 128),
            pe_dim=_encoder_hparams.get("pe_dim", None),
            use_sample_token=_encoder_hparams.get("use_sample_token", False),
            use_hierarchical_embed=_encoder_hparams.get("use_hierarchical_embed", False),
        )

        if _pretrained_encoder_sd is not None:
            # 首次创建:加载预训练 encoder 权重(strict=False 容忍 buffer 差异 / V5 新模块)
            missing, unexpected = self.encoder.load_state_dict(
                _pretrained_encoder_sd, strict=False,
            )
            if missing or unexpected:
                import warnings
                warnings.warn(
                    f"[MiCoFormerClassifier] encoder load_state_dict non-strict: "
                    f"missing={missing[:5]}{'...' if len(missing)>5 else ''}, "
                    f"unexpected={unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
                )
        # else: Lightning 从微调 ckpt 的 state_dict 恢复 encoder 权重

        # 冻结 encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ============ PMA(V5 暖启动)============
        self.pma: Optional[PMA] = None
        if pooling_mode == "pma":
            # 从 ckpt hparams 推断 PMA 参数(若未显式传)
            ckpt_pma_nhead = _encoder_hparams.get("pma_nhead", None)
            ckpt_pma_k = _encoder_hparams.get("pma_k", None)
            actual_pma_nhead = pma_nhead if pma_nhead is not None else (ckpt_pma_nhead or 4)
            actual_pma_k = pma_k if pma_k is not None else (ckpt_pma_k or 1)
            self.pma = PMA(d_model=d_model, nhead_pma=actual_pma_nhead, k=actual_pma_k)
            # 暖启动:加载预训练 PMA 权重
            if _pma_state_dict is not None:
                try:
                    self.pma.load_state_dict(_pma_state_dict, strict=True)
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"[MiCoFormerClassifier] PMA strict load_state_dict failed: {e}. "
                        "Retrying with strict=False."
                    )
                    self.pma.load_state_dict(_pma_state_dict, strict=False)
            # 若 _pma_state_dict 为 None(预训练时没用 PMA),则 PMA 保持随机初始化

        # V5: head 输入维度统一 d_model(无 sample_and_mean)
        head_input_dim = d_model

        # 构建多任务分类 head
        self.heads = nn.ModuleDict()
        for cfg in task_configs:
            self.heads[cfg["name"]] = ClassificationHead(
                input_dim=head_input_dim,
                num_classes=cfg["num_classes"],
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )

        # 每个任务单独的 loss 函数（nn.ModuleDict 确保正确注册为子模块）
        self.criteria = nn.ModuleDict({
            cfg["name"]: nn.CrossEntropyLoss(ignore_index=-1)
            for cfg in task_configs
        })

        # torchmetrics：每个任务、每个 split 独立维护
        self._task_configs = task_configs
        self._build_metrics(task_configs)

    def _build_metrics(self, task_configs: List[Dict[str, Any]]) -> None:
        """为每个任务和 split (val/test) 构建 torchmetrics。"""
        for split in ("val", "test"):
            for cfg in task_configs:
                name = cfg["name"]
                nc = cfg["num_classes"]
                metrics = MetricCollection({
                    "acc": MulticlassAccuracy(num_classes=nc, average="macro"),
                    "f1_macro": MulticlassF1Score(num_classes=nc, average="macro"),
                    "f1_weighted": MulticlassF1Score(num_classes=nc, average="weighted"),
                    "auroc": MulticlassAUROC(num_classes=nc, average="macro"),
                })
                # 注册为子模块，确保自动 device 迁移
                setattr(self, f"_metrics_{split}_{name}", metrics)

    def _get_metrics(self, split: str, task_name: str) -> MetricCollection:
        return getattr(self, f"_metrics_{split}_{task_name}")

    def on_train_epoch_start(self) -> None:
        # 冻结模式下 encoder 始终保持 eval（禁止 BN/Dropout 更新统计量）
        if self.hparams["freeze_encoder"]:
            self.encoder.eval()

    def _pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """根据 pooling_mode 从 encoder 输出中提取样本表示(V5:pma / mean_pool)。

        V5 默认 use_sample_token=False,encoder 输出 h 不含 [SAMPLE],直接对整个 [B, L, d] 做 pool。
        若旧 ckpt 使用 use_sample_token=True,encoder 输出 [B, L+1, d],需要先去掉第 0 位 [SAMPLE]。
        """
        # 兼容旧 ckpt:若 encoder 仍带 [SAMPLE],去掉第 0 位
        if self.encoder.use_sample_token:
            h_token = h[:, 1:, :]
        else:
            h_token = h

        mode = self.hparams["pooling_mode"]
        if mode == "pma":
            key_padding_mask = ~attention_mask  # True=PAD
            return self.pma(h_token, key_padding_mask=key_padding_mask)
        elif mode == "mean_pool":
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            return (h_token * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pooling_mode: {mode}")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        h, _ = self.encoder(
            token_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            abund_bins=batch.get("abund_bins"),
            abund_values=batch.get("abund_values"),
            mask_positions=None,  # 微调时无 MLM mask
            taxon_path_ids=batch.get("taxon_path_ids"),
            var_indices=batch.get("var_indices"),
        )
        pooled = self._pool(h, batch["attention_mask"])
        logits = {name: head(pooled) for name, head in self.heads.items()}
        return logits

    def _compute_loss_and_log(
        self,
        batch: Dict[str, Any],
        split: str,
    ) -> torch.Tensor:
        """计算多任务 loss 并记录指标。"""
        logits = self(batch)
        labels = batch["labels"]  # dict: task_name → [B]

        total_loss = torch.tensor(0.0, device=self.device)
        has_loss = False

        for cfg in self._task_configs:
            name = cfg["name"]
            if name not in labels:
                continue
            task_labels = labels[name]
            task_logits = logits[name]
            loss = self.criteria[name](task_logits, task_labels)

            # 如果整个 batch 对该任务全为 -1，loss 为 0（ignore_index 处理）
            valid_mask = task_labels >= 0
            n_valid = valid_mask.sum()

            if n_valid > 0:
                has_loss = True
                total_loss = total_loss + loss
                # sync_dist=True：多 GPU 时跨 rank 聚合 val/test 指标
                self.log(
                    f"{split}/{name}/loss",
                    loss,
                    prog_bar=True,
                    on_epoch=True,
                    batch_size=int(n_valid),
                    sync_dist=(split in ("val", "test")),
                )

                # 更新 torchmetrics（只在 val/test 时）
                if split in ("val", "test"):
                    metrics = self._get_metrics(split, name)
                    preds = task_logits[valid_mask]
                    tgts = task_labels[valid_mask]
                    metrics.update(preds, tgts)

                # 训练时也计算 accuracy 用于监控
                if split == "train":
                    with torch.no_grad():
                        pred = task_logits[valid_mask].argmax(dim=-1)
                        acc = (pred == task_labels[valid_mask]).float().mean()
                        self.log(f"train/{name}/acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        if not has_loss:
            # 极端情况：batch 中所有任务都无有效标签
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log(
            f"{split}/loss_total",
            total_loss,
            prog_bar=(split == "train"),
            on_epoch=True,
            sync_dist=(split in ("val", "test")),
        )
        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._compute_loss_and_log(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self._compute_loss_and_log(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self._compute_loss_and_log(batch, "test")

    def _log_epoch_metrics(self, split: str) -> None:
        """在 epoch 结束时 compute & log torchmetrics，然后 reset。"""
        for cfg in self._task_configs:
            name = cfg["name"]
            metrics = self._get_metrics(split, name)
            computed = metrics.compute()
            for metric_name, value in computed.items():
                # torchmetrics 内部已跨 rank 聚合，这里 sync_dist 主要是保证
                # log 出来的标量在每个 rank 上一致，避免 ModelCheckpoint / EarlyStopping
                # 在各 rank 观察到不同值
                self.log(
                    f"{split}/{name}/{metric_name}",
                    value,
                    prog_bar=True,
                    sync_dist=True,
                )
            metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        freeze_encoder = self.hparams["freeze_encoder"]
        no_decay_names = ["bias", "LayerNorm.weight", "norm.weight"]

        # 收集 head 参数
        head_decay, head_no_decay = [], []
        for name, param in self.heads.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay_names):
                head_no_decay.append(param)
            else:
                head_decay.append(param)

        param_groups = [
            {"params": head_decay, "lr": self.hparams["lr_head"], "weight_decay": self.hparams["weight_decay"]},
            {"params": head_no_decay, "lr": self.hparams["lr_head"], "weight_decay": 0.0},
        ]

        # V5: PMA 是可训练 sample 聚合层(预训练暖启动后仍需微调)
        # 与 head 同 lr (lr_head),与 encoder 解耦——即使 encoder 冻结,PMA 仍参与训练
        if self.pma is not None:
            pma_decay, pma_no_decay = [], []
            for name, param in self.pma.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in no_decay_names):
                    pma_no_decay.append(param)
                else:
                    pma_decay.append(param)
            param_groups.extend([
                {"params": pma_decay, "lr": self.hparams["lr_head"], "weight_decay": self.hparams["weight_decay"]},
                {"params": pma_no_decay, "lr": self.hparams["lr_head"], "weight_decay": 0.0},
            ])

        # 非冻结时加入 encoder 参数
        if not freeze_encoder:
            enc_decay, enc_no_decay = [], []
            for name, param in self.encoder.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in no_decay_names):
                    enc_no_decay.append(param)
                else:
                    enc_decay.append(param)
            param_groups.extend([
                {"params": enc_decay, "lr": self.hparams["lr_encoder"], "weight_decay": self.hparams["weight_decay"]},
                {"params": enc_no_decay, "lr": self.hparams["lr_encoder"], "weight_decay": 0.0},
            ])

        optimizer = torch.optim.AdamW(param_groups)

        total_steps = int(self.trainer.estimated_stepping_batches)
        lr_sched_cfg = build_lr_scheduler(
            optimizer,
            scheduler_type=self.hparams["lr_scheduler"],
            warmup_ratio=float(self.hparams["warmup_ratio"]),
            total_steps=total_steps,
            plateau_factor=float(self.hparams["plateau_factor"]),
            plateau_patience=int(self.hparams["plateau_patience"]),
            plateau_min_lr=float(self.hparams["plateau_min_lr"]),
            plateau_mode="max",
            plateau_monitor=self.hparams["monitor_metric"],
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_sched_cfg}
