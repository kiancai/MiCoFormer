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

from micoformer.models.pretrain_module import MiCoFormerModule
from micoformer.models.encoder import MiCoFormerEncoder
from micoformer.models.heads import ClassificationHead
from micoformer.utils.train_utils import build_lr_scheduler


class MiCoFormerClassifier(L.LightningModule):
    """多任务下游分类 Lightning Module。加载预训练 encoder，冻结或微调。"""

    def __init__(
        self,
        *,
        pretrained_ckpt_path: Optional[str] = None,
        task_configs: List[Dict[str, Any]],
        # 每项形如 {"name": "Phenotype", "num_classes": 2}
        pooling_mode: str = "mean_pool",
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
        # 从预训练 ckpt 提取的 encoder 架构参数（首次创建时自动填充，存入 hparams 以支持
        # 从微调 ckpt 直接恢复而无需预训练 ckpt 仍在原路径）
        _encoder_hparams: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if pretrained_ckpt_path is not None:
            # 首次创建：从预训练 ckpt 提取 encoder 架构参数和权重
            pretrained_module = MiCoFormerModule.load_from_checkpoint(
                pretrained_ckpt_path, map_location="cpu"
            )
            _encoder_hparams = dict(pretrained_module.hparams)
            _pretrained_encoder = pretrained_module.encoder
        elif _encoder_hparams is not None:
            # 从微调 ckpt 恢复：_encoder_hparams 来自已保存的 hparams，
            # Lightning 将从 state_dict 恢复 encoder 权重
            _pretrained_encoder = None
        else:
            raise RuntimeError(
                "Cannot create MiCoFormerClassifier: provide pretrained_ckpt_path or _encoder_hparams. "
                "Old checkpoints (before P1-1 fix) require the pretrain ckpt to still be accessible."
            )

        # 保存 hparams（忽略 pretrained_ckpt_path，保留 _encoder_hparams 以支持独立恢复）
        self.save_hyperparameters(ignore=["pretrained_ckpt_path"])

        d_model = _encoder_hparams["d_model"]

        if pooling_mode not in {"sample", "mean_pool", "sample_and_mean"}:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

        # 从架构参数重建 encoder（无论是首次创建还是从微调 ckpt 恢复）
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
            token_embedding_mode=_encoder_hparams["token_embedding_mode"],
            rank_vocab_sizes=dict(_encoder_hparams["rank_vocab_sizes"]),
            use_taxonomy_bias=_encoder_hparams.get("use_taxonomy_bias", False),
        )

        if _pretrained_encoder is not None:
            # 首次创建：加载预训练权重
            self.encoder.load_state_dict(_pretrained_encoder.state_dict())
        # else: Lightning 从微调 ckpt 的 state_dict 恢复 encoder 权重

        # 冻结 encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 根据 pooling 模式确定 head 输入维度
        if pooling_mode == "sample_and_mean":
            head_input_dim = 2 * d_model
        else:
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
        """根据 pooling_mode 从 encoder 输出中提取样本表示。"""
        mode = self.hparams["pooling_mode"]
        if mode == "sample":
            return h[:, 0, :]
        elif mode == "mean_pool":
            # attention_mask [B, L]，h[:, 1:, :] 是 taxon token 部分
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            return (h[:, 1:, :] * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif mode == "sample_and_mean":
            mask = attention_mask.unsqueeze(-1).float()
            mean = (h[:, 1:, :] * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.cat([h[:, 0, :], mean], dim=-1)
        else:
            raise ValueError(f"Unknown pooling_mode: {mode}")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        h, _ = self.encoder(
            token_ids=batch["token_ids"],
            abund_bins=batch["abund_bins"],
            taxon_path_ids=batch["taxon_path_ids"],
            attention_mask=batch["attention_mask"],
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
