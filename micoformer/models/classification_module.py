from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
)

from micoformer.models.pretrain_module import MiCoFormerModule
from micoformer.models.heads import ClassificationHead


class MiCoFormerClassifier(L.LightningModule):
    """多任务下游分类 Lightning Module。加载预训练 encoder，冻结或微调。"""

    def __init__(
        self,
        *,
        pretrained_ckpt_path: str,
        task_configs: List[Dict[str, Any]],
        # 每项形如 {"name": "Phenotype", "num_classes": 2}
        pooling_mode: str = "mean_pool",
        head_hidden_dim: int = 0,
        head_dropout: float = 0.1,
        freeze_encoder: bool = True,
        lr_head: float = 1e-3,
        lr_encoder: float = 1e-5,
        weight_decay: float = 1e-2,
        warmup_steps: int = 200,
        max_steps: int = 10000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if pooling_mode not in {"sample", "mean_pool", "sample_and_mean"}:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

        # 从预训练 checkpoint 加载 encoder
        pretrained_module = MiCoFormerModule.load_from_checkpoint(
            pretrained_ckpt_path, map_location="cpu"
        )
        self.encoder = pretrained_module.encoder
        d_model = pretrained_module.hparams["d_model"]

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

        # 每个任务单独的 loss 函数，ignore_index=-1 跳过无标签样本
        self.criteria = {
            cfg["name"]: nn.CrossEntropyLoss(ignore_index=-1)
            for cfg in task_configs
        }

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
                self.log(f"{split}/{name}/loss", loss, prog_bar=True, on_epoch=True, batch_size=int(n_valid))

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

        self.log(f"{split}/loss_total", total_loss, prog_bar=(split == "train"), on_epoch=True)
        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss_and_log(batch, "train")
        # 记录学习率
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss

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
                self.log(f"{split}/{name}/{metric_name}", value, prog_bar=True)
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

        # Warmup + Cosine Decay（与预训练一致）
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams["warmup_steps"],
        )
        decay_steps = max(1, self.hparams["max_steps"] - self.hparams["warmup_steps"])
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=decay_steps, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams["warmup_steps"]],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
