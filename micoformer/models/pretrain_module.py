from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from micoformer.models.encoder import MiCoFormerEncoder
from micoformer.models.heads import AbundanceBinHead


class WarmupThenPlateau:
    """线性 warmup 后交由 ReduceLROnPlateau 接管。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        factor: float,
        patience: int,
        min_lr: float,
        start_factor: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.start_factor = float(start_factor)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self.plateau = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        if self.warmup_steps > 0:
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * self.start_factor

    def _warmup_factor(self, step_index: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        progress = min(1.0, step_index / float(self.warmup_steps))
        return self.start_factor + (1.0 - self.start_factor) * progress

    def step_batch(self) -> None:
        if self.step_count >= self.warmup_steps:
            return
        self.step_count += 1
        factor = self._warmup_factor(self.step_count)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step_metric(self, metric: float) -> None:
        if self.step_count < self.warmup_steps:
            return
        self.plateau.step(metric)


class MiCoFormerModule(L.LightningModule):

    def __init__(
        self,
        *,
        genus_vocab_size: Optional[int] = None,   # taxon 模式必须提供；taxon_path 模式不需要
        total_abundance_bins: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_taxon_id: int = 0,
        pad_bin_id: int = 0,
        token_embedding_mode: str = "taxon_path",
        rank_vocab_sizes: Optional[Dict[str, int]] = None,  # taxon_path 模式必须提供
        use_taxonomy_bias: bool = False,  # R2：启用 taxonomy 距离注意力偏置
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.02,
        lr_scheduler: str = "cosine",
        plateau_factor: float = 0.5,
        plateau_patience: int = 2,
        plateau_min_lr: float = 1e-6,
        budget_mode: str = "epoch",
        warmup_steps: Optional[int] = None,  # legacy checkpoint compatibility
        max_steps: Optional[int] = None,     # legacy checkpoint compatibility
    ) -> None:
        super().__init__()

        if warmup_steps is not None and max_steps and max_steps > 0 and warmup_ratio == 0.02:
            warmup_ratio = float(warmup_steps) / float(max_steps)

        # 保存所有 __init__ 参数到 self.hparams，便于 checkpoint 保存和恢复
        self.save_hyperparameters()
        self._plateau_controller: Optional[WarmupThenPlateau] = None

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
            use_taxonomy_bias=use_taxonomy_bias,
        )

        # 预训练任务头
        self.head = AbundanceBinHead(d_model=d_model, num_bins=total_abundance_bins)

        # 损失函数 (不进行 reduce，保留每个样本/token的loss)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h, sample_repr = self.encoder(
            token_ids=batch["token_ids"],
            abund_bins=batch["abund_bins"],
            taxon_path_ids=batch.get("taxon_path_ids", None),
            attention_mask=batch["attention_mask"],
        )
        logits = self.head(h)
        return {"token_repr": h, "sample_repr": sample_repr, "abund_logits": logits}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # 前向传播
        out = self(batch)

        # 取出前向传播结果
        logits = out["abund_logits"]       # [B, L, Num_Bins]
        labels = batch["labels_abund"]     # [B, L]
        mask_pos = batch["mask_positions"] # [B, L]

        # 同样需要对齐长度，去掉 logits 的第 0 位 (SAMPLE)
        logits = logits[:, 1:, :]

        if mask_pos.any():
            # 取出 Mask 位置的预测 Logits 与真实 Labels (布尔索引筛选)
            masked_logits = logits[mask_pos] # [N_Masked, Num_Bins]
            masked_labels = labels[mask_pos] # [N_Masked]
            
            # 计算 Cross Entropy Loss
            loss_vec = self.criterion(masked_logits, masked_labels)
            loss = loss_vec.mean()

            # 计算 Top-1 准确率 (Accuracy)，用于监控模型学习进度
            with torch.no_grad():
                pred = masked_logits.argmax(dim=-1)
                acc = (pred == masked_labels).float().mean()
                self.log("train/acc_mask", acc, prog_bar=True, on_step=True, on_epoch=True)
        else:
            # 极少数情况下 (如 batch 很小且 mask_prob 很低)，可能没有采样到 mask，此时 loss 为 0
            loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # 记录当前的 Learning Rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", current_lr, prog_bar=True, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:

        out = self(batch)
        logits = out["abund_logits"]
        labels = batch["labels_abund"]
        mask_pos = batch["mask_positions"]
        
        # 注意：Encoder 输出的 h 现在包含了 [SAMPLE] 在第 0 位
        # 而 logits 是对 h 进行投影得到的，所以 logits 也是 [Batch, Length+1, Num_Bins]
        # 但是 labels 和 mask_pos 是原始数据的长度 [Batch, Length] (不含 SAMPLE)
        # 所以我们需要把 logits 的第 0 位去掉，对齐长度
        logits = logits[:, 1:, :]

        if mask_pos.any():
            masked_logits = logits[mask_pos]
            masked_labels = labels[mask_pos]
            loss = self.criterion(masked_logits, masked_labels).mean()
            
            pred = masked_logits.argmax(dim=-1)
            acc = (pred == masked_labels).float().mean()
            
            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            self.log("val/acc_mask", acc, prog_bar=True, on_epoch=True)
        else:
            self.log("val/loss", torch.tensor(0.0, device=logits.device), prog_bar=True, on_epoch=True)
            self.log("val/acc_mask", torch.tensor(0.0, device=logits.device), prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if self._plateau_controller is None or self.trainer.sanity_checking:
            return
        metric = self.trainer.callback_metrics.get("val/loss")
        if metric is None:
            return
        value = float(metric.item() if hasattr(metric, "item") else metric)
        self._plateau_controller.step_metric(value)

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        if self._plateau_controller is not None:
            self._plateau_controller.step_batch()
        super().optimizer_step(*args, **kwargs)

    def _estimated_total_steps(self) -> int:
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        if total_steps > 0:
            return total_steps

        trainer_max_steps = getattr(self.trainer, "max_steps", None)
        if isinstance(trainer_max_steps, int) and trainer_max_steps > 0:
            return trainer_max_steps

        raise RuntimeError("Unable to infer total training steps for LR scheduling.")

    def _build_cosine_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        eta_min: float = 1e-6,
        start_factor: float = 0.01,
    ) -> LambdaLR:
        base_lr = float(self.hparams.lr)
        min_lr_ratio = eta_min / base_lr if base_lr > 0 else 0.0

        def lr_lambda(current_step: int) -> float:
            if total_steps <= 0:
                return 1.0

            if warmup_steps > 0 and current_step < warmup_steps:
                progress = (current_step + 1) / float(warmup_steps)
                return start_factor + (1.0 - start_factor) * progress

            if total_steps <= warmup_steps:
                return min_lr_ratio

            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self):
        # 分离参数组：对 bias 和 LayerNorm 不使用 weight_decay，防止过度正则化
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

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr
        )

        total_steps = self._estimated_total_steps()
        warmup_steps = int(float(self.hparams.warmup_ratio) * total_steps)

        if self.hparams.lr_scheduler == "cosine":
            scheduler = self._build_cosine_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            self._plateau_controller = None
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        if self.hparams.lr_scheduler == "plateau":
            self._plateau_controller = WarmupThenPlateau(
                optimizer,
                warmup_steps=warmup_steps,
                factor=float(self.hparams.plateau_factor),
                patience=int(self.hparams.plateau_patience),
                min_lr=float(self.hparams.plateau_min_lr),
            )
            return {"optimizer": optimizer}

        raise ValueError(f"Unknown lr_scheduler: {self.hparams.lr_scheduler}")
