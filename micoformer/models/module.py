from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from micoformer.models.encoder import MiCoFormerEncoder
from micoformer.models.heads import AbundanceBinHead


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
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
    ) -> None:
        super().__init__()

        # 保存所有 __init__ 参数到 self.hparams，便于 checkpoint 保存和恢复
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

        # Learning Rate Scheduler: Warmup + Cosine Decay
        # 前 warmup_steps 步线性增加 LR，之后 Cosine 衰减
        # 注意：这里假设总步数为 max_steps，如果实际训练步数更少，Cosine 可能不会降到最低
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.hparams.warmup_steps
        )
        
        # 计算剩余步数用于 Cosine Decay
        decay_steps = max(1, self.hparams.max_steps - self.hparams.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=decay_steps, eta_min=1e-6
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # 每个 step 更新一次 LR
                "frequency": 1,
            },
        }
