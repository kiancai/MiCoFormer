"""
MiCoFormer 正式训练参数注册表。

来源：V2 超参数搜索 run_20260412_052731 的最终决策。
- Stage A top-1：共享主体架构 + 各变体预训练参数
- Stage C best：各变体第一次微调参数

不在本文件中保存的决策：
- full training 的 epoch 预算（在 run_full_training.py 的 CLI 默认值中）
- Stage 3 LOO 协议（在 evaluation-hub/scripts/ 中）
"""

from dataclasses import dataclass


# ─── 共享主体架构（所有变体共用）─────────────────────────────────────────────
SHARED_ARCH = {
    "d_model": 512,
    "num_layers": 12,
    "nhead": 16,          # fine regime (d_head=32)
    "ff_ratio": 4,
    "num_abundance_bins": 40,
    "dropout": 0.05,
}

# ─── 共享预训练固定参数 ───────────────────────────────────────────────────────
SHARED_PRETRAIN = {
    "warmup_ratio": 0.25,
    "mask_prob": 0.15,
    "min_abundance": 4e-6,
    "bias_grad_every_k": 4,
    "lr_scheduler_type": "cosine",
    "abundance_mode": "abs_log_bins",
    "max_seq_len": 1024,
}

# ─── 共享微调固定参数 ────────────────────────────────────────────────────────
SHARED_FINETUNE = {
    "pooling_mode": "sample_and_mean",
    "freeze_encoder": True,
    "weight_decay": 0.01,
    "batch_size": 32,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
}


@dataclass
class VariantConfig:
    """单个 MiCoFormer 变体的完整参数配置。"""

    # 变体标识
    name: str

    # 模型开关
    token_embedding_mode: str  # "taxon" 或 "taxon_path"
    use_taxonomy_bias: bool    # R2 开关

    # 预训练可适配参数（来自 V2 Stage A top-1）
    pretrain_lr: float
    pretrain_batch_size: int
    pretrain_weight_decay: float

    # 微调可适配参数（来自 V2 Stage C best）
    lr_head: float
    lr_encoder: float
    head_hidden_dim: int
    head_dropout: float


# ─── 四个变体的参数（来自 V2 结果）────────────────────────────────────────────
VARIANTS: dict[str, VariantConfig] = {
    "baseline": VariantConfig(
        name="baseline",
        token_embedding_mode="taxon",
        use_taxonomy_bias=False,
        pretrain_lr=3e-4,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=1e-3,
        lr_encoder=5e-6,
        head_hidden_dim=128,
        head_dropout=0.1,
    ),
    "r1": VariantConfig(
        name="r1",
        token_embedding_mode="taxon_path",
        use_taxonomy_bias=False,
        pretrain_lr=1e-3,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=3e-3,
        lr_encoder=5e-6,
        head_hidden_dim=128,
        head_dropout=0.1,
    ),
    "r2": VariantConfig(
        name="r2",
        token_embedding_mode="taxon",
        use_taxonomy_bias=True,
        pretrain_lr=1e-3,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=5e-4,
        lr_encoder=5e-6,
        head_hidden_dim=128,
        head_dropout=0.0,
    ),
    "r1r2": VariantConfig(
        name="r1r2",
        token_embedding_mode="taxon_path",
        use_taxonomy_bias=True,
        pretrain_lr=1e-3,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=5e-4,
        lr_encoder=5e-6,
        head_hidden_dim=256,
        head_dropout=0.2,
    ),
}
