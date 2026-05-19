"""
MiCoFormer 正式训练参数注册表（V4 / GG2 corpus）。

来源：
- 共享主体架构：V2 超参数搜索 run_20260412_052731 的 Stage A top-1
- 各变体预训练 / 微调可适配参数：V2 Stage A top-1 + Stage C best
- bias_type 字段：V4 新增（替代旧 use_taxonomy_bias）。
  taxo / phylo 两种新 R2 实现没有 V2 实测最佳超参，先复用 V2 旧 R2 (LCA 5-bucket)
  搜出来的参数；后续根据实测再调。

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
    "lr_scheduler_type": "cosine",
    "abundance_mode": "abs_log_bins",
    "max_seq_len": 1024,
    # V4 R2 phylo 模式专用：MLP 隐藏层维度
    # 默认 16，在 SDPA fallback 下控制 [B, L, L, hidden] 中间张量内存
    "phylo_mlp_hidden": 16,
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
    # V4 R2 类型：
    #   "none"  - baseline，无距离 bias
    #   "taxo"  - 离散 7-bucket 查 varp['taxo_dist']
    #   "phylo" - 连续 MLP 查 varp['phylo_dist']
    bias_type: str

    # 预训练可适配参数（来自 V2 Stage A top-1）
    pretrain_lr: float
    pretrain_batch_size: int
    pretrain_weight_decay: float

    # 微调可适配参数（来自 V2 Stage C best）
    lr_head: float
    lr_encoder: float
    head_hidden_dim: int
    head_dropout: float


# ─── 6 个变体（2×3 消融：R1 on/off × bias_type ∈ {none, taxo, phylo}）─────────
# 命名约定：
#   baseline      = R1=off, bias=none
#   r1            = R1=on,  bias=none
#   r2_taxo       = R1=off, bias=taxo   （新 7-bucket，替代旧 LCA 5-bucket）
#   r1r2_taxo     = R1=on,  bias=taxo
#   r2_phylo      = R1=off, bias=phylo  （V4 新增，连续 MLP）
#   r1r2_phylo    = R1=on,  bias=phylo
VARIANTS: dict[str, VariantConfig] = {
    "baseline": VariantConfig(
        name="baseline",
        token_embedding_mode="taxon",
        bias_type="none",
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
        bias_type="none",
        pretrain_lr=1e-3,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=3e-3,
        lr_encoder=5e-6,
        head_hidden_dim=128,
        head_dropout=0.1,
    ),
    "r2_taxo": VariantConfig(
        name="r2_taxo",
        token_embedding_mode="taxon",
        bias_type="taxo",
        pretrain_lr=1e-3,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=5e-4,
        lr_encoder=5e-6,
        head_hidden_dim=128,
        head_dropout=0.0,
    ),
    "r1r2_taxo": VariantConfig(
        name="r1r2_taxo",
        token_embedding_mode="taxon_path",
        bias_type="taxo",
        pretrain_lr=1e-3,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=5e-4,
        lr_encoder=5e-6,
        head_hidden_dim=256,
        head_dropout=0.2,
    ),
    # V4 新增：连续 phylo 距离变体。
    # 微调参数复用对应 taxo 变体（V2 没有为 phylo 单独搜过；后续实测再调）。
    "r2_phylo": VariantConfig(
        name="r2_phylo",
        token_embedding_mode="taxon",
        bias_type="phylo",
        pretrain_lr=1e-3,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=5e-4,
        lr_encoder=5e-6,
        head_hidden_dim=128,
        head_dropout=0.0,
    ),
    "r1r2_phylo": VariantConfig(
        name="r1r2_phylo",
        token_embedding_mode="taxon_path",
        bias_type="phylo",
        pretrain_lr=1e-3,
        pretrain_batch_size=64,
        pretrain_weight_decay=0.1,
        lr_head=5e-4,
        lr_encoder=5e-6,
        head_hidden_dim=256,
        head_dropout=0.2,
    ),
}
