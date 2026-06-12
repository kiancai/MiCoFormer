"""预训练 workflow：可复用的训练逻辑（不含 argparse）"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional

import anndata as ad
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info

from micoformer.datamodules.pretrain_datamodule import MiCoDataModule
from micoformer.models.pretrain_module import MiCoFormerModule
from micoformer.utils.train_utils import (
    choose_precision,
    inject_var_buffers,
    resolve_pretrain_ff_params,
    validate_index_arrays,
    validate_no_split_overlap,
    validate_pretrain_config,
)


TAG = "[train_pretrain]"


@dataclass
class PretrainRunConfig:
    # 0. 输入与切分参数
    h5ad_path: str

    # 1. 模型版本开关
    # V4 R2：距离驱动的 attention bias（'none' | 'taxo' | 'phylo'）
    bias_type: str = "phylo"           # V5 默认 phylo
    phylo_mlp_hidden: int = 64          # V5 默认 64(3 层 MLP);旧 alias phylo_bias_hidden 走同字段
    # phylo MLP 末层是否保留 bias 项。默认 True 保持兼容;新预训练推荐 False
    # (见 attn_bias.PhyloDistBias 注释,实测 bias 是 dead-weight 让 weight 学不出距离依赖)
    phylo_bias_last_layer_bias: bool = True

    # Tree loss(distance-preservation 辅助损失,见 utils/tree_loss.py 文献依据)
    # tree_loss_weight=0 时整条路径不创建,跟现状完全等价(默认 off);
    # tree_loss_weight>0 时要求 bias_type='phylo'(连续 patristic dist 拟合 cosine);
    # workflow 注入 dist_matrix buffer 之后会自动把 phylo_dist 引用挂到 helper 上。
    tree_loss_weight: float = 0.0
    tree_n_pairs: int = 256
    tree_n_triplets: int = 128
    tree_margin: float = 0.5

    # X2 多任务范式(2026-05-28 夜,详 decisions.md / roadmap §4.1 d):
    #   mlm_weight       : abundance huber 回归权重(0 关掉 MLM)
    #   x2_phylo_weight  : 预测 phylo coord MSE 权重
    #   x2_protein_weight: 预测 protein coord MSE 权重(等 bacformer 出 protein_pe)
    #   x2_head_hidden   : PriorCoordHead 中间层维度
    # phase 1 默认 mlm=1, x2_phylo=1, x2_protein=0(蛋白 off,无外部阻塞)
    # phase 2 蛋白完成后开:mlm=1, x2_phylo=1, x2_protein=1
    # 三个 weight 都=0 + tree_loss_weight=0 会触发 module __init__ ValueError
    mlm_weight: float = 1.0
    x2_phylo_weight: float = 0.0
    x2_protein_weight: float = 0.0
    x2_head_hidden: int = 128
    # 蛋白 PE 通道:phase 2 启用时 use_protein_pe=True + protein_pe_dim 从 varm['protein_pe'].shape[1] 读
    use_protein_pe: bool = False
    protein_pe_hidden: int = 128
    # protein_pe_dim 不在 config 里指定:由 datamodule 加载 varm['protein_pe'] 后自动决定
    # protein_pe_path:protein_feat.npy 外部路径([V_real, 480]),给 protein_pe **embedding** 输入用
    #   (我们的 protein_feat 在外部 npy,不在金标准语料 varm,所以走 workflow 传入而非 datamodule.varm)
    #   use_protein_pe=True 时必须给定
    protein_pe_path: Optional[str] = None

    # Phylo Soft-Target CE(2026-05-29,替代 X2 32d MSE — 实测 mean collapse)
    # phylo_ce_weight>0 时:① workflow 自动注入 dist_matrix(无论 bias_type) ② module 创建 vocab_head
    # phylo_ce_tau:soft target 温度,推荐 6.5
    phylo_ce_weight: float = 0.0
    phylo_ce_tau: float = 6.5
    # Phylo Tree-Wasserstein simplified(2026-05-29 phase 2,W-1 expected phylo distance loss)
    # 数学:W(p, δ_v*) = Σ p(v) × d(v, v*) — closed-form when target one-hot,无 hyperparameter
    # 跟 phylo_ce 共享 vocab_head + dist_matrix;两者互斥但允许同时(便于 ablation)
    phylo_w_weight: float = 0.0
    # Protein Tree-Wasserstein simplified(2026-05-30,phylo_w 精确镜像,把 phylo dist 换成蛋白距离)
    # protein_w_weight>0 时:① workflow 从 protein_dist_path 加载 protein_dist 并注入 encoder
    #   ② module 创建/共享 vocab_head + encoder 创建 protein_dist_matrix buffer
    # protein_dist_path:protein_dist.npy 外部路径([V_real, V_real] float32,对角=0、对称)
    #   protein_w_weight>0 时必须给定
    protein_w_weight: float = 0.0
    protein_dist_path: Optional[str] = None
    # 对比学习(2026-06-04,InfoNCE,保留 MLM 锚):contrastive_weight>0 开;两视图=同样本两套 abund-mask
    contrastive_weight: float = 0.0
    contrastive_temp: float = 0.1
    contrastive_proj_dim: int = 128
    contrastive_mask_prob: float = 0.15
    # JEPA(2026-06-04,潜空间预测被遮 genus 含义向量;详见 pretrain_module + PLAN.md)
    #   jepa_weight>0 开;target encoder=EMA 副本(看完整),predictor 用被遮 genus 坐标当地址 query。
    #   红线:坐标只当地址,target 是含义向量非坐标(否则退化成已证伪的 X2_phylo)。
    #   需 use_phylo_pe/use_protein_pe 提供 coords。
    #   ⚠️ JEPA 模式 genus_mask_token 闲置(mask_token_id_replace=False)→ 三卡须开
    #   ddp_find_unused_parameters=True(同纯 MLM 对照)。
    jepa_weight: float = 0.0
    jepa_mask_ratio: float = 0.5
    jepa_mlm_mask_prob: float = 0.15
    jepa_ema_decay: float = 0.996
    jepa_pred_dim: int = 256
    jepa_pred_depth: int = 2
    jepa_pred_heads: int = 4
    jepa_vicreg_weight: float = 0.0
    jepa_mask_mode: str = "structured"      # v2 默认 structured(样本内成簇遮,见 PLAN 结构化 mask)
    jepa_addr_mode: str = "coords"          # 'coords'(历史,phylo/protein 错图) | 'genus'(Cell-JEPA 身份地址,2026-06-09)
    jepa_n_seeds: int = 4                    # structured 时多少种子簇(I-JEPA multi-block;v2 默认 4)
    # JEPA v2(2026-06-06,删 MLM + 双自监督 + 防塌升级,详 pretrain_module)
    #   jepa_global_weight : 全局对齐 loss 权重(student PMA vs teacher PMA;默认 0.5)
    #   jepa_n_reg_tokens  : encoder 前缀 register token 数(T-JEPA 防塌;默认 4)
    #   jepa_ratio_start/end: structured mask ratio curriculum,按 epoch 线性 start→end(0.3→0.5)
    jepa_global_weight: float = 0.5
    jepa_n_reg_tokens: int = 4
    jepa_ratio_start: float = 0.3
    jepa_ratio_end: float = 0.5
    # JEPA v3(2026-06-11,set 级,全盘抄 GeneJEPA)
    jepa_setlevel: bool = False
    jepa_loss_type: str = "cosine"
    jepa_ema_end: float = 0.9995
    jepa_ema_warmup_steps: int = 0
    jepa_student_vicreg_weight: float = 0.0
    jepa_predict_residual: bool = False

    # 去批次(2026-06-08;study=Project_ID;默认全关=与现状等价):
    #   study_balanced         : train 用 StudyBalancedBatchSampler(每 batch 同一 study,CONCORD 对比用)
    #   use_study_conditioning : 条件 MLM 头(study_embed[study_id] 只进重建头,scVI 式);透传 dm.n_studies
    #   study_min_size         : >= 此样本数的 study 各占一 id,小尾巴并 UNK(0)
    study_balanced: bool = False
    use_study_conditioning: bool = False
    study_min_size: int = 64

    # 2.1. 模型主体参数
    d_model: int = 512          # 2026-06 对齐主线(旧默认 256;模型规模漏传后果严重故对齐 default)
    nhead: int = 16             # 旧默认 8
    num_layers: int = 12        # 旧默认 6
    # ff_dim 与 ff_ratio 互斥：指定其中一个，另一个保持 None
    ff_dim: int | None = None      # FeedForward 绝对维度
    ff_ratio: int | None = None    # FeedForward 比例（dim_ff = d_model × ff_ratio）；未指定时走默认 4
    num_abundance_bins: int = 40

    # 2.2. 模型主体参数的协议参数
    abundance_mode: str = "abs_log_bins"
    min_abundance: float = 4e-6
    max_seq_len: int = 512

    # 3.1. 预训练中的训练主体参数
    batch_size: int = 32
    mask_prob: float = 0.15
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.02

    # 3.2. 预训练中的协议参数
    lr_scheduler_type: str = "cosine"
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 2
    lr_plateau_min_lr: float = 1e-6

    # 4. 预算与验证协议参数
    budget_mode: str = "epoch"
    max_epochs: int | None = None
    max_steps: int | None = None
    val_interval_epochs: int | None = None
    val_interval_steps: int | None = None
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0

    # 4.1. Early stopping（0=禁用，与 finetune 对称）
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    # ModelCheckpoint 保留数：-1=保存每次验证的 ckpt（长训练"回头看"用），>0=只留最优 K 个
    save_top_k: int = 3

    # 5. 运行与工程参数
    devices: int = 1
    num_nodes: int = 1                 # 多节点 DDP 节点数(单节点保持 1)
    # 多卡 DDP find_unused_parameters 开关(默认 False=现状最快:所有参数都参与 forward)。
    # 仅当某配置下有参数天然不参与 loss 时打开 —— 如纯 MLM 对照(phylo_w=protein_w=0)
    # 关掉 mask_token_id_replace → genus_mask_token 闲置 → DDP reducer 报 unused-parameter。
    # 打开只影响 DDP 梯度同步记账,不改 forward/loss/学习,数值等价。
    ddp_find_unused_parameters: bool = False
    precision: str = "auto"
    seed: int = 42
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    num_workers: int = 4
    log_dir: str = "tmp/logs"
    no_progress_bar: bool = False
    # 激活重算(以时间换显存,留给未来对比学习);默认关,本次正式训练不用
    grad_checkpointing: bool = False

    # ============== V5 新增 ==============
    abundance_encoding: str = "mlp"             # "mlp" | "bin"
    abundance_value_transform: str = "rclr_sigma"  # V5 §4.2 present-only 写法（编码消融,见 datasets._VALID_VALUE_TRANSFORM）
    abundance_loss: str = "huber"               # "huber" | "bin_ce"
    use_phylo_pe: bool = True
    phylo_pe_hidden: int = 128
    pooling_mode: str = "pma"                   # "pma" | "mean_pool"
    pma_nhead: int = 4
    pma_k: int = 1
    use_metadata_task: bool = True
    metadata_loss_weight: float = 0.3
    metadata_num_classes: int = 6
    huber_beta: float = 1.0
    # 验证 val 监控 loss 名称(V5 默认 val/loss,跟现有 ModelCheckpoint 一致)
    val_monitor: str = "val/loss"

    # DAPT/续训用:从已有 ckpt 加载初始 state_dict(非 trainer resume,仅初始化权重)
    # None=正常从头训练;指定路径=load_state_dict(strict=False),允许 buffer 缺失
    init_from_ckpt: str | None = None


# 执行一次完整的预训练流程，返回结果字典
def run_pretrain_once(
    config: PretrainRunConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    log_subdir: str = "pretrain_stage0",
) -> dict[str, object]:
    validate_pretrain_config(config)
    L.seed_everything(config.seed, workers=True)

    # 索引越界 sanity check：避免使用了和 h5ad 不匹配的 splits 文件
    n_obs = validate_index_arrays(
        config.h5ad_path,
        train=train_indices,
        val=val_indices,
    )
    validate_no_split_overlap(train=train_indices, val=val_indices)

    # 计算有效的 dim_feedforward（ff_ratio 或 ff_dim 二选一）
    ff_dim, ff_ratio = resolve_pretrain_ff_params(config)
    if ff_dim is not None:
        effective_ff_dim = ff_dim
    else:
        effective_ff_dim = config.d_model * ff_ratio  # type: ignore[operator]

    chosen_precision = choose_precision(config.precision)
    rank_zero_info(f"{TAG} Using precision={chosen_precision}")
    rank_zero_info(f"{TAG} Train: {len(train_indices)}, Val: {len(val_indices)} (n_obs={n_obs})")
    rank_zero_info(f"{TAG} Budget mode: {config.budget_mode}")
    if config.budget_mode == "epoch":
        rank_zero_info(
            f"{TAG} Max epochs: {config.max_epochs}, "
            f"val_interval_epochs={config.val_interval_epochs}"
        )
    else:
        rank_zero_info(
            f"{TAG} Max steps: {config.max_steps}, "
            f"val_interval_steps={config.val_interval_steps}"
        )
    rank_zero_info(
        f"{TAG} LR scheduler: {config.lr_scheduler_type}, "
        f"warmup_ratio={config.warmup_ratio}"
    )

    # 1. 初始化数据模块
    rank_zero_info(f"{TAG} Initializing DataModule...")
    dm = MiCoDataModule(
        h5ad_path=config.h5ad_path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=None,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_seq_len=config.max_seq_len,
        mask_prob=config.mask_prob,
        num_abundance_bins=config.num_abundance_bins,
        min_abundance=config.min_abundance,
        abundance_mode=config.abundance_mode,
        # V5
        abundance_encoding=config.abundance_encoding,
        abundance_value_transform=config.abundance_value_transform,
        use_metadata_task=config.use_metadata_task,
        # 去批次(默认关)
        study_balanced=config.study_balanced,
        use_study_conditioning=config.use_study_conditioning,
        study_min_size=config.study_min_size,
    )

    # 2. 初始化模型
    rank_zero_info(
        f"{TAG} Initializing Model with d_model={config.d_model}, "
        f"layers={config.num_layers}, ff_dim={effective_ff_dim}"
    )
    rank_zero_info(
        f"{TAG} V5 flags: abundance_encoding={config.abundance_encoding} (loss={config.abundance_loss}), "
        f"use_phylo_pe={config.use_phylo_pe}, pooling={config.pooling_mode}, "
        f"use_metadata_task={config.use_metadata_task} (λ={config.metadata_loss_weight})"
    )
    # bias_type != 'none' 时把 var 表大小传给模型（占位 dist_matrix buffer 用），
    # 同时从 datamodule 拿对应的距离矩阵（phylo 或 taxo），注入 encoder
    _n_vars = 0
    _dist_matrix_to_inject = None
    # 2026-05-29:phylo_ce_weight 或 phylo_w_weight > 0 也需要 dist_matrix(即使 bias_type='none')
    # 优先级:taxo bias → taxo_dist;phylo bias / phylo_ce / phylo_w → phylo_dist
    _need_dist = (
        config.bias_type != "none"
        or config.phylo_ce_weight > 0
        or config.phylo_w_weight > 0
    )
    if _need_dist:
        if config.bias_type == "taxo":
            _dist_matrix_to_inject = dm.taxo_dist_matrix
            _dist_source = "taxo (bias)"
        else:
            # phylo bias OR phylo_ce(bias_type='none')都用 phylo_dist
            _dist_matrix_to_inject = dm.phylo_dist_matrix
            _dist_source = (
                "phylo (bias)" if config.bias_type == "phylo"
                else "phylo (for phylo_ce loss, bias_type=none)"
            )
        if _dist_matrix_to_inject is None:
            raise RuntimeError(
                f"need dist_matrix ({_dist_source}) but DataModule did not load it. "
                f"Check that MCFCorpus has the corresponding varp key."
            )
        _n_vars = int(_dist_matrix_to_inject.shape[0])
        rank_zero_info(
            f"{TAG} dist_matrix source={_dist_source}, n_vars={_n_vars}, "
            f"dtype={_dist_matrix_to_inject.dtype}"
        )

    # PE coords 检查(use_phylo_pe=True 时必须有 varm['position_encoding'])
    _pe_coords_to_inject = None
    _pe_dim = None
    if config.use_phylo_pe:
        if dm.phylo_pe_coords_raw is None:
            raise RuntimeError(
                "use_phylo_pe=True requires varm['position_encoding'] in h5ad, "
                "but DataModule did not load it."
            )
        _pe_coords_to_inject = dm.phylo_pe_coords_raw
        _pe_dim = dm.pe_dim
        rank_zero_info(
            f"{TAG} PhyloPE: pe_dim={_pe_dim}, coords shape={tuple(_pe_coords_to_inject.shape)}"
        )

    # Protein PE coords 检查(X2 phase 2:use_protein_pe=True 时必须有 protein_feat)
    # 优先用外部 protein_pe_path(.npy);否则回落到语料 varm['protein_pe']
    _protein_pe_coords_to_inject = None
    _protein_pe_dim = None
    if config.use_protein_pe:
        if config.protein_pe_path is not None:
            _ppe_arr = np.load(config.protein_pe_path).astype(np.float32)
            _protein_pe_coords_to_inject = torch.from_numpy(_ppe_arr)
            rank_zero_info(
                f"{TAG} ProteinPE: loaded from external npy {config.protein_pe_path}, "
                f"shape={tuple(_protein_pe_coords_to_inject.shape)}"
            )
        elif getattr(dm, "protein_pe_coords_raw", None) is not None:
            _protein_pe_coords_to_inject = dm.protein_pe_coords_raw
            rank_zero_info(
                f"{TAG} ProteinPE: loaded from varm['protein_pe'], "
                f"shape={tuple(_protein_pe_coords_to_inject.shape)}"
            )
        else:
            raise RuntimeError(
                "use_protein_pe=True requires protein_pe_path (.npy) or varm['protein_pe'] in h5ad, "
                "but neither was provided (likely bacformer_prior pipeline 未完成)."
            )
        _protein_pe_dim = int(_protein_pe_coords_to_inject.shape[1])
        rank_zero_info(f"{TAG} ProteinPE: pe_dim={_protein_pe_dim}")

    # Protein dist matrix 检查(protein_w_weight>0 时必须有 protein_dist_path)
    _protein_dist_to_inject = None
    if config.protein_w_weight > 0:
        if config.protein_dist_path is None:
            raise RuntimeError(
                "protein_w_weight > 0 requires protein_dist_path (.npy [V_real, V_real] float32), "
                "but it was not provided."
            )
        _pd_arr = np.load(config.protein_dist_path).astype(np.float32)
        _protein_dist_to_inject = torch.from_numpy(_pd_arr)
        # protein_w 也需要 n_vars > 0(创建 protein_dist_matrix buffer)
        if _n_vars == 0:
            _n_vars = int(_protein_dist_to_inject.shape[0])
        rank_zero_info(
            f"{TAG} protein_dist: loaded from {config.protein_dist_path}, "
            f"shape={tuple(_protein_dist_to_inject.shape)}, n_vars={_n_vars}"
        )

    # Metadata class weights
    _meta_weights_list = None
    if config.use_metadata_task and dm.env_class_weights is not None:
        _meta_weights_list = dm.env_class_weights.tolist()
        rank_zero_info(
            f"{TAG} EnvCategory class weights (sqrt-smoothed): "
            f"{[f'{w:.3f}' for w in _meta_weights_list]}"
        )

    model = MiCoFormerModule(
        genus_vocab_size=dm.genus_vocab_size,
        total_abundance_bins=dm.total_abundance_bins,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=effective_ff_dim,
        dropout=config.dropout,
        pad_taxon_id=dm.special_ids["pad_taxon_id"],
        pad_bin_id=dm.special_ids["pad_bin_id"],
        rank_vocab_sizes=dm.rank_vocab_sizes,
        bias_type=config.bias_type,
        phylo_mlp_hidden=config.phylo_mlp_hidden,
        phylo_bias_last_layer_bias=config.phylo_bias_last_layer_bias,
        # Tree loss
        tree_loss_weight=config.tree_loss_weight,
        tree_n_pairs=config.tree_n_pairs,
        tree_n_triplets=config.tree_n_triplets,
        tree_margin=config.tree_margin,
        # X2 多任务(2026-05-28 夜)
        mlm_weight=config.mlm_weight,
        x2_phylo_weight=config.x2_phylo_weight,
        x2_protein_weight=config.x2_protein_weight,
        x2_head_hidden=config.x2_head_hidden,
        use_protein_pe=config.use_protein_pe,
        protein_pe_hidden=config.protein_pe_hidden,
        protein_pe_dim=_protein_pe_dim,
        # Phylo Soft-Target CE(2026-05-29)
        phylo_ce_weight=config.phylo_ce_weight,
        phylo_ce_tau=config.phylo_ce_tau,
        # Phylo Tree-Wasserstein simplified(2026-05-29 phase 2)
        phylo_w_weight=config.phylo_w_weight,
        # Protein Tree-Wasserstein simplified(2026-05-30,phylo_w 镜像)
        protein_w_weight=config.protein_w_weight,
        # 对比学习(2026-06-04,InfoNCE)
        contrastive_weight=config.contrastive_weight,
        contrastive_temp=config.contrastive_temp,
        contrastive_proj_dim=config.contrastive_proj_dim,
        contrastive_mask_prob=config.contrastive_mask_prob,
        # JEPA(2026-06-04,潜空间预测)
        jepa_weight=config.jepa_weight,
        jepa_mask_ratio=config.jepa_mask_ratio,
        jepa_mlm_mask_prob=config.jepa_mlm_mask_prob,
        jepa_ema_decay=config.jepa_ema_decay,
        jepa_pred_dim=config.jepa_pred_dim,
        jepa_pred_depth=config.jepa_pred_depth,
        jepa_pred_heads=config.jepa_pred_heads,
        jepa_vicreg_weight=config.jepa_vicreg_weight,
        jepa_mask_mode=config.jepa_mask_mode,
        jepa_addr_mode=config.jepa_addr_mode,
        jepa_n_seeds=config.jepa_n_seeds,
        # JEPA v2(2026-06-06)
        jepa_global_weight=config.jepa_global_weight,
        jepa_n_reg_tokens=config.jepa_n_reg_tokens,
        jepa_ratio_start=config.jepa_ratio_start,
        jepa_ratio_end=config.jepa_ratio_end,
        # JEPA v3(2026-06-11,set 级)
        jepa_setlevel=config.jepa_setlevel,
        jepa_loss_type=config.jepa_loss_type,
        jepa_ema_end=config.jepa_ema_end,
        jepa_ema_warmup_steps=config.jepa_ema_warmup_steps,
        jepa_student_vicreg_weight=config.jepa_student_vicreg_weight,
        jepa_predict_residual=config.jepa_predict_residual,
        # 去批次条件 MLM(2026-06-08;n_studies 由 DataModule 派生后透传)
        use_study_conditioning=config.use_study_conditioning,
        n_studies=dm.n_studies,
        n_vars=_n_vars,
        # V5
        abundance_encoding=config.abundance_encoding,
        abundance_loss=config.abundance_loss,
        use_phylo_pe=config.use_phylo_pe,
        phylo_pe_hidden=config.phylo_pe_hidden,
        pe_dim=_pe_dim,
        grad_checkpointing=config.grad_checkpointing,
        pooling_mode=config.pooling_mode,
        pma_nhead=config.pma_nhead,
        pma_k=config.pma_k,
        use_metadata_task=config.use_metadata_task,
        metadata_loss_weight=config.metadata_loss_weight,
        metadata_num_classes=config.metadata_num_classes,
        metadata_class_weights=_meta_weights_list,
        huber_beta=config.huber_beta,
        # 优化器
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler=config.lr_scheduler_type,
        plateau_factor=config.lr_plateau_factor,
        plateau_patience=config.lr_plateau_patience,
        plateau_min_lr=config.lr_plateau_min_lr,
    )

    # 注入 var-level buffer:dist_matrix(R2) + phylo_pe coords(V5) + protein_pe coords(X2 phase 2)
    #   + protein_dist_matrix(protein_w loss,2026-05-30)
    inject_var_buffers(
        model.encoder,
        _dist_matrix_to_inject,
        _pe_coords_to_inject,
        _protein_pe_coords_to_inject,
        protein_dist_matrix=_protein_dist_to_inject,
    )

    # Tree loss helper 需要 phylo_dist 引用;在 encoder 注入之后立刻挂上
    # (用同一对象,避免重复占 263MB 显存)
    if model.tree_loss_helper is not None:
        if model.encoder.dist_matrix is None or not model.encoder._dist_matrix_loaded:
            raise RuntimeError(
                "tree_loss_weight>0 requires encoder.dist_matrix to be loaded; "
                "ensure bias_type='phylo' + inject_var_buffers ran successfully."
            )
        model.tree_loss_helper.set_phylo_dist(model.encoder.dist_matrix)
        rank_zero_info(
            f"{TAG} TreeLossHelper attached: weight={config.tree_loss_weight}, "
            f"n_pairs={config.tree_n_pairs}, n_triplets={config.tree_n_triplets}, "
            f"margin={config.tree_margin}"
        )

    # DAPT 续训:在 buffer 注入之后加载 ckpt 的 state_dict(strict=False 允许 non-persistent buffer 缺失)
    if config.init_from_ckpt:
        rank_zero_info(f"{TAG} Loading initial weights from {config.init_from_ckpt}")
        _state = torch.load(config.init_from_ckpt, map_location="cpu", weights_only=False)
        _sd = _state.get("state_dict", _state)
        _incompat = model.load_state_dict(_sd, strict=False)
        # 报告 missing/unexpected 帮助诊断 V5 架构兼容性
        _miss = [k for k in _incompat.missing_keys
                 if not (k.endswith(".coords") or k.endswith(".dist_matrix"))]
        _unexp = list(_incompat.unexpected_keys)
        rank_zero_info(
            f"{TAG} load_state_dict: missing={len(_miss)} unexpected={len(_unexp)} "
            f"(non-persistent buffers excluded)"
        )
        if _miss:
            rank_zero_info(f"{TAG}   first missing keys: {_miss[:5]}")
        if _unexp:
            rank_zero_info(f"{TAG}   first unexpected keys: {_unexp[:5]}")

    # 3. 设置日志记录器与回调
    # DDP subprocess 下各 rank 各自重跑本脚本,run_version 必须确定性生成
    # (不能含 uuid / 秒级时间,否则各 rank 算出不同目录 → 日志/ckpt 分裂)。
    # 指纹基于影响本次 run 的关键 config;所有 rank 拿到相同 config → 必然一致。
    # 同 config 重跑会落到同一目录(可接受;Lightning 会续写/覆盖)。
    _fp_src = "|".join(
        [
            log_subdir,
            str(config.seed),
            os.path.abspath(config.h5ad_path),
            config.budget_mode,
            str(config.max_epochs),
            str(config.max_steps),
            str(config.batch_size),
            str(config.devices),
            str(config.accumulate_grad_batches),
            str(config.lr),
            config.abundance_value_transform,  # §4.2:不同写法的 run 落不同目录,避免互相覆盖
        ]
    )
    _fp = hashlib.md5(_fp_src.encode("utf-8")).hexdigest()[:10]
    run_version = f"run_{time.strftime('%Y%m%d')}_{_fp}"
    csv_logger = CSVLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)

    # 显式指定 dirpath，避免依赖 "第一个 logger 的 save_dir" 这种隐式行为
    ckpt_dir = os.path.join(config.log_dir, log_subdir, run_version, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=config.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
        filename="micoformer-epoch{epoch:02d}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
        )

    # 4. 初始化 Trainer
    _use_gpu = torch.cuda.is_available()
    trainer_kwargs = dict(
        accelerator="gpu" if _use_gpu else "cpu",
        devices=config.devices,
        num_nodes=config.num_nodes,
        precision=chosen_precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        default_root_dir=config.log_dir,
    )
    # 多卡 DDP:显式 DDPStrategy(subprocess launcher,非 ddp_spawn)。
    #  - subprocess 让每 rank 重跑 run_pretrain_once → 各自本地 inject_var_buffers,
    #    避免 spawn 跨进程 pickle 263MB dist_matrix buffer。
    #  - broadcast_buffers=False:dist_matrix / phylo_pe.coords / _meta_class_weights 都是
    #    persistent=False 的冻结 buffer,各 rank 已本地注入/重建相同值;否则 DDP 每步
    #    broadcast 263MB 走 PCIe 严重拖慢。
    #  - find_unused_parameters=False:abund_embed 已改条件创建 + sample_embed 已删 →
    #    默认/各 flag 组合下所有参数都参与 forward,可关 find_unused。
    #    ⚠️ 仅 CPU 验证过,首次多卡 DDP 跑若报 unused-parameter 错,改回 True 并排查。
    if _use_gpu and config.devices > 1:
        trainer_kwargs["strategy"] = DDPStrategy(
            broadcast_buffers=False,
            find_unused_parameters=config.ddp_find_unused_parameters,
        )
    if config.budget_mode == "epoch":
        trainer_kwargs["max_epochs"] = config.max_epochs
        trainer_kwargs["check_val_every_n_epoch"] = config.val_interval_epochs
    else:
        trainer_kwargs["max_epochs"] = -1
        trainer_kwargs["max_steps"] = config.max_steps
        trainer_kwargs["val_check_interval"] = config.val_interval_steps
    if config.no_progress_bar:
        trainer_kwargs["enable_progress_bar"] = False

    trainer = L.Trainer(**trainer_kwargs)

    # 5. 开始训练
    rank_zero_info(f"{TAG} Starting training...")
    trainer.fit(model, datamodule=dm)

    best_score = checkpoint_callback.best_model_score
    best_model_path = checkpoint_callback.best_model_path
    val_metrics: dict[str, object] = {}

    # 当最终 epoch 不是固定验证周期的整数倍时，补一次显式验证，避免漏掉 epoch 20。
    if (
        config.budget_mode == "epoch"
        and config.val_interval_epochs is not None
        and config.max_epochs is not None
        and config.max_epochs % config.val_interval_epochs != 0
    ):
        rank_zero_info(
            f"{TAG} Final epoch {config.max_epochs} is not aligned with "
            f"val_interval_epochs={config.val_interval_epochs}; running one final validation."
        )
        final_validate = trainer.validate(model, datamodule=dm, verbose=False)
        if final_validate:
            val_metrics = final_validate[0]
            final_val_loss_raw = val_metrics.get("val/loss")
            if final_val_loss_raw is not None:
                final_val_loss = (
                    final_val_loss_raw.item()
                    if hasattr(final_val_loss_raw, "item")
                    else float(final_val_loss_raw)
                )
                current_best = best_score.item() if best_score is not None else None
                if current_best is None or final_val_loss < current_best:
                    best_model_path = os.path.join(ckpt_dir, "micoformer-final-validated.ckpt")
                    trainer.save_checkpoint(best_model_path)
                    best_score = None
                    rank_zero_info(
                        f"{TAG} Final validation improved best val/loss to {final_val_loss:.6f}; "
                        f"saved checkpoint to {best_model_path}"
                    )
                    return {
                        "best_model_path": best_model_path,
                        "best_score": final_val_loss,
                        "best_val_loss": final_val_loss,
                        "val_metrics": val_metrics,
                        "test_metrics": None,
                    }

    return {
        "best_model_path": best_model_path,
        "best_score": best_score.item() if best_score is not None else None,
        "best_val_loss": best_score.item() if best_score is not None else None,  # 向后兼容别名
        "val_metrics": val_metrics,
        "test_metrics": None,
    }
