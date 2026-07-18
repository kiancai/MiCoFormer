from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from micoformer.models.encoder import MiCoFormerEncoder
from micoformer.models.jepa import JEPAPredictor
from micoformer.models.heads import (
    AbundanceBinHead,
    AbundanceRegressionHead,
    MetadataHead,
    PriorCoordHead,
    SampleViewLinearHead,
)
from micoformer.models.pma import PMA
from micoformer.utils.train_utils import build_lr_scheduler
from micoformer.utils.tree_loss import TreeLossHelper


_VALID_ABUNDANCE_LOSS = {"huber", "bin_ce"}
_VALID_POOLING_MODE = {"pma", "mean_pool"}
_VALID_SAMPLE_VIEW_TARGETS = {"raw", "rclr_sigma", "rank", "func_bacformer", "phylo_32coord"}
_SAMPLE_VIEW_ALIASES = {
    "rclr": "rclr_sigma",
    "func": "func_bacformer",
    "phylo": "phylo_32coord",
}


def _normalize_sample_view_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [x.strip() for x in value.replace(",", " ").split()]
    else:
        raw = [str(x).strip() for x in value]
    out: list[str] = []
    for name in raw:
        if not name:
            continue
        canonical = _SAMPLE_VIEW_ALIASES.get(name, name)
        if canonical not in _VALID_SAMPLE_VIEW_TARGETS:
            raise ValueError(
                f"Unknown sample-view target {name!r}. Expected one of {sorted(_VALID_SAMPLE_VIEW_TARGETS)}."
            )
        if canonical not in out:
            out.append(canonical)
    return out


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
        sample_view_heads: Optional[List[str]] = None,
        sample_view_loss_weight: float = 0.0,
        sample_view_loss_weights: Optional[List[float]] = None,
        sample_view_n_vars: Optional[int] = None,
        sample_view_func_dim: Optional[int] = None,
        sample_view_phylo_dim: Optional[int] = None,
        sample_view_diversity_weight: float = 1e-3,
        sample_view_close_weight: float = 1e-3,
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
        # 2026-05-29 phase 1 实测 phylo_ce 在 ep0 就达到 H(target)≈2.10 数学下界 → saturate
        phylo_ce_weight: float = 0.0,
        phylo_ce_tau: float = 6.5,
        # Phylo Tree-Wasserstein simplified(2026-05-29 phase 2,替代 phylo-soft-CE):
        #   phylo_w_weight: Wasserstein-1 expected distance loss 权重(0 关掉)
        # 数学:当 target 是 one-hot at v* 时,W(p, δ_v*) = E_{v~p}[d(v, v*)] = Σ p(v) × d(v, v*)
        #     即 expected phylo distance loss = strict Tree-Wasserstein W-1 closed form
        #     (Yamada EACL 2021 / Le NeurIPS 2019)
        # 优点 vs phylo_ce:① 无 hyperparameter τ ② hard target floor=0 不 saturate
        #     ③ implementation 1 行(复用 dist_matrix buffer,无需 newick parse)
        # 要求:跟 phylo_ce 同样需 n_vars > 0;两者互斥(同时>0 会触发 warning 但允许)
        phylo_w_weight: float = 0.0,
        # Protein Tree-Wasserstein simplified(2026-05-30,phylo_w 的精确镜像):
        #   protein_w_weight: W-1 expected protein distance loss 权重(0 关掉)
        # 数学:loss = E_{v~p}[protein_dist(v, v*)] = Σ p(v) × protein_dist(v, v*)
        #     与 phylo_w 完全同构,只把 phylo patristic 距离换成蛋白距离矩阵 protein_dist。
        # 要求:need_protein_dist=True + n_vars > 0(encoder.protein_dist_matrix buffer 必须创建);
        #     workflow 须 inject 真实 protein_dist_matrix。与 phylo_w / phylo_ce 共享 vocab_head。
        protein_w_weight: float = 0.0,
        # 对比学习(2026-06-04,InfoNCE,保留 MLM 锚;详见 _shared_step):
        #   contrastive_weight   : InfoNCE 权重(0 关,>0 开;典型 0.1-0.5)
        #   contrastive_temp     : NT-Xent 温度 τ(典型 0.1)
        #   contrastive_proj_dim : projection head 输出维度(典型 128)
        #   contrastive_mask_prob: 第二视图 abund-mask 比例(默认 0.15,同 MLM)
        # 机制:同一样本两套不同 abund-mask → 两个 mean-pool 表征 → proj → InfoNCE,
        #   batch 内其他样本为负样本。mean-pool 不依赖 PMA(no_metadata/warm-start 兼容)。
        contrastive_weight: float = 0.0,
        contrastive_temp: float = 0.1,
        contrastive_proj_dim: int = 128,
        contrastive_mask_prob: float = 0.15,
        # JEPA(2026-06-04,潜空间预测;详见 _shared_step + on_train_batch_end + PLAN.md):
        #   jepa_weight        : JEPA latent-prediction loss 权重(0 关,>0 开;典型 1.0)
        #   jepa_mask_ratio    : target token 比例(从 context 移除、由 predictor 预测;典型 0.5)
        #   jepa_mlm_mask_prob : MLM 锚的 abund-mask 比例(在 context 内选;防塌锚,典型 0.15)
        #   jepa_ema_decay     : target encoder EMA 起步衰减(0.996,训练中 linear ramp→1)
        #   jepa_pred_dim/depth/heads : 窄 bottleneck predictor 配置(典型 256 / 2 / 4)
        #   jepa_vicreg_weight : VICReg variance 防塌正则权重(后备;起步 0,塌了抬,见 PLAN 防塌段)
        # 机制:context encoder 看可见 genus(target 屏蔽)→ h_ctx;EMA target encoder 看完整
        #   样本 → 含义向量(LN 后、stop-grad);predictor 用被遮 genus 的 phylo/protein 坐标当
        #   地址 query 预测其含义向量。MLM 当锚一起训(scJEPA:recon+latent > latent-alone)。
        # 红线:坐标只当地址(query 输入),target 是含义向量、绝非坐标本身(否则退化成 X2_phylo)。
        # 与 metadata/contrastive/x2/phylo_w 互斥使用(jepa baseline = 纯 MLM+JEPA)。
        jepa_weight: float = 0.0,
        jepa_mask_ratio: float = 0.5,
        jepa_mlm_mask_prob: float = 0.15,
        jepa_ema_decay: float = 0.996,
        jepa_pred_dim: int = 256,
        jepa_pred_depth: int = 2,
        jepa_pred_heads: int = 4,
        jepa_vicreg_weight: float = 0.0,
        # 结构化 mask(2026-06-04 讨论;v2 2026-06-06 改 default structured + n_seeds 4):
        #   jepa_mask_mode : 'random'(随机遮 ~ratio)| 'structured'(按 phylo/protein 坐标成簇遮)
        #   jepa_n_seeds   : structured 时多少个种子簇(I-JEPA multi-block;每簇遮 ratio/n_seeds 最近邻)
        # structured 糙版=样本内现算坐标距离、多种子各遮最近一撮、phylo/protein 每 batch 随机交替。
        # 红线:坐标当"出题人/脚手架"(决定遮哪片),非被拟合的答案。
        jepa_mask_mode: str = "structured",
        jepa_n_seeds: int = 4,
        # address query 用什么定位被遮 genus(2026-06-09):
        #   'coords' = phylo/protein 坐标(历史;已证 phylo/protein 是"错的图"——进化/序列 ≠ 行为共变)
        #   'genus'  = 被遮菌的 genus embedding 身份(Cell-JEPA 式;吃数据驱动的菌间共变结构,不用错图)
        jepa_addr_mode: str = "coords",
        # ============ JEPA v2(2026-06-06,删 MLM + 双自监督 + 防塌升级)============
        # 依据 Point-JEPA(无序点云+坐标,最同构)/ I-JEPA / T-JEPA。删 MLM 锚后换 3 重防塌:
        #   ① global align(teacher PMA 池化 + LN + EMA detach,样本级潜空间预测)
        #   ② register token(T-JEPA,在 encoder 内;n_reg_tokens 透传给 encoder)
        #   ③ VICReg variance+covariance(补全,作用在 predictor 输出)
        # jepa_global_weight : 全局对齐 loss 权重(student PMA vs teacher PMA,默认 0.5);>0 才建 head
        # jepa_n_reg_tokens  : encoder 前缀 register token 数(默认 4;透传 encoder.n_reg_tokens)
        # jepa_ratio_start/end: structured mask ratio curriculum,按 current_epoch 线性 start→end
        #                       (默认 0.3→0.5;拿不到 max_epochs 退回固定 jepa_mask_ratio)
        jepa_global_weight: float = 0.5,
        jepa_n_reg_tokens: int = 4,
        jepa_ratio_start: float = 0.3,
        jepa_ratio_end: float = 0.5,
        # ============ JEPA v3(2026-06-11,全盘抄 GeneJEPA set 级范式)============
        # jepa_setlevel : 开启纯 set 级 JEPA(GeneJEPA 式) —— student 看 context→PMA→z_s;
        #   teacher(EMA)**只看 target 子集**→PMA→z_t(detach+LN);global_predictor(z_s) 对齐 z_t。
        #   砍 token 级 predictor 路径(v1/v2 token-JEPA 已证失败);jepa_weight 当主 loss 权重。
        # jepa_loss_type : 'cosine'(GeneJEPA 式,默认)| 'mse'(I-JEPA 式)
        # jepa_ema_end   : EMA cosine 调度终点(GeneJEPA 0.9995;起点用 jepa_ema_decay=0.996)
        # jepa_ema_warmup_steps : 前 N 步 teacher 冻结不更新(GeneJEPA=2000;0=不 warmup)
        # jepa_student_vicreg_weight : student z_s 的 VICReg 权重(GeneJEPA 在 student_ctx 也加一遍防塌)
        jepa_setlevel: bool = False,
        jepa_loss_type: str = "cosine",
        jepa_ema_end: float = 0.9995,
        jepa_ema_warmup_steps: int = 0,
        jepa_student_vicreg_weight: float = 0.0,
        # jepa_predict_residual : token 级 JEPA target 减样本全局中心(predict residual,2026-06-11)
        #   逼模型预测"被遮菌相对样本整体的偏差"——铲掉"看整体组成"捷径(ep0/ep1 诊断坐实模型停均值档),
        #   残差只能靠相关邻居预测 = 强制学菌间共变(verify_sigma 证实结构存在但 JEPA 没学到)。
        jepa_predict_residual: bool = False,
        # ============ 去批次条件 MLM(2026-06-08;study = Project_ID)============
        #   use_study_conditioning: 重建头额外加 study_embed[study_id](encoder/PMA 输出**不给** study →
        #     逼样本级表征不必承载批次,scVI 式条件解码;study_embed 零初始化 = 起步等价纯 MLM、渐进涌现)
        #   n_studies: study 词表大小(含 UNK=0;由 DataModule.n_studies 透传)
        use_study_conditioning: bool = False,
        n_studies: int = 0,
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

        sample_view_names = _normalize_sample_view_names(sample_view_heads)
        sample_view_active = bool(sample_view_names) and float(sample_view_loss_weight) > 0
        if sample_view_names:
            if pooling_mode != "pma":
                raise ValueError("sample_view_heads requires pooling_mode='pma'.")
            if pma_k != len(sample_view_names):
                raise ValueError(
                    "sample_view_heads uses one independent PMA seed per view; "
                    f"pma_k must equal len(sample_view_heads)={len(sample_view_names)}, got {pma_k}."
                )
            if any(v in sample_view_names for v in ("raw", "rclr_sigma", "rank")) and (
                sample_view_n_vars is None or sample_view_n_vars <= 0
            ):
                raise ValueError("raw/rclr_sigma/rank sample views require sample_view_n_vars > 0.")
            if "func_bacformer" in sample_view_names and (
                sample_view_func_dim is None or sample_view_func_dim <= 0
            ):
                raise ValueError("func_bacformer sample view requires sample_view_func_dim > 0.")
            if "phylo_32coord" in sample_view_names and (
                sample_view_phylo_dim is None or sample_view_phylo_dim <= 0
            ):
                raise ValueError("phylo_32coord sample view requires sample_view_phylo_dim > 0.")
            if sample_view_loss_weights is not None and len(sample_view_loss_weights) != len(sample_view_names):
                raise ValueError(
                    "sample_view_loss_weights length must match sample_view_heads length: "
                    f"{len(sample_view_loss_weights)} != {len(sample_view_names)}."
                )

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
            or (tree_loss_weight > 0) or (phylo_ce_weight > 0) or (phylo_w_weight > 0)
            or (protein_w_weight > 0) or (jepa_weight > 0) or sample_view_active
        )
        if not _any_loss:
            raise ValueError(
                "All training losses are off (mlm/x2_phylo/x2_protein/tree_loss/phylo_ce/phylo_w/protein_w all=0). "
                "At least one loss weight must be > 0."
            )

        # Phylo-soft-CE / Phylo-W 都需要 dist_matrix buffer + n_vars
        _need_dist = (phylo_ce_weight > 0) or (phylo_w_weight > 0)
        if _need_dist and (n_vars is None or n_vars <= 0):
            raise ValueError(
                "phylo_ce_weight > 0 or phylo_w_weight > 0 requires n_vars > 0 "
                "to allocate encoder.dist_matrix buffer (workflow 必须 inject 真实 dist_matrix)."
            )

        # Protein-W 需要 protein_dist_matrix buffer + n_vars(镜像 phylo_w)
        if protein_w_weight > 0 and (n_vars is None or n_vars <= 0):
            raise ValueError(
                "protein_w_weight > 0 requires n_vars > 0 "
                "to allocate encoder.protein_dist_matrix buffer (workflow 必须 inject 真实 protein_dist)."
            )

        # 保存所有 __init__ 参数到 self.hparams,便于 checkpoint 保存和恢复
        self.save_hyperparameters()
        self.sample_view_names = sample_view_names

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
            # n_vars 需要在 bias_type!='none' 或 phylo_ce_weight/phylo_w_weight/protein_w_weight>0 时传入
            n_vars=n_vars if (bias_type != "none" or phylo_ce_weight > 0 or phylo_w_weight > 0 or protein_w_weight > 0) else None,
            # protein_w_weight>0 时创建 encoder.protein_dist_matrix buffer
            need_protein_dist=(protein_w_weight > 0),
            abundance_encoding=abundance_encoding,
            use_phylo_pe=use_phylo_pe,
            phylo_pe_hidden=phylo_pe_hidden,
            pe_dim=pe_dim,
            # X2 多任务:蛋白 PE 透传
            use_protein_pe=use_protein_pe,
            protein_pe_hidden=protein_pe_hidden,
            protein_pe_dim=protein_pe_dim,
            grad_checkpointing=grad_checkpointing,
            # JEPA v2:register token 仅 JEPA 开启时启用(n_reg_tokens=0 时 encoder 无此参数)
            n_reg_tokens=(jepa_n_reg_tokens if jepa_weight > 0 else 0),
        )

        # ============ MLM head ============
        if abundance_loss == "huber":
            self.mlm_head = AbundanceRegressionHead(d_model=d_model)
        else:
            self.mlm_head = AbundanceBinHead(d_model=d_model, num_bins=total_abundance_bins)
        self.bin_ce = nn.CrossEntropyLoss(reduction="none")  # bin 路径用
        self.huber_beta = huber_beta

        # ============ PMA(V5)或 mean_pool ============
        # 2026-05-29:加 and use_metadata_task 条件 — 之前 PMA 仅 use_metadata_task=True 时被调用,
        # 但 module __init__ 无条件创建,导致 phase 2 撤 metadata 时 PMA 是 1.1M dead params(不参与
        # forward 也不被 optimizer 更新,但占显存 + 进 ckpt 状态)。现修复让 PMA 跟 metadata 同生死。
        # 2026-06-06 JEPA v2:全局对齐(jepa_global_weight>0)也用 PMA 池化 student/teacher,
        #   故 JEPA global 开启时同样创建 PMA(MLM-free 范式下 use_metadata_task=False 也要有)。
        _pma_needed = (
            use_metadata_task
            or (jepa_weight > 0 and jepa_global_weight > 0)
            or (jepa_weight > 0 and jepa_setlevel)
        )
        self.pma: Optional[PMA] = None
        if pooling_mode == "pma" and _pma_needed:
            self.pma = PMA(d_model=d_model, nhead_pma=pma_nhead, k=1)

        # ============ sample-level multi-view shaping heads ============
        self.sample_view_pmas: Optional[nn.ModuleDict] = None
        self.sample_view_heads: Optional[nn.ModuleDict] = None
        self.sample_view_loss_weights: Optional[torch.Tensor] = None
        if sample_view_names:
            self.sample_view_pmas = nn.ModuleDict(
                {view: PMA(d_model=d_model, nhead_pma=pma_nhead, k=1) for view in sample_view_names}
            )
            dims = {
                "raw": int(sample_view_n_vars or 0),
                "rclr_sigma": int(sample_view_n_vars or 0),
                "rank": int(sample_view_n_vars or 0),
                "func_bacformer": int(sample_view_func_dim or 0),
                "phylo_32coord": int(sample_view_phylo_dim or 0),
            }
            self.sample_view_heads = nn.ModuleDict(
                {view: SampleViewLinearHead(d_model=d_model, out_dim=dims[view]) for view in sample_view_names}
            )
            weights = sample_view_loss_weights or [1.0 for _ in sample_view_names]
            self.register_buffer(
                "_sample_view_loss_weights",
                torch.tensor(weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer(
                "_sample_view_loss_weights",
                torch.ones(0, dtype=torch.float32),
                persistent=False,
            )

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

        # ============ Phylo Soft-Target CE / Tree-Wasserstein head(2026-05-29) ============
        # vocab_head: d_model → genus_vocab_size(含 PAD/UNK 共 V_real+2 dim)
        # phylo_ce: target 是 softmax(-dist/τ) soft 分布(KL loss)
        # phylo_w : target 是 one-hot,loss = E_{v~p}[d(v, v*)] (expected phylo distance, W-1 simplified)
        # 共享同一 vocab_head;weight=0 时不创建(避免 DDP find_unused)
        # protein_w 也共享同一 vocab_head(镜像 phylo_w)
        self.vocab_head: Optional[nn.Linear] = None
        if (phylo_ce_weight > 0) or (phylo_w_weight > 0) or (protein_w_weight > 0):
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

        # ============ 对比 projection head(2026-06-04,InfoNCE) ============
        # contrastive_weight=0 时不创建(避免 DDP find_unused)。SimCLR 式 2 层 MLP,
        # 接在 mean-pool 样本表征上(不走 PMA,no_metadata/warm-start 兼容)。
        self.contrastive_proj: Optional[nn.Module] = None
        if contrastive_weight > 0:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, contrastive_proj_dim),
            )

        # ============ JEPA target encoder(EMA)+ predictor(2026-06-04) ============
        # jepa_weight=0 时不创建(避免 DDP find_unused + 省显存)。
        # target_encoder = context encoder 的 deepcopy,requires_grad=False,由 EMA 更新
        #   (on_train_batch_end)。⚠️ deepcopy 发生在此处(coords 还是占位 0),inject_var_buffers
        #   只更新 self.encoder → 必须在 on_*_start 用 _sync_target_buffers() 把 coords 同步给
        #   target_encoder,否则 target forward 会因 PhyloPE._coords_loaded=False 直接 raise。
        self.target_encoder: Optional[MiCoFormerEncoder] = None
        self.jepa_predictor: Optional[JEPAPredictor] = None
        self._target_buffers_synced = False
        if jepa_weight > 0:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
        # token 级 predictor 仅非 setlevel(旧 v1/v2 token-JEPA)创建;setlevel 砍掉(避免 DDP unused)
        if jepa_weight > 0 and not jepa_setlevel:
            _addr_genus = (jepa_addr_mode == "genus")
            self.jepa_predictor = JEPAPredictor(
                d_model=d_model,
                pred_dim=jepa_pred_dim,
                depth=jepa_pred_depth,
                nhead=jepa_pred_heads,
                # genus 模式用身份地址,不建坐标投影(避免无用参数 / DDP unused)
                phylo_pe_dim=(pe_dim if (use_phylo_pe and not _addr_genus) else None),
                protein_pe_dim=(protein_pe_dim if (use_protein_pe and not _addr_genus) else None),
            )

        # ============ JEPA v2 全局对齐 predictor(2026-06-06) ============
        # 仅 jepa_global_weight>0 时创建(避免 DDP find_unused)。2 层 MLP 窄 bottleneck:
        # d_model → jepa_pred_dim → d_model。student PMA 池化向量过它,回归 teacher PMA 池化向量
        # (teacher 看完整样本、EMA detach)。红线:target 是 teacher 含义池化向量,非坐标。
        # 注意:全局对齐需要 PMA(student / teacher 都用 self.pma 池化);若 pma 不存在则 __init__ raise。
        self.jepa_global_predictor: Optional[nn.Module] = None
        if jepa_weight > 0 and (jepa_global_weight > 0 or jepa_setlevel):
            if self.pma is None:
                raise ValueError(
                    "jepa_global_weight>0 / jepa_setlevel requires PMA pooling (self.pma) for "
                    "sample-level alignment, but self.pma is None. Use pooling_mode='pma' + "
                    "(use_metadata_task or jepa_setlevel), or set jepa_global_weight=0."
                )
            # BYOL/GeneJEPA 式 predictor:带 LN(防塌)+ 窄 bottleneck;student 侧非对称(teacher 无 predictor)
            # 结尾 LN:稳住 pred_g 范数(≈√d),防 cosine 梯度含 1/‖pred‖ 在范数→0 时爆 NaN
            self.jepa_global_predictor = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, jepa_pred_dim),
                nn.GELU(),
                nn.LayerNorm(jepa_pred_dim),
                nn.Linear(jepa_pred_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ============ 去批次条件 MLM(2026-06-08,scVI 式)============
        # study_embed 只加到重建头输入(_shared_step MLM 分支),encoder/PMA 输出保持 batch-free。
        # 零初始化:起步 = 无 study bias = 等价纯 MLM,训练中渐进涌现(配 no_decay,见 configure_optimizers)。
        self.study_embed: Optional[nn.Embedding] = None
        if use_study_conditioning:
            if n_studies is None or n_studies <= 0:
                raise ValueError(
                    "use_study_conditioning=True requires n_studies > 0 (透传 DataModule.n_studies)."
                )
            self.study_embed = nn.Embedding(n_studies, d_model)
            nn.init.zeros_(self.study_embed.weight)

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
        # 2026-05-30 修泄露:phylo_ce/phylo_w/protein_w 也用 vocab_head 预测 mask 位置的 genus,
        # 若不屏蔽被预测 token 的输入身份(genus_embed + phylo_PE + protein_PE),模型直接读自己的
        # 输入 = 泄露(phylo_w/protein_w 退化成抄输入,非真任务)。故这些 loss 开启时同样需要
        # mask_token_id_replace(置 genus_mask_token + PE 输出乘 0),与 x2 一致。
        mask_id_replace = x2_active or (
            (self.hparams.phylo_ce_weight > 0)
            or (self.hparams.phylo_w_weight > 0)
            or (self.hparams.protein_w_weight > 0)
        )
        h = self.encoder(
            token_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            abund_bins=batch.get("abund_bins"),
            abund_values=batch.get("abund_values"),
            mask_positions=batch.get("mask_positions"),
            var_indices=batch.get("var_indices"),
            mask_token_id_replace=mask_id_replace,
        )
        return h

    def _pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """sample-level pooling: PMA(V5) 或 masked mean。"""
        h_token = h
        key_padding_mask = ~attention_mask  # True = PAD
        if self.hparams.pooling_mode == "pma":
            if self.pma is None:
                raise RuntimeError("pooling_mode='pma' requires self.pma for this code path.")
            pooled = self.pma(h_token, key_padding_mask=key_padding_mask)
            if pooled.ndim == 3:
                return pooled[:, 0, :]
            return pooled
        # mean_pool: 对非 PAD 位置求平均
        mask_f = attention_mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (h_token * mask_f).sum(dim=1) / denom

    def _pool_sample_views(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return view-specific sample vectors from independent PMA(k=1) modules."""
        if not self.sample_view_names or self.sample_view_pmas is None:
            raise RuntimeError("sample-view pooling requested but sample_view_pmas is not initialized.")
        key_padding_mask = ~attention_mask  # True = PAD
        pooled = []
        for view in self.sample_view_names:
            z = self.sample_view_pmas[view](h, key_padding_mask=key_padding_mask)
            if z.ndim != 2:
                raise RuntimeError(f"sample-view PMA {view!r} expected [B, d_model], got {tuple(z.shape)}.")
            pooled.append(z)
        return torch.stack(pooled, dim=1)

    @staticmethod
    def _info_nce(p1: torch.Tensor, p2: torch.Tensor, temp: float) -> torch.Tensor:
        """对称 NT-Xent(SimCLR):p1/p2 为同一样本两视图的投影,batch 内其他样本为负样本。"""
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        logits = (p1 @ p2.t()) / temp                       # [B, B] 相似度
        labels = torch.arange(p1.shape[0], device=p1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    @staticmethod
    def _vicreg_var_cov(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """标准 VICReg variance + covariance 正则(2026-06-06 v2 补全)。
        z: [N, d] —— 作用在 predictor 输出(去 NaN/inf 后)。
          variance   : hinge relu(1 - std).mean(每维 std,鼓励每维方差 ≥ 1,防维度塌)
          covariance : off-diagonal 协方差平方和 / d(去相关,防冗余 / 信息塌进低维)
        N<2 时返回 0(无法估方差/协方差)。"""
        if z.shape[0] < 2:
            zero = z.new_zeros(())
            return zero, zero
        std = torch.sqrt(z.var(dim=0) + 1e-4)                    # [d]
        loss_var = torch.relu(1.0 - std).mean()
        zc = z - z.mean(dim=0, keepdim=True)
        d = z.shape[1]
        cov = (zc.t() @ zc) / (z.shape[0] - 1)                   # [d, d] 协方差
        off_diag = cov - torch.diag(torch.diagonal(cov))        # 去对角
        loss_cov = off_diag.pow(2).sum() / d
        return loss_var, loss_cov

    @staticmethod
    def _effective_rank(mat: torch.Tensor) -> torch.Tensor:
        """有效秩 = exp(spectral entropy)(2026-06-06 诊断,抓 tgt_std 抓不到的"维度塌")。
        mat: [N, d] —— 对其奇异值谱算熵。p = s / s.sum(),effrank = exp(-Σ p log p)。
        effrank ∈ [1, min(N,d)];越小 = 谱越集中(塌到少数主方向)。NaN/inf 已在调用前清掉。"""
        if mat.shape[0] < 2 or mat.numel() == 0:
            return mat.new_tensor(1.0)
        # 奇异值(对 [N,d] 直接 SVD;只要 singular values)
        try:
            s = torch.linalg.svdvals(mat)
        except Exception:
            return mat.new_tensor(1.0)
        s = s[s > 0]
        if s.numel() == 0:
            return mat.new_tensor(1.0)
        p = s / s.sum()
        ent = -(p * torch.log(p + 1e-12)).sum()
        return torch.exp(ent)

    @staticmethod
    def _top_singular_fraction(mat: torch.Tensor) -> torch.Tensor:
        if mat.shape[0] < 2 or mat.numel() == 0:
            return mat.new_tensor(1.0)
        try:
            s = torch.linalg.svdvals(mat)
        except Exception:
            return mat.new_tensor(1.0)
        total = s.sum()
        if not torch.isfinite(total) or total <= 0:
            return mat.new_tensor(1.0)
        return s.max() / total

    # ------------------------------------------------------------------
    # JEPA 辅助(2026-06-04)
    # ------------------------------------------------------------------
    def _sync_target_buffers(self) -> None:
        """把 self.encoder 的 persistent=False buffer(phylo/protein coords)同步给 target_encoder。

        必须在 target forward 之前调:inject_var_buffers 只更新 self.encoder,而 target_encoder
        是 __init__ 时 deepcopy 的占位 coords(_coords_loaded=False,forward 会 raise)。coords 是
        frozen buffer,直接共享同一对象即可;幂等,on_train_start / on_validation_start 都调(便宜)。
        """
        if self.target_encoder is None:
            return
        with torch.no_grad():
            for src, dst in (
                (self.encoder.phylo_pe, self.target_encoder.phylo_pe),
                (self.encoder.protein_pe, self.target_encoder.protein_pe),
            ):
                if src is not None and dst is not None and getattr(src, "_coords_loaded", False):
                    dst.coords = src.coords            # 共享同一 frozen buffer(省显存)
                    dst._coords_loaded = True
        self._target_buffers_synced = True

    def _ema_decay_now(self) -> float:
        """EMA 衰减调度:setlevel(GeneJEPA)用 cosine ramp jepa_ema_decay(0.996)→jepa_ema_end(0.9995);
        否则 linear ramp jepa_ema_decay→1.0(I-JEPA)。"""
        base = float(self.hparams.jepa_ema_decay)
        try:
            total = max(1, int(self.trainer.estimated_stepping_batches))
        except Exception:
            total = 1
        frac = min(1.0, float(self.global_step) / total)
        if getattr(self.hparams, "jepa_setlevel", False):
            # GeneJEPA 式 cosine ramp:frac=0→base, frac=1→end(train.py:421-424 同式)
            end = min(float(self.hparams.jepa_ema_end), 0.9999)
            return end - (end - base) * (math.cos(math.pi * frac) + 1.0) / 2.0
        return base + (1.0 - base) * frac

    @torch.no_grad()
    def _ema_update_target(self) -> None:
        """EMA 更新 target_encoder 参数(coords buffer 由 _sync_target_buffers 处理,不参与 EMA)。
        setlevel warmup:前 jepa_ema_warmup_steps 步 teacher 冻结(GeneJEPA;先让 student 跑起来)。"""
        if self.target_encoder is None:
            return
        warmup = int(getattr(self.hparams, "jepa_ema_warmup_steps", 0))
        if warmup > 0 and self.global_step < warmup:
            return                              # warmup 期 teacher 冻结不更新
        decay = self._ema_decay_now()
        for p_ctx, p_tgt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.mul_(decay).add_(p_ctx.detach(), alpha=1.0 - decay)

    def _jepa_ratio_now(self) -> float:
        """JEPA v2 structured mask ratio curriculum(2026-06-06):
        按 self.current_epoch 从 jepa_ratio_start 线性爬到 jepa_ratio_end,跨 trainer.max_epochs。
        拿不到 max_epochs(无 trainer / max_epochs=-1 等)则退回固定 jepa_mask_ratio。"""
        start = float(getattr(self.hparams, "jepa_ratio_start", 0.3))
        end = float(getattr(self.hparams, "jepa_ratio_end", 0.5))
        try:
            max_ep = int(self.trainer.max_epochs)
        except Exception:
            max_ep = -1
        if max_ep is None or max_ep <= 1:
            return float(self.hparams.jepa_mask_ratio)
        frac = min(1.0, max(0.0, float(self.current_epoch) / float(max_ep - 1)))
        return start + (end - start) * frac

    def _jepa_split_batch(self, batch: Dict[str, torch.Tensor]):
        """把有效 genus 分 context / target 两组(v2 全程 structured + ratio curriculum)。

        v2(2026-06-06):删 MLM 锚 —— mlm_weight==0 时不在 context 内选 abund-mask
        (mask_positions 置全 False),JEPA 的 target_mask 结构化簇照常工作。
        mlm_weight>0(兼容旧 v1)时仍寄生 MLM 锚。

        Returns: (ctx_batch, jepa_ctx)
          ctx_batch : 浅拷贝并改了 attention_mask=ctx_mask、mask_positions 的 batch
          jepa_ctx  : {target_mask, ctx_mask, full_mask}
        """
        am = batch["attention_mask"]
        ratio = self._jepa_ratio_now()            # v2:动态 ratio(curriculum)
        mask_mode = getattr(self.hparams, "jepa_mask_mode", "random")
        if mask_mode.startswith("structured"):
            target_mask = self._jepa_structured_target(
                batch["token_ids"],
                am,
                ratio,
                abund_values=batch.get("abund_values"),
                mode=mask_mode,
            )
        else:
            target_mask = (torch.rand(am.shape, device=am.device) < ratio) & am
        ctx_mask = am & ~target_mask
        # 兜底:每个有效样本必须既有 context 又有 target —— setlevel teacher 看 target 子集,
        #   任一为空 → encoder 对全屏蔽样本 attention softmax 全 -inf → NaN(2026-06-11 修)。
        valid_row = am.any(dim=1)
        no_tgt = valid_row & ~target_mask.any(dim=1)          # 有效但无 target
        if no_tgt.any():
            sc = torch.rand(am.shape, device=am.device).masked_fill(~am, -1.0)
            pick = sc.argmax(1)                               # 每样本随机选 1 个有效位置
            rows = torch.nonzero(no_tgt, as_tuple=False).flatten()
            target_mask[rows, pick[rows]] = True
        ctx_mask = am & ~target_mask
        no_ctx = valid_row & ~ctx_mask.any(dim=1)             # 有效但无 context(target 占满)
        if no_ctx.any():
            sc = torch.rand(am.shape, device=am.device).masked_fill(~target_mask, -1.0)
            pick = sc.argmax(1)                               # 从 target 里挪 1 个回 context
            rows = torch.nonzero(no_ctx, as_tuple=False).flatten()
            target_mask[rows, pick[rows]] = False
        ctx_mask = am & ~target_mask
        # MLM 锚(v2:mlm_weight==0 时彻底关掉,jepa_mlm_mask_prob 失效;>0 才寄生)
        if float(self.hparams.mlm_weight) > 0:
            p = float(self.hparams.jepa_mlm_mask_prob)
            abund_mask = (torch.rand(am.shape, device=am.device) < p) & ctx_mask
        else:
            abund_mask = torch.zeros_like(am)
        ctx_batch = dict(batch)
        ctx_batch["attention_mask"] = ctx_mask
        ctx_batch["mask_positions"] = abund_mask
        return ctx_batch, {"target_mask": target_mask, "ctx_mask": ctx_mask, "full_mask": am}

    def _jepa_structured_target(self, token_ids, am, ratio, abund_values=None, mode="structured"):
        """结构化 mask(糙版):按 phylo/protein 坐标成簇遮——多种子,每种子遮样本内最近一撮。
        样本内现算坐标距离(坐标已在 encoder buffer,零注入);phylo/protein 每 batch 随机交替。
        见 PLAN.md"结构化 mask"。坐标当"出题人/脚手架"(决定遮哪片),非被拟合答案(红线)。

        structured_hi*:2026-06-11 新诊断分支。seed 不再随机抽,而从样本内高丰度 token
        开始,再沿 phylo/protein 邻域扩成 multi-block。目的:让结构化 mask 遮到 pooled 表征
        真正在乎的主导菌群,同时仍不把 phylo/protein 当 target/loss。
        """
        B, L = am.shape
        device = am.device
        n_seeds = max(1, int(getattr(self.hparams, "jepa_n_seeds", 3)))
        # phylo/protein 交替(每 batch 随机选一个坐标源;便宜门显示 protein 更成簇)
        if mode.endswith("_phylo"):
            use_protein = False
        elif mode.endswith("_protein"):
            use_protein = True
        else:
            use_protein = bool(
                self.hparams.use_protein_pe and self.encoder.protein_pe is not None
                and (torch.rand(1).item() < 0.5)
            )
        pe = self.encoder.protein_pe if use_protein else self.encoder.phylo_pe
        if pe is None or not getattr(pe, "_coords_loaded", False):
            return (torch.rand(am.shape, device=device) < ratio) & am   # 坐标没加载 → 回退随机
        coords = pe.coords[token_ids].float()                            # [B, L, d]
        D = torch.cdist(coords, coords)                                  # [B, L, L] 样本内两两距离
        D = D.masked_fill((~am).unsqueeze(1), float("inf"))              # PAD 列不被选为近邻
        n_valid = am.sum(1).clamp(min=1)                                 # [B]
        target_total = (ratio * n_valid.float()).long().clamp(min=1)
        per_seed = (target_total.float() / n_seeds).ceil().long().clamp(min=1)  # [B] 每簇遮几个
        target_mask = torch.zeros_like(am)
        high_abund_seed = mode.startswith("structured_hi") and abund_values is not None
        if high_abund_seed:
            # rclr_sigma 可正可负,但排序仍表示样本内相对主导程度。每个 seed 取下一高丰度 token。
            seed_order = abund_values.float().masked_fill(~am, -1e9).argsort(dim=1, descending=True)
        for seed_i in range(n_seeds):
            if high_abund_seed:
                seed = seed_order[:, min(seed_i, L - 1)]
            else:
                seed_score = torch.rand(B, L, device=device).masked_fill(~am, -1.0)
                seed = seed_score.argmax(1)                                  # [B] 每样本随机有效种子
            Dseed = D.gather(1, seed.view(B, 1, 1).expand(B, 1, L)).squeeze(1)  # [B, L] 到种子距离
            rank = Dseed.argsort(1).argsort(1)                           # [B, L] 距种子名次(0=种子自己)
            target_mask = target_mask | ((rank < per_seed.view(B, 1)) & am)    # 最近 per_seed 个
        return target_mask

    # ------------------------------------------------------------------
    # 训练 / 验证步
    # ------------------------------------------------------------------
    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,  # "train" / "val"
        batch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        # ============ JEPA 预处理(2026-06-04) ============
        # jepa_weight>0:先把有效 genus 分 context/target,context encoder 只看 context(target
        #   屏蔽),MLM 锚寄生在这次 context forward 上(iBOT 式 2-forward)。orig_batch 留给
        #   target encoder(看完整)+ predictor(取 token 坐标当地址 query)。
        jepa_on = float(self.hparams.jepa_weight) > 0 and self.target_encoder is not None
        jepa_ctx = None
        orig_batch = batch
        if jepa_on:
            batch, jepa_ctx = self._jepa_split_batch(batch)

        h = self._encode(batch)
        h_token = h

        # ============ MLM loss ============
        # v2(2026-06-06):mlm_weight==0 时**彻底跳过** MLM 路径 —— 不前向 mlm_head、不算 loss、
        #   不进 total(mlm_head 此时是真正未用参数,DDP 须开 find_unused;单进程无影响)。
        #   JEPA v2 = MLM-free,这里是删 MLM 的入口。
        # mlm_weight>0(兼容旧 V5 / v1)时按原逻辑前向 + 计 loss。
        mask_pos = batch["mask_positions"]
        mlm_w = float(self.hparams.mlm_weight)
        loss_mlm = torch.zeros((), device=h.device, dtype=h.dtype)
        if mlm_w > 0:
            # 去批次条件 MLM(2026-06-08):重建头额外加 study bias(study_embed[study_id]),
            # 只进重建头、不进 encoder/PMA 输出(后者保持 batch-free)。study_embed=None 时退化为纯 MLM。
            mlm_feat = h_token
            if self.study_embed is not None and "study_id" in batch:
                mlm_feat = h_token + self.study_embed(batch["study_id"]).unsqueeze(1)
            if self.hparams.abundance_loss == "huber":
                pred = self.mlm_head(mlm_feat)              # [B, L]
                target = batch["labels_abund_values"]       # [B, L] float32
                if mask_pos.any():
                    loss_mlm = F.smooth_l1_loss(
                        pred[mask_pos], target[mask_pos], beta=self.huber_beta, reduction="mean"
                    )
                    with torch.no_grad():
                        mae = (pred[mask_pos] - target[mask_pos]).abs().mean()
                else:
                    mae = torch.zeros((), device=h.device, dtype=h.dtype)
            else:
                logits = self.mlm_head(mlm_feat)             # [B, L, num_bins]
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
                    mae = None

            # log MLM(仅 mlm_weight>0 时记)
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

        # ============ Phylo Soft-Target CE / Tree-Wasserstein / Protein Tree-W loss ============
        # (phylo_ce/phylo_w: 2026-05-29;protein_w: 2026-05-30,phylo_w 精确镜像)
        # 三条 path 共享 vocab_head 计算 + (各自的)dist_to_true,各算各的 loss
        # phylo_ce : target = softmax(-dist/τ) soft 分布,loss = KL(soft target || pred)
        # phylo_w  : target = one-hot,loss = E_{v~p}[d(v, v*)] (W-1 expected phylo distance)
        # protein_w: 同 phylo_w,只把 phylo dist 换成 protein_dist(蛋白距离矩阵)
        loss_phylo_ce = torch.zeros((), device=h.device, dtype=h.dtype)
        loss_phylo_w = torch.zeros((), device=h.device, dtype=h.dtype)
        loss_protein_w = torch.zeros((), device=h.device, dtype=h.dtype)
        phylo_ce_w = float(self.hparams.phylo_ce_weight)
        phylo_w_w = float(self.hparams.phylo_w_weight)
        protein_w_w = float(self.hparams.protein_w_weight)
        if (phylo_ce_w > 0 or phylo_w_w > 0 or protein_w_w > 0) and self.vocab_head is not None:
            # phylo_ce / phylo_w 需要 dist_matrix(phylo);protein_w 需要 protein_dist_matrix
            if (phylo_ce_w > 0 or phylo_w_w > 0) and (
                self.encoder.dist_matrix is None or not self.encoder._dist_matrix_loaded
            ):
                raise RuntimeError(
                    "phylo_ce_weight>0 or phylo_w_weight>0 requires encoder.dist_matrix loaded "
                    "(call inject_var_buffers with dist_matrix before forward())."
                )
            if protein_w_w > 0 and (
                self.encoder.protein_dist_matrix is None or not self.encoder._protein_dist_loaded
            ):
                raise RuntimeError(
                    "protein_w_weight>0 requires encoder.protein_dist_matrix loaded "
                    "(call inject_var_buffers with protein_dist_matrix before forward())."
                )
            # vocab id 约定:0=PAD, 1=UNK, 2~V_real+1 = real genus(var_index = vocab_id - 2)
            mask_target_vocab = batch["token_ids"][mask_pos]   # [N] long
            valid_target = mask_target_vocab >= 2              # 跳过 UNK
            if valid_target.any():
                target_var_idx = (mask_target_vocab[valid_target] - 2).long()  # [n_valid]
                # 共用 logits 计算
                logits_full = self.vocab_head(h_token)                            # [B, L, V_real+2]
                mask_logits_full = logits_full[mask_pos]                          # [N, V_real+2]
                logits_real = mask_logits_full[valid_target, 2:]                  # [n_valid, V_real]
                # pred_probs 给 phylo_w / protein_w 共用(phylo_ce 单独走 log_softmax)
                pred_probs = None

                # phylo_ce / phylo_w 用 phylo dist_to_true
                if phylo_ce_w > 0 or phylo_w_w > 0:
                    dist_to_true = self.encoder.dist_matrix[target_var_idx].float()  # [n_valid, V_real]
                    # phylo_ce branch
                    if phylo_ce_w > 0:
                        tau = float(self.hparams.phylo_ce_tau)
                        target_dist = F.softmax(-dist_to_true / tau, dim=-1)
                        log_probs = F.log_softmax(logits_real, dim=-1)
                        loss_phylo_ce = -(target_dist * log_probs).sum(-1).mean()
                    # phylo_w (Tree-W simplified) branch
                    # 数学:W(p, δ_v*) = Σ_v p(v) × d(v, v*),Yamada 2021 closed-form 当 target=one-hot
                    # hard target floor=0(when p=one-hot at v*),不 saturate vs phylo_ce ep0 plateau
                    if phylo_w_w > 0:
                        pred_probs = F.softmax(logits_real, dim=-1)               # [n_valid, V_real]
                        # 2026-05-30:除以 dist_scale(非零距离均值)归一化到 ~1 量级,
                        # 否则 E[d]~125 会淹没 mlm~0.2 数百倍,weight=1 名义五五开实则 660:1。
                        loss_phylo_w = (pred_probs * dist_to_true).sum(-1).mean() / self.encoder.dist_scale

                # protein_w (蛋白 Tree-W simplified) branch — 精确镜像 phylo_w
                # 复用 logits_real / pred_probs,只把 dist 换成 protein_dist_matrix
                if protein_w_w > 0:
                    if pred_probs is None:
                        pred_probs = F.softmax(logits_real, dim=-1)               # [n_valid, V_real]
                    protein_dist_to_true = self.encoder.protein_dist_matrix[target_var_idx].float()  # [n_valid, V_real]
                    loss_protein_w = (
                        (pred_probs * protein_dist_to_true).sum(-1).mean()
                        / self.encoder.protein_dist_scale
                    )

            if phylo_ce_w > 0:
                self.log(
                    f"{stage}/loss_phylo_ce", loss_phylo_ce,
                    prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                    sync_dist=(stage == "val"),
                )
            if phylo_w_w > 0:
                self.log(
                    f"{stage}/loss_phylo_w", loss_phylo_w,
                    prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                    sync_dist=(stage == "val"),
                )
            if protein_w_w > 0:
                self.log(
                    f"{stage}/loss_protein_w", loss_protein_w,
                    prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True,
                    sync_dist=(stage == "val"),
                )

        # ============ Metadata loss ============
        total_loss = (
            mlm_w * loss_mlm
            + x2_phylo_w * loss_x2_phylo
            + x2_protein_w * loss_x2_protein
            + phylo_ce_w * loss_phylo_ce
            + phylo_w_w * loss_phylo_w
            + protein_w_w * loss_protein_w
        )

        # ============ Sample-level multi-view shaping loss ============
        sample_view_w = float(getattr(self.hparams, "sample_view_loss_weight", 0.0))
        if self.sample_view_names and sample_view_w > 0:
            if self.sample_view_pmas is None or self.sample_view_heads is None:
                raise RuntimeError("sample_view_heads requires sample_view_pmas and sample_view_heads modules.")
            if "sample_view_targets" not in batch:
                raise RuntimeError(
                    "sample_view_heads is enabled but batch does not contain sample_view_targets. "
                    "Check MiCoDataModule sample_view_heads / sample_view_protein_feat_path settings."
                )
            pooled_views = self._pool_sample_views(h_token, batch["attention_mask"])
            if pooled_views.ndim != 3 or pooled_views.shape[1] != len(self.sample_view_names):
                raise RuntimeError(
                    "sample_view_heads expects PMA output [B, n_views, d_model], got "
                    f"{tuple(pooled_views.shape)} for n_views={len(self.sample_view_names)}."
                )
            view_losses = []
            weights = self._sample_view_loss_weights.to(device=h.device, dtype=torch.float32)
            for view_i, view in enumerate(self.sample_view_names):
                if view not in batch["sample_view_targets"]:
                    raise RuntimeError(f"Missing sample-view target {view!r} in batch.")
                z_view = pooled_views[:, view_i, :]
                pred_view = self.sample_view_heads[view](z_view)
                target_view = batch["sample_view_targets"][view].to(
                    device=pred_view.device,
                    dtype=pred_view.dtype,
                    non_blocking=True,
                )
                loss_view = F.mse_loss(pred_view.float(), target_view.float(), reduction="mean")
                view_losses.append(loss_view * weights[view_i])
                self.log(
                    f"{stage}/loss_sample_view_{view}",
                    loss_view,
                    prog_bar=False,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    sync_dist=(stage == "val"),
                )
            loss_sample_views = sum(view_losses) / weights.sum().clamp(min=1e-6)
            total_loss = total_loss + sample_view_w * loss_sample_views
            self.log(
                f"{stage}/loss_sample_views",
                loss_sample_views,
                prog_bar=(stage == "train"),
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=(stage == "val"),
            )

            div_w = float(getattr(self.hparams, "sample_view_diversity_weight", 0.0))
            if div_w > 0 and pooled_views.shape[1] > 1:
                z = F.normalize(pooled_views.float(), dim=-1, eps=1e-6)
                sim = torch.einsum("bvd,bwd->bvw", z, z)
                eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
                offdiag = ~eye.expand_as(sim)
                loss_view_div = sim[offdiag].pow(2).mean()
                total_loss = total_loss + div_w * loss_view_div.to(total_loss.dtype)
                self.log(
                    f"{stage}/loss_sample_view_div",
                    loss_view_div,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    sync_dist=(stage == "val"),
                )

            close_w = float(getattr(self.hparams, "sample_view_close_weight", 0.0))
            if close_w > 0 and self.sample_view_pmas is not None and len(self.sample_view_pmas) > 1:
                q = torch.stack([self.sample_view_pmas[view].query.squeeze(0) for view in self.sample_view_names], dim=0).float()
                loss_view_close = (q - q.mean(dim=0, keepdim=True)).pow(2).mean()
                total_loss = total_loss + close_w * loss_view_close.to(total_loss.dtype)
                self.log(
                    f"{stage}/loss_sample_view_close",
                    loss_view_close,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    sync_dist=(stage == "val"),
                )

            if stage == "val" and batch_idx == 0 and ((int(self.current_epoch) + 1) % 5 == 0):
                with torch.no_grad():
                    for view_i, view in enumerate(self.sample_view_names):
                        z_rank = pooled_views[:, view_i, :].detach().float()
                        z_rank = z_rank[torch.isfinite(z_rank).all(dim=1)]
                        erank = self._effective_rank(z_rank)
                        top_frac = self._top_singular_fraction(z_rank)
                        self.log(
                            f"{stage}/erank_sample_view_{view}",
                            erank,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=(stage == "val"),
                        )
                        self.log(
                            f"{stage}/top_singular_frac_sample_view_{view}",
                            top_frac,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=(stage == "val"),
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

        # ============ 对比 InfoNCE loss(2026-06-04,仅 train,保留 MLM 锚) ============
        # 同一样本两套不同 abund-mask → 两个 mean-pool 表征 → InfoNCE 拉近,batch 内其他样本为负。
        # MLM 锚已在上面算(防表征塌缩)。第二视图重 encode 一次(~2× 前向)。仅 train。
        cw = float(self.hparams.contrastive_weight)
        if stage == "train" and cw > 0 and self.contrastive_proj is not None:
            am = batch["attention_mask"]
            mask_f = am.float().unsqueeze(-1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            z1 = (h_token * mask_f).sum(dim=1) / denom                      # 视图1 mean-pool(h 已算)
            # 视图2: genus dropout(随机保留 ~50% token,制造样本级大差异=rarefaction 快速代理)+ abund-mask
            # (2026-06-04:mask-twice 推不动样本级,试丢 genus 类增广;有用再参数化/做精确 rarefaction)
            keep2 = (torch.rand(am.shape, device=am.device) < 0.5) & am
            keep2 = keep2 | ((~keep2.any(dim=1, keepdim=True)) & am)        # 防整样本全屏蔽 → 回退保留全部
            rand2 = torch.rand(am.shape, device=am.device)
            mp2 = (rand2 < float(self.hparams.contrastive_mask_prob)) & keep2
            batch2 = dict(batch)
            batch2["attention_mask"] = keep2                                # genus dropout:dropped token 不参与 attention
            batch2["mask_positions"] = mp2
            h2 = self._encode(batch2)
            mf2 = keep2.float().unsqueeze(-1)
            z2 = (h2 * mf2).sum(dim=1) / mf2.sum(dim=1).clamp(min=1.0)       # 视图2 mean-pool(仅保留 token)
            loss_con = self._info_nce(
                self.contrastive_proj(z1), self.contrastive_proj(z2),
                float(self.hparams.contrastive_temp),
            )
            total_loss = total_loss + cw * loss_con
            self.log(f"{stage}/loss_contrastive", loss_con,
                     prog_bar=True, on_step=True, on_epoch=True)
            with torch.no_grad():                                          # 塌缩哨兵:表征每维 std 均值(趋 0=塌)
                repr_std = z1.detach().float().std(dim=0).mean()
            self.log(f"{stage}/repr_std", repr_std,
                     prog_bar=True, on_step=True, on_epoch=True)

        # ============ JEPA latent prediction(2026-06-04;v2 升级 2026-06-06) ============
        # 红线:坐标只当地址 query,target = EMA target encoder 看完整样本的含义向量(LN + stop-grad)。
        # h_token 是 context forward(target 屏蔽)的输出;predictor 在 target 位置用坐标 query 预测。
        # v2 双自监督:per-token JEPA(loss_jepa)+ 全局对齐(loss_jepa_global,student/teacher PMA)。
        # ============ JEPA v3 set 级(2026-06-11,全盘抄 GeneJEPA)============
        # student 看 context(h_token)→PMA→z_s;teacher(EMA)**只看 target 子集**→PMA→z_t(LN+detach);
        #   global_predictor(z_s) cosine 对齐 z_t。无坐标地址、无 token 级 predictor(v1/v2 已证失败)。
        if jepa_on and self.hparams.jepa_setlevel:
            jepa_w = float(self.hparams.jepa_weight)
            target_mask = jepa_ctx["target_mask"]
            ctx_mask = jepa_ctx["ctx_mask"]
            d_m = h_token.shape[-1]
            # student:context 池化(h_token=context forward 输出,target 已屏蔽)
            z_s = self._pool(h_token, ctx_mask)                       # [B, d]
            # teacher(EMA, no_grad):只让 target 子集可见 → 池化 → LN + detach(零泄露:teacher 没看 context)
            with torch.no_grad():
                h_tgt_raw = self.target_encoder(
                    token_ids=orig_batch["token_ids"],
                    attention_mask=target_mask,
                    abund_bins=orig_batch.get("abund_bins"),
                    abund_values=orig_batch.get("abund_values"),
                    mask_positions=None,
                    var_indices=orig_batch.get("var_indices"),
                    mask_token_id_replace=False,
                )
                z_t = self._pool(h_tgt_raw.to(h_token.dtype), target_mask)
                z_t = F.layer_norm(z_t.float(), (d_m,)).to(h_token.dtype)
            z_s_ln = F.layer_norm(z_s.float(), (d_m,)).to(h_token.dtype)
            pred_g = self.jepa_global_predictor(z_s_ln)               # BYOL/GeneJEPA predictor(带 LN)
            if self.hparams.jepa_loss_type == "cosine":
                # GeneJEPA 式稳健 cosine:显式 normalize(eps=1e-6,比 cosine_similarity 的 1e-8 稳)
                p_n = F.normalize(pred_g.float(), dim=-1, eps=1e-6)
                t_n = F.normalize(z_t.detach().float(), dim=-1, eps=1e-6)
                loss_jepa = (1.0 - (p_n * t_n).sum(dim=-1)).mean().to(h_token.dtype)
            else:
                loss_jepa = F.mse_loss(pred_g, z_t.detach())
            total_loss = total_loss + jepa_w * loss_jepa
            self.log(f"{stage}/loss_jepa", loss_jepa,
                     prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))
            # 防塌 VICReg(GeneJEPA 起步就开):predictor 输出 + student z_s 各加 var+cov
            vw = float(self.hparams.jepa_vicreg_weight)
            if vw > 0:
                var_p, cov_p = self._vicreg_var_cov(pred_g.float())
                total_loss = total_loss + vw * (var_p + cov_p).to(h.dtype)
                self.log(f"{stage}/loss_jepa_var", var_p, on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))
                self.log(f"{stage}/loss_jepa_cov", cov_p, on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))
            vw_s = float(self.hparams.jepa_student_vicreg_weight)
            if vw_s > 0:
                var_s, cov_s = self._vicreg_var_cov(z_s_ln.float())
                total_loss = total_loss + vw_s * (var_s + cov_s).to(h.dtype)
            # 防塌监控:teacher 池化向量每维 std(→0=塌)+ 有效秩(→1=维度塌)
            with torch.no_grad():
                _tgt_std = z_t.detach().float().std(dim=0).mean()
                _eff_rank = self._effective_rank(z_t.detach().float())
            self.log(f"{stage}/global_tgt_std", _tgt_std, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))
            self.log(f"{stage}/jepa_eff_rank", _eff_rank, prog_bar=True, on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))

        # ---- 旧 token 级 + global(v1/v2;jepa_setlevel=False 时走这里,保留对照)----
        if jepa_on and not self.hparams.jepa_setlevel:
            jepa_w = float(self.hparams.jepa_weight)
            am = jepa_ctx["full_mask"]
            target_mask = jepa_ctx["target_mask"]
            # target encoder(EMA, no_grad):看完整样本(不遮 abund),输出含义向量(token 级 h_tgt)
            with torch.no_grad():
                h_tgt_raw = self.target_encoder(
                    token_ids=orig_batch["token_ids"],
                    attention_mask=am,
                    abund_bins=orig_batch.get("abund_bins"),
                    abund_values=orig_batch.get("abund_values"),
                    mask_positions=None,
                    var_indices=orig_batch.get("var_indices"),
                    mask_token_id_replace=False,
                )
                # predict residual(2026-06-11):target 减样本全局中心 → 预测被遮菌相对整体的偏差。
                #   铲掉"看整体组成"捷径(ep0/ep1 诊断 shuf/norm≈1.05→1.10=模型停均值档,没学共变);
                #   残差只编码"该菌偏离样本中心多少",只能靠相关邻居推断。全局用 valid token 均值(detach)。
                if self.hparams.jepa_predict_residual:
                    _m = am.unsqueeze(-1).to(h_tgt_raw.dtype)            # [B, L, 1] valid mask
                    _g = (h_tgt_raw * _m).sum(1, keepdim=True) / _m.sum(1, keepdim=True).clamp(min=1.0)
                    h_tgt_raw = h_tgt_raw - _g                          # [B, L, d] 减样本全局 → 残差
                # target 归一化(parameter-free LN,I-JEPA/data2vec):稳定回归尺度
                h_tgt = F.layer_norm(h_tgt_raw.float(), (h_tgt_raw.shape[-1],)).to(h_token.dtype)
            # 地址 query(红线:地址不是答案,target 才是 EMA 含义向量)
            if self.hparams.jepa_addr_mode == "genus":
                # Cell-JEPA 式:用被遮菌的 genus embedding 身份当地址(吃数据驱动菌间共变,不用 phylo/protein 错图)
                # detach = frozen 身份地址(沿用"地址不可学"精神);genus_embed 仍经 encoder 路径更新
                genus_query = self.encoder.genus_embed(orig_batch["token_ids"]).detach()
                phylo_coords = protein_coords = None
            else:
                genus_query = None
                phylo_coords = (
                    self.encoder.phylo_pe.coords[orig_batch["token_ids"]]
                    if (self.hparams.use_phylo_pe and self.encoder.phylo_pe is not None) else None
                )
                protein_coords = (
                    self.encoder.protein_pe.coords[orig_batch["token_ids"]]
                    if (self.hparams.use_protein_pe and self.encoder.protein_pe is not None) else None
                )
            pred = self.jepa_predictor(h_token, target_mask, am, phylo_coords, protein_coords, genus_query)
            if target_mask.any():
                loss_jepa = F.mse_loss(pred[target_mask], h_tgt[target_mask].detach())
            else:
                loss_jepa = torch.zeros((), device=h.device, dtype=h.dtype)
            total_loss = total_loss + jepa_w * loss_jepa
            self.log(f"{stage}/loss_jepa", loss_jepa,
                     prog_bar=True, on_step=(stage == "train"), on_epoch=True,
                     sync_dist=(stage == "val"))

            # ---- 全局对齐(v2 新目标,2026-06-06):student PMA vs teacher PMA(LN + detach)----
            # student = context encoder 输出 h_token 对 valid genus token PMA 池化 → z_s
            # teacher = EMA target encoder 输出 h_tgt_raw(看完整样本)同样 PMA 池化 → z_t(detach)
            # 两者各过 parameter-free LayerNorm;loss = mse(global_predictor(z_s), z_t.detach())
            gw = float(self.hparams.jepa_global_weight)
            if gw > 0 and self.jepa_global_predictor is not None and self.pma is not None:
                kpm = ~am                                        # True=PAD(屏蔽);register 已在 encoder 内处理
                z_s = self.pma(h_token, key_padding_mask=kpm)    # [B, d] student 池化(带梯度)
                with torch.no_grad():
                    z_t = self.pma(h_tgt_raw.to(h_token.dtype), key_padding_mask=kpm)  # teacher 池化
                    z_t = F.layer_norm(z_t.float(), (z_t.shape[-1],)).to(h_token.dtype)
                z_s = F.layer_norm(z_s.float(), (z_s.shape[-1],)).to(h_token.dtype)
                g_pred = self.jepa_global_predictor(z_s)
                loss_global = F.mse_loss(g_pred, z_t.detach())
                total_loss = total_loss + gw * loss_global
                # 全局路径 VICReg variance(v2.1 2026-06-06:删 MLM 后 BYOL 全局裸奔易塌,
                # 复用 jepa_vicreg_weight 预防;逼 batch 内每维 std≥1,正交于 per-sample LayerNorm)
                vw_g = float(self.hparams.jepa_vicreg_weight)
                if vw_g > 0:
                    g_std = torch.sqrt(g_pred.float().var(dim=0) + 1e-4)
                    loss_global_var = torch.relu(1.0 - g_std).mean()
                    total_loss = total_loss + vw_g * loss_global_var.to(h.dtype)
                    self.log(f"{stage}/loss_global_var", loss_global_var,
                             on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))
                self.log(f"{stage}/loss_jepa_global", loss_global,
                         prog_bar=True, on_step=(stage == "train"), on_epoch=True,
                         sync_dist=(stage == "val"))
                with torch.no_grad():                            # teacher 池化向量每维 std(防全局塌)
                    global_tgt_std = z_t.detach().float().std(dim=0).mean()
                self.log(f"{stage}/global_tgt_std", global_tgt_std,
                         prog_bar=True, on_step=(stage == "train"), on_epoch=True,
                         sync_dist=(stage == "val"))

            # ---- VICReg variance + covariance 防塌(v2 补全 covariance,作用在 predictor 输出)----
            vw = float(self.hparams.jepa_vicreg_weight)
            if vw > 0 and target_mask.any():
                z = pred[target_mask].float()
                z = z[torch.isfinite(z).all(dim=1)]              # 去 NaN/inf 行
                loss_var, loss_cov = self._vicreg_var_cov(z)
                loss_vicreg = loss_var + loss_cov
                total_loss = total_loss + vw * loss_vicreg.to(h.dtype)
                self.log(f"{stage}/loss_jepa_var", loss_var,
                         on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))
                self.log(f"{stage}/loss_jepa_cov", loss_cov,
                         on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))

            # ---- 塌缩哨兵 + 有效秩诊断(debug 核心验收项)----
            # tgt_std/pred_std:每维 std 均值,趋 0 = 维度坍缩
            # jepa_effrank:对 h_tgt[target_mask] 算 exp(spectral entropy),抓"谱集中"型塌(std 抓不到)
            with torch.no_grad():
                if target_mask.any():
                    tgt_std = h_tgt[target_mask].float().std(dim=0).mean()
                    pred_std = pred[target_mask].float().std(dim=0).mean()
                    self.log(f"{stage}/jepa_tgt_std", tgt_std,
                             prog_bar=True, on_step=(stage == "train"), on_epoch=True,
                             sync_dist=(stage == "val"))
                    self.log(f"{stage}/jepa_pred_std", pred_std,
                             on_step=(stage == "train"), on_epoch=True, sync_dist=(stage == "val"))
                    h_eff = h_tgt[target_mask].float()
                    h_eff = h_eff[torch.isfinite(h_eff).all(dim=1)]   # 去 NaN/inf
                    effrank = self._effective_rank(h_eff)
                    self.log(f"{stage}/jepa_effrank", effrank,
                             prog_bar=True, on_step=(stage == "train"), on_epoch=True,
                             sync_dist=(stage == "val"))

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
        return self._shared_step(batch, "train", batch_idx=batch_idx)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val", batch_idx=batch_idx)

    # ------------------------------------------------------------------
    # JEPA lifecycle hooks(2026-06-04)
    # ------------------------------------------------------------------
    def on_train_start(self) -> None:
        # inject_var_buffers 已在 fit 前更新 self.encoder.coords → 同步给 target_encoder
        self._sync_target_buffers()

    def on_validation_start(self) -> None:
        # sanity-check / val-first 时 on_train_start 可能还没跑,这里兜底同步(幂等)
        if self.target_encoder is not None and not getattr(self, "_target_buffers_synced", False):
            self._sync_target_buffers()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # EMA 更新 target_encoder(在 optimizer step 之后)
        self._ema_update_target()

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
        no_decay_prefixes = (
            "encoder.phylo_pe.", "encoder.dist_bias.",
            "jepa_predictor.",          # predictor 别被 WD 压死(检索:predictor 是防塌主力,弱了诱发塌缩)
            "jepa_global_predictor.",   # v2 全局对齐 predictor 同理(防塌主力,不压)
            "study_embed.",             # 去批次 study bias:零初始化渐进涌现,WD 会压回 0(同 phylo_pe)
        )
        # encoder.reg_tokens(v2 register,ndim==2)默认会进 decay 组;它是类 [CLS] 可学 token,
        # 加进 no_decay_exact 跟 pma.query / mask_token 一致(零初始化偏小,WD 会压回弱化防塌作用)。
        no_decay_exact = {
            "encoder.abund_mask_token", "pma.query",
            "jepa_predictor.mask_token", "encoder.reg_tokens",
        }

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
