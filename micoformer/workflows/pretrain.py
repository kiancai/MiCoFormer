"""预训练 workflow：可复用的训练逻辑（不含 argparse）"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

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
    # 旧 alias:V5 推荐用 use_hierarchical_embed
    token_embedding_mode: str | None = None
    # V5 主开关
    use_hierarchical_embed: bool = False
    # V4 R2：距离驱动的 attention bias（'none' | 'taxo' | 'phylo'）
    bias_type: str = "phylo"           # V5 默认 phylo
    phylo_mlp_hidden: int = 64          # V5 默认 64(3 层 MLP);旧 alias phylo_bias_hidden 走同字段

    # 2.1. 模型主体参数
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    # ff_dim 与 ff_ratio 互斥：指定其中一个，另一个保持 None
    ff_dim: int | None = None      # FeedForward 绝对维度
    ff_ratio: int | None = None    # FeedForward 比例（dim_ff = d_model × ff_ratio）；未指定时走默认 4
    num_abundance_bins: int = 40

    # 2.2. 模型主体参数的协议参数
    abundance_mode: str = "abs_log_bins"
    min_abundance: float = 4e-6
    max_seq_len: int = 1024

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
    abundance_loss: str = "huber"               # "huber" | "bin_ce"
    use_phylo_pe: bool = True
    phylo_pe_hidden: int = 128
    use_sample_token: bool = False
    pooling_mode: str = "pma"                   # "pma" | "mean_pool"
    pma_nhead: int = 4
    pma_k: int = 1
    use_metadata_task: bool = True
    metadata_field: str = "EnvCategory"
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
        use_metadata_task=config.use_metadata_task,
    )

    # 2. 初始化模型
    rank_zero_info(
        f"{TAG} Initializing Model with d_model={config.d_model}, "
        f"layers={config.num_layers}, ff_dim={effective_ff_dim}"
    )
    rank_zero_info(
        f"{TAG} V5 flags: abundance_encoding={config.abundance_encoding} (loss={config.abundance_loss}), "
        f"use_phylo_pe={config.use_phylo_pe}, pooling={config.pooling_mode}, "
        f"use_sample_token={config.use_sample_token}, use_hierarchical_embed={config.use_hierarchical_embed}, "
        f"use_metadata_task={config.use_metadata_task} (λ={config.metadata_loss_weight})"
    )
    # bias_type != 'none' 时把 var 表大小传给模型（占位 dist_matrix buffer 用），
    # 同时从 datamodule 拿对应的距离矩阵（phylo 或 taxo），注入 encoder
    _n_vars = 0
    _dist_matrix_to_inject = None
    if config.bias_type != "none":
        if config.bias_type == "taxo":
            _dist_matrix_to_inject = dm.taxo_dist_matrix
        elif config.bias_type == "phylo":
            _dist_matrix_to_inject = dm.phylo_dist_matrix
        if _dist_matrix_to_inject is None:
            raise RuntimeError(
                f"bias_type={config.bias_type!r} requires varp['{config.bias_type}_dist'] in h5ad, "
                f"but DataModule did not load it. Check that MCFCorpus has the corresponding varp key."
            )
        _n_vars = int(_dist_matrix_to_inject.shape[0])
        rank_zero_info(
            f"{TAG} R2 bias_type={config.bias_type}, n_vars={_n_vars}, "
            f"dist_matrix dtype={_dist_matrix_to_inject.dtype}"
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
        token_embedding_mode=config.token_embedding_mode,
        rank_vocab_sizes=dm.rank_vocab_sizes,
        bias_type=config.bias_type,
        phylo_mlp_hidden=config.phylo_mlp_hidden,
        n_vars=_n_vars,
        # V5
        abundance_encoding=config.abundance_encoding,
        abundance_loss=config.abundance_loss,
        use_phylo_pe=config.use_phylo_pe,
        phylo_pe_hidden=config.phylo_pe_hidden,
        pe_dim=_pe_dim,
        use_hierarchical_embed=config.use_hierarchical_embed,
        use_sample_token=config.use_sample_token,
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

    # 注入 var-level buffer:dist_matrix(R2) + phylo_pe coords(V5)
    inject_var_buffers(model.encoder, _dist_matrix_to_inject, _pe_coords_to_inject)

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
    #  - find_unused_parameters=True:abund_embed(encoder 无条件创建)在 mlp 路径不参与
    #    forward,否则 DDP backward 会因检测到未用参数而报错。
    if _use_gpu and config.devices > 1:
        trainer_kwargs["strategy"] = DDPStrategy(
            broadcast_buffers=False,
            find_unused_parameters=True,
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
