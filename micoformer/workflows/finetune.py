"""微调 workflow：可复用的训练逻辑（不含 argparse）"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

import anndata as ad
import lightning as L
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info

from micoformer.datamodules.classification_datamodule import ClassificationDataModule
from micoformer.models.classification_module import MiCoFormerClassifier
from micoformer.utils.train_utils import (
    choose_precision,
    validate_finetune_config,
    validate_index_arrays,
    validate_no_split_overlap,
)


TAG = "[train_finetune]"


@dataclass
class FinetuneRunConfig:
    # 0. 输入与切分参数
    h5ad_path: str
    pretrained_ckpt: str

    # 1. 分类头 / pooling 参数
    pooling_mode: str = "mean_pool"
    head_hidden_dim: int = 0
    head_dropout: float = 0.1
    freeze_encoder: bool = False

    # 2.1. 数据协议参数
    batch_size: int = 32
    max_seq_len: int = 1024

    # 3.1. 微调训练主体参数
    lr_head: float = 1e-3
    lr_encoder: float = 1e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1

    # 3.2. 微调协议参数
    # lr_scheduler_type 决定学习率下降方式：
    # - cosine：warmup 后按 cosine 平滑衰减
    # - plateau：warmup 后根据监控指标（默认 val/{task}/f1_macro）是否停滞来自动降 LR
    lr_scheduler_type: str = "cosine"
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 2
    lr_plateau_min_lr: float = 1e-6

    # 4. 预算与验证协议参数
    # budget_mode 决定"训练预算"的单位：
    # - epoch：更适合当前这种数据规模不算特别大的实验
    # - step：更适合超大数据集或只想固定 optimizer 更新次数的场景
    budget_mode: str = "epoch"
    max_epochs: int | None = None
    max_steps: int | None = None
    val_interval_epochs: int | None = None
    val_interval_steps: int | None = None
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # 4.1. Trainer 控制参数（与 pretrain 对称）
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0

    # 5. 运行与工程参数
    devices: int = 1
    precision: str = "auto"
    seed: int = 42
    num_workers: int = 4
    log_dir: str = "tmp/logs"
    no_progress_bar: bool = False


def parse_label_values(label_values_str: str | None) -> dict[str, list[str]] | None:
    """
    解析 'Field=v1,v2' 格式的 label_values 字符串为 {field: [values]} 映射。

    格式说明：
    - 单个 token：'Phenotype=Health,Disease'
    - 多个 token（空格分隔）：'Phenotype=Health,Disease Smoking=Yes,No'
    - 多次调用（list 传入）也支持（由 CLI 的 nargs='*' 拼接后传入）

    示例：
    - parse_label_values("Phenotype=Health,Disease")
      → {"Phenotype": ["Health", "Disease"]}
    """
    if label_values_str is None:
        return None

    result: dict[str, list[str]] = {}
    for token in label_values_str.split():
        if "=" not in token:
            raise ValueError(
                f"label_values token must be 'Field=v1,v2', got {token!r}. "
                f"Example: --label_values \"Phenotype=Health,Disease\""
            )
        field, values_str = token.split("=", 1)
        field = field.strip()
        values = [v.strip() for v in values_str.split(",") if v.strip()]
        if not field:
            raise ValueError(f"Empty field name in label_values token: {token!r}")
        if not values:
            raise ValueError(f"No values specified for field '{field}' in label_values.")
        result[field] = values
    return result if result else None


def build_label_configs(
    label_fields: list[str],
    label_values_str: str | None = None,
) -> list[dict[str, object]]:
    """从标签字段列表和可选的 values 过滤构建 label_configs"""
    label_values_map = parse_label_values(label_values_str)

    label_configs: list[dict[str, object]] = []
    for field in label_fields:
        config: dict[str, object] = {"field": field}
        if label_values_map is not None and field in label_values_map:
            config["values"] = label_values_map[field]
        label_configs.append(config)
    return label_configs


def _extract_scalar_metrics(metrics: dict) -> dict[str, float | int]:
    extracted: dict[str, float | int] = {}
    for key, value in metrics.items():
        scalar_value = value.item() if hasattr(value, "item") else value
        if isinstance(scalar_value, (int, float)):
            extracted[key] = scalar_value
    return extracted


# 执行一次完整的微调流程，返回结果字典
def run_finetune_once(
    config: FinetuneRunConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None,
    label_configs: list[dict[str, object]],
    *,
    log_subdir: str = "finetune_stage0",
) -> dict[str, dict[str, float | int]]:
    validate_finetune_config(config)
    L.seed_everything(config.seed, workers=True)

    # 索引越界 sanity check：避免使用了和 h5ad 不匹配的 splits 文件
    n_obs = validate_index_arrays(
        config.h5ad_path,
        train=train_indices,
        val=val_indices,
        test=test_indices,
    )
    validate_no_split_overlap(train=train_indices, val=val_indices, test=test_indices)

    # 从预训练 checkpoint 读取数据协议参数（BUG-4：继承 abundance 参数）
    from micoformer.models.pretrain_module import MiCoFormerModule as _PretrainModule
    _phparams = _PretrainModule.load_from_checkpoint(
        config.pretrained_ckpt, map_location="cpu"
    ).hparams
    _num_bins = int(_phparams.get("total_abundance_bins", 42)) - 2
    _min_abund = float(_phparams.get("min_abundance", 4e-6))
    _abund_mode = str(_phparams.get("abundance_mode", "abs_log_bins"))
    rank_zero_info(
        f"{TAG} Inherited from pretrain ckpt: num_abundance_bins={_num_bins}, "
        f"min_abundance={_min_abund}, abundance_mode={_abund_mode}"
    )

    chosen_precision = choose_precision(config.precision)
    rank_zero_info(f"{TAG} Using precision={chosen_precision}")
    rank_zero_info(
        f"{TAG} Train: {len(train_indices)}, Val: {len(val_indices)}"
        + (f", Test: {len(test_indices)}" if test_indices is not None else "")
        + f" (n_obs={n_obs})"
    )
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

    # 1. 初始化数据模块（使用从预训练 ckpt 继承的 abundance 参数）
    rank_zero_info(f"{TAG} Initializing DataModule...")
    dm = ClassificationDataModule(
        h5ad_path=config.h5ad_path,
        label_configs=label_configs,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_seq_len=config.max_seq_len,
        num_abundance_bins=_num_bins,
        min_abundance=_min_abund,
        abundance_mode=_abund_mode,
    )

    # 2. 构建任务配置
    task_configs = [
        {"name": cfg["field"], "num_classes": cfg["num_classes"]}
        for cfg in dm.task_configs
    ]
    primary_task = task_configs[0]["name"]
    monitor_metric = f"val/{primary_task}/f1_macro"

    # 3. 初始化分类模型
    rank_zero_info(
        f"{TAG} Initializing classifier with pooling={config.pooling_mode}, "
        f"freeze_encoder={config.freeze_encoder}"
    )
    rank_zero_info(f"{TAG} Monitor metric: {monitor_metric}")
    model = MiCoFormerClassifier(
        pretrained_ckpt_path=config.pretrained_ckpt,
        task_configs=task_configs,
        pooling_mode=config.pooling_mode,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
        freeze_encoder=config.freeze_encoder,
        lr_head=config.lr_head,
        lr_encoder=config.lr_encoder,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler=config.lr_scheduler_type,
        plateau_factor=config.lr_plateau_factor,
        plateau_patience=config.lr_plateau_patience,
        plateau_min_lr=config.lr_plateau_min_lr,
        monitor_metric=monitor_metric,
    )

    # 4. 设置日志记录器与回调
    # 加 uuid 后缀避免同秒并行启动的时间戳碰撞（P2-14）
    run_version = time.strftime("run_%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
    csv_logger = CSVLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)

    # 显式指定 dirpath，避免依赖 "第一个 logger 的 save_dir" 这种隐式行为
    ckpt_dir = os.path.join(config.log_dir, log_subdir, run_version, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=monitor_metric,
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        filename="micoformer-epoch{epoch:02d}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                mode="max",
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
        )

    # 5. 初始化 Trainer
    trainer_kwargs = dict(
        devices=config.devices,
        precision=chosen_precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        default_root_dir=config.log_dir,
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

    # 6. 开始训练
    rank_zero_info(f"{TAG} Starting fine-tuning (log_subdir={log_subdir}) ...")
    trainer.fit(model, datamodule=dm)

    # 7. 收集结果（BUG-2：先 validate best ckpt，再 test，避免 callback_metrics 被 test 污染）
    results: dict[str, dict[str, float | int]] = {}

    val_best = trainer.validate(model, datamodule=dm, ckpt_path="best")
    if val_best:
        results["val"] = _extract_scalar_metrics(val_best[0])

    if test_indices is not None:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
        if test_results:
            results["test"] = _extract_scalar_metrics(test_results[0])

    best_score = checkpoint_callback.best_model_score
    results["best_model_path"] = checkpoint_callback.best_model_path  # type: ignore[assignment]
    results["best_score"] = best_score.item() if best_score is not None else None  # type: ignore[assignment]

    rank_zero_info(f"{TAG} Best checkpoint: {checkpoint_callback.best_model_path}")
    if best_score is not None:
        rank_zero_info(
            f"{TAG} Best {monitor_metric}: {best_score.item():.6f}"
        )
    return results
