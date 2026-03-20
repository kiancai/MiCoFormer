"""
超参数搜索脚本 — MiCoFormer 预训练 baseline (R1=off, R2=off)

三组串行搜索：arch → optim → data
每组使用 random search，前一组最优结果作为下一组的固定参数。

用法：
    python scripts/3.hyperparam_search.py \
        --h5ad data/processed/microbiome_dataset.h5ad \
        --train_indices data/processed/splits/split_group_A.npy \
        --val_indices data/processed/splits/split_group_B.npy \
        --group arch --num_trials 80 --max_steps 15000
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import Callback, LearningRateMonitor

from micoformer.datamodules.mico_datamodule import MiCoDataModule
from micoformer.models.module import MiCoFormerModule

# ---------------------------------------------------------------------------
# 搜索空间定义
# ---------------------------------------------------------------------------

SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    "arch": {
        "d_model": [64, 128, 256, 512, 768],
        "num_layers": [1, 2, 4, 6, 8, 12],
        "nhead": [2, 4, 8, 16],
        "ff_ratio": [2, 4, 8],
        "dropout": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    },
    "optim": {
        "lr": [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
        "batch_size": [16, 32, 64, 128, 256],
        "weight_decay": [0, 1e-3, 1e-2, 5e-2, 1e-1],
        "warmup_steps": [0, 500, 1000, 2000, 4000],
    },
    "data": {
        "num_abundance_bins": [10, 20, 40, 80, 160],
        "mask_prob": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        "min_abundance": [1e-8, 1e-6, 1e-5, 1e-4],
    },
}

# 全局默认值（不在当前搜索组内的参数使用此值）
DEFAULTS: Dict[str, Any] = {
    "d_model": 256,
    "num_layers": 6,
    "nhead": 8,
    "ff_ratio": 4,
    "dropout": 0.1,
    "lr": 3e-4,
    "batch_size": 32,
    "weight_decay": 1e-2,
    "warmup_steps": 2000,
    "num_abundance_bins": 40,
    "mask_prob": 0.15,
    "min_abundance": 4e-6,
}

# 所有超参数名称（用于 CSV 输出列顺序）
ALL_PARAM_NAMES = list(DEFAULTS.keys())


# ---------------------------------------------------------------------------
# 记录历史最佳 val/loss 的回调
# ---------------------------------------------------------------------------

class BestValLossTracker(Callback):
    """在每次验证结束后记录历史最优 val/loss"""

    def __init__(self) -> None:
        super().__init__()
        self.best_val_loss: float = float("inf")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if "val/loss" in metrics:
            current = float(metrics["val/loss"].item())
            if current < self.best_val_loss:
                self.best_val_loss = current


# ---------------------------------------------------------------------------
# run_name 生成
# ---------------------------------------------------------------------------

def _format_number(v: float) -> str:
    """将数值格式化为紧凑字符串，用于 run_name"""
    if isinstance(v, int) or (isinstance(v, float) and v == int(v)):
        return str(int(v))
    # 科学记数法格式
    s = f"{v:.0e}"
    return s


def make_run_name(group: str, config: Dict[str, Any]) -> str:
    """根据搜索组和超参数生成一目了然的 run_name"""
    if group == "arch":
        return (
            f"d{config['d_model']}"
            f"_L{config['num_layers']}"
            f"_h{config['nhead']}"
            f"_ff{config['ff_ratio']}x"
            f"_dp{config['dropout']}"
        )
    elif group == "optim":
        return (
            f"lr{_format_number(config['lr'])}"
            f"_bs{config['batch_size']}"
            f"_wd{_format_number(config['weight_decay'])}"
            f"_wu{config['warmup_steps']}"
        )
    elif group == "data":
        return (
            f"bins{config['num_abundance_bins']}"
            f"_mp{config['mask_prob']}"
            f"_minab{_format_number(config['min_abundance'])}"
        )
    else:
        raise ValueError(f"Unknown group: {group}")


# ---------------------------------------------------------------------------
# 超参数采样
# ---------------------------------------------------------------------------

def sample_configs(
    group: str,
    num_trials: int,
    seed: int,
    base_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    从搜索空间中采样配置。

    - arch / optim 组：随机采样 num_trials 个组合
    - data 组：逐个参数扫描（one-at-a-time），忽略 num_trials

    base_config：从上一组搜索结果继承的固定参数（覆盖 DEFAULTS）。
    """
    rng = random.Random(seed)
    space = SEARCH_SPACES[group]

    # 基础配置 = DEFAULTS + base_config 覆盖
    fixed = dict(DEFAULTS)
    if base_config:
        fixed.update(base_config)

    # data 组：逐个参数扫描，三个参数相互独立无需组合搜索
    if group == "data":
        configs: List[Dict[str, Any]] = []
        seen: set = set()
        for param_name, values in space.items():
            for v in values:
                config = dict(fixed)
                config[param_name] = v
                key = tuple(sorted(config.items()))
                if key in seen:
                    continue
                seen.add(key)
                configs.append(config)
        print(f"Data group: one-at-a-time sweep → {len(configs)} unique configs")
        return configs

    # arch / optim 组：随机采样
    configs: List[Dict[str, Any]] = []
    seen: set = set()

    # 最大尝试次数，防止无限循环
    max_attempts = num_trials * 20

    for _ in range(max_attempts):
        if len(configs) >= num_trials:
            break

        # 从搜索空间随机采样当前组的参数
        sampled = {k: rng.choice(v) for k, v in space.items()}

        # arch 组需要处理 nhead 整除 d_model 的约束
        if group == "arch":
            d_model = sampled["d_model"]
            nhead = sampled["nhead"]
            if d_model % nhead != 0:
                # rejection sampling：重新采样 nhead
                valid_nheads = [h for h in space["nhead"] if d_model % h == 0]
                if not valid_nheads:
                    continue  # 跳过这个 d_model（理论上不会发生）
                sampled["nhead"] = rng.choice(valid_nheads)

        # 合并固定参数和采样参数
        config = dict(fixed)
        config.update(sampled)

        # 去重
        key = tuple(sorted(config.items()))
        if key in seen:
            continue
        seen.add(key)

        configs.append(config)

    if len(configs) < num_trials:
        print(
            f"Warning: only sampled {len(configs)} unique configs "
            f"(requested {num_trials})"
        )

    return configs


# ---------------------------------------------------------------------------
# base_config 读取
# ---------------------------------------------------------------------------

def load_base_config(csv_path: str) -> Dict[str, Any]:
    """
    从上一组搜索的 summary CSV 读取 top-1 配置（按 val_loss 升序排列的第一行）。
    返回该行中所有能匹配到 DEFAULTS 键的参数。
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        row = next(reader)  # 第一行 = 最优配置

    config: Dict[str, Any] = {}
    for key in ALL_PARAM_NAMES:
        if key in row and row[key] != "":
            raw = row[key]
            # 尝试解析为数值
            try:
                # 先尝试 int
                if "." not in raw and "e" not in raw.lower():
                    config[key] = int(raw)
                else:
                    config[key] = float(raw)
            except ValueError:
                config[key] = raw

    return config


# ---------------------------------------------------------------------------
# resume 恢复：读取已有 summary CSV 中的历史结果
# ---------------------------------------------------------------------------

def load_existing_results(csv_path: str) -> List[Dict[str, Any]]:
    """从已有 summary CSV 中读取所有历史结果（用于 resume 时合并）"""
    results: List[Dict[str, Any]] = []
    if not Path(csv_path).exists():
        return results
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if k == "rank":
                    continue  # rank 会重新计算
                try:
                    if v == "inf":
                        parsed[k] = float("inf")
                    elif "." in v or "e" in v.lower():
                        parsed[k] = float(v)
                    else:
                        parsed[k] = int(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            results.append(parsed)
    return results


# ---------------------------------------------------------------------------
# 单次试验执行
# ---------------------------------------------------------------------------

def run_single_trial(
    config: Dict[str, Any],
    args: argparse.Namespace,
    run_name: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Dict[str, Any]:
    """
    执行单次试验，返回包含 val_loss 和 status 的结果字典。
    """
    t0 = time.time()

    # 计算 dim_feedforward（绝对值）
    dim_feedforward = config["d_model"] * config["ff_ratio"]

    # 自动精度选择
    precision = "16-mixed" if torch.cuda.is_available() else "32"

    # 动态调整 val_check_interval，防止 batch_size 大时 num_batches < val_check_interval
    num_train_batches = math.ceil(len(train_indices) / config["batch_size"])
    val_check_interval = min(args.val_check_interval, num_train_batches)
    if val_check_interval != args.val_check_interval:
        print(f"  val_check_interval adjusted: {args.val_check_interval} → {val_check_interval} "
              f"(batch_size={config['batch_size']}, num_batches={num_train_batches})")

    result: Dict[str, Any] = {**config, "run_name": run_name, "val_loss": float("inf"), "status": "UNKNOWN"}

    # 显式声明，确保 finally 块可以清理
    dm = None
    model = None
    trainer = None

    try:
        # 创建 DataModule（baseline 模式）
        dm = MiCoDataModule(
            h5ad_path=args.h5ad,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=None,
            batch_size=config["batch_size"],
            num_workers=args.num_workers,
            max_seq_len=1024,
            mask_prob=config["mask_prob"],
            num_abundance_bins=config["num_abundance_bins"],
            min_abundance=config["min_abundance"],
            abundance_mode="abs_log_bins",
            token_embedding_mode="taxon",       # baseline: R1=off
            use_taxonomy_bias=False,             # baseline: R2=off
        )

        # 创建模型
        model = MiCoFormerModule(
            genus_vocab_size=dm.genus_vocab_size,
            total_abundance_bins=dm.total_abundance_bins,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=dim_feedforward,
            dropout=config["dropout"],
            pad_taxon_id=dm.special_ids["pad_taxon_id"],
            pad_bin_id=dm.special_ids["pad_bin_id"],
            token_embedding_mode="taxon",
            rank_vocab_sizes=dm.rank_vocab_sizes,
            use_taxonomy_bias=False,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            warmup_steps=config["warmup_steps"],
            max_steps=args.max_steps,
        )

        # 日志目录：{log_dir}/hpsearch_{group}/{run_name}
        group_log_dir = f"hpsearch_{args.group}"
        tb_logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=group_log_dir,
            version=run_name,
        )
        csv_logger = CSVLogger(
            save_dir=args.log_dir,
            name=group_log_dir,
            version=run_name,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        # 用回调追踪历史最优 val/loss（而非仅取最后一次）
        val_tracker = BestValLossTracker()

        # GPU 设备选择
        if torch.cuda.is_available():
            devices = [args.gpu_id]
        else:
            devices = 1

        trainer = L.Trainer(
            max_steps=args.max_steps,
            max_epochs=10000,  # 不限制 epoch，仅靠 max_steps 控制
            devices=devices,
            precision=precision,
            gradient_clip_val=1.0,
            logger=[tb_logger, csv_logger],
            callbacks=[lr_monitor, val_tracker],
            enable_checkpointing=False,   # 搜索阶段不保存 checkpoint
            enable_progress_bar=False,    # 关闭进度条，避免刷屏
            val_check_interval=val_check_interval,
            enable_model_summary=False,
        )

        trainer.fit(model, datamodule=dm)

        # 从回调中获取历史最优 val/loss
        best_val_loss = val_tracker.best_val_loss
        if best_val_loss == float("inf"):
            # 验证从未执行（max_steps 太小导致没到 val_check_interval）
            result["status"] = "NO_VAL"
            print("  Warning: validation never ran (max_steps < val_check_interval?)")
        else:
            result["val_loss"] = best_val_loss
            result["status"] = "OK"

    except RuntimeError as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg:
            result["status"] = "OOM"
            print(f"  OOM detected, skipping this config")
        else:
            result["status"] = f"ERROR: {e}"
            print(f"  RuntimeError: {e}")
    except Exception as e:
        result["status"] = f"ERROR: {e}"
        print(f"  Exception: {e}")
    finally:
        # 显式清理引用，帮助 GC 回收 GPU 内存
        del model, trainer, dm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"
    result["elapsed"] = elapsed_str

    return result


# ---------------------------------------------------------------------------
# 结果汇总
# ---------------------------------------------------------------------------

def save_summary(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """按 val_loss 升序排列并保存 summary CSV"""
    # 按 val_loss 排序
    results_sorted = sorted(results, key=lambda r: r.get("val_loss", float("inf")))

    # CSV 列顺序
    fieldnames = ["rank", "run_name"] + ALL_PARAM_NAMES + ["val_loss", "status", "elapsed"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(results_sorted):
            row_out = dict(row)
            row_out["rank"] = i + 1
            writer.writerow(row_out)

    print(f"\nSummary saved to: {output_path}")


def print_results_table(results: List[Dict[str, Any]], group: str, top_n: int = 10) -> None:
    """打印排名表"""
    results_sorted = sorted(results, key=lambda r: r.get("val_loss", float("inf")))

    print(f"\n{'='*80}")
    print(f"  Top {min(top_n, len(results_sorted))} results for group '{group}'")
    print(f"{'='*80}")

    for i, r in enumerate(results_sorted[:top_n]):
        status = r.get("status", "?")
        val_loss = r.get("val_loss", float("inf"))
        elapsed = r.get("elapsed", "?")
        run_name = r.get("run_name", "?")

        if val_loss == float("inf"):
            loss_str = "   inf   "
        else:
            loss_str = f"{val_loss:.4f}"

        print(f"  #{i+1:3d}  {loss_str}  {status:<5s}  ({elapsed})  {run_name}")

    print(f"{'='*80}")

    # 打印最佳配置的完整参数
    if results_sorted and results_sorted[0].get("val_loss", float("inf")) < float("inf"):
        best = results_sorted[0]
        print(f"\nBest config (val/loss={best['val_loss']:.4f}):")
        for k in ALL_PARAM_NAMES:
            if k in best:
                print(f"  {k}: {best[k]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MiCoFormer 超参数搜索脚本（baseline R1=off, R2=off）"
    )

    # 数据参数
    p.add_argument("--h5ad", type=str, required=True, help="AnnData (.h5ad) 文件路径")
    p.add_argument("--train_indices", type=str, required=True, help="训练集索引 .npy")
    p.add_argument("--val_indices", type=str, required=True, help="验证集索引 .npy")

    # 搜索控制
    p.add_argument(
        "--group", type=str, required=True, choices=["arch", "optim", "data"],
        help="搜索组: arch / optim / data",
    )
    p.add_argument("--num_trials", type=int, default=80, help="随机采样试验数")
    p.add_argument("--max_steps", type=int, default=15000, help="每次试验的训练步数")
    p.add_argument("--log_dir", type=str, default="tmp/logs", help="日志根目录")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader 进程数")
    p.add_argument(
        "--base_config", type=str, default=None,
        help="上一组搜索的 summary CSV 路径，自动读取 top-1 作为固定参数",
    )
    p.add_argument(
        "--resume_from_trial", type=int, default=0,
        help="中断恢复：跳过前 N 个试验（会自动从已有 summary CSV 加载历史结果）",
    )
    p.add_argument(
        "--val_check_interval", type=int, default=500,
        help="每多少步验证一次（会自动调低到不超过每 epoch 的 batch 数）",
    )
    p.add_argument(
        "--gpu_id", type=int, default=0,
        help="指定使用哪张 GPU（默认 0）。多卡并行时启动多个进程各用不同 gpu_id",
    )
    p.add_argument(
        "--retry_errors", action="store_true",
        help="重跑模式：只重跑 summary CSV 中状态为 ERROR 的 trial（保留已有 OK 结果）",
    )

    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = build_argparser().parse_args()

    print(f"Hyperparameter search: group={args.group}, "
          f"trials={args.num_trials}, max_steps={args.max_steps}")

    # 设置全局种子
    L.seed_everything(args.seed, workers=True)

    # 加载分割索引
    train_indices = np.load(args.train_indices)
    val_indices = np.load(args.val_indices)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # 加载 base_config（如果指定）
    base_config = None
    if args.base_config:
        base_config = load_base_config(args.base_config)
        print(f"Loaded base config from {args.base_config}:")
        for k, v in base_config.items():
            print(f"  {k}: {v}")

    # 采样超参数配置
    configs = sample_configs(
        group=args.group,
        num_trials=args.num_trials,
        seed=args.seed,
        base_config=base_config,
    )
    print(f"Sampled {len(configs)} unique configs")

    # 确保日志目录存在
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    summary_path = str(Path(args.log_dir) / f"hpsearch_{args.group}_summary.csv")

    # --retry_errors 模式：只重跑失败的 trial
    if args.retry_errors:
        existing = load_existing_results(summary_path)
        if not existing:
            print(f"No existing results found at {summary_path}, nothing to retry.")
            return

        ok_results = []
        error_configs = []
        for r in existing:
            status = str(r.get("status", ""))
            val_loss = r.get("val_loss", float("inf"))
            if status == "OK":
                ok_results.append(r)
            else:
                # 从结果行中重建 config
                cfg = {k: r[k] for k in ALL_PARAM_NAMES if k in r}
                run_name = r.get("run_name", make_run_name(args.group, cfg))
                error_configs.append((cfg, run_name))

        print(f"Loaded {len(existing)} existing results: "
              f"{len(ok_results)} OK, {len(error_configs)} to retry")

        if not error_configs:
            print("All trials succeeded, nothing to retry.")
            return

        results = list(ok_results)
        for i, (cfg, run_name) in enumerate(error_configs):
            print(f"\n[RETRY {i+1}/{len(error_configs)}] {run_name}")
            result = run_single_trial(
                config=cfg, args=args, run_name=run_name,
                train_indices=train_indices, val_indices=val_indices,
            )
            results.append(result)

            val_loss = result.get("val_loss", float("inf"))
            status = result.get("status", "?")
            elapsed = result.get("elapsed", "?")
            loss_str = "inf" if val_loss == float("inf") else f"{val_loss:.4f}"
            print(f"  => val/loss={loss_str}  {status}  ({elapsed})")

            save_summary(results, summary_path)

        print_results_table(results, args.group)
        return

    # 正常搜索模式
    results: List[Dict[str, Any]] = []
    if args.resume_from_trial > 0:
        existing = load_existing_results(summary_path)
        if existing:
            results.extend(existing)
            print(f"Loaded {len(existing)} existing results from {summary_path}")

    # 执行试验
    for i, config in enumerate(configs):
        # 中断恢复：跳过前 N 个试验
        if i < args.resume_from_trial:
            print(f"[{i+1}/{len(configs)}] SKIPPED (resume_from_trial={args.resume_from_trial})")
            continue

        run_name = make_run_name(args.group, config)
        print(f"\n[{i+1}/{len(configs)}] {run_name}")

        result = run_single_trial(
            config=config,
            args=args,
            run_name=run_name,
            train_indices=train_indices,
            val_indices=val_indices,
        )

        results.append(result)

        # 实时打印结果
        val_loss = result.get("val_loss", float("inf"))
        status = result.get("status", "?")
        elapsed = result.get("elapsed", "?")
        if val_loss == float("inf"):
            print(f"  => val/loss=inf  {status}  ({elapsed})")
        else:
            print(f"  => val/loss={val_loss:.4f}  {status}  ({elapsed})")

        # 每个 trial 结束后立即保存 summary（防止中途崩溃丢失所有结果）
        save_summary(results, summary_path)

    # 最终汇总
    if results:
        save_summary(results, summary_path)
        print_results_table(results, args.group)
    else:
        print("No trials were executed.")


if __name__ == "__main__":
    main()
