"""V2.5 超参数验证协议 runtime helpers.

职责：
- 从 full_training/config.py 读取 V2 确定的固定参数
- 构建 pretrain / finetune / evaluate_c 三种 stage block 的任务列表
- 多 GPU 调度执行
- CSV / JSON / YAML 持久化
- R2 bias_table 验证工具
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:
    yaml = None


# ─── 路径定位 ──────────────────────────────────────────────────────────────

PROTOCOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 从 full_training/config.py 导入 V2 确定的参数
from protocols.full_training.config import (  # noqa: E402
    SHARED_ARCH,
    SHARED_FINETUNE,
    SHARED_PRETRAIN,
    VARIANTS,
)


# ─── 全局常量 ──────────────────────────────────────────────────────────────

DEFAULT_SEED = 42
DEFAULT_LABEL_FIELD = "Is_Healthy"
DEFAULT_LABEL_VALUES = ["True", "False"]
DEFAULT_GPU_COOLDOWN_SECONDS = 15
DEFAULT_CPU_THREADS = 1

STAGE_BLOCK_SPECS: dict[str, dict[str, Any]] = {
    "pretrain": {
        "stage_name": "pretrain",
        "plan_filename": "pretrain_plan.csv",
        "live_status_filename": "pretrain_live_status.csv",
        "summary_filename": "pretrain_summary.csv",
        "dashboard_filename": "pretrain_dashboard.json",
        "log_dirname": "pretrain",
        "tb_port": "6201",
    },
    "finetune": {
        "stage_name": "finetune",
        "plan_filename": "finetune_plan.csv",
        "live_status_filename": "finetune_live_status.csv",
        "summary_filename": "finetune_summary.csv",
        "dashboard_filename": "finetune_dashboard.json",
        "log_dirname": "finetune",
        "tb_port": "6202",
    },
    "evaluate_c": {
        "stage_name": "evaluate_c",
        "plan_filename": "evaluate_c_plan.csv",
        "live_status_filename": "evaluate_c_live_status.csv",
        "summary_filename": "evaluate_c_summary.csv",
        "dashboard_filename": "evaluate_c_dashboard.json",
        "log_dirname": "evaluate_c",
        "tb_port": "6203",
    },
}


# ─── CSV I/O ───────────────────────────────────────────────────────────────

def _coerce_scalar(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return ""
    if raw in {"True", "False"}:
        return raw == "True"
    if raw in {"None", "null"}:
        return None
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except Exception:
            pass
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{k: _coerce_scalar(v) for k, v in row.items()} for row in reader]


def write_rows(rows: list[dict[str, Any]], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    preferred = [
        "trial_id", "model_variant", "task_kind", "status", "seed",
        "best_val_loss", "checkpoint_path", "elapsed_seconds", "gpu_id",
        "val_macro_f1", "val_auroc", "val_accuracy",
        "test_macro_f1", "test_auroc", "test_accuracy",
        "study_id", "run_name", "error_message",
    ]
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    serialized[key] = json.dumps(value, ensure_ascii=True)
                else:
                    serialized[key] = value
            writer.writerow(serialized)


# ─── Manifest / JSON / YAML I/O ───────────────────────────────────────────

def write_manifest(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML required for YAML output")
        with path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    elif path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2, ensure_ascii=True, default=str))
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=True, default=str))


def read_manifest(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML required for YAML input")
        with path.open("r") as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        return json.loads(path.read_text())
    else:
        return json.loads(path.read_text())


# ─── Dashboard ─────────────────────────────────────────────────────────────

def _write_dashboard(rows: list[dict[str, Any]], dashboard_path: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        s = str(row.get("status", "UNKNOWN"))
        counts[s] = counts.get(s, 0) + 1
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_trials": len(rows),
        "counts": counts,
        "ok_trials": counts.get("OK", 0),
        "failed_trials": sum(v for k, v in counts.items() if k not in ("OK", "PENDING", "RUNNING")),
    }
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


# ─── 环境快照 ──────────────────────────────────────────────────────────────

def capture_environment_snapshot(output_path: str | Path, gpu_ids: list[int] | None = None) -> None:
    output_path = Path(output_path)
    try:
        import torch
    except Exception:
        torch = None

    lines = [
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version.split()[0]}",
    ]

    def _try_version(mod: str) -> str:
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "unknown")
        except Exception as exc:
            return f"unavailable ({exc})"

    lines.append(f"torch_version: {_try_version('torch')}")
    lines.append(f"lightning_version: {_try_version('lightning')}")
    lines.append(f"anndata_version: {_try_version('anndata')}")

    if gpu_ids is not None:
        lines.append(f"gpu_ids: {gpu_ids}")
    if torch is not None:
        try:
            lines.append(f"cuda_available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
                names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                lines.append(f"cuda_device_names: {names}")
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


# ─── 目录与布局 ────────────────────────────────────────────────────────────

def make_run_id(prefix: str = "v25") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def init_run_dir(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    layout = {
        "run_dir": run_dir,
        "config": run_dir / "config",
        "decisions": run_dir / "decisions",
        "splits": run_dir / "splits",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def get_stage_block_paths(run_dir: str | Path, stage_block: str) -> dict[str, Path]:
    spec = get_stage_block_spec(stage_block)
    run_dir = Path(run_dir)
    stage_dir = run_dir / spec["stage_name"]
    return {
        "stage_dir": stage_dir,
        "plan": stage_dir / spec["plan_filename"],
        "live_status": stage_dir / spec["live_status_filename"],
        "summary": stage_dir / spec["summary_filename"],
        "dashboard": stage_dir / spec["dashboard_filename"],
        "log_dir": stage_dir / spec["log_dirname"],
    }


def get_stage_block_spec(stage_block: str) -> dict[str, Any]:
    if stage_block not in STAGE_BLOCK_SPECS:
        raise ValueError(f"Unknown stage_block: {stage_block}. Available: {list(STAGE_BLOCK_SPECS.keys())}")
    return STAGE_BLOCK_SPECS[stage_block]


def detect_available_cpu_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def resolve_cpu_runtime_settings(
    requested_num_workers: int | None = None,
    requested_cpu_threads: int | None = None,
) -> dict[str, int]:
    available = detect_available_cpu_cores()
    workers = requested_num_workers or 4
    threads = requested_cpu_threads or DEFAULT_CPU_THREADS
    safe_workers = max(1, min(workers, available - 1))
    safe_threads = max(1, min(threads, available))
    return {
        "available_cpu_cores": available,
        "requested_num_workers": workers,
        "safe_num_workers": safe_workers,
        "safe_cpu_threads": safe_threads,
    }


def apply_cpu_runtime_settings(cpu_threads: int) -> None:
    try:
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    except Exception:
        pass


# ─── 任务构建：Pretrain ───────────────────────────────────────────────────

def build_pretrain_tasks(
    variant_names: list[str],
    seed: int,
    h5ad_path: str,
    run_dir: Path,
    splits_dir: Path,
    num_workers: int = 4,
) -> list[dict[str, Any]]:
    """构建 4 个 pretrain task dict（每个 variant 1 个）。"""
    run_dir = Path(run_dir)
    splits_dir = Path(splits_dir)
    train_path = splits_dir / "pretrain_train.npy"
    val_path = splits_dir / "pretrain_val.npy"

    tasks = []
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        trial_id = f"{variant_name}_pretrain_seed{seed}"

        task = {
            "trial_id": trial_id,
            "task_kind": "pretrain",
            "model_variant": variant_name,
            "run_name": trial_id,
            "seed": seed,
            # 数据
            "h5ad_path": str(h5ad_path),
            "train_indices_path": str(train_path),
            "val_indices_path": str(val_path),
            # 模型开关
            "token_embedding_mode": variant.token_embedding_mode,
            "use_taxonomy_bias": variant.use_taxonomy_bias,
            "bias_grad_every_k": SHARED_PRETRAIN["bias_grad_every_k"],
            # 共享架构
            "d_model": SHARED_ARCH["d_model"],
            "num_layers": SHARED_ARCH["num_layers"],
            "nhead": SHARED_ARCH["nhead"],
            "ff_ratio": SHARED_ARCH["ff_ratio"],
            "num_abundance_bins": SHARED_ARCH["num_abundance_bins"],
            "dropout": SHARED_ARCH["dropout"],
            # 共享预训练参数
            "warmup_ratio": SHARED_PRETRAIN["warmup_ratio"],
            "mask_prob": SHARED_PRETRAIN["mask_prob"],
            "min_abundance": SHARED_PRETRAIN["min_abundance"],
            "abundance_mode": SHARED_PRETRAIN["abundance_mode"],
            "max_seq_len": SHARED_PRETRAIN["max_seq_len"],
            "lr_scheduler_type": SHARED_PRETRAIN["lr_scheduler_type"],
            # 变体预训练参数
            "lr": variant.pretrain_lr,
            "batch_size": variant.pretrain_batch_size,
            "weight_decay": variant.pretrain_weight_decay,
            # 预算
            "budget_mode": "epoch",
            "max_epochs": 20,
            "val_interval_epochs": 2,
            "early_stopping_patience": 0,
            "early_stopping_min_delta": 0.0,
            # 运行时
            "devices": 1,
            "precision": "auto",
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "limit_train_batches": 1.0,
            "limit_val_batches": 1.0,
            "num_workers": num_workers,
            "no_progress_bar": True,
            # 日志
            "log_dir": str(run_dir / "pretrain"),
            "log_subdir": f"{variant_name}/seed{seed}",
        }
        tasks.append(task)

    return tasks


# ─── 任务构建：Finetune ───────────────────────────────────────────────────

def build_finetune_tasks(
    variant_names: list[str],
    pretrained_ckpts: dict[str, str],
    seed: int,
    h5ad_path: str,
    run_dir: Path,
    splits_dir: Path,
    num_workers: int = 4,
) -> list[dict[str, Any]]:
    """构建 4 个 finetune task dict（每个 variant 1 个）。"""
    run_dir = Path(run_dir)
    splits_dir = Path(splits_dir)
    train_path = splits_dir / "pretrain_train.npy"
    val_path = splits_dir / "pretrain_val.npy"

    tasks = []
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        trial_id = f"{variant_name}_finetune_seed{seed}"

        task = {
            "trial_id": trial_id,
            "task_kind": "finetune",
            "model_variant": variant_name,
            "run_name": trial_id,
            "seed": seed,
            # 数据
            "h5ad_path": str(h5ad_path),
            "train_indices_path": str(train_path),
            "val_indices_path": str(val_path),
            "test_indices_path": str(val_path),  # 同时在 B 上评估
            "pretrained_ckpt": pretrained_ckpts[variant_name],
            # 标签
            "label_field": DEFAULT_LABEL_FIELD,
            "label_values": DEFAULT_LABEL_VALUES,
            # 共享微调参数
            "pooling_mode": SHARED_FINETUNE["pooling_mode"],
            "freeze_encoder": SHARED_FINETUNE["freeze_encoder"],
            "batch_size": SHARED_FINETUNE["batch_size"],
            "weight_decay": SHARED_FINETUNE["weight_decay"],
            "warmup_ratio": SHARED_FINETUNE["warmup_ratio"],
            "lr_scheduler_type": SHARED_FINETUNE["lr_scheduler_type"],
            "max_seq_len": SHARED_PRETRAIN["max_seq_len"],
            # 变体微调参数
            "lr_head": variant.lr_head,
            "lr_encoder": variant.lr_encoder,
            "head_hidden_dim": variant.head_hidden_dim,
            "head_dropout": variant.head_dropout,
            # 预算
            "budget_mode": "epoch",
            "max_epochs": 20,
            "val_interval_epochs": 1,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
            # 运行时
            "devices": 1,
            "precision": "auto",
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0,
            "limit_train_batches": 1.0,
            "limit_val_batches": 1.0,
            "num_workers": num_workers,
            "no_progress_bar": True,
            # 日志
            "log_dir": str(run_dir / "finetune"),
            "log_subdir": f"{variant_name}/seed{seed}",
        }
        tasks.append(task)

    return tasks


# ─── 任务构建：Evaluate C ─────────────────────────────────────────────────

def build_evaluate_c_tasks(
    variant_names: list[str],
    finetuned_ckpts: dict[str, str],
    seed: int,
    h5ad_path: str,
    run_dir: Path,
    splits_dir: Path,
    label_field: str = DEFAULT_LABEL_FIELD,
    label_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    """构建 per-study 评估 task dict。每个 (variant, study_id) 一个 task。"""
    if label_values is None:
        label_values = DEFAULT_LABEL_VALUES

    run_dir = Path(run_dir)
    splits_dir = Path(splits_dir)

    # 读取 C 的全局索引
    c_indices = np.load(str(splits_dir / "pretrain_test_c.npy"))

    # 打开 h5ad 按 Project_ID 分组
    import anndata as ad
    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    try:
        obs_c = adata.obs.iloc[c_indices]
        study_groups = obs_c.groupby("Project_ID", sort=True)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    # 为每个 study 保存 test indices
    evaluate_dir = run_dir / "evaluate_c"
    evaluate_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for study_id, group_df in study_groups:
        study_test_indices = c_indices[group_df.index.values]
        study_npy_path = splits_dir / f"eval_c_{study_id}.npy"
        np.save(str(study_npy_path), study_test_indices)

        for variant_name in variant_names:
            trial_id = f"{variant_name}_eval_{study_id}_seed{seed}"

            task = {
                "trial_id": trial_id,
                "task_kind": "evaluate",
                "model_variant": variant_name,
                "run_name": trial_id,
                "seed": seed,
                "study_id": study_id,
                "n_test": len(study_test_indices),
                # 数据
                "h5ad_path": str(h5ad_path),
                "test_indices_path": str(study_npy_path),
                "finetuned_ckpt": finetuned_ckpts[variant_name],
                # 标签
                "label_field": label_field,
                "label_values": label_values,
                # 运行时
                "batch_size": SHARED_FINETUNE["batch_size"],
                "num_workers": 4,
                # 日志
                "log_dir": str(evaluate_dir),
                "log_subdir": f"{variant_name}/{study_id}",
            }
            tasks.append(task)

    return tasks


# ─── 配置构建 ──────────────────────────────────────────────────────────────

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return bool(value)


def _prepare_pretrain_config(task: dict[str, Any]) -> Any:
    from micoformer.workflows.pretrain import PretrainRunConfig

    return PretrainRunConfig(
        h5ad_path=str(task["h5ad_path"]),
        token_embedding_mode=str(task["token_embedding_mode"]),
        use_taxonomy_bias=_as_bool(task["use_taxonomy_bias"]),
        bias_grad_every_k=int(task.get("bias_grad_every_k", 1)),
        d_model=int(task["d_model"]),
        nhead=int(task["nhead"]),
        num_layers=int(task["num_layers"]),
        ff_dim=None,
        ff_ratio=int(task["ff_ratio"]),
        num_abundance_bins=int(task["num_abundance_bins"]),
        abundance_mode=str(task["abundance_mode"]),
        min_abundance=float(task["min_abundance"]),
        max_seq_len=int(task["max_seq_len"]),
        batch_size=int(task["batch_size"]),
        mask_prob=float(task["mask_prob"]),
        dropout=float(task["dropout"]),
        lr=float(task["lr"]),
        weight_decay=float(task["weight_decay"]),
        warmup_ratio=float(task["warmup_ratio"]),
        lr_scheduler_type=str(task.get("lr_scheduler_type", "cosine")),
        lr_plateau_factor=0.5,
        lr_plateau_patience=2,
        lr_plateau_min_lr=1e-6,
        budget_mode=str(task["budget_mode"]),
        max_epochs=int(task["max_epochs"]),
        max_steps=None,
        val_interval_epochs=int(task["val_interval_epochs"]),
        val_interval_steps=None,
        limit_train_batches=float(task.get("limit_train_batches", 1.0)),
        limit_val_batches=float(task.get("limit_val_batches", 1.0)),
        devices=int(task.get("devices", 1)),
        precision=str(task.get("precision", "auto")),
        seed=int(task["seed"]),
        accumulate_grad_batches=int(task.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(task.get("gradient_clip_val", 1.0)),
        num_workers=int(task.get("num_workers", 4)),
        log_dir=str(task["log_dir"]),
        no_progress_bar=_as_bool(task.get("no_progress_bar", True)),
        early_stopping_patience=int(task.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(task.get("early_stopping_min_delta", 0.0)),
    )


def _prepare_finetune_config(task: dict[str, Any]) -> Any:
    from micoformer.workflows.finetune import FinetuneRunConfig

    return FinetuneRunConfig(
        h5ad_path=str(task["h5ad_path"]),
        pretrained_ckpt=str(task["pretrained_ckpt"]),
        pooling_mode=str(task["pooling_mode"]),
        head_hidden_dim=int(task["head_hidden_dim"]),
        head_dropout=float(task["head_dropout"]),
        freeze_encoder=_as_bool(task["freeze_encoder"]),
        batch_size=int(task["batch_size"]),
        max_seq_len=int(task.get("max_seq_len", 1024)),
        lr_head=float(task["lr_head"]),
        lr_encoder=float(task["lr_encoder"]),
        weight_decay=float(task["weight_decay"]),
        warmup_ratio=float(task["warmup_ratio"]),
        lr_scheduler_type=str(task.get("lr_scheduler_type", "cosine")),
        lr_plateau_factor=0.5,
        lr_plateau_patience=2,
        lr_plateau_min_lr=1e-6,
        budget_mode=str(task["budget_mode"]),
        max_epochs=int(task["max_epochs"]),
        max_steps=None,
        val_interval_epochs=int(task["val_interval_epochs"]),
        val_interval_steps=None,
        early_stopping_patience=int(task.get("early_stopping_patience", 5)),
        early_stopping_min_delta=float(task.get("early_stopping_min_delta", 0.0)),
        gradient_clip_val=float(task.get("gradient_clip_val", 1.0)),
        accumulate_grad_batches=int(task.get("accumulate_grad_batches", 1)),
        limit_train_batches=float(task.get("limit_train_batches", 1.0)),
        limit_val_batches=float(task.get("limit_val_batches", 1.0)),
        devices=int(task.get("devices", 1)),
        precision=str(task.get("precision", "auto")),
        seed=int(task["seed"]),
        num_workers=int(task.get("num_workers", 4)),
        log_dir=str(task["log_dir"]),
        no_progress_bar=_as_bool(task.get("no_progress_bar", True)),
    )


# ─── 任务执行 ──────────────────────────────────────────────────────────────

def _timestamp_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _execute_task(task: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    """执行单个 task，根据 task_kind 分发到 pretrain / finetune / evaluate。"""
    start = time.time()
    row = dict(task)
    row["gpu_id"] = gpu_id
    row["start_time"] = _timestamp_now()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        task_kind = str(task["task_kind"])

        if task_kind == "pretrain":
            from micoformer.workflows.pretrain import run_pretrain_once

            config = _prepare_pretrain_config(task)
            train_indices = np.load(str(task["train_indices_path"]))
            val_indices = np.load(str(task["val_indices_path"]))
            result = run_pretrain_once(
                config, train_indices, val_indices,
                log_subdir=str(task["log_subdir"]),
            )
            row["best_val_loss"] = result.get("best_val_loss")
            row["best_score"] = result.get("best_score")
            row["checkpoint_path"] = result.get("best_model_path", "")

        elif task_kind == "finetune":
            from micoformer.workflows.finetune import build_label_configs, run_finetune_once

            config = _prepare_finetune_config(task)
            train_indices = np.load(str(task["train_indices_path"]))
            val_indices = np.load(str(task["val_indices_path"]))
            test_indices = np.load(str(task["test_indices_path"]))
            label_field = str(task["label_field"])
            label_values = task.get("label_values", DEFAULT_LABEL_VALUES)
            label_values_str = f"{label_field}={','.join(label_values)}"
            label_configs = build_label_configs([label_field], label_values_str)

            result = run_finetune_once(
                config,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                label_configs=label_configs,
                log_subdir=str(task["log_subdir"]),
            )
            val_metrics = result.get("val", {})
            test_metrics = result.get("test", {})
            row["checkpoint_path"] = result.get("best_model_path", "")
            row["best_score"] = result.get("best_score")
            row["val_macro_f1"] = val_metrics.get(f"val/{label_field}/f1_macro")
            row["val_auroc"] = val_metrics.get(f"val/{label_field}/auroc")
            row["val_accuracy"] = val_metrics.get(f"val/{label_field}/acc")
            row["test_macro_f1"] = test_metrics.get(f"test/{label_field}/f1_macro")
            row["test_auroc"] = test_metrics.get(f"test/{label_field}/auroc")
            row["test_accuracy"] = test_metrics.get(f"test/{label_field}/acc")

        elif task_kind == "evaluate":
            test_indices = np.load(str(task["test_indices_path"]))
            label_field = str(task["label_field"])
            label_values = task.get("label_values", DEFAULT_LABEL_VALUES)

            result = evaluate_checkpoint(
                finetuned_ckpt=str(task["finetuned_ckpt"]),
                test_indices=test_indices,
                h5ad_path=str(task["h5ad_path"]),
                label_field=label_field,
                label_values=label_values,
                seed=int(task["seed"]),
                num_workers=int(task.get("num_workers", 4)),
                batch_size=int(task.get("batch_size", 32)),
            )
            row["test_macro_f1"] = result.get(f"test/{label_field}/f1_macro")
            row["test_auroc"] = result.get(f"test/{label_field}/auroc")
            row["test_accuracy"] = result.get(f"test/{label_field}/acc")
            row["test_weighted_f1"] = result.get(f"test/{label_field}/f1_weighted")
            row["checkpoint_path"] = str(task["finetuned_ckpt"])
        else:
            raise ValueError(f"Unknown task_kind: {task_kind}")

        row["status"] = "OK"
        row["error_message"] = ""

    except RuntimeError as exc:
        message = str(exc)
        row["status"] = "OOM" if "out of memory" in message.lower() else "ERROR"
        row["error_message"] = message
        row["checkpoint_path"] = ""
    except Exception as exc:
        row["status"] = "ERROR"
        row["error_message"] = f"{exc}\n{traceback.format_exc()}"
        row["checkpoint_path"] = ""
    finally:
        row["elapsed_seconds"] = round(time.time() - start, 2)
        row["end_time"] = _timestamp_now()
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return row


# ─── 纯测试评估 ────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    finetuned_ckpt: str,
    test_indices: np.ndarray,
    h5ad_path: str,
    label_field: str = DEFAULT_LABEL_FIELD,
    label_values: list[str] | None = None,
    seed: int = 42,
    num_workers: int = 4,
    batch_size: int = 32,
) -> dict[str, float]:
    """纯测试评估：加载 finetuned MiCoFormerClassifier，运行 trainer.test()。"""
    if label_values is None:
        label_values = DEFAULT_LABEL_VALUES

    import lightning as L
    from micoformer.datamodules.classification_datamodule import ClassificationDataModule
    from micoformer.models.classification_module import MiCoFormerClassifier
    from micoformer.workflows.finetune import build_label_configs

    # 从 finetuned ckpt 加载模型（无需 pretrain ckpt，_encoder_hparams 已保存在 hparams 中）
    model = MiCoFormerClassifier.load_from_checkpoint(finetuned_ckpt, map_location="cpu")

    # 从保存的 encoder hparams 提取 abundance 参数
    enc_hparams = model.hparams.get("_encoder_hparams", {})
    num_abundance_bins = enc_hparams.get("num_abundance_bins", 40)
    min_abundance = enc_hparams.get("min_abundance", 4e-6)
    abundance_mode = enc_hparams.get("abundance_mode", "abs_log_bins")

    # 构建 label configs
    label_values_str = f"{label_field}={','.join(label_values)}"
    label_configs = build_label_configs([label_field], label_values_str)

    # 构建 test-only datamodule
    dm = ClassificationDataModule(
        h5ad_path=h5ad_path,
        label_configs=label_configs,
        train_indices=None,
        val_indices=None,
        test_indices=test_indices.tolist() if isinstance(test_indices, np.ndarray) else test_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_len=1024,
        num_abundance_bins=num_abundance_bins,
        min_abundance=min_abundance,
        abundance_mode=abundance_mode,
    )

    # 运行测试
    trainer = L.Trainer(
        devices=1,
        logger=False,
        enable_progress_bar=False,
        accelerator="auto",
    )
    results = trainer.test(model, datamodule=dm)

    return results[0] if results else {}


# ─── R2 bias_table 验证 ───────────────────────────────────────────────────

def verify_bias_table(
    checkpoint_path: str,
    expected_shape: tuple[int, int] = (16, 5),
    min_mean_abs: float = 1e-3,
) -> dict[str, Any]:
    """加载 checkpoint，检查 encoder.taxonomy_bias_params.bias_table。"""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # 查找 bias_table key
    bias_table = None
    for key, value in state_dict.items():
        if "bias_table" in key and "taxonomy_bias" in key:
            bias_table = value
            break

    if bias_table is None:
        return {
            "has_bias_table": False,
            "shape": None,
            "all_zeros": True,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "per_head_mean_abs": [],
            "passed": False,
        }

    abs_vals = bias_table.abs()
    per_head = [abs_vals[h].mean().item() for h in range(bias_table.shape[0])]

    return {
        "has_bias_table": True,
        "shape": tuple(bias_table.shape),
        "all_zeros": bool((bias_table == 0).all()),
        "mean_abs": float(abs_vals.mean().item()),
        "max_abs": float(abs_vals.max().item()),
        "per_head_mean_abs": per_head,
        "passed": bool(not (bias_table == 0).all() and abs_vals.mean().item() >= min_mean_abs),
    }


# ─── Live Status 管理 ─────────────────────────────────────────────────────

def _apply_live_status_defaults(
    tasks: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    *,
    retry_failed: bool,
) -> list[dict[str, Any]]:
    existing_by_id = {str(row["trial_id"]): dict(row) for row in existing_rows}
    initialized: list[dict[str, Any]] = []
    for task in tasks:
        trial_id = str(task["trial_id"])
        if trial_id in existing_by_id:
            row = existing_by_id[trial_id]
            status = str(row.get("status", ""))
            if status == "OK":
                initialized.append(row)
                continue
            if status == "ERROR" and not retry_failed:
                initialized.append(row)
                continue
            # 重置为 PENDING
            row["status"] = "PENDING"
            row["start_time"] = ""
            row["end_time"] = ""
            row["elapsed_seconds"] = ""
            row["error_message"] = ""
            initialized.append(row)
        else:
            row = dict(task)
            row["status"] = "PENDING"
            row["start_time"] = ""
            row["end_time"] = ""
            row["elapsed_seconds"] = ""
            row["gpu_id"] = ""
            row["checkpoint_path"] = ""
            row["error_message"] = ""
            initialized.append(row)
    return initialized


def validate_existing_task_definitions(
    tasks: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    summary_path: Path,
) -> None:
    task_ids = {str(t["trial_id"]) for t in tasks}
    for row in existing_rows:
        rid = str(row.get("trial_id", ""))
        if rid not in task_ids:
            raise ValueError(
                f"Summary contains trial_id={rid!r} not in current plan. "
                f"Delete {summary_path} and re-run."
            )


def _sorted_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status_order = {"ERROR": 0, "OOM": 1, "NO_CKPT": 2, "RUNNING": 3, "PENDING": 4, "OK": 5}
    return sorted(rows, key=lambda r: (status_order.get(str(r.get("status", "")), 9), str(r.get("trial_id", ""))))


def merge_rows(
    existing: list[dict[str, Any]],
    new_row: dict[str, Any],
) -> list[dict[str, Any]]:
    trial_id = str(new_row.get("trial_id", ""))
    merged = [r for r in existing if str(r.get("trial_id", "")) != trial_id]
    merged.append(new_row)
    return merged


# ─── Stage Block 调度 ──────────────────────────────────────────────────────

def prepare_stage_block(
    run_dir: str | Path,
    stage_block: str,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """准备 stage block：加载 summary，初始化 live status，返回 paths。"""
    run_dir = Path(run_dir)
    paths = get_stage_block_paths(run_dir, stage_block)
    summary_path = paths["summary"]

    # 从 summary 加载已有结果
    existing = load_rows(summary_path)
    return {"paths": paths, "existing": existing}


def run_stage_block(
    run_dir: str | Path,
    stage_block: str,
    gpu_ids: list[int],
    num_workers: int | None = None,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    gpu_cooldown_seconds: int = DEFAULT_GPU_COOLDOWN_SECONDS,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """从 plan CSV 加载任务并执行。由 run_stage.py CLI 调用。"""
    run_dir = Path(run_dir)
    paths = get_stage_block_paths(run_dir, stage_block)

    if not paths["plan"].exists():
        raise FileNotFoundError(
            f"Plan CSV not found: {paths['plan']}\n"
            "Run the notebook to generate the plan first."
        )

    tasks = load_rows(paths["plan"])
    if num_workers is not None:
        for t in tasks:
            t["num_workers"] = num_workers

    return run_tasks(
        tasks,
        gpu_ids=gpu_ids,
        summary_path=paths["summary"],
        live_status_path=paths["live_status"],
        dashboard_path=paths["dashboard"],
        cpu_threads=cpu_threads,
        gpu_cooldown_seconds=gpu_cooldown_seconds,
        retry_failed=retry_failed,
    )


def run_tasks(
    tasks: list[dict[str, Any]],
    *,
    gpu_ids: list[int],
    summary_path: str | Path,
    live_status_path: str | Path | None = None,
    dashboard_path: str | Path | None = None,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    gpu_cooldown_seconds: int = DEFAULT_GPU_COOLDOWN_SECONDS,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """多 GPU 并行执行任务列表。"""
    apply_cpu_runtime_settings(cpu_threads)
    summary_path = Path(summary_path)
    live_status_path = (
        Path(live_status_path)
        if live_status_path is not None
        else summary_path.with_name(summary_path.stem.replace("_summary", "_live_status") + summary_path.suffix)
    )
    dashboard_path = (
        Path(dashboard_path)
        if dashboard_path is not None
        else summary_path.with_name(summary_path.stem.replace("_summary", "_dashboard") + ".json")
    )

    existing = load_rows(summary_path)
    validate_existing_task_definitions(tasks, existing, summary_path=summary_path)
    live_rows = _apply_live_status_defaults(tasks, existing, retry_failed=retry_failed)
    write_rows(live_rows, live_status_path)
    _write_dashboard(live_rows, dashboard_path)

    pending = [row for row in live_rows if str(row.get("status")) in ("PENDING", "ERROR", "RUNNING")]
    if not pending:
        print(f"[scheduler] All {len(tasks)} tasks already completed (OK)")
        return load_rows(summary_path)

    print(f"[scheduler] {len(pending)} pending / {len(tasks)} total tasks, {len(gpu_ids)} GPU(s)")

    if len(gpu_ids) == 1:
        return _run_tasks_in_process(
            pending,
            existing=existing,
            gpu_id=gpu_ids[0],
            summary_path=summary_path,
            live_status_path=live_status_path,
            dashboard_path=dashboard_path,
        )

    return _run_tasks_with_external_processes(
        pending,
        existing=existing,
        gpu_ids=gpu_ids,
        cpu_threads=cpu_threads,
        gpu_cooldown_seconds=gpu_cooldown_seconds,
        summary_path=summary_path,
        live_status_path=live_status_path,
        dashboard_path=dashboard_path,
    )


def _run_tasks_in_process(
    pending: list[dict[str, Any]],
    *,
    existing: list[dict[str, Any]],
    gpu_id: int,
    summary_path: Path,
    live_status_path: Path,
    dashboard_path: Path,
) -> list[dict[str, Any]]:
    rows = existing[:]
    live_rows_by_id = {
        str(row["trial_id"]): row
        for row in load_rows(live_status_path)
    }

    for task in pending:
        trial_id = str(task["trial_id"])
        live_row = live_rows_by_id[trial_id]
        live_row["status"] = "RUNNING"
        live_row["gpu_id"] = gpu_id
        live_row["start_time"] = _timestamp_now()
        write_rows(list(live_rows_by_id.values()), live_status_path)
        _write_dashboard(list(live_rows_by_id.values()), dashboard_path)

        row = _execute_task(task, gpu_id)
        rows = merge_rows(rows, row)
        live_rows_by_id[trial_id] = dict(row)
        write_rows(_sorted_summary_rows(rows), summary_path)
        write_rows(list(live_rows_by_id.values()), live_status_path)
        _write_dashboard(list(live_rows_by_id.values()), dashboard_path)

    return load_rows(summary_path)


def _run_tasks_with_external_processes(
    pending: list[dict[str, Any]],
    *,
    existing: list[dict[str, Any]],
    gpu_ids: list[int],
    cpu_threads: int,
    gpu_cooldown_seconds: int,
    summary_path: Path,
    live_status_path: Path,
    dashboard_path: Path,
) -> list[dict[str, Any]]:
    rows = existing[:]
    live_rows_by_id = {
        str(row["trial_id"]): row
        for row in load_rows(live_status_path)
    }
    pending_queue = list(pending)
    active_by_trial: dict[str, dict[str, Any]] = {}
    gpu_ready_at = {int(gpu_id): 0.0 for gpu_id in gpu_ids}
    task_dir = dashboard_path.parent / ".task_runner"
    task_dir.mkdir(parents=True, exist_ok=True)

    def finalize_row(row: dict[str, Any]) -> None:
        nonlocal rows
        trial_id = str(row["trial_id"])
        rows = merge_rows(rows, row)
        live_rows_by_id[trial_id] = dict(row)
        write_rows(_sorted_summary_rows(rows), summary_path)
        write_rows(list(live_rows_by_id.values()), live_status_path)
        _write_dashboard(list(live_rows_by_id.values()), dashboard_path)

    def start_task(task: dict[str, Any], gpu_id: int) -> None:
        trial_id = str(task["trial_id"])
        live_row = live_rows_by_id[trial_id]
        live_row["status"] = "RUNNING"
        live_row["gpu_id"] = gpu_id
        live_row["start_time"] = _timestamp_now()
        write_rows(list(live_rows_by_id.values()), live_status_path)
        _write_dashboard(list(live_rows_by_id.values()), dashboard_path)

        task_path = task_dir / f"{trial_id}.task.json"
        result_path = task_dir / f"{trial_id}.result.json"
        log_path = task_dir / f"{trial_id}.log"
        write_manifest(task_path, task)
        if result_path.exists():
            result_path.unlink()
        log_handle = log_path.open("w")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(PROTOCOL_ROOT / "run_trial.py"),
                "--task-json", str(task_path),
                "--gpu-id", str(gpu_id),
                "--result-json", str(result_path),
                "--cpu-threads", str(cpu_threads),
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        active_by_trial[trial_id] = {
            "proc": proc,
            "gpu_id": gpu_id,
            "task": dict(task),
            "task_path": task_path,
            "result_path": result_path,
            "log_path": log_path,
            "log_handle": log_handle,
            "launched_at": time.time(),
        }

    def start_ready_tasks() -> None:
        now = time.time()
        busy_gpu_ids = {int(info["gpu_id"]) for info in active_by_trial.values()}
        for gpu_id in gpu_ids:
            gpu_id = int(gpu_id)
            if not pending_queue:
                break
            if gpu_id in busy_gpu_ids:
                continue
            if now < gpu_ready_at[gpu_id]:
                continue
            start_task(pending_queue.pop(0), gpu_id)

    start_ready_tasks()

    while active_by_trial or pending_queue:
        finished_trials: list[tuple[str, dict[str, Any]]] = []
        for trial_id, info in list(active_by_trial.items()):
            proc = info["proc"]
            exitcode = proc.poll()
            if exitcode is None:
                continue
            finished_trials.append((trial_id, info))

        if not finished_trials:
            start_ready_tasks()
            time.sleep(0.5)
            continue

        for trial_id, info in finished_trials:
            proc = info["proc"]
            gpu_id = int(info["gpu_id"])
            task = dict(info["task"])
            task_path = Path(info["task_path"])
            result_path = Path(info["result_path"])
            log_path = Path(info["log_path"])
            log_handle = info["log_handle"]
            launched_at = float(info["launched_at"])
            log_handle.close()
            exitcode = proc.returncode
            active_by_trial.pop(trial_id)

            if result_path.exists():
                row = read_manifest(result_path)
            else:
                live_row = live_rows_by_id[trial_id]
                start_time = str(live_row.get("start_time", "")) or _timestamp_now()
                tail = _tail_text(log_path).strip()
                error_message = (
                    "External task process exited before producing a result "
                    f"(exitcode={exitcode})."
                )
                if tail:
                    error_message = f"{error_message}\n{tail}"
                row = dict(task)
                row.update({
                    "status": "ERROR",
                    "gpu_id": gpu_id,
                    "start_time": start_time,
                    "end_time": _timestamp_now(),
                    "elapsed_seconds": round(time.time() - launched_at, 2),
                    "checkpoint_path": "",
                    "error_message": error_message,
                })

            finalize_row(row)
            gpu_ready_at[gpu_id] = time.time() + max(0, int(gpu_cooldown_seconds))

            if task_path.exists():
                task_path.unlink()
            if result_path.exists():
                result_path.unlink()

        start_ready_tasks()

    return load_rows(summary_path)


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]
