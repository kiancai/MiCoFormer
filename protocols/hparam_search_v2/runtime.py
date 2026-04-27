"""Hyperparameter search V2 runtime helpers.

This module is intentionally small in scope:

- stage task generation
- multi-GPU trial scheduling
- summary / manifest persistence
- promotion of selected checkpoints

The experiment control flow remains in `hparam_search_v2.py`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


PROTOCOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "baseline": {
        "token_embedding_mode": "taxon",
        "use_taxonomy_bias": False,
    },
    "r1": {
        "token_embedding_mode": "taxon_path",
        "use_taxonomy_bias": False,
    },
    "r2": {
        "token_embedding_mode": "taxon",
        "use_taxonomy_bias": True,
    },
    "r1r2": {
        "token_embedding_mode": "taxon_path",
        "use_taxonomy_bias": True,
    },
}
VARIANT_DISPATCH_PRIORITY = ["baseline", "r2", "r1r2", "r1"]

HEAD_REGIMES: dict[str, dict[int, int]] = {
    "standard": {256: 4, 512: 8, 768: 12},
    "fine": {256: 8, 512: 16, 768: 24},
}

SHARED_D_MODELS = [256, 512, 768]
SHARED_NUM_LAYERS = [4, 6, 8, 12]
STAGE_A_STANDARD_BATCH_GRID = [128, 256]
STAGE_A_SAFE_BATCH_GRID = [64, 128]
STAGE_A_LR_GRID = [3e-4, 1e-3, 3e-3]
STAGE_A_WD_GRID = [0.05, 0.1]
STAGE_A_COVERAGE_MIN_OK = 8
STAGE_A_TRAIN_PARAM_MIN_OK = 9
STAGE_C_MIN_OK = 12
# R2 优化：每隔 k 步才对 bias_table 反传一次梯度，大幅降低 r2/r1r2 变体的训练开销。
# k=4 是经过 profile 后的推荐值（可将 bias_table 梯度开销降低 ~4x，对模型质量影响极小）。
# 仅在 use_taxonomy_bias=True 时生效；baseline/r1 使用此参数但不影响行为。
BIAS_GRAD_EVERY_K: int = 4
STAGE_B_FIXED_CONFIG = {
    "pooling_mode": "sample_and_mean",
    "freeze_encoder": False,
    "lr_head": 1e-3,
    "lr_encoder": 1e-5,
    "weight_decay": 1e-2,
    "head_hidden_dim": 128,
    "head_dropout": 0.1,
    "batch_size": 32,
    "warmup_ratio": 0.1,
    "budget_mode": "epoch",
    "max_epochs": 20,
    "val_interval_epochs": 2,
    "early_stopping_patience": 0,
}
STAGE_C_MODE_POOLING = ["sample", "mean_pool", "sample_and_mean"]
STAGE_C_MODE_FREEZE = [True, False]
STAGE_C_LR_HEAD = [5e-4, 1e-3, 3e-3]
STAGE_C_LR_ENCODER = [5e-6, 1e-5, 3e-5]
STAGE_C_HEAD_HIDDEN = [0, 128, 256]
STAGE_C_HEAD_DROPOUT = [0.0, 0.1, 0.2]
DEFAULT_LABEL_FIELD = "Is_Healthy"
DEFAULT_LABEL_VALUES = ["True", "False"]
DEFAULT_CPU_THREADS = 1
DEFAULT_GPU_COOLDOWN_SECONDS = 15
STAGE_BLOCK_SPECS: dict[str, dict[str, str]] = {
    "a1_coverage": {
        "stage_name": "stage_a",
        "plan_filename": "a1_coverage_plan.csv",
        "live_status_filename": "a1_coverage_live_status.csv",
        "summary_filename": "a1_coverage_summary.csv",
        "dashboard_filename": "a1_coverage_dashboard.json",
        "log_dirname": "a1_coverage",
        "tb_port": "6006",
    },
    "a2_nhead": {
        "stage_name": "stage_a",
        "plan_filename": "a2_nhead_plan.csv",
        "live_status_filename": "a2_nhead_live_status.csv",
        "summary_filename": "a2_nhead_summary.csv",
        "dashboard_filename": "a2_nhead_dashboard.json",
        "log_dirname": "a2_nhead",
        "tb_port": "6007",
    },
    "a3_train_params": {
        "stage_name": "stage_a",
        "plan_filename": "a3_train_params_plan.csv",
        "live_status_filename": "a3_train_params_live_status.csv",
        "summary_filename": "a3_train_params_summary.csv",
        "dashboard_filename": "a3_train_params_dashboard.json",
        "log_dirname": "a3_train_params",
        "tb_port": "6008",
    },
    "b_screen": {
        "stage_name": "stage_b",
        "plan_filename": "b_screen_plan.csv",
        "live_status_filename": "b_screen_live_status.csv",
        "summary_filename": "b_screen_summary.csv",
        "dashboard_filename": "b_screen_dashboard.json",
        "log_dirname": "b_screen",
        "tb_port": "6101",
    },
    "c1a_mode": {
        "stage_name": "stage_c",
        "plan_filename": "c1a_mode_plan.csv",
        "live_status_filename": "c1a_mode_live_status.csv",
        "summary_filename": "c1a_mode_summary.csv",
        "dashboard_filename": "c1a_mode_dashboard.json",
        "log_dirname": "c1a_mode",
        "tb_port": "6102",
    },
    "c1b_lr": {
        "stage_name": "stage_c",
        "plan_filename": "c1b_lr_plan.csv",
        "live_status_filename": "c1b_lr_live_status.csv",
        "summary_filename": "c1b_lr_summary.csv",
        "dashboard_filename": "c1b_lr_dashboard.json",
        "log_dirname": "c1b_lr",
        "tb_port": "6103",
    },
    "c1c_head": {
        "stage_name": "stage_c",
        "plan_filename": "c1c_head_plan.csv",
        "live_status_filename": "c1c_head_live_status.csv",
        "summary_filename": "c1c_head_summary.csv",
        "dashboard_filename": "c1c_head_dashboard.json",
        "log_dirname": "c1c_head",
        "tb_port": "6104",
    },
    "c2_final_compare": {
        "stage_name": "stage_c",
        "plan_filename": "c2_final_compare_plan.csv",
        "live_status_filename": "c2_final_compare_live_status.csv",
        "summary_filename": "c2_final_compare_summary.csv",
        "dashboard_filename": "c2_final_compare_dashboard.json",
        "log_dirname": "c2_final_compare",
        "tb_port": "6105",
    },
}


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def init_run_dir(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    layout = {
        "run_dir": run_dir,
        "config": run_dir / "config",
        "splits": run_dir / "splits",
        "decisions": run_dir / "decisions",
        "stage_a": run_dir / "stage_a",
        "stage_b": run_dir / "stage_b",
        "stage_c": run_dir / "stage_c",
    }
    for key in ("stage_a", "stage_b", "stage_c"):
        layout[f"{key}_logs"] = layout[key] / "logs"
        layout[f"{key}_checkpoints"] = layout[key] / "checkpoints"

    for path in layout.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    return layout


def detect_available_cpu_cores() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        count = os.cpu_count() or 1
        return max(1, int(count))


def resolve_cpu_runtime_settings(
    *,
    requested_num_workers: int | None,
    requested_cpu_threads: int | None = None,
) -> dict[str, int]:
    available_cores = detect_available_cpu_cores()
    requested_workers = 0 if requested_num_workers is None else max(0, int(requested_num_workers))
    requested_threads = (
        DEFAULT_CPU_THREADS if requested_cpu_threads is None else max(1, int(requested_cpu_threads))
    )

    safe_num_workers = min(requested_workers, max(0, available_cores - 1))
    safe_cpu_threads = min(requested_threads, available_cores)
    return {
        "available_cpu_cores": available_cores,
        "requested_num_workers": requested_workers,
        "safe_num_workers": safe_num_workers,
        "requested_cpu_threads": requested_threads,
        "safe_cpu_threads": safe_cpu_threads,
    }


def apply_cpu_runtime_settings(cpu_threads: int, *, touch_torch: bool = True) -> None:
    resolved_threads = max(1, int(cpu_threads))
    env_values = {
        "OMP_NUM_THREADS": str(resolved_threads),
        "MKL_NUM_THREADS": str(resolved_threads),
        "OPENBLAS_NUM_THREADS": str(resolved_threads),
        "NUMEXPR_NUM_THREADS": str(resolved_threads),
        "VECLIB_MAXIMUM_THREADS": str(resolved_threads),
    }
    for key, value in env_values.items():
        os.environ[key] = value

    if not touch_torch:
        return

    try:
        import torch

        torch.set_num_threads(resolved_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


def get_stage_block_spec(stage_block: str) -> dict[str, str]:
    if stage_block not in STAGE_BLOCK_SPECS:
        available = ", ".join(sorted(STAGE_BLOCK_SPECS))
        raise ValueError(f"Unknown stage_block {stage_block!r}. Available: {available}")
    return STAGE_BLOCK_SPECS[stage_block]


def get_stage_block_paths(run_dir: str | Path, stage_block: str) -> dict[str, Path]:
    layout = init_run_dir(run_dir)
    spec = get_stage_block_spec(stage_block)
    stage_root = layout[spec["stage_name"]]
    return {
        "plan": stage_root / spec["plan_filename"],
        "live_status": stage_root / spec["live_status_filename"],
        "summary": stage_root / spec["summary_filename"],
        "dashboard": stage_root / spec["dashboard_filename"],
        "log_dir": stage_root / "logs" / spec["log_dirname"],
    }


def _try_import_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic helper
        return f"unavailable ({exc})"


def capture_environment_snapshot(output_path: str | Path, gpu_ids: list[int]) -> None:
    output_path = Path(output_path)
    cpu_info = resolve_cpu_runtime_settings(requested_num_workers=0, requested_cpu_threads=DEFAULT_CPU_THREADS)
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    lines = [
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"python_executable: {sys.executable}",
        f"python_version: {sys.version.split()[0]}",
        f"available_cpu_cores: {cpu_info['available_cpu_cores']}",
        f"torch_version: {_try_import_version('torch')}",
        f"lightning_version: {_try_import_version('lightning')}",
        f"anndata_version: {_try_import_version('anndata')}",
        f"gpu_ids: {gpu_ids}",
    ]
    if torch is not None:
        try:
            lines.append(f"cuda_available: {torch.cuda.is_available()}")
            lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                lines.append(f"cuda_device_names: {names}")
        except Exception as exc:  # pragma: no cover - diagnostic helper
            lines.append(f"cuda_probe_error: {exc}")
    output_path.write_text("\n".join(lines) + "\n")


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "1", "yes", "y", "on"}:
            return True
        if raw in {"false", "0", "no", "n", "off", ""}:
            return False
    return bool(value)


TASK_RESULT_FIELDS = {
    "status",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "gpu_id",
    "error_message",
    "checkpoint_path",
    "promoted_checkpoint_path",
    "best_val_loss",
    "best_score",
    "val_macro_f1",
    "val_auroc",
    "val_accuracy",
    "test_macro_f1",
    "test_auroc",
    "test_accuracy",
    "num_seeds",
    "macro_f1_mean",
    "macro_f1_std",
    "macro_f1_best",
    "auroc_mean",
    "auroc_std",
    "accuracy_mean",
    "accuracy_std",
}
TASK_PATHLIKE_FIELDS = {
    "h5ad_path",
    "train_indices_path",
    "val_indices_path",
    "test_indices_path",
    "pretrained_ckpt",
    "source_checkpoint_path",
    "log_dir",
    "log_subdir",
}


def _normalize_task_signature_value(key: str, value: Any) -> Any:
    if key in TASK_PATHLIKE_FIELDS and isinstance(value, str) and value.strip():
        path = Path(value)
        if key == "log_subdir":
            return value.replace("\\", "/")
        if len(path.parts) >= 2:
            return "/".join(path.parts[-2:])
        return path.name
    if isinstance(value, list):
        return [_normalize_task_signature_value(key, item) for item in value]
    if isinstance(value, dict):
        return {
            sub_key: _normalize_task_signature_value(sub_key, sub_value)
            for sub_key, sub_value in sorted(value.items())
        }
    return value


def compute_task_signature(row: dict[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if key in TASK_RESULT_FIELDS or key == "task_signature":
            continue
        payload[key] = _normalize_task_signature_value(key, value)
    if not _as_bool(payload.get("use_taxonomy_bias", False)):
        payload.pop("bias_grad_every_k", None)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def attach_task_signature(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["task_signature"] = compute_task_signature(enriched)
    return enriched


def write_rows(rows: list[dict[str, Any]], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    preferred = [
        "trial_id",
        "stage_block",
        "task_kind",
        "stage",
        "search_block",
        "trial_block",
        "model_variant",
        "run_name",
        "status",
        "start_time",
        "end_time",
        "best_val_loss",
        "val_macro_f1",
        "val_auroc",
        "val_accuracy",
        "checkpoint_path",
        "promoted_checkpoint_path",
        "elapsed_seconds",
        "gpu_id",
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


def write_manifest(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json" or yaml is None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def read_manifest(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def merge_rows(existing: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    merged = [row for row in existing if row.get("trial_id") != new_row.get("trial_id")]
    merged.append(new_row)
    return merged


def build_default_splits(h5ad_path: str | Path, split_dir: str | Path) -> dict[str, str]:
    from micoformer.workflows.splits import make_split

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "pretrain_train": split_dir / "pretrain_train_A.npy",
        "pretrain_val": split_dir / "pretrain_val_B.npy",
        "finetune_train": split_dir / "finetune_train_A.npy",
        "finetune_val": split_dir / "finetune_val_B.npy",
        "finetune_test": split_dir / "finetune_test_C.npy",
    }

    filter_map = {
        "pretrain_train": [("Split_Group", ["A"])],
        "pretrain_val": [("Split_Group", ["B"])],
        "finetune_train": [("Split_Group", ["A"])],
        "finetune_val": [("Split_Group", ["B"])],
        "finetune_test": [("Split_Group", ["C"])],
    }
    for key, path in split_paths.items():
        if not path.exists():
            make_split(h5ad=str(h5ad_path), filters=filter_map[key], output=str(path))
    return {key: str(path) for key, path in split_paths.items()}


def build_label_configs(
    field: str = DEFAULT_LABEL_FIELD,
    values: list[str] | None = None,
) -> list[dict[str, Any]]:
    config: dict[str, Any] = {"field": field}
    if values is not None:
        config["values"] = list(values)
    return [config]


def load_run_context(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config_dir = run_dir / "config"
    decisions_dir = run_dir / "decisions"

    run_config = read_manifest(config_dir / "run_config.yaml")
    split_paths = read_manifest(config_dir / "split_paths.yaml")

    return {
        "run_dir": run_dir,
        "run_config": run_config,
        "split_paths": split_paths,
        "cpu_runtime": resolve_cpu_runtime_settings(
            requested_num_workers=run_config.get("num_workers", 4),
            requested_cpu_threads=run_config.get("cpu_threads", DEFAULT_CPU_THREADS),
        ),
        "gpu_cooldown_seconds": int(run_config.get("gpu_cooldown_seconds", DEFAULT_GPU_COOLDOWN_SECONDS)),
        "shortlist_confirmed": (
            read_manifest(decisions_dir / "shortlist_confirmed.yaml")
            if (decisions_dir / "shortlist_confirmed.yaml").exists()
            else None
        ),
        "locked_arch_confirmed": (
            read_manifest(decisions_dir / "locked_arch_confirmed.yaml")
            if (decisions_dir / "locked_arch_confirmed.yaml").exists()
            else None
        ),
        "stage_a_top3": (
            read_manifest(decisions_dir / "stage_a_top3.yaml")
            if (decisions_dir / "stage_a_top3.yaml").exists()
            else None
        ),
        "stage_b_representatives": (
            read_manifest(decisions_dir / "stage_b_representatives.yaml")
            if (decisions_dir / "stage_b_representatives.yaml").exists()
            else None
        ),
        "final_candidates": (
            read_manifest(decisions_dir / "final_candidates.yaml")
            if (decisions_dir / "final_candidates.yaml").exists()
            else None
        ),
    }


def validate_variant_ok_counts(
    rows: list[dict[str, Any]],
    *,
    min_ok: int,
    stage_label: str,
    expected_variants: list[str] | None = None,
) -> dict[str, int]:
    expected_variants = expected_variants or list(VARIANT_SPECS)
    ok_counts = {variant: 0 for variant in expected_variants}
    total_counts = {variant: 0 for variant in expected_variants}

    for row in rows:
        variant = str(row.get("model_variant", ""))
        if variant not in ok_counts:
            continue
        total_counts[variant] += 1
        if row.get("status") == "OK":
            ok_counts[variant] += 1

    failures = {
        variant: {"ok": ok_counts[variant], "total": total_counts[variant]}
        for variant in expected_variants
        if ok_counts[variant] < min_ok
    }
    if failures:
        diagnostics = summarize_variant_failures(rows, failures.keys())
        raise RuntimeError(
            f"{stage_label} has insufficient OK trials: {failures}. "
            f"V2 requires at least {min_ok} OK runs per variant before continuing. "
            f"Failure details: {diagnostics}"
        )
    return ok_counts


def summarize_variant_failures(
    rows: list[dict[str, Any]],
    variants: list[str] | tuple[str, ...] | Any,
) -> dict[str, dict[str, Any]]:
    requested = {str(variant) for variant in variants}
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("model_variant", ""))
        if variant in requested:
            grouped_rows[variant].append(row)

    diagnostics: dict[str, dict[str, Any]] = {}
    for variant in sorted(requested):
        variant_rows = grouped_rows.get(variant, [])
        status_counts: dict[str, int] = defaultdict(int)
        error_counts: dict[str, int] = defaultdict(int)
        recent_trials: list[dict[str, str]] = []

        for row in variant_rows:
            status = str(row.get("status", "UNKNOWN"))
            status_counts[status] += 1
            error_message = str(row.get("error_message", "")).strip()
            if error_message:
                first_line = error_message.splitlines()[0].strip()
                error_counts[first_line] += 1

        def _sort_key(row: dict[str, Any]) -> tuple[str, str]:
            return (str(row.get("end_time", "")), str(row.get("start_time", "")))

        for row in sorted(variant_rows, key=_sort_key, reverse=True)[:3]:
            recent_trials.append({
                "trial_id": str(row.get("trial_id", "")),
                "status": str(row.get("status", "")),
                "gpu_id": str(row.get("gpu_id", "")),
                "end_time": str(row.get("end_time", "")),
                "error": str(row.get("error_message", "")).splitlines()[0].strip(),
            })

        top_errors = [
            {"message": message, "count": count}
            for message, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        ]

        diagnostics[variant] = {
            "status_counts": dict(sorted(status_counts.items())),
            "top_errors": top_errors,
            "recent_trials": recent_trials,
        }
    return diagnostics


def _base_pretrain_task(
    *,
    run_dir: str | Path,
    stage_name: str,
    model_variant: str,
    h5ad_path: str,
    train_indices_path: str,
    val_indices_path: str,
    seed: int,
    num_workers: int,
    trial_id: str,
    run_name: str,
) -> dict[str, Any]:
    variant = VARIANT_SPECS[model_variant]
    stage_root = Path(run_dir) / stage_name
    return {
        "task_kind": "pretrain",
        "stage": stage_name,
        "model_variant": model_variant,
        "trial_id": trial_id,
        "run_name": run_name,
        "h5ad_path": h5ad_path,
        "train_indices_path": train_indices_path,
        "val_indices_path": val_indices_path,
        "seed": seed,
        "devices": 1,
        "precision": "auto",
        "num_workers": num_workers,
        "log_dir": str(stage_root / "logs"),
        "log_subdir": model_variant,
        "budget_mode": "epoch",
        "max_epochs": 20,
        "val_interval_epochs": 3,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "no_progress_bar": True,
        "abundance_mode": "abs_log_bins",
        "max_seq_len": 1024,
        "ff_ratio": 4,
        "num_abundance_bins": 40,
        "warmup_ratio": 0.25,
        "min_abundance": 4e-6,
        "token_embedding_mode": variant["token_embedding_mode"],
        "use_taxonomy_bias": variant["use_taxonomy_bias"],
        "bias_grad_every_k": BIAS_GRAD_EVERY_K,
    }


def build_stage_a_coverage_tasks(
    *,
    run_dir: str | Path,
    h5ad_path: str,
    split_paths: dict[str, str],
    seed: int = 42,
    num_workers: int = 4,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for model_variant in VARIANT_SPECS:
        idx = 0
        for d_model in SHARED_D_MODELS:
            for num_layers in SHARED_NUM_LAYERS:
                idx += 1
                nhead = HEAD_REGIMES["standard"][d_model]
                trial_id = f"{model_variant}_a1_{idx:02d}"
                task = _base_pretrain_task(
                    run_dir=run_dir,
                    stage_name="stage_a",
                    model_variant=model_variant,
                    h5ad_path=h5ad_path,
                    train_indices_path=split_paths["pretrain_train"],
                    val_indices_path=split_paths["pretrain_val"],
                    seed=seed,
                    num_workers=num_workers,
                    trial_id=trial_id,
                    run_name=f"{model_variant}_a1_d{d_model}_L{num_layers}_standard",
                )
                task.update({
                    "stage_block": "a1_coverage",
                    "trial_block": "coverage",
                    "shared_shortlist_id": "",
                    "locked_shared_arch_id": "",
                    "log_subdir": "a1_coverage/" + model_variant,
                    "nhead_regime": "standard",
                    "d_model": d_model,
                    "num_layers": num_layers,
                    "nhead": nhead,
                    "dropout": 0.05,
                    "lr": 3e-4,
                    "batch_size": 128,
                    "weight_decay": 0.05,
                    "mask_prob": 0.15,
                })
                tasks.append(task)
    return tasks


def build_stage_a_nhead_tasks(
    *,
    run_dir: str | Path,
    h5ad_path: str,
    split_paths: dict[str, str],
    shortlist: list[dict[str, Any]],
    seed: int = 42,
    num_workers: int = 4,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if len(shortlist) != 4:
        raise RuntimeError(
            f"Stage A-2 requires exactly 4 shortlist entries, got {len(shortlist)}."
        )
    for model_variant in VARIANT_SPECS:
        for idx, combo in enumerate(shortlist, start=1):
            d_model = int(combo["d_model"])
            num_layers = int(combo["num_layers"])
            trial_id = f"{model_variant}_a2_{idx:02d}"
            task = _base_pretrain_task(
                run_dir=run_dir,
                stage_name="stage_a",
                model_variant=model_variant,
                h5ad_path=h5ad_path,
                train_indices_path=split_paths["pretrain_train"],
                val_indices_path=split_paths["pretrain_val"],
                seed=seed,
                num_workers=num_workers,
                trial_id=trial_id,
                run_name=f"{model_variant}_a2_d{d_model}_L{num_layers}_fine",
            )
            task.update({
                "stage_block": "a2_nhead",
                "trial_block": "nhead_regime",
                "shared_shortlist_id": idx,
                "locked_shared_arch_id": "",
                "log_subdir": "a2_nhead/" + model_variant,
                "nhead_regime": "fine",
                "d_model": d_model,
                "num_layers": num_layers,
                "nhead": HEAD_REGIMES["fine"][d_model],
                "dropout": 0.05,
                "lr": 3e-4,
                "batch_size": 128,
                "weight_decay": 0.05,
                "mask_prob": 0.15,
            })
            tasks.append(task)
    return tasks


def build_stage_a_train_param_tasks(
    *,
    run_dir: str | Path,
    h5ad_path: str,
    split_paths: dict[str, str],
    locked_arch: dict[str, Any],
    use_safe_batch_grid: bool = False,
    seed: int = 42,
    num_workers: int = 4,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    batch_grid = STAGE_A_SAFE_BATCH_GRID if use_safe_batch_grid else STAGE_A_STANDARD_BATCH_GRID
    locked_arch_id = locked_arch.get("locked_shared_arch_id", "locked_arch")
    for model_variant in VARIANT_SPECS:
        idx = 0
        for lr in STAGE_A_LR_GRID:
            for batch_size in batch_grid:
                for weight_decay in STAGE_A_WD_GRID:
                    idx += 1
                    trial_id = f"{model_variant}_a3_{idx:02d}"
                    task = _base_pretrain_task(
                        run_dir=run_dir,
                        stage_name="stage_a",
                        model_variant=model_variant,
                        h5ad_path=h5ad_path,
                        train_indices_path=split_paths["pretrain_train"],
                        val_indices_path=split_paths["pretrain_val"],
                        seed=seed,
                        num_workers=num_workers,
                        trial_id=trial_id,
                        run_name=(
                            f"{model_variant}_a3_d{locked_arch['d_model']}"
                            f"_L{locked_arch['num_layers']}_{locked_arch['nhead_regime']}"
                            f"_lr{lr:g}_bs{batch_size}_wd{weight_decay:g}"
                        ),
                    )
                    task.update({
                        "stage_block": "a3_train_params",
                        "trial_block": "train_params",
                        "shared_shortlist_id": "",
                        "locked_shared_arch_id": locked_arch_id,
                        "log_subdir": "a3_train_params/" + model_variant,
                        "nhead_regime": locked_arch["nhead_regime"],
                        "d_model": int(locked_arch["d_model"]),
                        "num_layers": int(locked_arch["num_layers"]),
                        "nhead": int(locked_arch["nhead"]),
                        "dropout": 0.05,
                        "lr": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "mask_prob": 0.15,
                    })
                    tasks.append(task)
    return tasks


def _base_finetune_task(
    *,
    run_dir: str | Path,
    stage_name: str,
    model_variant: str,
    h5ad_path: str,
    split_paths: dict[str, str],
    pretrained_ckpt: str,
    seed: int,
    num_workers: int,
    trial_id: str,
    run_name: str,
    label_field: str,
    label_values: list[str],
) -> dict[str, Any]:
    stage_root = Path(run_dir) / stage_name
    return {
        "task_kind": "finetune",
        "stage": stage_name,
        "model_variant": model_variant,
        "trial_id": trial_id,
        "run_name": run_name,
        "h5ad_path": h5ad_path,
        "train_indices_path": split_paths["finetune_train"],
        "val_indices_path": split_paths["finetune_val"],
        "test_indices_path": split_paths["finetune_test"],
        "pretrained_ckpt": pretrained_ckpt,
        "seed": seed,
        "devices": 1,
        "precision": "auto",
        "num_workers": num_workers,
        "log_dir": str(stage_root / "logs"),
        "log_subdir": model_variant,
        "budget_mode": "epoch",
        "max_epochs": 20,
        "val_interval_epochs": 2,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "no_progress_bar": True,
        "label_field": label_field,
        "label_values": list(label_values),
        "batch_size": 32,
        "pooling_mode": "sample_and_mean",
        "freeze_encoder": False,
        "lr_head": 1e-3,
        "lr_encoder": 1e-5,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.1,
        "head_hidden_dim": 128,
        "head_dropout": 0.1,
        "lr_scheduler_type": "cosine",
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "max_seq_len": 1024,
    }


def build_stage_b_tasks(
    *,
    run_dir: str | Path,
    h5ad_path: str,
    split_paths: dict[str, str],
    candidate_rows: list[dict[str, Any]],
    seed: int = 42,
    num_workers: int = 4,
    label_field: str = DEFAULT_LABEL_FIELD,
    label_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    label_values = label_values or DEFAULT_LABEL_VALUES
    tasks: list[dict[str, Any]] = []
    variant_counts: dict[str, int] = defaultdict(int)
    for row in candidate_rows:
        model_variant = str(row["model_variant"])
        variant_counts[model_variant] += 1
        rank_idx = variant_counts[model_variant]
        task = _base_finetune_task(
            run_dir=run_dir,
            stage_name="stage_b",
            model_variant=model_variant,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            pretrained_ckpt=str(row["checkpoint_path"]),
            seed=seed,
            num_workers=num_workers,
            trial_id=f"{model_variant}_b_{rank_idx:02d}",
            run_name=f"{model_variant}_b_rank{rank_idx}_seed{seed}",
            label_field=label_field,
            label_values=label_values,
        )
        task.update({
            "stage_block": "b_screen",
            "search_block": "screen",
            "pretrain_rank": row.get("selected_rank", row.get("rank_within_variant", rank_idx)),
            "log_subdir": "b_screen/" + model_variant,
            "source_checkpoint_path": row["checkpoint_path"],
        })
        tasks.append(task)
    return tasks


def build_stage_c_mode_tasks(
    *,
    run_dir: str | Path,
    h5ad_path: str,
    split_paths: dict[str, str],
    representatives: list[dict[str, Any]],
    seed: int = 42,
    num_workers: int = 4,
    label_field: str = DEFAULT_LABEL_FIELD,
    label_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    label_values = label_values or DEFAULT_LABEL_VALUES
    tasks: list[dict[str, Any]] = []
    for rep in representatives:
        model_variant = str(rep["model_variant"])
        checkpoint_alias = f"{model_variant}_rep"
        pretrain_checkpoint = str(rep.get("source_checkpoint_path", rep["checkpoint_path"]))
        idx = 0
        for pooling_mode in STAGE_C_MODE_POOLING:
            for freeze_encoder in STAGE_C_MODE_FREEZE:
                idx += 1
                task = _base_finetune_task(
                    run_dir=run_dir,
                    stage_name="stage_c",
                    model_variant=model_variant,
                    h5ad_path=h5ad_path,
                    split_paths=split_paths,
                    pretrained_ckpt=pretrain_checkpoint,
                    seed=seed,
                    num_workers=num_workers,
                    trial_id=f"{model_variant}_c1a_{idx:02d}",
                    run_name=f"{model_variant}_c1a_{checkpoint_alias}_{pooling_mode}_frz{int(freeze_encoder)}",
                    label_field=label_field,
                    label_values=label_values,
                )
                task.update({
                    "stage_block": "c1a_mode",
                    "search_block": "mode",
                    "source_checkpoint_path": pretrain_checkpoint,
                    "log_subdir": "c1a_mode/" + model_variant,
                    "pooling_mode": pooling_mode,
                    "freeze_encoder": freeze_encoder,
                    "head_hidden_dim": 128,
                    "head_dropout": 0.1,
                    "lr_head": 1e-3,
                    "lr_encoder": 1e-5,
                    "weight_decay": 1e-2,
                })
                tasks.append(task)
    return tasks


def build_stage_c_lr_tasks(
    *,
    run_dir: str | Path,
    h5ad_path: str,
    split_paths: dict[str, str],
    best_mode_rows: list[dict[str, Any]],
    seed: int = 42,
    num_workers: int = 4,
    label_field: str = DEFAULT_LABEL_FIELD,
    label_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    label_values = label_values or DEFAULT_LABEL_VALUES
    tasks: list[dict[str, Any]] = []
    for row in best_mode_rows:
        model_variant = str(row["model_variant"])
        idx = 0
        for lr_head in STAGE_C_LR_HEAD:
            for lr_encoder in STAGE_C_LR_ENCODER:
                idx += 1
                task = _base_finetune_task(
                    run_dir=run_dir,
                    stage_name="stage_c",
                    model_variant=model_variant,
                    h5ad_path=h5ad_path,
                    split_paths=split_paths,
                    pretrained_ckpt=str(row["source_checkpoint_path"]),
                    seed=seed,
                    num_workers=num_workers,
                    trial_id=f"{model_variant}_c1b_{idx:02d}",
                    run_name=f"{model_variant}_c1b_lrhead{lr_head:g}_lrenc{lr_encoder:g}",
                    label_field=label_field,
                    label_values=label_values,
                )
                task.update({
                    "stage_block": "c1b_lr",
                    "search_block": "lr",
                    "source_checkpoint_path": row["source_checkpoint_path"],
                    "log_subdir": "c1b_lr/" + model_variant,
                    "pooling_mode": row["pooling_mode"],
                    "freeze_encoder": row["freeze_encoder"],
                    "head_hidden_dim": 128,
                    "head_dropout": 0.1,
                    "lr_head": lr_head,
                    "lr_encoder": lr_encoder,
                    "weight_decay": 1e-2,
                })
                tasks.append(task)
    return tasks


def build_stage_c_head_tasks(
    *,
    run_dir: str | Path,
    h5ad_path: str,
    split_paths: dict[str, str],
    best_lr_rows: list[dict[str, Any]],
    seed: int = 42,
    num_workers: int = 4,
    label_field: str = DEFAULT_LABEL_FIELD,
    label_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    label_values = label_values or DEFAULT_LABEL_VALUES
    tasks: list[dict[str, Any]] = []
    for row in best_lr_rows:
        model_variant = str(row["model_variant"])
        idx = 0
        for head_hidden_dim in STAGE_C_HEAD_HIDDEN:
            for head_dropout in STAGE_C_HEAD_DROPOUT:
                idx += 1
                task = _base_finetune_task(
                    run_dir=run_dir,
                    stage_name="stage_c",
                    model_variant=model_variant,
                    h5ad_path=h5ad_path,
                    split_paths=split_paths,
                    pretrained_ckpt=str(row["source_checkpoint_path"]),
                    seed=seed,
                    num_workers=num_workers,
                    trial_id=f"{model_variant}_c1c_{idx:02d}",
                    run_name=f"{model_variant}_c1c_hd{head_hidden_dim}_dp{head_dropout:g}",
                    label_field=label_field,
                    label_values=label_values,
                )
                task.update({
                    "stage_block": "c1c_head",
                    "search_block": "head",
                    "source_checkpoint_path": row["source_checkpoint_path"],
                    "log_subdir": "c1c_head/" + model_variant,
                    "pooling_mode": row["pooling_mode"],
                    "freeze_encoder": row["freeze_encoder"],
                    "lr_head": row["lr_head"],
                    "lr_encoder": row["lr_encoder"],
                    "weight_decay": 1e-2,
                    "head_hidden_dim": head_hidden_dim,
                    "head_dropout": head_dropout,
                })
                tasks.append(task)
    return tasks


def build_final_compare_tasks(
    *,
    run_dir: str | Path,
    h5ad_path: str,
    split_paths: dict[str, str],
    finalists: list[dict[str, Any]],
    seeds: list[int] | None = None,
    num_workers: int = 4,
    label_field: str = DEFAULT_LABEL_FIELD,
    label_values: list[str] | None = None,
) -> list[dict[str, Any]]:
    label_values = label_values or DEFAULT_LABEL_VALUES
    seeds = seeds or [42, 52, 62]
    tasks: list[dict[str, Any]] = []
    for finalist in finalists:
        role = finalist["role"]
        for seed in seeds:
            task = _base_finetune_task(
                run_dir=run_dir,
                stage_name="stage_c",
                model_variant=str(finalist["model_variant"]),
                h5ad_path=h5ad_path,
                split_paths=split_paths,
                pretrained_ckpt=str(finalist["source_checkpoint_path"]),
                seed=seed,
                num_workers=num_workers,
                trial_id=f"{role}_seed{seed}",
                run_name=f"{role}_seed{seed}",
                label_field=label_field,
                label_values=label_values,
            )
            task.update({
                "stage_block": "c2_final_compare",
                "search_block": "final_compare",
                "role": role,
                "source_checkpoint_path": finalist["source_checkpoint_path"],
                "log_subdir": f"c2_final_compare/{role}",
                "pooling_mode": finalist["pooling_mode"],
                "freeze_encoder": finalist["freeze_encoder"],
                "lr_head": finalist["lr_head"],
                "lr_encoder": finalist["lr_encoder"],
                "weight_decay": finalist["weight_decay"],
                "head_hidden_dim": finalist["head_hidden_dim"],
                "head_dropout": finalist["head_dropout"],
            })
            tasks.append(task)
    return tasks


def _require_manifest_value(value: Any, path_label: str, stage_block: str) -> Any:
    if value is None:
        raise RuntimeError(
            f"{stage_block} requires {path_label}, but it does not exist yet. "
            f"Finish the prerequisite notebook step first."
        )
    return value


def build_tasks_for_stage_block(
    run_dir: str | Path,
    stage_block: str,
    *,
    num_workers: int | None = None,
) -> list[dict[str, Any]]:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    split_paths = context["split_paths"]
    seed = int(run_config.get("seed", 42))
    requested_num_workers = run_config.get("num_workers", 4) if num_workers is None else num_workers
    resolved_cpu = resolve_cpu_runtime_settings(
        requested_num_workers=requested_num_workers,
        requested_cpu_threads=run_config.get("cpu_threads", DEFAULT_CPU_THREADS),
    )
    resolved_workers = int(resolved_cpu["safe_num_workers"])
    h5ad_path = str(run_config["h5ad_path"])
    label_field = str(run_config.get("label_field", DEFAULT_LABEL_FIELD))
    label_values = list(run_config.get("label_values", DEFAULT_LABEL_VALUES))

    if stage_block == "a1_coverage":
        return build_stage_a_coverage_tasks(
            run_dir=run_dir,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            seed=seed,
            num_workers=resolved_workers,
        )
    if stage_block == "a2_nhead":
        shortlist = _require_manifest_value(
            context["shortlist_confirmed"],
            "decisions/shortlist_confirmed.yaml",
            stage_block,
        )
        return build_stage_a_nhead_tasks(
            run_dir=run_dir,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            shortlist=shortlist,
            seed=seed,
            num_workers=resolved_workers,
        )
    if stage_block == "a3_train_params":
        locked_arch = _require_manifest_value(
            context["locked_arch_confirmed"],
            "decisions/locked_arch_confirmed.yaml",
            stage_block,
        )
        return build_stage_a_train_param_tasks(
            run_dir=run_dir,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            locked_arch=locked_arch,
            use_safe_batch_grid=bool(run_config.get("use_safe_stage_a_batch_grid", False)),
            seed=seed,
            num_workers=resolved_workers,
        )
    if stage_block == "b_screen":
        stage_a_top3 = _require_manifest_value(
            context["stage_a_top3"],
            "decisions/stage_a_top3.yaml",
            stage_block,
        )
        return build_stage_b_tasks(
            run_dir=run_dir,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            candidate_rows=stage_a_top3,
            seed=seed,
            num_workers=resolved_workers,
            label_field=label_field,
            label_values=label_values,
        )
    if stage_block == "c1a_mode":
        representatives = _require_manifest_value(
            context["stage_b_representatives"],
            "decisions/stage_b_representatives.yaml",
            stage_block,
        )
        return build_stage_c_mode_tasks(
            run_dir=run_dir,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            representatives=representatives,
            seed=seed,
            num_workers=resolved_workers,
            label_field=label_field,
            label_values=label_values,
        )
    if stage_block == "c1b_lr":
        best_mode_rows = _require_manifest_value(
            read_manifest(Path(run_dir) / "decisions" / "stage_c_best_mode.yaml")
            if (Path(run_dir) / "decisions" / "stage_c_best_mode.yaml").exists()
            else None,
            "decisions/stage_c_best_mode.yaml",
            stage_block,
        )
        return build_stage_c_lr_tasks(
            run_dir=run_dir,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            best_mode_rows=best_mode_rows,
            seed=seed,
            num_workers=resolved_workers,
            label_field=label_field,
            label_values=label_values,
        )
    if stage_block == "c1c_head":
        best_lr_rows = _require_manifest_value(
            read_manifest(Path(run_dir) / "decisions" / "stage_c_best_lr.yaml")
            if (Path(run_dir) / "decisions" / "stage_c_best_lr.yaml").exists()
            else None,
            "decisions/stage_c_best_lr.yaml",
            stage_block,
        )
        return build_stage_c_head_tasks(
            run_dir=run_dir,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            best_lr_rows=best_lr_rows,
            seed=seed,
            num_workers=resolved_workers,
            label_field=label_field,
            label_values=label_values,
        )
    if stage_block == "c2_final_compare":
        finalists = _require_manifest_value(
            context["final_candidates"],
            "decisions/final_candidates.yaml",
            stage_block,
        )
        return build_final_compare_tasks(
            run_dir=run_dir,
            h5ad_path=h5ad_path,
            split_paths=split_paths,
            finalists=finalists,
            seeds=list(run_config.get("final_compare_seeds", [42, 52, 62])),
            num_workers=resolved_workers,
            label_field=label_field,
            label_values=label_values,
        )
    raise ValueError(f"Unsupported stage_block: {stage_block}")


def prepare_stage_block(
    run_dir: str | Path,
    stage_block: str,
    *,
    num_workers: int | None = None,
) -> dict[str, Any]:
    tasks = [
        attach_task_signature(task)
        for task in build_tasks_for_stage_block(run_dir, stage_block, num_workers=num_workers)
    ]
    paths = get_stage_block_paths(run_dir, stage_block)
    write_rows(tasks, paths["plan"])
    return {"tasks": tasks, "paths": paths}


def _task_sort_key(row: dict[str, Any], metric_key: str, reverse: bool) -> tuple[Any, ...]:
    metric = row.get(metric_key)
    if metric in (None, ""):
        metric = float("-inf") if reverse else float("inf")
    aux_1 = row.get("val_auroc", float("-inf") if reverse else float("inf"))
    aux_2 = row.get("val_accuracy", float("-inf") if reverse else float("inf"))
    if reverse:
        return (-float(metric), -float(aux_1), -float(aux_2), str(row.get("trial_id", "")))
    return (float(metric), str(row.get("trial_id", "")))


def select_top_k_per_variant(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    k: int,
    reverse: bool,
    min_ok_per_variant: int | None = None,
    stage_label: str | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "OK":
            grouped[str(row["model_variant"])].append(row)

    if min_ok_per_variant is not None:
        missing = {
            variant: len(grouped.get(variant, []))
            for variant in VARIANT_SPECS
            if len(grouped.get(variant, [])) < min_ok_per_variant
        }
        if missing:
            stage_name = stage_label or metric_key
            raise RuntimeError(
                f"{stage_name} has insufficient OK trials for final selection: {missing}. "
                f"Require at least {min_ok_per_variant} OK rows per variant."
            )

    selected: list[dict[str, Any]] = []
    for model_variant, group_rows in grouped.items():
        ordered = sorted(group_rows, key=lambda row: _task_sort_key(row, metric_key, reverse))
        for rank, row in enumerate(ordered[:k], start=1):
            enriched = dict(row)
            enriched["selected_rank"] = rank
            selected.append(enriched)
    return selected


def compute_stage_a_coverage_overview(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage_rows = [row for row in rows if row.get("trial_block") == "coverage"]
    variant_ranks: dict[str, dict[tuple[int, int], int]] = {}
    for model_variant in VARIANT_SPECS:
        variant_ok = [
            row for row in coverage_rows
            if row.get("model_variant") == model_variant and row.get("status") == "OK"
        ]
        variant_ok = sorted(variant_ok, key=lambda row: float(row["best_val_loss"]))
        variant_ranks[model_variant] = {
            (int(row["d_model"]), int(row["num_layers"])): idx
            for idx, row in enumerate(variant_ok, start=1)
        }

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in coverage_rows:
        grouped[(int(row["d_model"]), int(row["num_layers"]))].append(row)

    overview: list[dict[str, Any]] = []
    for (d_model, num_layers), group_rows in grouped.items():
        ok_rows = [row for row in group_rows if row.get("status") == "OK"]
        mean_loss = (
            statistics.mean(float(row["best_val_loss"]) for row in ok_rows)
            if ok_rows else None
        )
        ranks = [
            variant_ranks[str(row["model_variant"])][(d_model, num_layers)]
            for row in ok_rows
            if (d_model, num_layers) in variant_ranks[str(row["model_variant"])]
        ]
        overview.append({
            "d_model": d_model,
            "num_layers": num_layers,
            "nhead_regime": "standard",
            "ok_count": len(ok_rows),
            "oom_count": sum(1 for row in group_rows if row.get("status") == "OOM"),
            "error_count": sum(
                1 for row in group_rows if row.get("status") not in ("OK", "OOM")
            ),
            "mean_best_val_loss": mean_loss if mean_loss is not None else "",
            "mean_rank": statistics.mean(ranks) if ranks else "",
        })

    overview.sort(
        key=lambda row: (
            -int(row["ok_count"]),
            float(row["mean_rank"]) if row["mean_rank"] != "" else float("inf"),
            float(row["mean_best_val_loss"]) if row["mean_best_val_loss"] != "" else float("inf"),
            int(row["d_model"]),
            int(row["num_layers"]),
        )
    )
    return overview


def suggest_shortlist(overview_rows: list[dict[str, Any]], shortlist_size: int = 4) -> list[dict[str, Any]]:
    shortlist: list[dict[str, Any]] = []
    for idx, row in enumerate(overview_rows[:shortlist_size], start=1):
        shortlist.append({
            "shared_shortlist_id": idx,
            "d_model": int(row["d_model"]),
            "num_layers": int(row["num_layers"]),
            "nhead_regime": "standard",
            "nhead": HEAD_REGIMES["standard"][int(row["d_model"])],
            "ff_ratio": 4,
            "num_abundance_bins": 40,
        })
    return shortlist


def compute_locked_arch_overview(
    coverage_rows: list[dict[str, Any]],
    nhead_rows: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shortlist_keys = {(int(item["d_model"]), int(item["num_layers"])) for item in shortlist}
    candidate_rows = []
    for row in coverage_rows:
        if (
            row.get("trial_block") == "coverage"
            and (int(row["d_model"]), int(row["num_layers"])) in shortlist_keys
        ):
            candidate_rows.append(row)
    candidate_rows.extend(row for row in nhead_rows if row.get("trial_block") == "nhead_regime")

    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        key = (int(row["d_model"]), int(row["num_layers"]), str(row["nhead_regime"]))
        grouped[key].append(row)

    overview: list[dict[str, Any]] = []
    for (d_model, num_layers, regime), group_rows in grouped.items():
        ok_rows = [row for row in group_rows if row.get("status") == "OK"]
        overview.append({
            "d_model": d_model,
            "num_layers": num_layers,
            "nhead_regime": regime,
            "nhead": HEAD_REGIMES[regime][d_model],
            "ff_ratio": 4,
            "num_abundance_bins": 40,
            "ok_count": len(ok_rows),
            "mean_best_val_loss": (
                statistics.mean(float(row["best_val_loss"]) for row in ok_rows)
                if ok_rows else ""
            ),
        })

    overview.sort(
        key=lambda row: (
            -int(row["ok_count"]),
            float(row["mean_best_val_loss"]) if row["mean_best_val_loss"] != "" else float("inf"),
            int(row["d_model"]),
            int(row["num_layers"]),
            str(row["nhead_regime"]),
        )
    )
    return overview


def suggest_locked_arch(overview_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = dict(overview_rows[0])
    best["locked_shared_arch_id"] = (
        f"d{best['d_model']}_L{best['num_layers']}_{best['nhead_regime']}"
    )
    best["ff_ratio"] = int(best.get("ff_ratio", 4))
    best["num_abundance_bins"] = int(best.get("num_abundance_bins", 40))
    return best


def promote_checkpoint(row: dict[str, Any], destination_dir: str | Path, alias: str) -> str:
    source = Path(str(row["checkpoint_path"]))
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest = destination_dir / f"{alias}.ckpt"
    shutil.copy2(source, dest)
    return str(dest)


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
        lr_scheduler_type="cosine",
        lr_plateau_factor=0.5,
        lr_plateau_patience=2,
        lr_plateau_min_lr=1e-6,
        budget_mode=str(task["budget_mode"]),
        max_epochs=int(task["max_epochs"]),
        max_steps=None,
        val_interval_epochs=int(task["val_interval_epochs"]),
        val_interval_steps=None,
        limit_train_batches=float(task["limit_train_batches"]),
        limit_val_batches=float(task["limit_val_batches"]),
        devices=int(task["devices"]),
        precision=str(task["precision"]),
        seed=int(task["seed"]),
        accumulate_grad_batches=int(task["accumulate_grad_batches"]),
        gradient_clip_val=float(task["gradient_clip_val"]),
        num_workers=int(task["num_workers"]),
        log_dir=str(task["log_dir"]),
        no_progress_bar=_as_bool(task["no_progress_bar"]),
        early_stopping_patience=int(task["early_stopping_patience"]),
        early_stopping_min_delta=float(task["early_stopping_min_delta"]),
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
        max_seq_len=int(task["max_seq_len"]),
        lr_head=float(task["lr_head"]),
        lr_encoder=float(task["lr_encoder"]),
        weight_decay=float(task["weight_decay"]),
        warmup_ratio=float(task["warmup_ratio"]),
        lr_scheduler_type=str(task["lr_scheduler_type"]),
        lr_plateau_factor=0.5,
        lr_plateau_patience=2,
        lr_plateau_min_lr=1e-6,
        budget_mode=str(task["budget_mode"]),
        max_epochs=int(task["max_epochs"]),
        max_steps=None,
        val_interval_epochs=int(task["val_interval_epochs"]),
        val_interval_steps=None,
        early_stopping_patience=int(task["early_stopping_patience"]),
        early_stopping_min_delta=float(task["early_stopping_min_delta"]),
        gradient_clip_val=float(task["gradient_clip_val"]),
        accumulate_grad_batches=int(task["accumulate_grad_batches"]),
        limit_train_batches=float(task["limit_train_batches"]),
        limit_val_batches=float(task["limit_val_batches"]),
        devices=int(task["devices"]),
        precision=str(task["precision"]),
        seed=int(task["seed"]),
        num_workers=int(task["num_workers"]),
        log_dir=str(task["log_dir"]),
        no_progress_bar=_as_bool(task["no_progress_bar"]),
    )


def _timestamp_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
        merged = dict(task)
        previous = existing_by_id.get(trial_id, {})
        if previous.get("status") == "OK" and not retry_failed:
            merged.update(previous)
        else:
            merged.update({
                "status": "PENDING",
                "gpu_id": "",
                "start_time": "",
                "end_time": "",
                "elapsed_seconds": previous.get("elapsed_seconds", ""),
                "error_message": "",
                "checkpoint_path": previous.get("checkpoint_path", ""),
            })
        initialized.append(merged)
    return initialized


def validate_existing_task_definitions(
    tasks: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    *,
    summary_path: Path,
) -> None:
    expected_by_id = {str(task["trial_id"]): str(task["task_signature"]) for task in tasks}
    mismatches: list[str] = []
    for row in existing_rows:
        trial_id = str(row.get("trial_id", ""))
        if trial_id not in expected_by_id:
            continue
        existing_signature = str(row.get("task_signature") or compute_task_signature(row))
        if existing_signature != expected_by_id[trial_id]:
            mismatches.append(trial_id)

    if mismatches:
        sample = ", ".join(sorted(mismatches)[:5])
        raise RuntimeError(
            f"Existing summary at {summary_path} does not match the current task definitions. "
            f"Mismatched trial_ids: {sample}. Use a new run_dir or clear the stale summary/live-status files "
            "before resuming."
        )


def _write_dashboard(rows: list[dict[str, Any]], dashboard_path: str | Path) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status", "UNKNOWN"))] += 1
    payload = {
        "updated_at": _timestamp_now(),
        "total_trials": len(rows),
        "counts": dict(sorted(counts.items())),
        "completed_trials": sum(count for status, count in counts.items() if status not in {"PENDING", "RUNNING"}),
        "ok_trials": counts.get("OK", 0),
        "failed_trials": sum(
            counts.get(status, 0)
            for status in ("OOM", "ERROR", "NO_CKPT", "NO_VAL")
        ),
    }
    write_manifest(dashboard_path, payload)


def _sorted_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("status", "")) not in {"OK", "RUNNING", "PENDING"},
            str(row.get("status", "")) == "PENDING",
            str(row.get("status", "")) == "RUNNING",
            str(row.get("trial_id", "")),
        ),
    )


def prioritize_pending_tasks(pending: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for task in pending:
        variant = str(task.get("model_variant", ""))
        if variant:
            grouped[variant].append(task)
        else:
            passthrough.append(task)

    if len(grouped) <= 1:
        return pending

    variant_order = [variant for variant in VARIANT_DISPATCH_PRIORITY if grouped.get(variant)]
    variant_order.extend(
        variant for variant in sorted(grouped) if variant not in variant_order
    )

    reordered: list[dict[str, Any]] = []
    while True:
        progressed = False
        for variant in variant_order:
            if grouped[variant]:
                reordered.append(grouped[variant].pop(0))
                progressed = True
        if not progressed:
            break

    reordered.extend(passthrough)
    return reordered


def _execute_task(task: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    start = time.time()
    row = dict(task)
    row["gpu_id"] = gpu_id
    row["start_time"] = _timestamp_now()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        import numpy as np

        if task["task_kind"] == "pretrain":
            from micoformer.workflows.pretrain import run_pretrain_once

            config = _prepare_pretrain_config(task)
            train_indices = np.load(str(task["train_indices_path"]))
            val_indices = np.load(str(task["val_indices_path"]))
            result = run_pretrain_once(config, train_indices, val_indices, log_subdir=str(task["log_subdir"]))
            row["best_val_loss"] = result.get("best_val_loss")
            row["best_score"] = result.get("best_score")
            row["checkpoint_path"] = result.get("best_model_path")
        else:
            from micoformer.workflows.finetune import run_finetune_once

            config = _prepare_finetune_config(task)
            train_indices = np.load(str(task["train_indices_path"]))
            val_indices = np.load(str(task["val_indices_path"]))
            test_indices = np.load(str(task["test_indices_path"]))
            label_configs = build_label_configs(str(task["label_field"]), list(task["label_values"]))
            result = run_finetune_once(
                config,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
                label_configs=label_configs,
                log_subdir=str(task["log_subdir"]),
            )
            task_name = str(task["label_field"])
            val_metrics = result.get("val", {})
            test_metrics = result.get("test", {})
            row["checkpoint_path"] = result.get("best_model_path")
            row["best_score"] = result.get("best_score")
            row["val_macro_f1"] = val_metrics.get(f"val/{task_name}/f1_macro")
            row["val_auroc"] = val_metrics.get(f"val/{task_name}/auroc")
            row["val_accuracy"] = val_metrics.get(f"val/{task_name}/acc")
            row["test_macro_f1"] = test_metrics.get(f"test/{task_name}/f1_macro")
            row["test_auroc"] = test_metrics.get(f"test/{task_name}/auroc")
            row["test_accuracy"] = test_metrics.get(f"test/{task_name}/acc")

        row["status"] = "OK" if row.get("checkpoint_path") else "NO_CKPT"
        row["error_message"] = ""
    except RuntimeError as exc:
        message = str(exc)
        row["status"] = "OOM" if "out of memory" in message.lower() else "ERROR"
        row["error_message"] = message
        row["checkpoint_path"] = ""
    except Exception as exc:  # pragma: no cover - exercised in live runs
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


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


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
                "--task-json",
                str(task_path),
                "--gpu-id",
                str(gpu_id),
                "--result-json",
                str(result_path),
                "--cpu-threads",
                str(cpu_threads),
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


def run_tasks(
    tasks: list[dict[str, Any]],
    *,
    summary_path: str | Path,
    gpu_ids: list[int],
    cpu_threads: int = DEFAULT_CPU_THREADS,
    gpu_cooldown_seconds: int = DEFAULT_GPU_COOLDOWN_SECONDS,
    live_status_path: str | Path | None = None,
    dashboard_path: str | Path | None = None,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    apply_cpu_runtime_settings(cpu_threads, touch_torch=False)
    summary_path = Path(summary_path)
    live_status_path = Path(live_status_path) if live_status_path is not None else summary_path.with_name(
        summary_path.stem.replace("_summary", "_live_status") + summary_path.suffix
    )
    dashboard_path = Path(dashboard_path) if dashboard_path is not None else summary_path.with_name(
        summary_path.stem.replace("_summary", "_dashboard") + ".json"
    )
    existing = load_rows(summary_path)
    validate_existing_task_definitions(tasks, existing, summary_path=summary_path)
    live_rows = _apply_live_status_defaults(tasks, existing, retry_failed=retry_failed)
    write_rows(live_rows, live_status_path)
    _write_dashboard(live_rows, dashboard_path)

    completed = {
        str(row["trial_id"]): row
        for row in existing
        if row.get("status") == "OK" and not retry_failed
    }
    pending = [task for task in tasks if str(task["trial_id"]) not in completed]
    pending = prioritize_pending_tasks(pending)
    if not pending:
        return existing

    if len(gpu_ids) == 1:
        return _run_tasks_in_process(
            pending,
            existing=existing,
            gpu_id=int(gpu_ids[0]),
            summary_path=summary_path,
            live_status_path=live_status_path,
            dashboard_path=dashboard_path,
        )

    return _run_tasks_with_external_processes(
        pending,
        existing=existing,
        gpu_ids=gpu_ids[: min(len(gpu_ids), len(pending))],
        cpu_threads=cpu_threads,
        gpu_cooldown_seconds=gpu_cooldown_seconds,
        summary_path=summary_path,
        live_status_path=live_status_path,
        dashboard_path=dashboard_path,
    )


def write_plan(tasks: list[dict[str, Any]], plan_path: str | Path) -> None:
    write_rows(tasks, plan_path)


def select_stage_b_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return select_top_k_per_variant(
        rows,
        metric_key="val_macro_f1",
        k=1,
        reverse=True,
        min_ok_per_variant=1,
        stage_label="Stage B representative selection",
    )


def select_best_stage_c_block(rows: list[dict[str, Any]], search_block: str) -> list[dict[str, Any]]:
    block_rows = [row for row in rows if row.get("search_block") == search_block]
    return select_top_k_per_variant(
        block_rows,
        metric_key="val_macro_f1",
        k=1,
        reverse=True,
        min_ok_per_variant=1,
        stage_label=f"Stage C1 {search_block} selection",
    )


def select_stage_c_finalists(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_counts = {
        variant: 0
        for variant in VARIANT_SPECS
    }
    for row in rows:
        variant = str(row.get("model_variant", ""))
        if variant in ok_counts and row.get("status") == "OK":
            ok_counts[variant] += 1

    insufficient = {
        variant: ok_count
        for variant, ok_count in ok_counts.items()
        if ok_count < STAGE_C_MIN_OK
    }
    if insufficient:
        raise RuntimeError(
            f"Stage C1 has insufficient OK trials: {insufficient}. "
            f"V2 requires at least {STAGE_C_MIN_OK} OK runs per variant before selecting finalists."
        )

    best_per_variant = select_best_stage_c_block(rows, "head")
    baseline_rows = [row for row in best_per_variant if row.get("model_variant") == "baseline"]
    non_baseline_rows = [row for row in best_per_variant if row.get("model_variant") != "baseline"]
    if not baseline_rows or not non_baseline_rows:
        raise RuntimeError(
            "Could not determine strongest baseline or strongest final model from Stage C1. "
            "Check Stage C summaries and rerun the failed variant search block."
        )

    strongest_baseline = baseline_rows[0]
    strongest_final_model = sorted(
        non_baseline_rows,
        key=lambda row: _task_sort_key(row, "val_macro_f1", True),
    )[0]
    return [
        {"role": "strongest_baseline", **strongest_baseline},
        {"role": "strongest_final_model", **strongest_final_model},
    ]


def summarize_final_compare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = [row for row in rows if row.get("search_block") == "final_compare" and row.get("status") == "OK"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        grouped[str(row["role"])].append(row)

    summary: list[dict[str, Any]] = []
    for role, group_rows in grouped.items():
        macro = [float(row["val_macro_f1"]) for row in group_rows if row.get("val_macro_f1") is not None]
        auroc = [float(row["val_auroc"]) for row in group_rows if row.get("val_auroc") is not None]
        acc = [float(row["val_accuracy"]) for row in group_rows if row.get("val_accuracy") is not None]
        exemplar = group_rows[0]
        summary.append({
            "role": role,
            "model_variant": exemplar["model_variant"],
            "source_checkpoint_path": exemplar["source_checkpoint_path"],
            "num_seeds": len(group_rows),
            "macro_f1_mean": statistics.mean(macro) if macro else "",
            "macro_f1_std": statistics.stdev(macro) if len(macro) > 1 else 0.0,
            "macro_f1_best": max(macro) if macro else "",
            "auroc_mean": statistics.mean(auroc) if auroc else "",
            "auroc_std": statistics.stdev(auroc) if len(auroc) > 1 else 0.0,
            "accuracy_mean": statistics.mean(acc) if acc else "",
            "accuracy_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
        })
    summary.sort(key=lambda row: row["role"])
    return summary


def validate_stage_block_results(stage_block: str, rows: list[dict[str, Any]]) -> None:
    if stage_block == "a1_coverage":
        validate_variant_ok_counts(rows, min_ok=STAGE_A_COVERAGE_MIN_OK, stage_label="Stage A-1 coverage")
        return
    if stage_block == "a2_nhead":
        validate_variant_ok_counts(rows, min_ok=4, stage_label="Stage A-2 nhead comparison")
        return
    if stage_block == "a3_train_params":
        validate_variant_ok_counts(rows, min_ok=STAGE_A_TRAIN_PARAM_MIN_OK, stage_label="Stage A-3 train-parameter search")
        return
    if stage_block == "b_screen":
        select_stage_b_representatives(rows)
        return
    if stage_block == "c1a_mode":
        validate_variant_ok_counts(rows, min_ok=1, stage_label="Stage C1-A mode search")
        return
    if stage_block == "c1b_lr":
        validate_variant_ok_counts(rows, min_ok=1, stage_label="Stage C1-B lr search")
        return
    if stage_block == "c1c_head":
        validate_variant_ok_counts(rows, min_ok=1, stage_label="Stage C1-C head search")
        return
    if stage_block == "c2_final_compare":
        grouped: dict[str, int] = defaultdict(int)
        for row in rows:
            if row.get("status") == "OK":
                grouped[str(row.get("role", ""))] += 1
        missing = {
            role: count
            for role, count in grouped.items()
            if count < 3
        }
        expected_roles = {"strongest_baseline", "strongest_final_model"}
        for role in expected_roles:
            if grouped.get(role, 0) < 3:
                missing[role] = grouped.get(role, 0)
        if missing:
            raise RuntimeError(
                f"Stage C2 final compare has insufficient successful seeds: {missing}. "
                "Each role must complete 3 seeds before final aggregation."
            )


def run_stage_block(
    run_dir: str | Path,
    stage_block: str,
    *,
    gpu_ids: list[int],
    num_workers: int | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    context = load_run_context(run_dir)
    cpu_runtime = resolve_cpu_runtime_settings(
        requested_num_workers=context["run_config"].get("num_workers", 4) if num_workers is None else num_workers,
        requested_cpu_threads=context["run_config"].get("cpu_threads", DEFAULT_CPU_THREADS),
    )
    prepared = prepare_stage_block(run_dir, stage_block, num_workers=num_workers)
    if len(gpu_ids) > 1:
        adjusted_workers = min(
            int(cpu_runtime["safe_num_workers"]),
            int(context["run_config"].get("multi_gpu_max_num_workers_per_trial", 0)),
        )
        for task in prepared["tasks"]:
            task["num_workers"] = adjusted_workers
            task["task_signature"] = compute_task_signature(task)
        write_rows(prepared["tasks"], prepared["paths"]["plan"])
    rows = run_tasks(
        prepared["tasks"],
        summary_path=prepared["paths"]["summary"],
        live_status_path=prepared["paths"]["live_status"],
        dashboard_path=prepared["paths"]["dashboard"],
        gpu_ids=gpu_ids,
        cpu_threads=int(cpu_runtime["safe_cpu_threads"]),
        gpu_cooldown_seconds=int(context["run_config"].get("gpu_cooldown_seconds", DEFAULT_GPU_COOLDOWN_SECONDS)),
        retry_failed=retry_failed,
    )
    validate_stage_block_results(stage_block, rows)
    return {
        "rows": rows,
        "paths": prepared["paths"],
        "tasks": prepared["tasks"],
        "cpu_runtime": cpu_runtime,
    }
