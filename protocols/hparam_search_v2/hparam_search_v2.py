# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (MiCoFormerV2)
#     language: python
#     name: micoformerv2
# ---

# %% [markdown]
# # MiCoFormer Hyperparameter Search V2
#
# 这个 notebook 是 V2 的控制台，不是长训练的执行器。
#
# 使用方式：
#
# 1. 在服务器上打开 JupyterLab，运行短 cell 做初始化、预览 plan、查看结果。
# 2. 遇到 `→ 复制下方命令到 tmux 执行` 的 cell，只在 notebook 里打印命令。
# 3. 真正的长训练在 tmux 里运行；关闭浏览器或 notebook 不会中断训练。
# 4. 回到 notebook 运行“刷新状态 / 读取结果”类 cell，查看成功、失败、OOM、TensorBoard 命令。

# %%
from __future__ import annotations

from pathlib import Path
import sys

try:
    import pandas as pd
except Exception:  # pragma: no cover - notebook helper
    pd = None


def locate_protocol_root() -> Path:
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    cwd = Path.cwd().resolve()
    candidates.extend([
        cwd,
        cwd / "MiCoFormer" / "protocols" / "hparam_search_v2",
        cwd / "protocols" / "hparam_search_v2",
    ])
    for candidate in candidates:
        if (candidate / "runtime.py").exists():
            return candidate
    raise RuntimeError("Could not locate protocols/hparam_search_v2 directory.")


PROTOCOL_ROOT = locate_protocol_root()
if str(PROTOCOL_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_ROOT))

import runtime


def as_table(rows: list[dict], columns: list[str] | None = None):
    if pd is None:
        if columns:
            return [{key: row.get(key, "") for key in columns} for row in rows]
        return rows
    frame = pd.DataFrame(rows)
    if columns:
        existing = [col for col in columns if col in frame.columns]
        frame = frame[existing]
    return frame


def stage_command(stage_block: str, retry_failed: bool = False) -> str:
    command = [
        sys.executable,
        str(PROTOCOL_ROOT / "run_stage.py"),
        "--run-dir",
        str(RUN_DIR),
        "--stage-block",
        stage_block,
        "--gpu-ids",
        ",".join(str(gpu_id) for gpu_id in GPU_IDS),
    ]
    if NUM_WORKERS is not None:
        command.extend(["--num-workers", str(NUM_WORKERS)])
    if retry_failed:
        command.append("--retry-failed")
    return " ".join(command)


def prepare_and_print_stage(stage_block: str, retry_failed: bool = False):
    prepared = runtime.prepare_stage_block(RUN_DIR, stage_block, num_workers=NUM_WORKERS)
    spec = runtime.get_stage_block_spec(stage_block)
    cpu_runtime = runtime.resolve_cpu_runtime_settings(
        requested_num_workers=NUM_WORKERS,
        requested_cpu_threads=CPU_THREADS,
    )
    print("# ================================================================")
    print(f"# Stage block: {stage_block}")
    print(f"# Trials     : {len(prepared['tasks'])}")
    print(f"# Plan       : {prepared['paths']['plan']}")
    print(f"# Live status: {prepared['paths']['live_status']}")
    print(f"# Summary    : {prepared['paths']['summary']}")
    print(
        f"# CPU        : available={cpu_runtime['available_cpu_cores']}, "
        f"requested_workers={cpu_runtime['requested_num_workers']}, "
        f"safe_workers={cpu_runtime['safe_num_workers']}, "
        f"safe_threads={cpu_runtime['safe_cpu_threads']}"
    )
    if cpu_runtime["available_cpu_cores"] <= 1 and len(GPU_IDS) > 1:
        print("# WARNING    : only 1 CPU core detected; prefer a single GPU for stable runs")
    print("# ================================================================")
    print()
    print(stage_command(stage_block, retry_failed=retry_failed))
    print()
    print("# TensorBoard")
    print(f"tensorboard --logdir {prepared['paths']['log_dir']} --port {spec['tb_port']}")
    return prepared


def show_stage_status(stage_block: str, top_n: int = 10):
    paths = runtime.get_stage_block_paths(RUN_DIR, stage_block)
    live_rows = runtime.load_rows(paths["live_status"])
    summary_rows = runtime.load_rows(paths["summary"])
    dashboard = runtime.read_manifest(paths["dashboard"]) if paths["dashboard"].exists() else {}

    print(f"stage_block = {stage_block}")
    print(f"live_status = {paths['live_status']}")
    print(f"summary     = {paths['summary']}")
    if dashboard:
        print("dashboard   =", dashboard)

    latest = sorted(
        live_rows,
        key=lambda row: (
            str(row.get("status", "")) != "RUNNING",
            str(row.get("end_time", "")),
            str(row.get("start_time", "")),
            str(row.get("trial_id", "")),
        ),
        reverse=True,
    )
    failures = [
        row for row in latest
        if row.get("status") not in ("OK", "PENDING", "RUNNING")
    ]
    cols = [
        "trial_id",
        "model_variant",
        "status",
        "gpu_id",
        "best_val_loss",
        "val_macro_f1",
        "val_auroc",
        "elapsed_seconds",
        "error_message",
    ]
    return {
        "live": as_table(latest[:top_n], cols),
        "failures": as_table(failures[:top_n], cols),
        "summary": as_table(summary_rows[:top_n], cols),
    }


def load_summary(stage_block: str) -> list[dict]:
    return runtime.load_rows(runtime.get_stage_block_paths(RUN_DIR, stage_block)["summary"])


# %% [markdown]
# ## Run Configuration
#
# 唯一需要人工改的通常是 `GPU_IDS` 和可选的 `RUN_ID`。
# 其余配置会写入 `runs/<run_id>/config/`，供 tmux 里的 CLI 读取。

# %%
PROJECT_ROOT = runtime.PROJECT_ROOT
H5AD_PATH = PROJECT_ROOT / "data" / "processed" / "microbiome_dataset.h5ad"

GPU_IDS = [0, 1, 2]
NUM_WORKERS = 4
CPU_THREADS = 1
GPU_COOLDOWN_SECONDS = runtime.DEFAULT_GPU_COOLDOWN_SECONDS
MULTI_GPU_MAX_NUM_WORKERS_PER_TRIAL = 4
SEED = 42

LABEL_FIELD = runtime.DEFAULT_LABEL_FIELD
LABEL_VALUES = runtime.DEFAULT_LABEL_VALUES

USE_SAFE_STAGE_A_BATCH_GRID = True
FINAL_COMPARE_SEEDS = [42, 52, 62]
# R2 优化参数：bias_table 每隔 k 步才反传一次梯度（由 runtime.BIAS_GRAD_EVERY_K 统一控制）
# 如需覆盖，直接在此处改：BIAS_GRAD_EVERY_K = 8
BIAS_GRAD_EVERY_K = runtime.BIAS_GRAD_EVERY_K

RUN_ID = "run_20260412_052731"

if "RUN_ID" not in globals():
    RUN_ID = runtime.make_run_id()
RUN_DIR = PROTOCOL_ROOT / "runs" / RUN_ID

print("PROTOCOL_ROOT =", PROTOCOL_ROOT)
print("PROJECT_ROOT  =", PROJECT_ROOT)
print("RUN_ID        =", RUN_ID)
print("RUN_DIR       =", RUN_DIR)
print("H5AD_PATH     =", H5AD_PATH)
print("GPU_IDS       =", GPU_IDS)
print("NUM_WORKERS   =", NUM_WORKERS)
print("CPU_THREADS   =", CPU_THREADS)
print("GPU_COOLDOWN  =", GPU_COOLDOWN_SECONDS)
print("MULTI_GPU_NUM_WORKERS =", MULTI_GPU_MAX_NUM_WORKERS_PER_TRIAL)
print("AVAILABLE_CPU =", runtime.detect_available_cpu_cores())


# %% [markdown]
# ## Initialize Run Layout
#
# 这一步只做轻量初始化，生成目录、配置和环境快照。

# %%
layout = runtime.init_run_dir(RUN_DIR)
runtime.capture_environment_snapshot(layout["config"] / "env_snapshot.txt", GPU_IDS)
runtime.write_manifest(
    layout["config"] / "run_config.yaml",
    {
        "run_id": RUN_ID,
        "h5ad_path": str(H5AD_PATH),
        "gpu_ids": GPU_IDS,
        "num_workers": NUM_WORKERS,
        "cpu_threads": CPU_THREADS,
        "gpu_cooldown_seconds": GPU_COOLDOWN_SECONDS,
        "multi_gpu_max_num_workers_per_trial": MULTI_GPU_MAX_NUM_WORKERS_PER_TRIAL,
        "seed": SEED,
        "label_field": LABEL_FIELD,
        "label_values": LABEL_VALUES,
        "use_safe_stage_a_batch_grid": USE_SAFE_STAGE_A_BATCH_GRID,
        "final_compare_seeds": FINAL_COMPARE_SEEDS,
        "bias_grad_every_k": BIAS_GRAD_EVERY_K,
    },
)
print("Initialized run directory:", RUN_DIR)


# %% [markdown]
# ## Build Default Splits

# %%
split_paths = runtime.build_default_splits(H5AD_PATH, layout["splits"])
runtime.write_manifest(layout["config"] / "split_paths.yaml", split_paths)
split_paths


# %% [markdown]
# ## Stage A-1
#
# 先运行下面 cell 生成命令，然后复制到 tmux。训练过程中可随时运行“刷新状态”。

# %%
prepared = prepare_and_print_stage("a1_coverage")


# %%
show = show_stage_status("a1_coverage")


# %% [markdown]
# ## Stage A-1 Summary And Shortlist Suggestion

# %%
coverage_rows = load_summary("a1_coverage")
coverage_overview = runtime.compute_stage_a_coverage_overview(coverage_rows) if coverage_rows else []
if coverage_overview:
    runtime.write_rows(coverage_overview, layout["decisions"] / "shortlist_suggestion.csv")
    SUGGESTED_SHORTLIST = runtime.suggest_shortlist(coverage_overview, shortlist_size=4)
    runtime.write_manifest(layout["decisions"] / "shortlist_suggestion.yaml", SUGGESTED_SHORTLIST)
else:
    SUGGESTED_SHORTLIST = []

as_table(
    coverage_overview,
    ["d_model", "num_layers", "ok_count", "oom_count", "mean_rank", "mean_best_val_loss"],
), SUGGESTED_SHORTLIST


# %% [markdown]
# ## Stage A-1 Manual Confirmation
#
# 如果你不想用自动建议，直接修改 `SHORTLIST_OVERRIDE`。

# %%
SHORTLIST_OVERRIDE = None

CONFIRMED_SHORTLIST = SHORTLIST_OVERRIDE or SUGGESTED_SHORTLIST
runtime.write_manifest(layout["decisions"] / "shortlist_confirmed.yaml", CONFIRMED_SHORTLIST)
CONFIRMED_SHORTLIST


# %% [markdown]
# ## Stage A-2

# %%
prepared = prepare_and_print_stage("a2_nhead")


# %%
show_stage_status("a2_nhead")


# %% [markdown]
# ## Stage A-2 Summary And Locked-Arch Suggestion

# %%
nhead_rows = load_summary("a2_nhead")
locked_arch_overview = (
    runtime.compute_locked_arch_overview(
        coverage_rows=coverage_rows,
        nhead_rows=nhead_rows,
        shortlist=CONFIRMED_SHORTLIST,
    )
    if coverage_rows and nhead_rows and CONFIRMED_SHORTLIST
    else []
)
if locked_arch_overview:
    runtime.write_rows(locked_arch_overview, layout["decisions"] / "locked_arch_suggestion.csv")
    SUGGESTED_LOCKED_ARCH = runtime.suggest_locked_arch(locked_arch_overview)
    runtime.write_manifest(layout["decisions"] / "locked_arch_suggestion.yaml", SUGGESTED_LOCKED_ARCH)
else:
    SUGGESTED_LOCKED_ARCH = {}

as_table(
    locked_arch_overview,
    ["d_model", "num_layers", "nhead_regime", "nhead", "ok_count", "mean_best_val_loss"],
), SUGGESTED_LOCKED_ARCH


# %% [markdown]
# ## Stage A-2 Manual Confirmation

# %%
LOCKED_ARCH_OVERRIDE = None

CONFIRMED_LOCKED_ARCH = LOCKED_ARCH_OVERRIDE or SUGGESTED_LOCKED_ARCH
runtime.write_manifest(layout["decisions"] / "locked_arch_confirmed.yaml", CONFIRMED_LOCKED_ARCH)
CONFIRMED_LOCKED_ARCH


# %% [markdown]
# ## Stage A-3

# %%
prepared = prepare_and_print_stage("a3_train_params")


# %%
show = show_stage_status("a3_train_params")


# %% [markdown]
# ## Stage A Top-3 Promotion

# %%
train_param_rows = load_summary("a3_train_params")
stage_a_top3_rows = runtime.select_top_k_per_variant(
    train_param_rows,
    metric_key="best_val_loss",
    k=3,
    reverse=False,
    min_ok_per_variant=runtime.STAGE_A_TRAIN_PARAM_MIN_OK,
    stage_label="Stage A top-3 selection",
) if train_param_rows else []

promoted_top3 = []
for row in stage_a_top3_rows:
    promoted = dict(row)
    alias = f"{row['model_variant']}_rank{row['selected_rank']}"
    promoted["checkpoint_path"] = runtime.promote_checkpoint(
        row,
        layout["stage_a_checkpoints"] / str(row["model_variant"]),
        alias,
    )
    promoted["promoted_checkpoint_path"] = promoted["checkpoint_path"]
    promoted_top3.append(promoted)

runtime.write_rows(promoted_top3, layout["stage_a"] / "top3_candidates.csv")
runtime.write_manifest(layout["decisions"] / "stage_a_top3.yaml", promoted_top3)
as_table(promoted_top3, ["model_variant", "selected_rank", "best_val_loss", "checkpoint_path"])


# %% [markdown]
# ## Stage B

# %%
prepared = prepare_and_print_stage("b_screen")


# %%
show_stage_status("b_screen")


# %% [markdown]
# ## Stage B Representatives

# %%
stage_b_rows = load_summary("b_screen")
stage_b_representatives = runtime.select_stage_b_representatives(stage_b_rows) if stage_b_rows else []
promoted_representatives = []
for row in stage_b_representatives:
    promoted = dict(row)
    alias = f"{row['model_variant']}_representative"
    promoted["checkpoint_path"] = runtime.promote_checkpoint(
        row,
        layout["stage_b_checkpoints"] / str(row["model_variant"]),
        alias,
    )
    promoted["promoted_checkpoint_path"] = promoted["checkpoint_path"]
    promoted_representatives.append(promoted)

runtime.write_rows(promoted_representatives, layout["stage_b"] / "representatives.csv")
runtime.write_manifest(layout["decisions"] / "stage_b_representatives.yaml", promoted_representatives)
as_table(promoted_representatives, ["model_variant", "val_macro_f1", "val_auroc", "checkpoint_path"])


# %% [markdown]
# ## Stage C1-A

# %%
prepared = prepare_and_print_stage("c1a_mode")


# %%
show = show_stage_status("c1a_mode")


# %% [markdown]
# ## Stage C1-A Best Mode

# %%
stage_c_mode_rows = load_summary("c1a_mode")
stage_c_best_mode = runtime.select_best_stage_c_block(stage_c_mode_rows, "mode") if stage_c_mode_rows else []
runtime.write_manifest(layout["decisions"] / "stage_c_best_mode.yaml", stage_c_best_mode)
as_table(stage_c_best_mode, ["model_variant", "pooling_mode", "freeze_encoder", "val_macro_f1", "val_auroc"])


# %%
# 这里为了强行保持所有模型架构采用相同的模式，决定运行下面这个 cell，来要求大家都采用 sample and mean

stage_c_mode_rows = load_summary("c1a_mode")

# 1. 先保留“各自最优”的自动结果，方便回溯
stage_c_best_mode_auto = (
  runtime.select_best_stage_c_block(stage_c_mode_rows, "mode")
  if stage_c_mode_rows else []
)
runtime.write_manifest(
  layout["decisions"] / "stage_c_best_mode_auto.yaml",
  stage_c_best_mode_auto,
)

# 2. 决定是否强制共享 mode
USE_SHARED_MODE = True
SHARED_POOLING_MODE = "sample_and_mean"
SHARED_FREEZE_ENCODER = True

if USE_SHARED_MODE:
  stage_c_best_mode = [
      row for row in stage_c_mode_rows
      if row.get("search_block") == "mode"
      and row.get("pooling_mode") == SHARED_POOLING_MODE
      and bool(row.get("freeze_encoder")) is SHARED_FREEZE_ENCODER
      and row.get("status") == "OK"
  ]

  stage_c_best_mode = runtime.select_top_k_per_variant(
      stage_c_best_mode,
      metric_key="val_macro_f1",
      k=1,
      reverse=True,
      min_ok_per_variant=1,
      stage_label="Stage C1-A shared-mode selection",
  )
else:
  stage_c_best_mode = stage_c_best_mode_auto

runtime.write_manifest(
  layout["decisions"] / "stage_c_best_mode.yaml",
  stage_c_best_mode,
)

as_table(
  stage_c_best_mode,
  ["model_variant", "pooling_mode", "freeze_encoder", "val_macro_f1", "val_auroc"],
)

# %% [markdown]
# ## Stage C1-B

# %%
prepared = prepare_and_print_stage("c1b_lr")


# %%
show = show_stage_status("c1b_lr")


# %% [markdown]
# ## Stage C1-B Best LR

# %%
stage_c_lr_rows = load_summary("c1b_lr")
stage_c_best_lr = runtime.select_best_stage_c_block(stage_c_lr_rows, "lr") if stage_c_lr_rows else []
runtime.write_manifest(layout["decisions"] / "stage_c_best_lr.yaml", stage_c_best_lr)
as_table(stage_c_best_lr, ["model_variant", "lr_head", "lr_encoder", "val_macro_f1", "val_auroc"])


# %% [markdown]
# ## Stage C1-B-supp

# %% [markdown]
# 刚刚 C1B 强行让所有模型架构都是用了 sample and mean 的方式，所以现在补充再看一下 最优方式是 mean 的两个模型，是什么情况 

# %%
from pathlib import Path
import shutil
import sys

SUPP_RUN_ID = RUN_ID + "_supp_c1b_mode"
SUPP_RUN_DIR = PROTOCOL_ROOT / "runs" / SUPP_RUN_ID
supp_layout = runtime.init_run_dir(SUPP_RUN_DIR)

for rel in [
  "config/run_config.yaml",
  "config/split_paths.yaml",
]:
  src = RUN_DIR / rel
  dst = SUPP_RUN_DIR / rel
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)

print("SUPP_RUN_DIR =", SUPP_RUN_DIR)
print("RUN_DIR      =", RUN_DIR)

# %%
stage_c_mode_rows = load_summary("c1a_mode")
stage_c_best_mode_auto = (
  runtime.select_best_stage_c_block(stage_c_mode_rows, "mode")
  if stage_c_mode_rows else []
)
main_best_mode = runtime.read_manifest(RUN_DIR / "decisions" / "stage_c_best_mode.yaml")

auto_by_variant = {row["model_variant"]: row for row in stage_c_best_mode_auto}
main_by_variant = {row["model_variant"]: row for row in main_best_mode}

stage_c_best_mode_supp = [
  main_by_variant["baseline"],
  auto_by_variant["r1"],
  main_by_variant["r2"],
  auto_by_variant["r1r2"],
]

runtime.write_manifest(
  SUPP_RUN_DIR / "decisions" / "stage_c_best_mode.yaml",
  stage_c_best_mode_supp,
)

main_c1b_rows = runtime.load_rows(RUN_DIR / "stage_c" / "c1b_lr_summary.csv")
seed_rows = [
  row for row in main_c1b_rows
  if row.get("model_variant") in {"baseline", "r2"}
]

supp_c1b_paths = runtime.get_stage_block_paths(SUPP_RUN_DIR, "c1b_lr")
runtime.write_rows(seed_rows, supp_c1b_paths["summary"])

as_table(
  stage_c_best_mode_supp,
  ["model_variant", "pooling_mode", "freeze_encoder", "val_macro_f1", "val_auroc"],
)

# %%
supp_prepared = runtime.prepare_stage_block(SUPP_RUN_DIR, "c1b_lr", num_workers=NUM_WORKERS)
supp_spec = runtime.get_stage_block_spec("c1b_lr")

print("# Trials     :", len(supp_prepared["tasks"]))
print("# Plan       :", supp_prepared["paths"]["plan"])
print("# Live status:", supp_prepared["paths"]["live_status"])
print("# Summary    :", supp_prepared["paths"]["summary"])
print()
print(" ".join([
  sys.executable,
  str(PROTOCOL_ROOT / "run_stage.py"),
  "--run-dir", str(SUPP_RUN_DIR),
  "--stage-block", "c1b_lr",
  "--gpu-ids", ",".join(str(gpu_id) for gpu_id in GPU_IDS),
  "--num-workers", str(NUM_WORKERS),
]))
print()
print("# TensorBoard")
print(f"tensorboard --logdir {supp_prepared['paths']['log_dir']} --port {supp_spec['tb_port']}")

# %% [markdown]
# ## Stage C1-C

# %%
prepared = prepare_and_print_stage("c1c_head")


# %%
show_stage_status("c1c_head")


# %% [markdown]
# ## Stage C Finalists

# %%
stage_c_head_rows = load_summary("c1c_head")
stage_c_all_rows = (
    load_summary("c1a_mode")
    + load_summary("c1b_lr")
    + stage_c_head_rows
)
stage_c_finalists = runtime.select_stage_c_finalists(stage_c_all_rows) if stage_c_all_rows else []
runtime.write_manifest(layout["decisions"] / "final_candidates.yaml", stage_c_finalists)
as_table(stage_c_finalists, ["role", "model_variant", "val_macro_f1", "val_auroc", "source_checkpoint_path"])


# %% [markdown]
# ## Stage C2

# %%
prepared = prepare_and_print_stage("c2_final_compare")


# %%
show_stage_status("c2_final_compare")


# %% [markdown]
# ## Final Comparison Table

# %%
stage_c_final_rows = load_summary("c2_final_compare")
final_compare = runtime.summarize_final_compare(stage_c_final_rows) if stage_c_final_rows else []
runtime.write_rows(final_compare, layout["stage_c"] / "final_compare.csv")
as_table(final_compare)


# %% [markdown]
# ## Run Outputs
#
# 常用文件：
#
# - `config/run_config.yaml`
# - `decisions/*.yaml`
# - `stage_a/*_plan.csv`
# - `stage_a/*_live_status.csv`
# - `stage_a/*_summary.csv`
# - `stage_b/*_live_status.csv`
# - `stage_c/*_live_status.csv`
