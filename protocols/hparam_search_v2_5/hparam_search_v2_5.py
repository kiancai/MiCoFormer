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
# # V2.5 超参数验证协议
#
# 使用修复后的 R2 代码（`bias_grad_every_k` 机制），在 V2 确定的参数下，
# 完成四个模型版本（baseline / r1 / r2 / r1r2）的完整预训练 → 微调 → C 数据集 per-study 预测流程。
#
# 使用方式：
# 1. 在服务器上打开 JupyterLab，运行短 cell 做初始化、预览 plan、查看结果。
# 2. 遇到 `→ 复制下方命令到 tmux 执行` 的 cell，只在 notebook 里打印命令。
# 3. 真正的长训练在 tmux 里运行；关闭浏览器或 notebook 不会中断训练。
# 4. 回到 notebook 运行"刷新状态 / 读取结果"类 cell，查看成功、失败、OOM、TensorBoard 命令。
# 5. 每个 stage 完成后有验证 cell，确保 R2 bias_table 等关键指标正常。

# %%
from __future__ import annotations

import shutil
from pathlib import Path
import sys

import numpy as np

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
        cwd / "MiCoFormer" / "protocols" / "hparam_search_v2_5",
        cwd / "protocols" / "hparam_search_v2_5",
    ])
    for candidate in candidates:
        if (candidate / "runtime.py").exists():
            return candidate
    raise RuntimeError("Could not locate protocols/hparam_search_v2_5 directory.")


PROTOCOL_ROOT = locate_protocol_root()
if str(PROTOCOL_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_ROOT))

import runtime
from protocols.full_training.config import SHARED_ARCH, SHARED_PRETRAIN, SHARED_FINETUNE, VARIANTS


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
    """构建 tasks、写 plan CSV、打印 tmux 命令。"""
    paths = runtime.get_stage_block_paths(RUN_DIR, stage_block)

    # 根据 stage block 构建 tasks 并写入 plan CSV
    if stage_block == "pretrain":
        tasks = runtime.build_pretrain_tasks(
            variant_names=VARIANTS_TO_RUN,
            seed=SEED,
            h5ad_path=str(H5AD_PATH),
            run_dir=RUN_DIR,
            splits_dir=SPLITS_DIR,
            num_workers=NUM_WORKERS or 4,
        )
    elif stage_block == "finetune":
        pretrain_summary = runtime.load_rows(
            runtime.get_stage_block_paths(RUN_DIR, "pretrain")["summary"]
        )
        pretrain_ckpts = {}
        for row in pretrain_summary:
            if row.get("status") == "OK" and row.get("checkpoint_path"):
                pretrain_ckpts[row["model_variant"]] = row["checkpoint_path"]
        missing = [v for v in VARIANTS_TO_RUN if v not in pretrain_ckpts]
        if missing:
            print(f"WARNING: missing pretrain checkpoints for {missing}; skipping finetune plan.")
            return {"paths": paths, "tasks": []}
        tasks = runtime.build_finetune_tasks(
            variant_names=VARIANTS_TO_RUN,
            pretrained_ckpts=pretrain_ckpts,
            seed=SEED,
            h5ad_path=str(H5AD_PATH),
            run_dir=RUN_DIR,
            splits_dir=SPLITS_DIR,
            num_workers=NUM_WORKERS or 4,
        )
    elif stage_block == "evaluate_c":
        finetune_summary = runtime.load_rows(
            runtime.get_stage_block_paths(RUN_DIR, "finetune")["summary"]
        )
        finetune_ckpts = {}
        for row in finetune_summary:
            if row.get("status") == "OK" and row.get("checkpoint_path"):
                finetune_ckpts[row["model_variant"]] = row["checkpoint_path"]
        missing = [v for v in VARIANTS_TO_RUN if v not in finetune_ckpts]
        if missing:
            print(f"WARNING: missing finetune checkpoints for {missing}; skipping evaluate_c plan.")
            return {"paths": paths, "tasks": []}
        tasks = runtime.build_evaluate_c_tasks(
            variant_names=VARIANTS_TO_RUN,
            finetuned_ckpts=finetune_ckpts,
            seed=SEED,
            h5ad_path=str(H5AD_PATH),
            run_dir=RUN_DIR,
            splits_dir=SPLITS_DIR,
            label_field=LABEL_FIELD,
            label_values=LABEL_VALUES,
        )
    else:
        raise ValueError(f"Unknown stage_block: {stage_block}")

    runtime.write_rows(tasks, paths["plan"])

    # 打印 tmux 命令
    cpu_runtime = runtime.resolve_cpu_runtime_settings(
        requested_num_workers=NUM_WORKERS,
        requested_cpu_threads=CPU_THREADS,
    )
    spec = runtime.get_stage_block_spec(stage_block)
    print("# ================================================================")
    print(f"# Stage block: {stage_block}")
    print(f"# Trials     : {len(tasks)}")
    print(f"# Plan       : {paths['plan']}")
    print(f"# Live status: {paths['live_status']}")
    print(f"# Summary    : {paths['summary']}")
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
    print(f"tensorboard --logdir {paths['stage_dir']} --port {spec['tb_port']}")
    return {"paths": paths, "tasks": tasks}


def show_stage_status(stage_block: str, top_n: int = 15):
    paths = runtime.get_stage_block_paths(RUN_DIR, stage_block)
    summary_rows = runtime.load_rows(paths["summary"])
    dashboard = runtime.read_manifest(paths["dashboard"]) if paths["dashboard"].exists() else {}

    print(f"stage_block = {stage_block}")
    print(f"summary     = {paths['summary']}")
    print(f"total_rows  = {len(summary_rows)}")
    if dashboard:
        print("dashboard   =", dashboard)

    cols = [
        "trial_id", "model_variant", "status", "gpu_id",
        "best_val_loss", "val_macro_f1", "val_auroc",
        "test_macro_f1", "test_auroc", "test_accuracy",
        "elapsed_seconds", "error_message",
    ]
    # 只显示实际存在的列
    cols = [c for c in cols if any(c in row for row in summary_rows)]
    failures = [r for r in summary_rows if r.get("status") not in ("OK", "PENDING", "RUNNING")]
    return {
        "summary": as_table(summary_rows[:top_n], cols),
        "failures": as_table(failures[:top_n], cols),
    }


def load_summary(stage_block: str) -> list[dict]:
    return runtime.load_rows(runtime.get_stage_block_paths(RUN_DIR, stage_block)["summary"])


# %% [markdown]
# ## Run Configuration
#
# 唯一需要人工改的通常是 `GPU_IDS`。其余配置会写入 `runs/<run_id>/config/`。

# %%
PROJECT_ROOT = runtime.PROJECT_ROOT
H5AD_PATH = PROJECT_ROOT / "data" / "processed" / "microbiome_dataset.h5ad"

GPU_IDS = [0, 1]
NUM_WORKERS = 4
CPU_THREADS = 1
SEED = 42

LABEL_FIELD = runtime.DEFAULT_LABEL_FIELD
LABEL_VALUES = runtime.DEFAULT_LABEL_VALUES

VARIANTS_TO_RUN = ["baseline", "r1", "r2", "r1r2"]

RUN_ID = runtime.make_run_id(prefix="v25")
RUN_DIR = PROTOCOL_ROOT / "runs" / RUN_ID
SPLITS_DIR = RUN_DIR / "splits"

print("PROTOCOL_ROOT =", PROTOCOL_ROOT)
print("PROJECT_ROOT  =", PROJECT_ROOT)
print("RUN_ID        =", RUN_ID)
print("RUN_DIR       =", RUN_DIR)
print("SPLITS_DIR    =", SPLITS_DIR)
print("H5AD_PATH     =", H5AD_PATH)
print("GPU_IDS       =", GPU_IDS)
print("NUM_WORKERS   =", NUM_WORKERS)
print("SEED          =", SEED)
print("VARIANTS      =", VARIANTS_TO_RUN)
print("LABEL_FIELD   =", LABEL_FIELD)
print("LABEL_VALUES  =", LABEL_VALUES)
print("AVAILABLE_CPU =", runtime.detect_available_cpu_cores())
print()
print("共享架构:", dict(SHARED_ARCH))
print("共享预训练:", {k: v for k, v in SHARED_PRETRAIN.items() if k != "abundance_mode"})
print("共享微调:", dict(SHARED_FINETUNE))
for vname in VARIANTS_TO_RUN:
    v = VARIANTS[vname]
    print(f"  {vname}: emb={v.token_embedding_mode}, bias={v.use_taxonomy_bias}, "
          f"lr={v.pretrain_lr}, lr_head={v.lr_head}")


# %% [markdown]
# ## Initialize Run Layout
#
# 轻量初始化：创建目录、设置 splits、写配置和环境快照。

# %%
layout = runtime.init_run_dir(RUN_DIR)

# 设置 split 文件：symlink 或 copy 原始 split 到 run_dir/splits/
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
split_source = PROJECT_ROOT / "data" / "processed" / "splits"
split_mapping = {
    "split_group_A.npy": "pretrain_train.npy",
    "split_group_B.npy": "pretrain_val.npy",
    "split_group_C.npy": "pretrain_test_c.npy",
}
for src_name, dst_name in split_mapping.items():
    src = split_source / src_name
    dst = SPLITS_DIR / dst_name
    if src.exists():
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        print(f"  {dst_name} -> {src}")
    else:
        print(f"  WARNING: {src} not found!")

# 写 run_config.yaml
runtime.write_manifest(
    layout["config"] / "run_config.yaml",
    {
        "run_id": RUN_ID,
        "h5ad_path": str(H5AD_PATH),
        "gpu_ids": GPU_IDS,
        "num_workers": NUM_WORKERS,
        "cpu_threads": CPU_THREADS,
        "seed": SEED,
        "label_field": LABEL_FIELD,
        "label_values": LABEL_VALUES,
        "variants": VARIANTS_TO_RUN,
        "shared_arch": dict(SHARED_ARCH),
        "bias_grad_every_k": SHARED_PRETRAIN["bias_grad_every_k"],
    },
)
runtime.capture_environment_snapshot(layout["config"] / "env_snapshot.txt", GPU_IDS)
print("\nInitialized run directory:", RUN_DIR)


# %% [markdown]
# ## Pre-flight Validation
#
# 验证数据文件、config 参数、split 重叠。**必须全部通过才能继续。**

# %%
passed = True

# 1. 数据文件存在
for path, name in [(H5AD_PATH, "h5ad"), (SPLITS_DIR / "pretrain_train.npy", "train split"),
                    (SPLITS_DIR / "pretrain_val.npy", "val split"),
                    (SPLITS_DIR / "pretrain_test_c.npy", "C split")]:
    if not path.exists():
        print(f"FAIL: {name} not found: {path}")
        passed = False
    else:
        print(f"OK:   {name} = {path}")

# 2. 共享架构参数
checks = [
    (SHARED_ARCH["d_model"] == 512, f"d_model={SHARED_ARCH['d_model']} (expected 512)"),
    (SHARED_ARCH["nhead"] == 16, f"nhead={SHARED_ARCH['nhead']} (expected 16, fine regime)"),
    (SHARED_ARCH["num_layers"] == 12, f"num_layers={SHARED_ARCH['num_layers']} (expected 12)"),
    (SHARED_ARCH["ff_ratio"] == 4, f"ff_ratio={SHARED_ARCH['ff_ratio']} (expected 4)"),
]
for ok, msg in checks:
    status = "OK" if ok else "FAIL"
    if not ok:
        passed = False
    print(f"{status}: SHARED_ARCH {msg}")

# 3. R2 修复验证
bgk = SHARED_PRETRAIN["bias_grad_every_k"]
if bgk > 0:
    print(f"OK:   bias_grad_every_k={bgk} (R2 fix active)")
else:
    print(f"FAIL: bias_grad_every_k={bgk} (R2 fix NOT active!)")
    passed = False

# 4. 变体配置
for vname in VARIANTS_TO_RUN:
    v = VARIANTS[vname]
    is_r2 = v.use_taxonomy_bias
    if vname in ("r2", "r1r2") and not is_r2:
        print(f"FAIL: {vname} should have use_taxonomy_bias=True")
        passed = False
    elif vname in ("baseline", "r1") and is_r2:
        print(f"FAIL: {vname} should have use_taxonomy_bias=False")
        passed = False
    else:
        print(f"OK:   {vname} emb={v.token_embedding_mode}, bias={is_r2}")

# 5. Split 重叠检查
train_idx = np.load(str(SPLITS_DIR / "pretrain_train.npy"))
val_idx = np.load(str(SPLITS_DIR / "pretrain_val.npy"))
test_c_idx = np.load(str(SPLITS_DIR / "pretrain_test_c.npy"))
print(f"\nSplit sizes: train={len(train_idx)}, val={len(val_idx)}, test_C={len(test_c_idx)}")

overlap_ab = set(train_idx) & set(val_idx)
overlap_ac = set(train_idx) & set(test_c_idx)
overlap_bc = set(val_idx) & set(test_c_idx)
if overlap_ab:
    print(f"FAIL: train ∩ val = {len(overlap_ab)} samples")
    passed = False
else:
    print("OK:   train ∩ val = empty")
if overlap_ac:
    print(f"FAIL: train ∩ C = {len(overlap_ac)} samples")
    passed = False
else:
    print("OK:   train ∩ C = empty")
if overlap_bc:
    print(f"FAIL: val ∩ C = {len(overlap_bc)} samples")
    passed = False
else:
    print("OK:   val ∩ C = empty")

print("\n" + ("=" * 50))
print("PRE-FLIGHT:", "PASSED" if passed else "FAILED — fix issues above before continuing")
print("=" * 50)


# %% [markdown]
# ## Stage 1: Pretrain
#
# 4 个 variant 各 1 个预训练 trial（baseline / r1 / r2 / r1r2）。
# 预算：20 epochs，每 2 epochs 验证。
#
# **→ 复制下方命令到 tmux 执行**

# %%
prepared = prepare_and_print_stage("pretrain")


# %% [markdown]
# ## 刷新 Pretrain 状态

# %%
show = show_stage_status("pretrain")
show["summary"]


# %%
show["failures"]


# %% [markdown]
# ## Post-Pretrain Validation
#
# **最关键的验证点**：检查 R2 bias_table 是否非零。
# 如果 bias_table 全零，说明 R2 bug 仍在，必须停止并排查。

# %%
pretrain_summary = load_summary("pretrain")
pretrain_paths = runtime.get_stage_block_paths(RUN_DIR, "pretrain")

print(f"Pretrain trials: {len(pretrain_summary)}")
ok_count = sum(1 for r in pretrain_summary if r.get("status") == "OK")
print(f"OK: {ok_count}/{len(pretrain_summary)}")

# 收集 checkpoint 路径
pretrain_ckpts = {}
for row in pretrain_summary:
    variant = row.get("model_variant", "?")
    status = row.get("status", "?")
    ckpt = row.get("checkpoint_path", "")
    loss = row.get("best_val_loss", "N/A")
    print(f"  {variant}: status={status}, val_loss={loss}, ckpt={ckpt}")
    if status == "OK" and ckpt:
        pretrain_ckpts[variant] = ckpt

print(f"\nCollected checkpoints: {len(pretrain_ckpts)}/{len(VARIANTS_TO_RUN)}")

# R2 bias_table 验证
print("\n" + "=" * 60)
print("R2 BIAS TABLE VERIFICATION")
print("=" * 60)
bias_passed = True
for variant_name in ["r2", "r1r2"]:
    ckpt_path = pretrain_ckpts.get(variant_name, "")
    if not ckpt_path:
        print(f"  {variant_name}: SKIP (no checkpoint)")
        continue
    report = runtime.verify_bias_table(ckpt_path)
    status_str = "PASS" if report["passed"] else "FAIL"
    all_zero_str = "ALL ZEROS!" if report["all_zeros"] else "non-zero"
    print(f"  {variant_name}: {status_str}")
    print(f"    has_bias_table={report['has_bias_table']}, shape={report['shape']}")
    print(f"    {all_zero_str}, mean_abs={report['mean_abs']:.6f}, max_abs={report['max_abs']:.6f}")
    print(f"    per_head_mean_abs={[f'{x:.6f}' for x in report['per_head_mean_abs']]}")
    if not report["passed"]:
        bias_passed = False
        print(f"    *** R2 BUG DETECTED for {variant_name} — DO NOT PROCEED ***")

# 验证 baseline/r1 不含 bias_table
print()
for variant_name in ["baseline", "r1"]:
    ckpt_path = pretrain_ckpts.get(variant_name, "")
    if not ckpt_path:
        continue
    report = runtime.verify_bias_table(ckpt_path)
    if report["has_bias_table"]:
        print(f"  WARNING: {variant_name} unexpectedly has bias_table")
    else:
        print(f"  OK: {variant_name} has no bias_table (as expected)")

# 验证 val_loss 非 NaN
print("\nVal loss checks:")
loss_passed = True
for row in pretrain_summary:
    variant = row.get("model_variant", "?")
    loss = row.get("best_val_loss")
    if loss is None or (isinstance(loss, float) and np.isnan(loss)):
        print(f"  FAIL: {variant} val_loss is NaN")
        loss_passed = False
    else:
        print(f"  OK:   {variant} val_loss={loss}")

print("\n" + "=" * 60)
print("POST-PRETRAIN VALIDATION:", "PASSED" if (bias_passed and loss_passed) else "FAILED")
if not bias_passed:
    print("*** R2 BIAS TABLE BUG DETECTED — STOP AND INVESTIGATE ***")
print("=" * 60)


# %% [markdown]
# ## Stage 2: Finetune
#
# 4 个 variant 各 1 个微调 trial。从 Stage 1 的 pretrain checkpoint 出发。
# 预算：20 epochs，patience=5（monitor=val/macro_f1）。
# **仅在 post-pretrain validation 通过后继续。**
#
# **→ 复制下方命令到 tmux 执行**

# %%
prepared = prepare_and_print_stage("finetune")


# %% [markdown]
# ## 刷新 Finetune 状态

# %%
show = show_stage_status("finetune")
show["summary"]


# %%
show["failures"]


# %% [markdown]
# ## Post-Finetune Validation

# %%
finetune_summary = load_summary("finetune")

print(f"Finetune trials: {len(finetune_summary)}")
ok_count = sum(1 for r in finetune_summary if r.get("status") == "OK")
print(f"OK: {ok_count}/{len(finetune_summary)}")

# 收集 finetune checkpoint
finetune_ckpts = {}
for row in finetune_summary:
    variant = row.get("model_variant", "?")
    status = row.get("status", "?")
    ckpt = row.get("checkpoint_path", "")
    f1 = row.get("val_macro_f1", "N/A")
    auroc = row.get("val_auroc", "N/A")
    print(f"  {variant}: status={status}, f1={f1}, auroc={auroc}, ckpt={ckpt}")
    if status == "OK" and ckpt:
        finetune_ckpts[variant] = ckpt

# 验证 val_macro_f1 > 0.5（基本 sanity check）
print("\nMacro F1 checks:")
f1_passed = True
for row in finetune_summary:
    variant = row.get("model_variant", "?")
    f1 = row.get("val_macro_f1")
    if f1 is None:
        print(f"  FAIL: {variant} val_macro_f1 is None")
        f1_passed = False
    elif float(f1) < 0.5:
        print(f"  WARN: {variant} val_macro_f1={f1} (< 0.5)")
    else:
        print(f"  OK:   {variant} val_macro_f1={f1}")

# 验证 r2/r1r2 的 bias_table 在微调后未丢失
print("\nPost-finetune bias_table check:")
bias_ok = True
for variant_name in ["r2", "r1r2"]:
    ckpt_path = finetune_ckpts.get(variant_name, "")
    if not ckpt_path:
        continue
    report = runtime.verify_bias_table(ckpt_path)
    if report["passed"]:
        print(f"  OK:   {variant_name} bias_table intact (mean_abs={report['mean_abs']:.6f})")
    else:
        print(f"  FAIL: {variant_name} bias_table lost or zero after finetune!")
        bias_ok = False

# 打印 4 variant 的 val metrics 对比
print("\n" + "=" * 60)
print("FINETUNE RESULTS COMPARISON")
print("=" * 60)
as_table(
    finetune_summary,
    ["model_variant", "status", "val_macro_f1", "val_auroc", "val_accuracy",
     "test_macro_f1", "test_auroc", "test_accuracy"],
)

print("\n" + "=" * 60)
print("POST-FINETUNE VALIDATION:", "PASSED" if (f1_passed and bias_ok) else "FAILED")
print("=" * 60)


# %% [markdown]
# ## Stage 3: Evaluate C (Per-Study Prediction)
#
# 在 C 数据集上按 Project_ID 分组，直接评估 Stage 2 的 finetuned checkpoint。
# 每个 (variant, study) 一个 task — 不做二次微调，不做 LOO。
#
# **→ 复制下方命令到 tmux 执行**

# %%
prepared = prepare_and_print_stage("evaluate_c")


# %% [markdown]
# ## 刷新 Evaluate C 状态

# %%
show = show_stage_status("evaluate_c", top_n=30)
show["summary"]


# %%
show["failures"]


# %% [markdown]
# ## Final Results

# %%
eval_summary = load_summary("evaluate_c")

if not eval_summary:
    print("No evaluate_c results found. Run Stage 3 first.")
else:
    ok_rows = [r for r in eval_summary if r.get("status") == "OK"]
    fail_rows = [r for r in eval_summary if r.get("status") != "OK"]
    print(f"Evaluate C trials: {len(eval_summary)} total, {len(ok_rows)} OK, {len(fail_rows)} failed")

    # 按 study 展示 per-variant results
    studies = sorted(set(r.get("study_id", "") for r in ok_rows))
    print(f"\nStudies: {len(studies)}")
    print()

    results_by_study = []
    for study_id in studies:
        study_rows = [r for r in ok_rows if r.get("study_id") == study_id]
        for row in study_rows:
            results_by_study.append({
                "study_id": study_id,
                "model_variant": row.get("model_variant", "?"),
                "n_test": row.get("n_test", "?"),
                "test_macro_f1": row.get("test_macro_f1"),
                "test_auroc": row.get("test_auroc"),
                "test_accuracy": row.get("test_accuracy"),
            })

    if results_by_study:
        print("Per-study results:")
        display(as_table(results_by_study))

    # 按 variant 聚合
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS BY VARIANT")
    print("=" * 60)

    for variant_name in VARIANTS_TO_RUN:
        variant_rows = [r for r in ok_rows if r.get("model_variant") == variant_name]
        if not variant_rows:
            print(f"\n{variant_name}: no OK results")
            continue

        f1_vals = [float(r["test_macro_f1"]) for r in variant_rows if r.get("test_macro_f1") is not None]
        auroc_vals = [float(r["test_auroc"]) for r in variant_rows if r.get("test_auroc") is not None]
        acc_vals = [float(r["test_accuracy"]) for r in variant_rows if r.get("test_accuracy") is not None]

        print(f"\n{variant_name} ({len(variant_rows)} studies):")
        if f1_vals:
            print(f"  Macro F1 : {np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}  (range: {min(f1_vals):.4f} ~ {max(f1_vals):.4f})")
        if auroc_vals:
            print(f"  AUROC    : {np.mean(auroc_vals):.4f} ± {np.std(auroc_vals):.4f}  (range: {min(auroc_vals):.4f} ~ {max(auroc_vals):.4f})")
        if acc_vals:
            print(f"  Accuracy : {np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}  (range: {min(acc_vals):.4f} ~ {max(acc_vals):.4f})")

    # R2 影响分析
    print("\n" + "=" * 60)
    print("R2 IMPACT ANALYSIS")
    print("=" * 60)

    def _mean_metric(rows, metric):
        vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
        return np.mean(vals) if vals else None

    for pair_label, v_a, v_b in [("baseline vs r2", "baseline", "r2"), ("r1 vs r1r2", "r1", "r1r2")]:
        rows_a = [r for r in ok_rows if r.get("model_variant") == v_a]
        rows_b = [r for r in ok_rows if r.get("model_variant") == v_b]
        f1_a = _mean_metric(rows_a, "test_macro_f1")
        f1_b = _mean_metric(rows_b, "test_macro_f1")
        if f1_a is not None and f1_b is not None:
            delta = f1_b - f1_a
            direction = "better" if delta > 0 else "worse"
            print(f"  {pair_label}: {v_a} F1={f1_a:.4f}, {v_b} F1={f1_b:.4f} → R2 {direction} by {abs(delta):.4f}")
        else:
            print(f"  {pair_label}: insufficient data")


# %% [markdown]
# ## Save Final Results

# %%
eval_summary = load_summary("evaluate_c")
if eval_summary:
    final_path = RUN_DIR / "final_results.csv"
    runtime.write_rows(eval_summary, final_path)
    print(f"Saved {len(eval_summary)} rows to {final_path}")

    # 也保存一份聚合结果
    ok_rows = [r for r in eval_summary if r.get("status") == "OK"]
    if ok_rows:
        agg_rows = []
        for variant_name in VARIANTS_TO_RUN:
            variant_rows = [r for r in ok_rows if r.get("model_variant") == variant_name]
            if not variant_rows:
                continue
            f1_vals = [float(r["test_macro_f1"]) for r in variant_rows if r.get("test_macro_f1") is not None]
            auroc_vals = [float(r["test_auroc"]) for r in variant_rows if r.get("test_auroc") is not None]
            acc_vals = [float(r["test_accuracy"]) for r in variant_rows if r.get("test_accuracy") is not None]
            agg_rows.append({
                "model_variant": variant_name,
                "n_studies": len(variant_rows),
                "macro_f1_mean": np.mean(f1_vals) if f1_vals else None,
                "macro_f1_std": np.std(f1_vals) if f1_vals else None,
                "auroc_mean": np.mean(auroc_vals) if auroc_vals else None,
                "auroc_std": np.std(auroc_vals) if auroc_vals else None,
                "accuracy_mean": np.mean(acc_vals) if acc_vals else None,
                "accuracy_std": np.std(acc_vals) if acc_vals else None,
            })
        agg_path = RUN_DIR / "final_results_aggregated.csv"
        runtime.write_rows(agg_rows, agg_path)
        print(f"Saved aggregated results to {agg_path}")
        as_table(agg_rows)
else:
    print("No results to save.")
