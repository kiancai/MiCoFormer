# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: notebooks///ipynb,notebooks///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # MiCoFormer 超参数搜索 Pipeline
#
# ## 本 Notebook 的使用方式
#
# 本 Notebook 引导你完成 MiCoFormer 预训练的超参数搜索全流程：
#
# ```
# 分割索引生成 → Dry Run 验证 → 架构搜索 → 优化参数搜索 → 数据参数搜索 → 结果分析 → 正式预训练 → 2×2 消融
# ```
#
# **哪些 cell 直接在 Notebook 里运行？**
#
# - 路径配置、环境检查、分割索引生成、Dry Run 验证 → 这些都是**几秒到几分钟**的短任务，直接运行。
# - 搜索结果分析（读 CSV、画图、提取最优参数）→ 搜索跑完后回来运行。
#
# **哪些 cell 不在 Notebook 里运行？**
#
# - 超参搜索（80 trials × 15k 步，几小时到几十小时）→ cell 会 **打印出一条完整的终端命令**，
#   你复制到服务器的 **tmux** 终端中执行。
# - 正式预训练、2×2 消融 → 同上，打印命令后去终端执行。
#
# **具体操作流程：**
#
# 1. 在服务器上开启 JupyterLab，打开本 Notebook
# 2. 从上到下依次执行每个 cell
# 3. 遇到「**→ 复制下方输出到 tmux 执行**」标记的 cell 时：
#    - 运行 cell → 它会打印一条命令
#    - 切到 tmux 终端，粘贴执行
#    - 等命令跑完后（可用 TensorBoard 监控），回到 Notebook 继续往下执行
# 4. 分析 cell 会自动读取搜索产出的 CSV 文件，展示排名和最优参数
#
# **前置条件：**
#
# - 已完成数据预处理（`protocols/data/prepare_resmicrodb.py`），`data/processed/microbiome_dataset.h5ad` 存在
# - conda 环境已激活（`conda activate MiCoFormerV2`）
# - 项目已安装（`pip install -e .`）

# %% [markdown]
# ---
# ## 0. 路径配置与环境检查
#
# 修改 `PROJECT_DIR` 为你的实际项目根目录。
# 其他所有路径都基于它自动拼接，不需要手动改。

# %%
import os
import sys
from pathlib import Path

# ===================== 唯一需要修改的地方 =====================
PROJECT_DIR = Path(os.getcwd()).resolve()
# 如果你的工作目录不是项目根目录，取消注释并修改下面这行：
# PROJECT_DIR = Path("/home/yourname/MiCoFormer")
# ==============================================================

# 数据文件
H5AD = PROJECT_DIR / "data" / "processed" / "microbiome_dataset.h5ad"

# 分割索引目录
SPLITS_DIR = PROJECT_DIR / "data" / "processed" / "splits"

# 日志输出目录（TensorBoard 事件、CSV 汇总表都在这里）
LOG_DIR = PROJECT_DIR / "outputs" / "protocols" / "pretrain_hparam"

# 确保目录存在
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 环境检查
print("=" * 60)
print(f"PROJECT_DIR : {PROJECT_DIR}")
print(f"H5AD        : {H5AD}")
print(f"  exists    : {H5AD.exists()}")
print(f"SPLITS_DIR  : {SPLITS_DIR}")
print(f"LOG_DIR     : {LOG_DIR}")
print(f"Python      : {sys.executable}")
print("=" * 60)

if not H5AD.exists():
    print("\n[WARNING] .h5ad 文件不存在！请先运行 protocols/data/prepare_resmicrodb.py")

# %%
# 检查关键依赖是否可用
try:
    import torch
    import lightning as L
    import anndata as ad
    import pandas as pd
    print(f"torch     : {torch.__version__}")
    print(f"lightning : {L.__version__}")
    print(f"anndata   : {ad.__version__}")
    print(f"CUDA      : {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print("  (will use CPU — 搜索会很慢)")
except ImportError as e:
    print(f"[ERROR] 缺少依赖: {e}")
    print("请确认已 conda activate MiCoFormerV2 && pip install -e .")

# %% [markdown]
# ---
# ## 1. 生成分割索引
#
# 用 `.h5ad` 中 `obs.Split_Group` 字段将样本分为三组：
#
# | Split_Group 值 | 用途 | 输出文件 |
# |---|---|---|
# | A | 训练集 (train) | `split_group_A.npy` |
# | B | 验证集 (val) | `split_group_B.npy` |
# | C | 测试集 (test) | `split_group_C.npy` |
#
# 每个 `.npy` 文件保存的是样本在 AnnData 中的**整数索引数组**。
#
# 如果文件已存在会自动跳过，不会重复生成。

# %%
import subprocess

splits = [
    ("A", SPLITS_DIR / "split_group_A.npy"),
    ("B", SPLITS_DIR / "split_group_B.npy"),
    ("C", SPLITS_DIR / "split_group_C.npy"),
]

for value, output_path in splits:
    if output_path.exists():
        arr = __import__("numpy").load(output_path)
        print(f"[SKIP] {output_path.name} already exists ({len(arr)} samples)")
        continue

    cmd = [
        sys.executable, str(PROJECT_DIR / "scripts" / "1.make_pretrain_splits.py"),
        "--h5ad", str(H5AD),
        "--field", "Split_Group",
        "--values", value,
        "--output", str(output_path),
    ]
    print(f"Generating {output_path.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
    else:
        print(result.stdout.strip())

# %% [markdown]
# ---
# ## 2. Dry Run — 验证搜索脚本能正常运行
#
# 在正式搜索前，先用**极少配置**跑一遍，确认没有报错：
# - 2 个 trial（2 种随机超参组合）
# - 每个 trial 只训练 100 步
# - 每 50 步验证一次
# - num_workers=0 避免多进程问题
#
# 预计耗时 1-3 分钟。如果这步通过，后面的正式搜索就不会因为脚本 bug 而浪费时间。

# %%
import subprocess

dryrun_log_dir = LOG_DIR / "dryrun"

cmd = [
    sys.executable, str(PROJECT_DIR / "protocols" / "pretrain" / "run_hparam_search.py"),
    "--h5ad", str(H5AD),
    "--train_indices", str(SPLITS_DIR / "split_group_A.npy"),
    "--val_indices",   str(SPLITS_DIR / "split_group_B.npy"),
    "--group", "arch",
    "--num_trials", "2",
    "--max_steps", "100",
    "--val_check_interval", "50",
    "--log_dir", str(dryrun_log_dir),
    "--num_workers", "0",
    "--seed", "42",
]

print("Running dry run ...")
print(f"Command: {' '.join(cmd)}\n")

result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

# 打印输出（截取最后部分，避免太长）
output = result.stdout
if len(output) > 5000:
    print("... (truncated) ...\n")
    print(output[-5000:])
else:
    print(output)

if result.returncode != 0:
    print(f"\n[FAILED] return code={result.returncode}")
    print(f"STDERR:\n{result.stderr[-3000:]}")
else:
    # 检查产出文件
    summary = dryrun_log_dir / "hpsearch_arch_summary.csv"
    if summary.exists():
        print(f"\n[OK] Dry run succeeded!")
        print(f"  Summary CSV: {summary}")
        import pandas as pd
        df = pd.read_csv(summary)
        print(f"  Trials completed: {len(df)}")
        print(f"  Best val_loss: {df['val_loss'].min()}")
    else:
        print(f"\n[WARNING] Dry run finished but summary CSV not found at {summary}")

# %% [markdown]
# ---
# ## 3. 超参数搜索
#
# ### 搜索策略总览
#
# 搜索在 **Baseline 配置**（R1=off, R2=off）上进行，共分三组串行搜索。
# 前一组的最优结果自动传给下一组作为固定参数。
#
# | 阶段 | 搜索组 | 搜索的参数 | 固定参数来源 |
# |---|---|---|---|
# | Step 3.1 | **arch** | d_model, num_layers, nhead, ff_ratio, dropout | 全部用默认值 |
# | Step 3.2 | **optim** | lr, batch_size, weight_decay, warmup_steps | arch 最优 |
# | Step 3.3 | **data** | num_abundance_bins, mask_prob, min_abundance | arch+optim 最优 |
#
# 每组搜索 = **80 个随机超参组合 × 15,000 步训练**，耗时取决于 GPU 性能。
#
# ### 搜索空间一览
#
# **架构参数 (arch):**
# - `d_model`: 64, 128, 256, 512, 768
# - `num_layers`: 1, 2, 4, 6, 8, 12
# - `nhead`: 2, 4, 8, 16（必须整除 d_model，不满足会自动重采样）
# - `ff_ratio`: 2, 4, 8（实际 FFN 维度 = d_model × ff_ratio）
# - `dropout`: 0.0, 0.05, 0.1, 0.2, 0.3, 0.5
#
# **优化参数 (optim):**
# - `lr`: 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3
# - `batch_size`: 16, 32, 64, 128, 256
# - `weight_decay`: 0, 1e-3, 1e-2, 5e-2, 1e-1
# - `warmup_steps`: 0, 500, 1000, 2000, 4000
#
# **数据/任务参数 (data):**
# - `num_abundance_bins`: 10, 20, 40, 80, 160
# - `mask_prob`: 0.05, 0.1, 0.15, 0.2, 0.3, 0.4
# - `min_abundance`: 1e-8, 1e-6, 1e-5, 1e-4
#
# ### 产出文件
#
# 每组搜索完成后会在 `LOG_DIR` 下产生：
#
# | 文件 | 说明 |
# |---|---|
# | `hpsearch_{group}_summary.csv` | 所有 trial 按 val_loss 排序的汇总表（第 1 行 = 最优） |
# | `hpsearch_{group}/{run_name}/` | 每个 trial 的 TensorBoard 事件文件和 CSV 日志 |
#
# ### 实时监控
#
# 搜索过程中，在另一个终端运行：
# ```bash
# tensorboard --logdir tmp/logs/hpsearch_arch --port 6006
# ```
# 在浏览器中打开 `http://服务器IP:6006` 即可实时查看所有 trial 的 loss 曲线。
# run 名称编码了关键超参，例如 `d256_L6_h8_ff4x_dp0.1`，一目了然。

# %% [markdown]
# ### 辅助函数
#
# 下面这个 cell 定义了「构建搜索命令」和「展示搜索结果」的辅助函数，
# 后续所有搜索步骤都会用到。直接运行即可。

# %%
def build_search_cmd(
    group: str,
    num_trials: int = 80,
    max_steps: int = 15000,
    val_check_interval: int = 500,
    seed: int = 42,
    num_workers: int = 4,
    base_config: Path = None,
    resume_from_trial: int = 0,
    num_gpus: int = 1,
    retry_errors: bool = False,
) -> str:
    """
    构建 protocols/pretrain/run_hparam_search.py 的完整终端命令字符串。

    Parameters
    ----------
    group : str
        搜索组，"arch" / "optim" / "data"
    num_trials : int
        随机采样的试验数（默认 80）。data 组忽略此参数（自动逐个扫描）
    max_steps : int
        每个试验的训练步数（默认 15000）
    val_check_interval : int
        每多少步做一次验证（默认 500，会自动调低到不超过 batch 数）
    seed : int
        随机种子（默认 42）
    num_workers : int
        DataLoader 进程数（默认 4）
    base_config : Path, optional
        上一组搜索的 summary CSV 路径，用于继承最优参数
    resume_from_trial : int
        中断恢复：跳过前 N 个 trial（默认 0，不跳过）
    num_gpus : int
        并行使用的 GPU 数量（默认 1 = 串行）
    retry_errors : bool
        是否只重跑 ERROR 的 trial（默认 False）
    """
    parts = [
        f"python {PROJECT_DIR / 'protocols' / 'pretrain' / 'run_hparam_search.py'}",
        f"    --h5ad {H5AD}",
        f"    --train_indices {SPLITS_DIR / 'split_group_A.npy'}",
        f"    --val_indices   {SPLITS_DIR / 'split_group_B.npy'}",
        f"    --group {group}",
        f"    --num_trials {num_trials}",
        f"    --max_steps {max_steps}",
        f"    --val_check_interval {val_check_interval}",
        f"    --log_dir {LOG_DIR}",
        f"    --seed {seed}",
        f"    --num_workers {num_workers}",
    ]
    if num_gpus > 1:
        parts.append(f"    --num_gpus {num_gpus}")
    if base_config is not None:
        parts.append(f"    --base_config {base_config}")
    if resume_from_trial > 0:
        parts.append(f"    --resume_from_trial {resume_from_trial}")
    if retry_errors:
        parts.append(f"    --retry_errors")
    return " \\\n".join(parts)


def show_search_results(group: str, top_n: int = 15):
    """
    读取某组搜索的 summary CSV，展示排名表和最优配置。

    Parameters
    ----------
    group : str
        搜索组名 ("arch" / "optim" / "data")
    top_n : int
        显示前多少名（默认 15）
    """
    import pandas as pd

    summary_path = LOG_DIR / f"hpsearch_{group}_summary.csv"
    if not summary_path.exists():
        print(f"[NOT FOUND] {summary_path}")
        print(f"→ {group} 组搜索还未完成，请先在终端中运行搜索命令。")
        return None

    df = pd.read_csv(summary_path)

    # 统计信息
    total = len(df)
    ok = (df["status"] == "OK").sum()
    oom = (df["status"] == "OOM").sum()
    no_val = (df["status"] == "NO_VAL").sum()
    errors = total - ok - oom - no_val

    print(f"{'=' * 70}")
    print(f"  {group.upper()} 搜索结果  |  总计 {total} trials  |  OK={ok}  OOM={oom}  NO_VAL={no_val}  ERROR={errors}")
    print(f"{'=' * 70}")

    # 按搜索组选择显示的列
    if group == "arch":
        param_cols = ["d_model", "num_layers", "nhead", "ff_ratio", "dropout"]
    elif group == "optim":
        param_cols = ["lr", "batch_size", "weight_decay", "warmup_steps"]
    elif group == "data":
        param_cols = ["num_abundance_bins", "mask_prob", "min_abundance"]
    else:
        param_cols = []

    display_cols = ["rank"] + param_cols + ["val_loss", "status", "elapsed"]
    existing_cols = [c for c in display_cols if c in df.columns]
    display(df[existing_cols].head(top_n))

    # 打印最优配置
    best = df.iloc[0]
    if best["val_loss"] != float("inf") and str(best["val_loss"]) != "inf":
        print(f"\n最优配置 (val_loss = {best['val_loss']:.4f}):")
        for col in param_cols:
            print(f"  {col}: {best[col]}")

    return df


def build_pretrain_cmd(
    config: dict,
    token_embedding_mode: str = "taxon",
    use_taxonomy_bias: bool = False,
    max_steps: int = 100000,
) -> str:
    """
    构建 scripts/2.train_pretrain.py 的完整终端命令字符串。

    Parameters
    ----------
    config : dict
        超参数字典，必须包含 d_model, nhead, num_layers, ff, dropout,
        lr, batch_size, weight_decay, warmup_steps,
        num_abundance_bins, mask_prob, min_abundance
    token_embedding_mode : str
        "taxon" (R1=off) 或 "taxon_path" (R1=on)
    use_taxonomy_bias : bool
        True = R2=on
    max_steps : int
        训练步数
    """
    parts = [
        f"python {PROJECT_DIR / 'scripts' / '2.train_pretrain.py'}",
        f"    --h5ad {H5AD}",
        f"    --train_indices {SPLITS_DIR / 'split_group_A.npy'}",
        f"    --val_indices   {SPLITS_DIR / 'split_group_B.npy'}",
        f"    --d_model {config['d_model']}",
        f"    --nhead {config['nhead']}",
        f"    --num_layers {config['num_layers']}",
        f"    --ff {config['ff']}",
        f"    --dropout {config['dropout']}",
        f"    --lr {config['lr']}",
        f"    --batch_size {config['batch_size']}",
        f"    --weight_decay {config['weight_decay']}",
        f"    --warmup_steps {config['warmup_steps']}",
        f"    --num_abundance_bins {config['num_abundance_bins']}",
        f"    --mask_prob {config['mask_prob']}",
        f"    --min_abundance {config['min_abundance']}",
        f"    --max_steps {max_steps}",
        f"    --token_embedding_mode {token_embedding_mode}",
        f"    --no_progress_bar",
        f"    --log_dir {LOG_DIR}",
    ]
    if use_taxonomy_bias:
        parts.append(f"    --use_taxonomy_bias")
    return " \\\n".join(parts)

# %% [markdown]
# ---
# ### Step 3.1 架构参数搜索 (arch)
#
# 搜索 `d_model / num_layers / nhead / ff_ratio / dropout`。
# 其余参数保持默认值（lr=3e-4, batch_size=32, ...）。
#
# **→ 运行下方 cell，复制输出的命令到 tmux 执行。**

# %%
print("# ====================================================================")
print("# Step 3.1: 架构参数搜索")
print("# 搜索: d_model, num_layers, nhead, ff_ratio, dropout")
print("# 固定: lr=3e-4, batch_size=32, weight_decay=1e-2, warmup_steps=2000,")
print("#       num_abundance_bins=40, mask_prob=0.15, min_abundance=4e-6")
print("# 共 80 trials × 15k steps")
print("# ====================================================================")
print()
print(build_search_cmd("arch"))
print()
print("# TensorBoard 监控（另一个终端）：")
print(f"# tensorboard --logdir {LOG_DIR / 'hpsearch_arch'} --port 6006")

# %% [markdown]
# **架构搜索完成后**，运行下方 cell 查看结果。
#
# 如果搜索还在跑，运行这个 cell 会提示 "NOT FOUND"，等搜索完了再来执行就行。

# %%
df_arch = show_search_results("arch")

# %% [markdown]
# ---
# ### Step 3.2 优化参数搜索 (optim)
#
# 搜索 `lr / batch_size / weight_decay / warmup_steps`。
# 架构参数固定为 Step 3.1 的最优值（通过 `--base_config` 自动从 summary CSV 第 1 行读取）。
#
# **前提**：Step 3.1 已完成，`hpsearch_arch_summary.csv` 存在。
#
# **→ 运行下方 cell，复制输出的命令到 tmux 执行。**

# %%
arch_summary = LOG_DIR / "hpsearch_arch_summary.csv"

if not arch_summary.exists():
    print("[ERROR] 架构搜索结果不存在，请先完成 Step 3.1")
else:
    print("# ====================================================================")
    print("# Step 3.2: 优化参数搜索")
    print("# 搜索: lr, batch_size, weight_decay, warmup_steps")
    print(f"# 架构参数固定来源: {arch_summary.name}")
    print("# 共 80 trials × 15k steps")
    print("# ====================================================================")
    print()
    print(build_search_cmd("optim", base_config=arch_summary))
    print()
    print("# TensorBoard 监控（另一个终端）：")
    print(f"# tensorboard --logdir {LOG_DIR / 'hpsearch_optim'} --port 6007")

# %% [markdown]
# **优化参数搜索完成后**，运行下方 cell 查看结果。

# %%
df_optim = show_search_results("optim")

# %% [markdown]
# ---
# ### Step 3.3 数据/任务参数搜索 (data)
#
# 搜索 `num_abundance_bins / mask_prob / min_abundance`。
# 架构 + 优化参数固定为前两步的最优值。
#
# **搜索策略：逐个参数扫描（one-at-a-time）**
#
# 这三个参数相互独立（mask_prob 控制训练信号密度、bins 控制编码分辨率、min_abundance 控制过滤阈值），
# 不需要组合搜索。脚本会自动逐个扫描：固定其他两个，遍历一个的所有候选值。
# 共 5 + 6 + 4 = 15 trials（去重后约 13 个），远比 60 组合高效。
#
# **前提**：Step 3.2 已完成，`hpsearch_optim_summary.csv` 存在。
#
# **→ 运行下方 cell，复制输出的命令到 tmux 执行。**

# %%
optim_summary = LOG_DIR / "hpsearch_optim_summary.csv"

if not optim_summary.exists():
    print("[ERROR] 优化参数搜索结果不存在，请先完成 Step 3.2")
else:
    print("# ====================================================================")
    print("# Step 3.3: 数据/任务参数搜索（逐个参数扫描）")
    print("# 搜索: num_abundance_bins, mask_prob, min_abundance")
    print(f"# 架构+优化参数固定来源: {optim_summary.name}")
    print("# 约 13~15 trials × 15k steps（one-at-a-time，非组合搜索）")
    print("# ====================================================================")
    print()
    print(build_search_cmd("data", base_config=optim_summary))
    print()
    print("# TensorBoard 监控（另一个终端）：")
    print(f"# tensorboard --logdir {LOG_DIR / 'hpsearch_data'} --port 6008")

# %% [markdown]
# **数据参数搜索完成后**，运行下方 cell 查看结果。

# %%
df_data = show_search_results("data")

# %% [markdown]
# ---
# ### Step 3.4 Round 2 精搜（可选）
#
# 如果 Round 1 的结果显示某些参数区间特别有希望，
# 可以手动修改 `protocols/pretrain/run_hparam_search.py` 中的 `SEARCH_SPACES` 缩小范围，
# 然后用更长步数（50k）重新搜索。
#
# 注意换一个 `seed`，避免和 Round 1 采样到完全相同的配置。

# %%
print("# ====================================================================")
print("# Round 2 精搜示例（需先手动缩小 SEARCH_SPACES）")
print("# ====================================================================")
print()
print(build_search_cmd("arch", num_trials=40, max_steps=50000, val_check_interval=1000, seed=123))

# %% [markdown]
# ---
# ### 中断恢复说明
#
# 如果搜索中途被杀（OOM / 断电 / 手动 kill），已完成的 trial 结果不会丢失
# （脚本每完成一个 trial 就保存一次 summary CSV）。
#
# 恢复方法：
# 1. 查看已完成多少个 trial
# 2. 用 `--resume_from_trial N` 跳过前 N 个，继续剩余的

# %%
# 查看各组搜索的完成进度和失败 trial 数量
import pandas as pd

print("各组搜索进度：")
print("-" * 60)
for group in ["arch", "optim", "data"]:
    p = LOG_DIR / f"hpsearch_{group}_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        n_ok = (df["status"] == "OK").sum()
        n_err = len(df) - n_ok
        print(f"  {group:6s}: {len(df)} total, {n_ok} OK, {n_err} errors")
    else:
        print(f"  {group:6s}: not started")

print()
print("如需恢复中断的搜索，示例命令：")
print(build_search_cmd("arch", resume_from_trial=35))

# %% [markdown]
# ---
# ### 重跑失败的 trial + 多 GPU 并行
#
# 搜索脚本支持两个关键功能：
#
# **`--retry_errors`**：只重跑 summary CSV 中失败的 trial，保留已有 OK 结果。
# 用于修复 bug 后补跑失败的实验（如之前 `val_check_interval` 导致的 ERROR）。
#
# **`--num_gpus N`**：将待执行的 trial 自动分配到 N 张 GPU 并行执行。
# 脚本内部用 `multiprocessing` spawn 子进程，每个子进程绑定一张 GPU。
# **一条命令搞定，不需要手动开多个终端。**
#
# 两者可以组合使用：`--retry_errors --num_gpus 2` = 用 2 张卡并行重跑所有失败 trial。
#
# **→ 运行下方 cell 查看当前需要重跑的情况，按需修改 group 和 num_gpus。**

# %%
# 生成重跑命令（按需修改 group 和 num_gpus）
arch_summary = LOG_DIR / "hpsearch_arch_summary.csv"

# 检查 optim 组是否有失败 trial
optim_csv = LOG_DIR / "hpsearch_optim_summary.csv"
if optim_csv.exists():
    df_optim = pd.read_csv(optim_csv)
    n_err = len(df_optim) - (df_optim["status"] == "OK").sum()
    if n_err > 0:
        print(f"optim 组有 {n_err} 个失败 trial，生成重跑命令：")
        print()
        print(build_search_cmd(
            "optim",
            base_config=arch_summary,
            retry_errors=True,
            num_gpus=2,  # ← 按你的 GPU 数量修改
        ))
    else:
        print("optim 组全部成功，无需重跑。")
else:
    print("optim 组尚未开始。")

# %% [markdown]
# ---
# ## 4. 汇总最优超参数
#
# 三组搜索全部完成后，运行此 cell 自动从最完整的 summary 中提取全部最优参数。
#
# 提取逻辑：
# - 优先从 `hpsearch_data_summary.csv` 读取（包含 arch + optim + data 的最优值）
# - 如果 data 搜索还没做，退而从 `hpsearch_optim_summary.csv` 读取
# - 再退而从 `hpsearch_arch_summary.csv` 读取
#
# 提取出来的 `BEST` 字典会在后续 Step 5、6 中自动使用。

# %%
import pandas as pd

# 按完整度从高到低查找
BEST = None
for name in ["hpsearch_data_summary.csv", "hpsearch_optim_summary.csv", "hpsearch_arch_summary.csv"]:
    p = LOG_DIR / name
    if p.exists():
        df = pd.read_csv(p)
        best_row = df.iloc[0]

        if str(best_row["val_loss"]) == "inf":
            print(f"[WARNING] {name} 的最优 val_loss 是 inf，所有 trial 可能都失败了。")
            continue

        # ff_ratio → ff（训练脚本使用绝对值）
        BEST = {
            "d_model":            int(best_row["d_model"]),
            "num_layers":         int(best_row["num_layers"]),
            "nhead":              int(best_row["nhead"]),
            "ff":                 int(best_row["d_model"] * best_row["ff_ratio"]),
            "dropout":            float(best_row["dropout"]),
            "lr":                 float(best_row["lr"]),
            "batch_size":         int(best_row["batch_size"]),
            "weight_decay":       float(best_row["weight_decay"]),
            "warmup_steps":       int(best_row["warmup_steps"]),
            "num_abundance_bins": int(best_row["num_abundance_bins"]),
            "mask_prob":          float(best_row["mask_prob"]),
            "min_abundance":      float(best_row["min_abundance"]),
        }

        print(f"来源: {name}")
        print(f"val_loss: {best_row['val_loss']:.4f}")
        print(f"\n最优超参数（BEST 字典）：")
        print("-" * 40)
        for k, v in BEST.items():
            print(f"  {k:24s}: {v}")
        print("-" * 40)
        print(f"\n注意: ff = d_model({int(best_row['d_model'])}) × ff_ratio({int(best_row['ff_ratio'])}) = {BEST['ff']}")
        break
else:
    print("[ERROR] 没有找到任何搜索结果文件。")
    print("请先完成至少一组超参搜索（Step 3.1）。")

# %% [markdown]
# ---
# ## 5. 正式预训练（Baseline）
#
# 使用 Step 4 提取的最优超参数，在 Baseline 配置（R1=off, R2=off）上进行完整训练。
#
# 训练步数默认 100k 步，可以根据需要调整。
#
# **→ 运行下方 cell，复制输出的命令到 tmux 执行。**

# %%
if BEST is None:
    print("[ERROR] BEST 字典为空，请先运行 Step 4 提取最优参数。")
else:
    print("# ====================================================================")
    print("# Step 5: 正式预训练 — Baseline (R1=off, R2=off)")
    print(f"# 超参来源: Step 4 提取的最优配置")
    print(f"# 训练步数: 100,000")
    print("# ====================================================================")
    print()
    print(build_pretrain_cmd(BEST, token_embedding_mode="taxon", use_taxonomy_bias=False))
    print()
    print("# TensorBoard 监控（另一个终端）：")
    print(f"# tensorboard --logdir {LOG_DIR / 'pretrain_stage0'} --port 6006")

# %% [markdown]
# ---
# ## 6. 2×2 消融实验
#
# 在确定最优超参后，分别在 4 种配置上训练，用于比较 R1 和 R2 的独立/联合效果：
#
# |  | R2=off | R2=on |
# |---|---|---|
# | **R1=off** (`--token_embedding_mode taxon`) | Baseline | +R2 |
# | **R1=on** (`--token_embedding_mode taxon_path`) | +R1 | +R1+R2 |
#
# - **R1 (Taxonomy-path embedding)**：5 个 rank 的 Embedding 逐元素相加，解决长尾 taxon 问题
# - **R2 (Taxonomy bias)**：Graphormer-style 注意力偏置，将系统发育距离注入注意力矩阵
#
# 4 条命令相互独立，可以同时在 4 个 tmux window 中并行运行（如果 GPU 显存足够），
# 也可以串行依次运行。
#
# **→ 运行下方 cell，复制输出的 4 条命令分别到 tmux 执行。**

# %%
if BEST is None:
    print("[ERROR] BEST 字典为空，请先运行 Step 4。")
else:
    ablations = [
        ("Baseline (R1=off, R2=off)", "taxon",      False),
        ("+R1 (R1=on, R2=off)",       "taxon_path", False),
        ("+R2 (R1=off, R2=on)",       "taxon",      True),
        ("+R1+R2 (both on)",          "taxon_path", True),
    ]

    for name, embed_mode, use_bias in ablations:
        print(f"# === {name} ===")
        print(build_pretrain_cmd(BEST, token_embedding_mode=embed_mode, use_taxonomy_bias=use_bias))
        print()

    print("# TensorBoard 监控（所有消融实验共享同一目录）：")
    print(f"# tensorboard --logdir {LOG_DIR / 'pretrain_stage0'} --port 6006")

# %% [markdown]
# ---
# ## 附录：常用监控命令速查
#
# 以下命令在服务器终端中使用。
#
# ### TensorBoard
# ```bash
# # 超参搜索（每组一个端口）
# tensorboard --logdir tmp/logs/hpsearch_arch  --port 6006
# tensorboard --logdir tmp/logs/hpsearch_optim --port 6007
# tensorboard --logdir tmp/logs/hpsearch_data  --port 6008
#
# # 正式训练 / 消融实验
# tensorboard --logdir tmp/logs/pretrain_stage0 --port 6006
# ```
#
# ### 进程与 GPU
# ```bash
# # 实时查看 GPU 使用率（每 2 秒刷新）
# watch -n 2 nvidia-smi
#
# # 查看后台 Python 进程
# ps aux | grep hyperparam_search
# ```
#
# ### 搜索进度与结果
# ```bash
# # 查看某组搜索跑到第几个 trial（每完成一个 trial 会打印一行 "=>"）
# grep "=>" tmp/logs/hpsearch_arch.log | wc -l
#
# # 查看 summary CSV 前几行（column 命令对齐列）
# head -11 tmp/logs/hpsearch_arch_summary.csv | column -t -s,
#
# # 查看实时日志输出
# tail -f tmp/logs/hpsearch_arch.log
# ```
