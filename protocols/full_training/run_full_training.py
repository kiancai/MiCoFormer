"""
MiCoFormer 正式训练协议：Stage 1 预训练 + Stage 2 第一次微调。

对每个 (variant, seed) 依次执行：
  1. 预训练（A train / B val）→ pretrain checkpoint
  2. 第一次微调（A train / B val）→ finetune checkpoint

输出目录结构：
  {output_dir}/
    {variant}/
      seed{seed}/
        pretrain/          # Lightning 日志 + best checkpoint
        finetune/          # Lightning 日志 + best checkpoint

Usage:
  # 所有 6 个变体 × 3 seeds
  python MiCoFormer/protocols/full_training/run_full_training.py \\
      --h5ad       data/gg2/MCFCorpus.gg2.h5ad \\
      --splits-dir data/processed/splits \\
      --output-dir MiCoFormer/protocols/full_training/runs \\
      --variants   baseline r1 r2_taxo r1r2_taxo r2_phylo r1r2_phylo \\
      --seeds      42 52 62 \\
      --gpu-ids    0,1,2,3

  # 单个变体调试
  python MiCoFormer/protocols/full_training/run_full_training.py \\
      --h5ad       data/gg2/MCFCorpus.gg2.h5ad \\
      --splits-dir data/processed/splits \\
      --output-dir MiCoFormer/protocols/full_training/runs \\
      --variants   baseline \\
      --seeds      42 \\
      --gpu-ids    0 \\
      --pretrain-epochs 50 \\
      --finetune-epochs 30
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 确保 MiCoFormer 包在 path 中
_HERE = Path(__file__).resolve()
_MICO_ROOT = _HERE.parents[2]  # MiCoFormer/
sys.path.insert(0, str(_MICO_ROOT))

from micoformer.workflows.pretrain import PretrainRunConfig, run_pretrain_once
from micoformer.workflows.finetune import FinetuneRunConfig, build_label_configs, run_finetune_once

# 从协议的 config.py 读取参数注册表
sys.path.insert(0, str(_HERE.parent))
from config import VARIANTS, SHARED_ARCH, SHARED_PRETRAIN, SHARED_FINETUNE


# ─── 标签配置（使用 benchmark 专用标签列）──────────────────────────────────────
LABEL_FIELD = "Is_Healthy_benchmark"
# False=0 (Disease), True=1 (Healthy)
# MiCoFormer 的 label_values 接受字符串，需要与 obs 中的值匹配
# Is_Healthy_benchmark 为 pandas BooleanDtype，字符串表示为 "True"/"False"
LABEL_VALUES_STR = f"{LABEL_FIELD}=False,True"


def run_one_variant_seed(
    variant_name: str,
    seed: int,
    h5ad_path: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    output_dir: Path,
    pretrain_epochs: int,
    finetune_epochs: int,
    gpu_id: int,
    no_progress_bar: bool,
) -> dict:
    """
    运行单个 (variant, seed) 的完整 Stage 1+2 流程。

    Returns
    -------
    dict with keys: 'pretrain_ckpt', 'finetune_ckpt'
    """
    variant = VARIANTS[variant_name]
    run_dir = output_dir / variant_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Variant={variant_name}  Seed={seed}  GPU={gpu_id}")
    print(f"{'='*60}")

    # ─── Stage 1: 预训练 ────────────────────────────────────────────────────
    pretrain_cfg = PretrainRunConfig(
        h5ad_path=h5ad_path,
        # 模型开关
        token_embedding_mode=variant.token_embedding_mode,
        bias_type=variant.bias_type,
        phylo_mlp_hidden=SHARED_PRETRAIN["phylo_mlp_hidden"],
        # 共享主体
        d_model=SHARED_ARCH["d_model"],
        num_layers=SHARED_ARCH["num_layers"],
        nhead=SHARED_ARCH["nhead"],
        ff_ratio=SHARED_ARCH["ff_ratio"],
        num_abundance_bins=SHARED_ARCH["num_abundance_bins"],
        dropout=SHARED_ARCH["dropout"],
        # 预训练训练参数
        lr=variant.pretrain_lr,
        batch_size=variant.pretrain_batch_size,
        weight_decay=variant.pretrain_weight_decay,
        warmup_ratio=SHARED_PRETRAIN["warmup_ratio"],
        mask_prob=SHARED_PRETRAIN["mask_prob"],
        min_abundance=SHARED_PRETRAIN["min_abundance"],
        lr_scheduler_type=SHARED_PRETRAIN["lr_scheduler_type"],
        abundance_mode=SHARED_PRETRAIN["abundance_mode"],
        max_seq_len=SHARED_PRETRAIN["max_seq_len"],
        # 预算
        budget_mode="epoch",
        max_epochs=pretrain_epochs,
        val_interval_epochs=2,
        early_stopping_patience=10,
        # 运行参数
        devices=1,
        seed=seed,
        log_dir=str(run_dir / "pretrain"),
        no_progress_bar=no_progress_bar,
    )

    # 通过 CUDA_VISIBLE_DEVICES 指定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"\n[Stage 1] Pretraining {variant_name} seed={seed}...")
    pretrain_result = run_pretrain_once(
        pretrain_cfg,
        train_indices,
        val_indices,
        log_subdir=f"pretrain_{variant_name}_seed{seed}",
    )
    pretrain_ckpt = pretrain_result["best_model_path"]
    print(f"[Stage 1] Done. best_ckpt={pretrain_ckpt}")

    # ─── Stage 2: 第一次微调 ────────────────────────────────────────────────
    label_configs = build_label_configs(
        label_fields=[LABEL_FIELD],
        label_values_str=LABEL_VALUES_STR,
    )

    finetune_cfg = FinetuneRunConfig(
        h5ad_path=h5ad_path,
        pretrained_ckpt=pretrain_ckpt,
        # 微调头/pooling 参数
        pooling_mode=SHARED_FINETUNE["pooling_mode"],
        freeze_encoder=SHARED_FINETUNE["freeze_encoder"],
        head_hidden_dim=variant.head_hidden_dim,
        head_dropout=variant.head_dropout,
        # 训练参数
        lr_head=variant.lr_head,
        lr_encoder=variant.lr_encoder,
        weight_decay=SHARED_FINETUNE["weight_decay"],
        batch_size=SHARED_FINETUNE["batch_size"],
        warmup_ratio=SHARED_FINETUNE["warmup_ratio"],
        lr_scheduler_type=SHARED_FINETUNE["lr_scheduler_type"],
        # 预算
        budget_mode="epoch",
        max_epochs=finetune_epochs,
        val_interval_epochs=1,
        early_stopping_patience=7,
        # 运行参数
        devices=1,
        seed=seed,
        log_dir=str(run_dir / "finetune"),
        no_progress_bar=no_progress_bar,
    )

    print(f"\n[Stage 2] Finetuning {variant_name} seed={seed}...")
    finetune_result = run_finetune_once(
        finetune_cfg,
        train_indices,
        val_indices,
        test_indices=None,
        label_configs=label_configs,
        log_subdir=f"finetune_{variant_name}_seed{seed}",
    )
    # run_finetune_once 返回的 dict key 是 'val'/'test'
    finetune_ckpt = finetune_cfg.log_dir  # checkpoint 在 log_dir/best.ckpt 中
    # 实际 best checkpoint 路径从 Lightning ModelCheckpoint 写入磁盘
    # 使用 glob 找到它
    ckpt_files = list(Path(run_dir / "finetune").rglob("best*.ckpt"))
    if ckpt_files:
        finetune_ckpt = str(sorted(ckpt_files)[-1])
    print(f"[Stage 2] Done. val_metrics={finetune_result.get('val', {})}")
    print(f"  best_ckpt (approx)={finetune_ckpt}")

    return {
        "pretrain_ckpt": pretrain_ckpt,
        "finetune_ckpt": finetune_ckpt,
        "val_metrics": finetune_result.get("val", {}),
    }


def main():
    parser = argparse.ArgumentParser(
        description="MiCoFormer 正式训练协议：Stage 1 预训练 + Stage 2 微调"
    )
    parser.add_argument("--h5ad", required=True, help="Benchmark 派生 h5ad 路径")
    parser.add_argument(
        "--splits-dir",
        required=True,
        help="存放 split_group_A.npy / split_group_B.npy 的目录",
    )
    parser.add_argument("--output-dir", required=True, help="输出根目录")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "r1", "r2_taxo", "r1r2_taxo", "r2_phylo", "r1r2_phylo"],
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 52, 62]
    )
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="逗号分隔的 GPU ID 列表，按 (variant, seed) 轮询分配，如 '0,1,2,3'",
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=50, help="预训练最大 epoch 数（含 early stopping）"
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=30, help="第一次微调最大 epoch 数（含 early stopping）"
    )
    parser.add_argument(
        "--no-progress-bar", action="store_true", help="禁用 Lightning 进度条"
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    train_indices = np.load(splits_dir / "split_group_A.npy")
    val_indices = np.load(splits_dir / "split_group_B.npy")
    print(f"Split A (train): {len(train_indices)}, Split B (val): {len(val_indices)}")

    output_dir = Path(args.output_dir)
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")]

    results = []
    job_idx = 0
    for variant_name in args.variants:
        for seed in args.seeds:
            gpu_id = gpu_ids[job_idx % len(gpu_ids)]
            result = run_one_variant_seed(
                variant_name=variant_name,
                seed=seed,
                h5ad_path=args.h5ad,
                train_indices=train_indices,
                val_indices=val_indices,
                output_dir=output_dir,
                pretrain_epochs=args.pretrain_epochs,
                finetune_epochs=args.finetune_epochs,
                gpu_id=gpu_id,
                no_progress_bar=args.no_progress_bar,
            )
            results.append(
                {
                    "variant": variant_name,
                    "seed": seed,
                    "gpu": gpu_id,
                    **result,
                }
            )
            job_idx += 1

    print(f"\n{'='*60}")
    print("All training complete. Summary:")
    for r in results:
        print(
            f"  {r['variant']} seed={r['seed']} | "
            f"pretrain_ckpt={Path(r['pretrain_ckpt']).name} | "
            f"finetune_ckpt={Path(r['finetune_ckpt']).name if r['finetune_ckpt'] else 'N/A'}"
        )


if __name__ == "__main__":
    main()
