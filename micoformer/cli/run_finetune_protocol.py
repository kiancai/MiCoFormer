from __future__ import annotations

import argparse
import os

from lightning.pytorch.utilities import rank_zero_info

from micoformer.cli.train_finetune import (
    TAG,
    add_finetune_common_args,
    namespace_to_config,
    print_results,
)
from micoformer.runners.finetune import (
    collect_kfold_indices,
    load_indices,
    run_finetune_once,
    summarize_kfold_results,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer finetune protocol runner")
    p.add_argument("--protocol", type=str, required=True, choices=["kfold", "holdout", "ood"])
    p.add_argument("--kfold_dir", type=str, default=None)
    p.add_argument("--train_indices_path", type=str, default=None)
    p.add_argument("--val_indices_path", type=str, default=None)
    p.add_argument("--test_indices_path", type=str, default=None)
    return add_finetune_common_args(p, default_log_dir="outputs/protocols/finetune")


def validate_protocol_args(args: argparse.Namespace) -> None:
    if args.protocol == "kfold":
        if args.kfold_dir is None:
            raise ValueError("--kfold_dir is required when --protocol=kfold.")
        if any(path is not None for path in (args.train_indices_path, args.val_indices_path, args.test_indices_path)):
            raise ValueError("--kfold_dir is incompatible with explicit train/val/test index paths.")
        return

    if args.train_indices_path is None or args.val_indices_path is None:
        raise ValueError("--train_indices_path and --val_indices_path are required for holdout / ood protocols.")
    if args.protocol == "ood" and args.test_indices_path is None:
        raise ValueError("--test_indices_path is required when --protocol=ood.")


def run_single_protocol(args: argparse.Namespace) -> None:
    config = namespace_to_config(args)
    train_indices = load_indices(args.train_indices_path, "train")
    val_indices = load_indices(args.val_indices_path, "val")
    test_indices = None
    if args.test_indices_path is not None:
        test_indices = load_indices(args.test_indices_path, "test")
    results = run_finetune_once(
        config,
        train_indices,
        val_indices,
        test_indices=test_indices,
        log_subdir=args.protocol,
    )
    print_results(results)


def run_kfold_protocol(args: argparse.Namespace) -> None:
    config = namespace_to_config(args)
    fold_indices = collect_kfold_indices(args.kfold_dir)
    rank_zero_info(f"{TAG} Detected {len(fold_indices)} folds: {fold_indices}")

    all_results = []
    for fold_i in fold_indices:
        train_path = os.path.join(args.kfold_dir, f"fold_{fold_i}_train.npy")
        val_path = os.path.join(args.kfold_dir, f"fold_{fold_i}_val.npy")
        rank_zero_info(f"{TAG} {'=' * 50}")
        rank_zero_info(f"{TAG} Fold {fold_i}")
        rank_zero_info(f"{TAG} {'=' * 50}")

        train_indices = load_indices(train_path, f"fold_{fold_i} train")
        val_indices = load_indices(val_path, f"fold_{fold_i} val")
        results = run_finetune_once(
            config,
            train_indices,
            val_indices,
            log_subdir=f"fold_{fold_i}",
        )
        all_results.append(results)

    summary = summarize_kfold_results(all_results)
    rank_zero_info(f"{TAG} {'=' * 50}")
    rank_zero_info(f"{TAG} K-fold Summary")
    rank_zero_info(f"{TAG} {'=' * 50}")
    for key, stats in summary.items():
        rank_zero_info(f"{TAG}   {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


def main() -> None:
    args = build_argparser().parse_args()
    validate_protocol_args(args)
    if args.protocol == "kfold":
        run_kfold_protocol(args)
        return
    run_single_protocol(args)


if __name__ == "__main__":
    main()
