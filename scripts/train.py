#!/usr/bin/env python3
"""Configure and run MiCoFormer pretraining.
设置参数并运行 MiCoFormer 预训练。
"""
from __future__ import annotations
import argparse
from pathlib import Path
from micoformer.config import (
    ModelConfig,
    TrainConfig,
    DataConfig,
    RunConfig,
)


def make_config(
    corpus: Path,
    prepared: Path,
    output: Path,
) -> RunConfig:
    """Collect data paths, model parameters, and training settings.
    汇总数据路径、模型参数和训练设置。
    """
    model = ModelConfig()
    # Set per-GPU batch size here. / 在这里设置每卡批大小。
    training = TrainConfig(
        batch_size_per_rank=128,
    )

    data = DataConfig(
        corpus=corpus,
        train_rows=prepared / "train_rows.npy",
        val_rows=prepared / "val_rows.npy",
        preparation_manifest=prepared / "preparation.json",
    )

    return RunConfig(
        model=model,
        training=training,
        data=data,
        output_root=output,
    ).normalized()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure and run MiCoFormer pretraining with prepared data.",
        epilog="Run from the repository root: "
        "torchrun --nproc_per_node=NUM_GPUS -m scripts.train. "
        "Set NUM_GPUS to the number of GPUs to use. "
        "Relative paths start from the current working directory.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/MCFCorpusV4.1.gg2.h5ad"),
        help="Path to the corpus used for sample preparation (default: %(default)s).",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/prepared"),
        help="Directory containing train_rows.npy, val_rows.npy, and preparation.json (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/micoformer"),
        help="Directory for run records, logs, and checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint to resume training from. Omit to start a new training run.",
    )
    args = parser.parse_args()
    config = make_config(
        args.corpus,
        args.prepared,
        args.output_root,
    )

    from micoformer.workflows.pretrain import run_full_pretraining
    print(run_full_pretraining(config, resume=args.resume))


if __name__ == "__main__":
    main()
