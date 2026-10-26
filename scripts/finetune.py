#!/usr/bin/env python3
"""Configure and run MiCoFormer unlabeled domain adaptation.
设置参数并运行 MiCoFormer 无标签领域微调。
"""
from __future__ import annotations
import argparse
from pathlib import Path
from micoformer.config import (
    TrainConfig,
    DataConfig,
    RunConfig,
)
from micoformer.workflows.checkpoints import load_model_config


def make_config(
    corpus: Path,
    prepared: Path,
    output: Path,
    checkpoint: Path,
    *,
    resume: Path | None = None,
) -> RunConfig:
    model = load_model_config(resume or checkpoint).for_finetuning()
    training = TrainConfig.for_finetuning()

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
        initialize_from=checkpoint,
    ).normalized()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure and run MiCoFormer unlabeled domain adaptation with prepared data.",
        epilog="Run from the repository root: "
        "torchrun --nproc_per_node=NUM_GPUS -m scripts.finetune. "
        "Set NUM_GPUS to the number of GPUs to use. "
        "Relative paths start from the current working directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Base checkpoint for this adaptation run. Keep the original path when resuming.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/MCFCorpusV4.3.gg2.h5ad"),
        help="Path to the corpus used for sample preparation (default: %(default)s).",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Domain split directory containing train_rows.npy, val_rows.npy, and preparation.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/finetune"),
        help="Directory for run records, logs, and checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint to resume training from. Omit to start a new adaptation run.",
    )
    args = parser.parse_args()
    config = make_config(
        args.corpus,
        args.prepared,
        args.output_root,
        args.checkpoint,
        resume=args.resume,
    )

    from micoformer.workflows.pretrain import run_training
    print(run_training(config, resume=args.resume))


if __name__ == "__main__":
    main()
