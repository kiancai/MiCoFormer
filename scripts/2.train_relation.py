#!/usr/bin/env python3
"""Prepare, train, or export the frozen V3-RM relation-only pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from micoformer.relation_pretraining.extract import extract_relation_embeddings
from micoformer.relation_pretraining.module import ARM_SPECS
from micoformer.relation_pretraining.smoke import (
    RelationSmokeConfig,
    run_relation_cuda_smoke,
)
from micoformer.relation_pretraining.workflow import (
    RelationRunConfig,
    prepare_relation_initialization,
    run_relation_pretraining,
)


def _shared_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--disease-rows", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-init",
        help="single-process epoch0 main/PMA checkpoints plus val/test/disease exports",
    )
    _shared_data_arguments(prepare)
    prepare.add_argument("--device", default="cuda:0")

    train = subparsers.add_parser("train", help="run one exact single-GPU arm")
    _shared_data_arguments(train)
    train.add_argument("--arm", choices=sorted(ARM_SPECS), required=True)
    train.add_argument(
        "--smoke-dir",
        type=Path,
        required=True,
        help="fresh B32 CUDA smoke directory containing a hash-valid passed .complete",
    )
    train.add_argument("--device-index", type=int, default=0)
    train.add_argument("--resume", type=Path)

    smoke = subparsers.add_parser(
        "smoke",
        help="run the fail-closed B32 real-CUDA launch gate without production outputs",
    )
    smoke.add_argument("--h5ad", type=Path, required=True)
    smoke.add_argument("--schedule-root", type=Path, required=True)
    smoke.add_argument("--cache-root", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--device-index", type=int, default=0)

    extract = subparsers.add_parser("extract", help="strict row-addressed checkpoint export")
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--h5ad", type=Path, required=True)
    extract.add_argument("--rows", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-init":
        path = prepare_relation_initialization(
            h5ad_path=args.h5ad,
            schedule_root=args.schedule_root,
            cache_root=args.cache_root,
            output_root=args.output_root,
            disease_rows_path=args.disease_rows,
            extraction_device=args.device,
            num_workers=args.num_workers,
        )
        print(path)
        return
    if args.command == "train":
        path = run_relation_pretraining(
            RelationRunConfig(
                h5ad_path=args.h5ad,
                schedule_root=args.schedule_root,
                cache_root=args.cache_root,
                output_root=args.output_root,
                arm_name=args.arm,
                disease_rows_path=args.disease_rows,
                smoke_dir=args.smoke_dir,
                resume_checkpoint=args.resume,
                num_workers=args.num_workers,
                device_index=args.device_index,
            )
        )
        print(path)
        return
    if args.command == "smoke":
        path = run_relation_cuda_smoke(
            RelationSmokeConfig(
                h5ad_path=args.h5ad,
                schedule_root=args.schedule_root,
                cache_root=args.cache_root,
                output_dir=args.output_dir,
                device_index=args.device_index,
            )
        )
        print(path)
        return

    rows = np.asarray(np.load(args.rows, allow_pickle=False))
    path = extract_relation_embeddings(
        checkpoint_path=args.checkpoint,
        h5ad_path=args.h5ad,
        row_ids=rows,
        row_source_path=args.rows,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(path)


if __name__ == "__main__":
    main()
