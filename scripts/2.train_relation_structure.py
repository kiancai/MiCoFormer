#!/usr/bin/env python3
"""Prepare, smoke, train and export fresh matched C0/C1/C2 relation arms."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from micoformer.relation_structure_pretraining.extract import extract_structure_embeddings
from micoformer.relation_structure_pretraining.model import STRUCTURE_ARMS
from micoformer.relation_structure_pretraining.workflow import (
    StructureRunConfig,
    prepare_structure_initialization,
    run_structure_cuda_smoke,
    run_structure_pretraining,
)


def _data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--schedule-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-init")
    _data_arguments(prepare)
    prepare.add_argument("--disease-rows", type=Path, required=True)
    prepare.add_argument("--num-workers", type=int, default=0)

    smoke = subparsers.add_parser("smoke")
    _data_arguments(smoke)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--device-index", type=int, default=0)

    train = subparsers.add_parser("train")
    _data_arguments(train)
    train.add_argument("--arm", choices=STRUCTURE_ARMS, required=True)
    train.add_argument("--disease-rows", type=Path, required=True)
    train.add_argument("--smoke-dir", type=Path, required=True)
    train.add_argument("--device-index", type=int, default=0)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--resume", type=Path)

    extract = subparsers.add_parser("extract")
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
        print(
            prepare_structure_initialization(
                h5ad_path=args.h5ad,
                schedule_root=args.schedule_root,
                cache_root=args.cache_root,
                output_root=args.output_root,
                disease_rows_path=args.disease_rows,
                num_workers=args.num_workers,
            )
        )
        return
    if args.command == "smoke":
        print(
            run_structure_cuda_smoke(
                h5ad_path=args.h5ad,
                schedule_root=args.schedule_root,
                cache_root=args.cache_root,
                output_root=args.output_root,
                output_dir=args.output_dir,
                device_index=args.device_index,
            )
        )
        return
    if args.command == "train":
        print(
            run_structure_pretraining(
                StructureRunConfig(
                    h5ad_path=args.h5ad,
                    schedule_root=args.schedule_root,
                    cache_root=args.cache_root,
                    output_root=args.output_root,
                    arm=args.arm,
                    disease_rows_path=args.disease_rows,
                    smoke_dir=args.smoke_dir,
                    device_index=args.device_index,
                    num_workers=args.num_workers,
                    resume_checkpoint=args.resume,
                )
            )
        )
        return
    rows = np.asarray(np.load(args.rows, allow_pickle=False))
    print(
        extract_structure_embeddings(
            checkpoint_path=args.checkpoint,
            h5ad_path=args.h5ad,
            row_ids=rows,
            row_source_path=args.rows,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )


if __name__ == "__main__":
    main()

