#!/usr/bin/env python3
"""Prepare, calibrate, smoke and train the frozen full-data MPC model."""
from __future__ import annotations

import argparse
from pathlib import Path

from micoformer.mpc_pretraining.workflow import (
    MPCRunConfig,
    calibrate_loss_weights,
    prepare_run,
    run_ddp_smoke,
    run_full_pretraining,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "calibrate", "smoke", "train"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--val-rows", type=Path, required=True)
    parser.add_argument("--prior-assets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MPCRunConfig(
        corpus=args.corpus,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        prior_assets=args.prior_assets,
        output_root=args.output_root,
    )
    if args.command != "train" and args.resume is not None:
        raise ValueError("--resume is valid only for the train command")
    if args.command == "prepare":
        print(prepare_run(config))
    elif args.command == "calibrate":
        print(calibrate_loss_weights(config))
    elif args.command == "smoke":
        print(run_ddp_smoke(config))
    else:
        print(run_full_pretraining(config, resume=args.resume))


if __name__ == "__main__":
    main()
