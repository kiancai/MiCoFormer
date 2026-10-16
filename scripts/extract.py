#!/usr/bin/env python3
"""Extract unmasked MiCoFormer sample representations from the unified corpus.

Use row indices from prepare_data.py; output follows their order.
The corpus supplies sample abundances; the checkpoint contains model priors.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from micoformer.workflows.extract import extract_representations

# Configure inference paths, device, and batch settings.
# 设置推理路径、设备与批次参数。
def main() -> None:
    """Parse extraction options and invoke the workflow.
    读取提取参数，并调用实际提取流程。
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run from the repository root: python -m scripts.extract. "
        "Relative paths start from the current working directory.",
    )
    # Require paths to the weights, corpus, row indices, and a new output directory.
    # 必须提供权重、语料、行号文件和新输出目录。
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a training checkpoint or model snapshot.",
    )
    parser.add_argument(
        "--corpus", type=Path, required=True,
        help="Path to the H5AD corpus supplying sample abundances.",
    )
    parser.add_argument(
        "--rows", type=Path, required=True,
        help="Path to a prepared row-index NPY file; output follows this order.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="New directory for embeddings.npy, rows.npy, and manifest.json.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device, such as cpu or cuda:0 (default: %(default)s).",
    )
    # Extraction batch size and workers are independent of training settings.
    # 提取批大小和 worker 数独立于训练设置。
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Samples per extraction batch (default: %(default)s).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="CPU data-loading subprocesses; 0 loads in the main process "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bf16"), default="auto",
        help="Inference precision; auto uses float32 on CPU and bf16 on CUDA "
        "(default: %(default)s).",
    )
    args = parser.parse_args()
    # Run extraction and print the completed result directory.
    # 执行提取，并输出完成后的结果目录。
    print(
        extract_representations(
            checkpoint=args.checkpoint,
            corpus=args.corpus,
            rows_path=args.rows,
            output=args.output,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            precision=args.precision,
        )
    )

if __name__ == "__main__":
    main()
