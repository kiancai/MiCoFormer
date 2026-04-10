from __future__ import annotations

import argparse
from pathlib import Path
import sys


def locate_protocol_root() -> Path:
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    cwd = Path.cwd().resolve()
    candidates.extend([
        cwd,
        cwd / "MiCoFormer" / "protocols" / "hparam_search_v2",
        cwd / "protocols" / "hparam_search_v2",
    ])
    for candidate in candidates:
        if (candidate / "runtime.py").exists():
            return candidate
    raise RuntimeError("Could not locate protocols/hparam_search_v2 directory.")


PROTOCOL_ROOT = locate_protocol_root()
if str(PROTOCOL_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_ROOT))

import runtime


def parse_gpu_ids(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    return [int(item) for item in values]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one V2 stage block from tmux/CLI.")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run_id>")
    parser.add_argument("--stage-block", required=True, choices=sorted(runtime.STAGE_BLOCK_SPECS))
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids, e.g. 0,1,2")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run failed rows instead of only skipping existing OK rows.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    result = runtime.run_stage_block(
        run_dir=args.run_dir,
        stage_block=args.stage_block,
        gpu_ids=gpu_ids,
        num_workers=args.num_workers,
        retry_failed=args.retry_failed,
    )
    rows = result["rows"]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1

    paths = result["paths"]
    print(f"stage_block = {args.stage_block}")
    print(f"run_dir     = {Path(args.run_dir).resolve()}")
    print(f"gpu_ids     = {gpu_ids}")
    print(f"summary     = {paths['summary']}")
    print(f"live_status = {paths['live_status']}")
    print(f"dashboard   = {paths['dashboard']}")
    print(f"status      = {counts}")


if __name__ == "__main__":
    main()
