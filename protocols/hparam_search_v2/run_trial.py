from __future__ import annotations

import argparse
import json
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single V2 trial in an isolated Python process.")
    parser.add_argument("--task-json", required=True, help="Path to a serialized task json file")
    parser.add_argument("--gpu-id", required=True, type=int, help="Physical GPU id to expose")
    parser.add_argument("--result-json", required=True, help="Path to write the finished row json")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU thread limit for this task")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    runtime.apply_cpu_runtime_settings(args.cpu_threads)
    task = json.loads(Path(args.task_json).read_text())
    row = runtime._execute_task(task, args.gpu_id)
    runtime.write_manifest(args.result_json, row)


if __name__ == "__main__":
    main()
