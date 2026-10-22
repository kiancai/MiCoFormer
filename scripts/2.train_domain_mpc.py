#!/usr/bin/env python3
"""Prepare, submit and run the four single-GPU domain corpus ablations."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def submit(manifest_path: Path, expected_sha: str, arms=None):
    from micoformer.domain_pretraining.prepare import validate_manifest, REPO, SUBMISSION_ORDER
    from micoformer.mpc_unifrac import workflow as common
    manifest_path = manifest_path.resolve()
    manifest = validate_manifest(manifest_path, expected_sha)
    output = manifest_path.parents[1]
    receipt_path = output / "audit" / f"submitted_jobs_{manifest_path.stem}.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(
        manifest=str(manifest_path), manifest_sha256=expected_sha,
        commit=manifest["source"]["commit"], machine="CmlLab", jobs={})
    if receipt["manifest_sha256"] != expected_sha:
        raise RuntimeError("submission receipt belongs to a different manifest")
    logs = output / "logs/slurm"
    logs.mkdir(parents=True, exist_ok=True)
    if set(manifest["arms"]) != set(SUBMISSION_ORDER):
        raise RuntimeError("manifest must contain the four approved domain arms")
    for name in SUBMISSION_ORDER:
        arm = manifest["arms"][name]
        if arms and name not in arms:
            continue
        if name in receipt["jobs"]:
            print(f"already submitted {name}: {receipt['jobs'][name]['job_id']}", flush=True)
            continue
        label = "mcfd-" + name.replace("respiratory", "resp").replace("_", "-")
        command = ["sbatch", "--parsable", "--job-name=" + label,
            "--output=" + str(logs / "%x-%j.out"), "--error=" + str(logs / "%x-%j.err"),
            str(REPO / "scripts/train_domain_mpc.sbatch"), str(manifest_path), expected_sha,
            name, manifest["source"]["commit"], str(REPO)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"sbatch failed for {name}: {result.stderr.strip()}")
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"unrecognized sbatch response; inspect before retrying: {result.stdout!r}")
        receipt["jobs"][name] = dict(job_id=job, submitted_at=datetime.now().astimezone().isoformat(),
            command=command, resources=dict(partition="gpu", gres="gpu:A100:1", cpus=4,
                memory="24G", time_limit="7-00:00:00"), output=arm["output"])
        # Persist each accepted job immediately; a partial retry does not resubmit.
        common._atomic_json(receipt_path, receipt)
        print(f"submitted {name}: {job}", flush=True)
    return receipt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output-root", type=Path, required=True)
    for name in ("submit", "job", "rebind"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--manifest-sha256", required=True)
        if name == "submit":
            p.add_argument("--arms", nargs="+", choices=("gut_12l", "gut_6l", "respiratory_12l", "respiratory_6l"))
        if name == "rebind":
            p.add_argument("--output-manifest", type=Path, required=True)
            p.add_argument("--reason", default="source-only maintenance; unchanged scientific inputs and method")
        if name == "job":
            p.add_argument("--arm", required=True,
                choices=("gut_12l", "gut_6l", "respiratory_12l", "respiratory_6l"))
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "prepare":
        from micoformer.domain_pretraining.prepare import prepare_experiment
        print(prepare_experiment(args.output_root))
    elif args.command == "submit":
        print(submit(args.manifest, args.manifest_sha256, args.arms))
    elif args.command == "rebind":
        from micoformer.domain_pretraining.prepare import rebind_source
        print(rebind_source(args.manifest, args.manifest_sha256, args.output_manifest, reason=args.reason))
    else:
        from micoformer.domain_pretraining.workflow import run_job
        run_job(args.manifest, args.manifest_sha256, args.arm, resume=args.resume)


if __name__ == "__main__":
    main()
