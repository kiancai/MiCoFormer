#!/usr/bin/env python3
"""占位673251内的呼吸道无标签LoRA组批对照；绝不控制allocation生命周期。"""
import argparse
import json
import os
from pathlib import Path
import tempfile

from micoformer.domain_bucket_adaptation.prepare import prepare, verify, RUN, now
from micoformer.domain_bucket_adaptation.workflow import preflight, train
from micoformer.domain_bucket_adaptation import gates
from micoformer.mpc_unifrac import workflow as common


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","gates","sequence"])
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.action=="prepare":
        prepare()
        return
    if not args.manifest or not args.manifest_sha256:
        parser.error("manifest path and fixed SHA256 required")
    manifest=verify(args.manifest,args.manifest_sha256)
    device=preflight(RUN/f"audit/{args.action}_gpu_preflight.json")
    if args.action=="gates":
        gates.run(manifest,args.manifest_sha256,device)
        gates.benchmark(manifest,args.manifest_sha256,device)
    else:
        for name in ("cuda_gates.json","throughput.json"):
            receipt=json.loads((RUN/"audit"/name).read_text())
            if receipt["status"]!="PASS" or receipt["manifest_sha256"]!=args.manifest_sha256:
                raise RuntimeError("same sealed recipe must pass CUDA gates and throughput first")
        for name in ("random","window4096"):
            result=train(manifest["arms"][name],manifest,args.manifest_sha256,device,resume=args.resume)
            if result is False:
                return
        common._atomic_json(RUN/"results/sequence_status.json",dict(status="COMPLETED",time=now(),
            manifest_sha256=args.manifest_sha256,arms=["random","window4096"]))


if __name__=="__main__":
    # Python tensor IPC uses Unix sockets: keep the owned directory short.
    with tempfile.TemporaryDirectory(prefix="mcf-rsp-",dir="/tmp") as ipc:
        os.environ["TMPDIR"]=ipc
        tempfile.tempdir=ipc
        try:
            main()
        except BaseException as error:
            if isinstance(error,SystemExit) and error.code in (0,None):
                raise
            common._atomic_json(RUN/"audit/last_failure.json",dict(status="FAILED",time=now(),
                pid=os.getpid(),error=repr(error)))
            raise
