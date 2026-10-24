#!/usr/bin/env python3
"""Finite per-GPU method pipeline; never controls the placeholder lifecycle."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from micoformer.domain_method_adaptation.prepare import FAMILY, RUN, prepare, verify, now
from micoformer.domain_method_adaptation.workflow import preflight, train
from micoformer.domain_method_adaptation import gates
from micoformer.domain_method_adaptation.evaluation import extract
from micoformer.mpc_unifrac import workflow as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'pipeline'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(); return
    if not args.manifest or not args.manifest_sha256:
        parser.error('sealed manifest and digest required')
    manifest = verify(args.manifest, args.manifest_sha256)
    device = preflight(RUN/'audit/pipeline_gpu_preflight.json')
    prior = {'adapter':['r128_s42_random', 'r128_s42_window4096'],
             'loraplus':['r128_s43_random', 'r128_s43_window4096'],
             'dora':['r64_s42_random', 'r64_s42_window4096']}[FAMILY]
    for key in prior:
        extract(key, device)
    gates.run(manifest, args.manifest_sha256, device)
    gates.benchmark(manifest, args.manifest_sha256, device)
    for name, arm in manifest['arms'].items():
        if not train(arm, manifest, args.manifest_sha256, device):
            return
        extract(name, device)
    common._atomic_json(RUN/'results/sequence_status.json', dict(status='COMPLETED', time=now(),
        manifest_sha256=args.manifest_sha256, arms=list(manifest['arms']), preserve_placeholder=True))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='mcf-method-', dir='/tmp') as ipc:
        os.environ['TMPDIR'] = ipc; tempfile.tempdir = ipc
        try:
            main()
        except BaseException as error:
            if isinstance(error, SystemExit) and error.code in (0, None): raise
            common._atomic_json(RUN/'audit/last_failure.json', dict(status='FAILED', time=now(), error=repr(error)))
            raise
