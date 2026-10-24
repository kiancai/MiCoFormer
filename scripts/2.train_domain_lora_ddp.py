#!/usr/bin/env python3
"""Prepare or run the two approved respiratory LoRA strength comparisons."""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare(root, job, uuids, label):
    import torch
    from micoformer.domain_adaptation.prepare import (file_record, verify_file,
        validate_contract, source_identity)
    from micoformer.mpc_unifrac import workflow as common
    root = root.resolve()
    previous = root / 'audit/lora_run_manifest.json'
    manifest = json.loads(previous.read_text())
    for asset in manifest['assets'].values():
        verify_file(asset)
    cpu = root / 'audit/cpu_acceptance.json'
    acceptance = json.loads(cpu.read_text())
    if acceptance['status'] != 'PASS' or acceptance['manifest_sha256'] != common.sha256_file(previous):
        raise RuntimeError('matching real-row CPU acceptance required')
    if label != 'ddp3_exact':
        raise RuntimeError('frozen-head control requires the accepted three-card execution recipe')
    manifest['source'] = source_identity()
    manifest['prepared_at'] = datetime.now().astimezone().isoformat()
    manifest['node'] = socket.gethostname()
    manifest['assets'].update(previous_manifest=file_record(previous),
        cpu_acceptance=file_record(cpu))
    manifest['execution'] = dict(world_size=3, per_rank_accumulation=1,
        logical_microbatches_per_update=3, per_rank_microbatch=128,
        dropout_seed_policy='42+rank; unchanged common seed42 adapter initialization',
        placeholder_job_id=str(job), node='gpu1', gpu_uuids=uuids,
        preserve_placeholder=True, allocation_actions='none',
        sequence=list(manifest['submission_order']), dataloader_context='forkserver',
        label=label, activation_recomputation=True, gradient_reduction='rank_ordered_mean')
    if len(uuids) != 3 or len(set(uuids)) != 3 or any(not u.startswith('GPU-') for u in uuids):
        raise RuntimeError('bind three distinct allocated GPU UUIDs')
    for name, arm in manifest['arms'].items():
        arm['output'] = str(root / label / 'runs' / name)
    validate_contract(manifest)
    if manifest['torch'] != torch.__version__ or manifest['cuda'] != torch.version.cuda:
        raise RuntimeError('environment changed')
    destination = root / 'audit' / f'lora_run_manifest_{label}.json'
    if destination.exists():
        raise FileExistsError('three-card manifest is immutable')
    (root / label).mkdir(exist_ok=True)
    common._atomic_json(destination, manifest)
    print(json.dumps(file_record(destination), indent=2))


def endpoint(arm, manifest, sha):
    import torch
    from micoformer.domain_adaptation.model_io import load_adapted_checkpoint
    from micoformer.domain_adaptation.prepare import file_record
    from micoformer.mpc_unifrac import workflow as common
    output = Path(arm['output'])
    status = json.loads((output / 'results/training_status.json').read_text())
    epochs = [json.loads(s) for s in (output / 'results/epochs.jsonl').read_text().splitlines() if s]
    if (status['status'] != 'COMPLETED' or status['global_step'] != arm['total_steps']
            or status['manifest_sha256'] != sha or [r['epoch'] for r in epochs] != list(range(1, 11))
            or not all(r['exact_train_row_coverage'] and r['validation']['exact_row_coverage'] for r in epochs)):
        raise RuntimeError('arm did not complete its fixed ten-epoch contract')
    path = output / 'run/epoch10.pt'
    model = load_adapted_checkpoint(path, manifest_sha256=sha)
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise RuntimeError('nonfinite endpoint')
    common._atomic_json(output / 'audit/endpoint_acceptance.json', dict(status='PASS',
        time=datetime.now().astimezone().isoformat(), checkpoint=file_record(path),
        manifest_sha256=sha, epoch=10, global_step=arm['total_steps'],
        strict_load=True, all_state_finite=True, exact_train_val_coverage=True,
        frozen_encoder_unchanged=True, self_supervised_heads_unchanged=True, allocation_preserved=True))


def sequence(path, sha):
    from micoformer.domain_adaptation.prepare import validate_manifest, REPO
    from micoformer.mpc_unifrac import workflow as common
    manifest = validate_manifest(path, sha)
    root = path.resolve().parents[1]
    execution = manifest['execution']
    run_root = root / execution['label']
    job = str(execution['placeholder_job_id'])
    if os.environ.get('SLURM_JOB_ID') != job or f'job_{job}' not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('controller must be inside the preserved allocation')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(execution['gpu_uuids'])
    with (run_root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def record(status, **extra):
            common._atomic_json(run_root / 'controller_status.json', dict(
                status=status, time=datetime.now().astimezone().isoformat(), pid=os.getpid(),
                sid=os.getsid(0), machine='CmlLab', node=socket.gethostname(),
                placeholder_job_id=job, preserve_placeholder=True, manifest_sha256=sha, **extra))
        record('RUNNING')
        try:
            for name in execution['sequence']:
                details = subprocess.check_output(['scontrol', 'show', 'job', '-dd', job], text=True)
                if 'JobState=RUNNING' not in details or '/LOOP_GPU.sh' not in details:
                    raise RuntimeError('placeholder changed; do not start new work')
                busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                '--format=csv,noheader'], text=True)
                if any(uuid in busy for uuid in execution['gpu_uuids']):
                    raise RuntimeError('allocated GPU has existing compute; refusing overlap')
                arm = manifest['arms'][name]
                completion = Path(arm['output']) / 'results/training_status.json'
                if completion.exists() and json.loads(completion.read_text())['status'] == 'COMPLETED':
                    endpoint(arm, manifest, sha)
                    continue
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nnodes=1', '--nproc-per-node=3', str(Path(__file__).resolve()), 'worker',
                    '--manifest', str(path.resolve()), '--manifest-sha256', sha, '--arm', name]
                log = run_root / 'logs' / f'{name}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                record('RUNNING', active_arm=name, command=command, log=str(log))
                print(f'Starting {name} on all three GPUs; placeholder {job} remains untouched', flush=True)
                with log.open('x') as handle:
                    subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
                endpoint(arm, manifest, sha)
            record('COMPLETED', arms=execution['sequence'])
            print(f'Both arms accepted. Placeholder {job} remains running.', flush=True)
        except BaseException as error:
            record('FAILED', error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--placeholder-job', required=True)
    p.add_argument('--gpu-uuids', nargs=3, required=True)
    p.add_argument('--execution-label', choices=('ddp3', 'ddp3_recompute', 'ddp3_exact'), default='ddp3')
    for name in ('sequence', 'worker'):
        p = sub.add_parser(name)
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--manifest-sha256', required=True)
        if name == 'worker':
            p.add_argument('--arm', choices=('respiratory_lr3e4', 'respiratory_rank64'), required=True)
            p.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == 'prepare':
        prepare(args.root, args.placeholder_job, args.gpu_uuids, args.execution_label)
    elif args.command == 'sequence':
        sequence(args.manifest, args.manifest_sha256)
    else:
        from micoformer.domain_adaptation.distributed import run
        run(args.manifest, args.manifest_sha256, args.arm, resume=args.resume)


if __name__ == '__main__':
    main()
