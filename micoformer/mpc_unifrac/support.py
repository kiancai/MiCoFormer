"""Exact DDP coverage, explicit loss weights, and observational gradient probes."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import (
    FullSupportRowDataset, DualCorruptionCollator, SharedEpoch, batch_to_device,
)

NAMES = ("mlm", "continuous", "presence", "unifrac")


class ExactDistributedSampler(DistributedSampler):
    """Partition each permutation once, with neither padding nor dropping."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.drop_last:
            raise ValueError("dropping rows is forbidden")
        self.num_samples = len(range(self.rank, len(self.dataset), self.num_replicas))
        self.total_size = len(self.dataset)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        return iter(indices[self.rank::self.num_replicas])


def distributed_training_loss(output, weights, world, device):
    # DDP averages gradients over ranks. Correct each component by its actual
    # target count so the result is the global target mean, including tails.
    local_counts = torch.tensor([getattr(output, n + "_count") for n in NAMES],
                                dtype=torch.float64, device=device)
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts)
    if bool((global_counts <= 0).any()):
        raise RuntimeError("empty loss component in global batch")
    factors = world * local_counts / global_counts
    return sum(float(weights[n]) * factors[i].float() * getattr(output, n + "_loss")
               for i, n in enumerate(NAMES))


def verify_epoch_rows(local_rows, expected, rank, world):
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(np.concatenate(local_rows), gathered, dst=0)
    valid = True
    if rank == 0:
        combined = np.concatenate(gathered)
        valid = np.array_equal(np.sort(combined), np.sort(expected))
    flag = torch.tensor(int(valid), device=torch.device("cuda", rank))
    dist.broadcast(flag, 0)
    if not flag.item():
        raise RuntimeError("training omitted or duplicated rows")


def _array_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _state_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_state_equal(x, y) for x, y in zip(a, b))
    return a == b


def _state_close(a, b, *, atol=1e-7):
    """Only post-update floating results use tolerance; restore itself is exact."""
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        a,b=a.detach().cpu(),b.detach().cpu()
        return bool(torch.allclose(a,b,atol=atol,rtol=1e-6)) if a.is_floating_point() else torch.equal(a,b)
    if isinstance(a, dict):
        return a.keys()==b.keys() and all(_state_close(a[k],b[k],atol=atol) for k in a)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(_state_close(x,y,atol=atol) for x,y in zip(a,b))
    return a==b


def run_fixed_gradient_probe(config, model, observed, weights, *, epoch, step, rank, world, device):
    """Two fixed global B384 batches; eval mode; no .grad or RNG mutation."""
    from . import workflow as w
    del observed
    rng = w._capture_rng_state(device)
    mode = model.training
    parameters = model.shared_encoder_parameters()
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("probe must start with cleared training gradients")
    before_state = w._model_state_sha256(model)
    model.eval()
    records = []
    try:
        for batch_index in range(2):
            path = config.output_root / f"audit/probe/rank{rank}_batch{batch_index}.npz"
            with np.load(path, allow_pickle=False) as z:
                batch = {k: np.array(z[k]) for k in z.files}
            arguments = batch_to_device(batch, device)
            local_counts = torch.tensor([arguments['mlm_mask'].sum(), arguments['query_mask'].sum(),
                                         arguments['query_mask'].sum()*2, len(arguments['genus_ids'])], device=device, dtype=torch.float64)
            counts = local_counts.clone(); dist.all_reduce(counts)
            gradients, losses = {}, {}
            for i, n in enumerate(NAMES):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model.component_loss(n, **arguments)
                local_sum = loss.detach().double()*local_counts[i]
                dist.all_reduce(local_sum)
                losses[n] = float(local_sum / counts[i])
                gradients[n] = w._averaged_encoder_gradients(
                    loss * (world * local_counts[i] / counts[i]).float(), parameters, world)
                del loss
            norms = {n: w._gradient_norm(gradients[n]) for n in NAMES}
            weighted_norms = {n: norms[n]*weights[n] for n in NAMES}
            cosines = {a+'__'+b: w._gradient_cosine(gradients[a], gradients[b])
                       for i,a in enumerate(NAMES) for b in NAMES[i+1:]}
            combined_sq = torch.zeros((), device=device, dtype=torch.float64)
            for parts in zip(*(gradients[n] for n in NAMES)):
                combined = sum(weights[n]*g.double() for n,g in zip(NAMES, parts))
                combined_sq += combined.square().sum()
            mpc_sq=torch.zeros((),device=device,dtype=torch.float64)
            for parts in zip(*(gradients[n] for n in NAMES[:3])):
                g=sum(weights[n]*v.double() for n,v in zip(NAMES[:3],parts));mpc_sq+=g.square().sum()
            mpc_norm=float(mpc_sq.sqrt())
            ratio=weighted_norms['unifrac']/max(mpc_norm,1e-30)
            record = dict(unifrac_over_mpc_gradient_norm=ratio,mpc_weighted_vector_sum_norm=mpc_norm,epoch=epoch, global_step=step, probe_batch=batch_index,
                mode='fixed_rows_fixed_corruption_eval_dropout_off_global_target_mean',
                raw_components=losses, weighted_components={n: weights[n]*losses[n] for n in NAMES},
                raw_encoder_grad_norm=norms, weighted_encoder_grad_norm=weighted_norms,
                component_cosines=cosines, weighted_vector_sum_norm=float(combined_sq.sqrt()),
                weighted_component_norm_sum=sum(weighted_norms.values()),
                query_norm_sum_over_mlm=(weighted_norms['continuous']+weighted_norms['presence'])/weighted_norms['mlm'],
                continuous_norm_over_presence=weighted_norms['continuous']/weighted_norms['presence'],
                weights=weights, target_counts={n:int(counts[i]) for i,n in enumerate(NAMES)},
                local_batch_sha256=w.sha256_file(path), optimizer_update=False)
            if not all(math.isfinite(x) for x in [*norms.values(), *losses.values(), *cosines.values(), record['weighted_vector_sum_norm']]):
                raise RuntimeError('nonfinite fixed gradient probe')
            # Observational paired views only: no gradients, no optimizer update;
            # outer RNG save/restore and parameter hash checks cover both forwards.
            from .teacher import weighted_unifrac,neighborhood_details
            with torch.no_grad():
                distances=weighted_unifrac(arguments['edge_mass'],model.unifrac_lengths)
                diagnostics={}
                for view in ['query_masked','clean']:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        if view=='clean':
                            _,h=model.unmasked_representations(arguments['genus_ids'],arguments['rclr'],arguments['padding_mask'])
                        else:
                            h=model._query_output(arguments['genus_ids'],arguments['rclr'],arguments['padding_mask'],arguments['query_mask'],arguments['negative_genus_ids']).sample_h
                    ce,entropy,kl=neighborhood_details(h,distances)
                    sums=torch.stack([ce.double(),entropy.double(),kl.double()])*len(h)
                    dist.all_reduce(sums)
                    diagnostics[view]=dict(zip(['ce','teacher_entropy','kl'],(sums/counts[3]).cpu().tolist()))
                    del h
                if abs(diagnostics['query_masked']['ce']-losses['unifrac'])>1e-5:
                    raise RuntimeError('fixed masked diagnostic drifted from gradient probe')
                record['unifrac_view_diagnostics']=diagnostics
            records.append(record)
            del gradients, arguments, combined
        if before_state != w._model_state_sha256(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('probe changed model or training gradients')
    finally:
        model.train(mode)
        w._restore_rng_state(rng, device)
    if not _state_equal(rng, w._capture_rng_state(device)):
        raise RuntimeError('probe changed training RNG')
    if rank == 0:
        for record in records:
            record['rng_and_parameters_preserved'] = True
            w._append_jsonl(config.output_root/'results/gradient_probes.jsonl', record)
    dist.barrier()
    if epoch==0 and any(r['unifrac_over_mpc_gradient_norm']>.05 for r in records):
        raise RuntimeError('INITIAL_UNIFRAC_GRADIENT_EXCEEDS_5_PERCENT: discussion required; no automatic reweighting')
    return records


def integrated_smoke(config):
    """Real B128 DDP, full and uneven-tail batches, validation and next-update restore."""
    from . import workflow as w
    config = config.normalized()
    sealed = w.validate_sealed_run(config)
    bound = w._load_calibration(config, sealed)
    rank, world, _, device = w._init_distributed()
    try:
        w._seed_all(config.seed)
        _, observed = w._load_prior(config)
        model = w._load_epoch0_model(config, sealed, bound, device)
        w._seed_all(config.seed + 10000*rank)
        loaders = w._build_loaders(config, rank=rank, world=world, observed=observed, workers=0)
        ddp = DDP(model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False,
                  gradient_as_bucket_view=True, static_graph=True)
        opt = torch.optim.AdamW(ddp.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        sched = w._make_scheduler(opt, total_steps=50*5270, warmup_ratio=.02, base_lr=3e-4, minimum_lr=1e-6)
        weights = config.loss_weights
        torch.cuda.reset_peak_memory_stats(device)
        losses = []
        initial_probe=run_fixed_gradient_probe(config,model,observed,weights,epoch=0,step=0,rank=rank,world=world,device=device)
        def update(net, optimizer, scheduler, batch):
            net.train(); optimizer.zero_grad(set_to_none=True)
            args = batch_to_device(batch, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = net(**args)
                loss = distributed_training_loss(out, weights, world, device)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite smoke loss')
            loss.backward()
            if not w._finite_gradients(net): raise RuntimeError('nonfinite smoke gradient')
            gs = w._assert_gradient_sync(net, world)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite smoke gradient norm')
            optimizer.step(); scheduler.step()
            ps = w._assert_parameter_sync(net, world)
            stats = w._loss_statistics_tensor(out, weights, device=device); dist.all_reduce(stats)
            return dict(loss=w._unpack_loss_statistics(stats,weights), gradient_sync=gs,
                        parameter_sync=ps, preclip=float(norm), batch_rows=len(batch['row_ids']))
        iterator = iter(loaders.train_loader)
        for _ in range(2): losses.append(update(ddp,opt,sched,next(iterator)))
        # Exercise the same exact sampler at the actual full-data tail sizes.
        local_indices = list(loaders.train_sampler)
        tail_indices = local_indices[(len(loaders.train_loader)-1)*128:]
        tail = loaders.train_loader.collate_fn([loaders.train_loader.dataset[i] for i in tail_indices])
        if len(tail['row_ids']) != (95 if rank < 2 else 94): raise RuntimeError('wrong tail')
        val_rows = loaders.val_rows[:767]
        ds = FullSupportRowDataset(config.corpus, val_rows)
        sampler = ExactDistributedSampler(ds,num_replicas=world,rank=rank,shuffle=False,drop_last=False)
        val_loader = DataLoader(ds,batch_size=128,sampler=sampler,num_workers=0,
            collate_fn=DualCorruptionCollator(observed_genus_ids=observed,seed=42,stream=2,shared_epoch=SharedEpoch(0)))
        validation = w._validate_epoch(ddp,val_loader,expected_rows=val_rows,lambdas=weights,rank=rank,world=world,device=device)
        opt.zero_grad(set_to_none=True)
        probe = run_fixed_gradient_probe(config,model,observed,weights,epoch=-1,step=2,rank=rank,world=world,device=device)
        rng = w._capture_rng_state(device); gathered=[None]*world if rank==0 else None
        dist.gather_object(rng,gathered,dst=0)
        cp_path=config.output_root/'audit/smoke_resume.pt'
        if rank==0:
            payload=w._checkpoint_payload(ddp=ddp,optimizer=opt,scheduler=sched,epoch=0,global_step=2,
                best_score=1.,meaningful_anchor=1.,stale_epochs=0,lambdas=weights,sealed=sealed,
                rng_by_rank=gathered,validation=validation,config=config)
            w._atomic_torch(cp_path,payload)
            del payload
        dist.barrier()
        baseline = update(ddp,opt,sched,tail)
        baseline_rng=w._capture_rng_state(device)
        baseline_path=config.output_root/f'audit/smoke_next_rank{rank}.pt'
        w._atomic_torch(baseline_path,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict()})
        del ddp,model,opt,sched,iterator,loaders
        gc.collect(); torch.cuda.empty_cache()
        model=w._load_epoch0_model(config,sealed,bound,device)
        ddp=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True,static_graph=True)
        opt=torch.optim.AdamW(ddp.parameters(),lr=3e-4,weight_decay=.01)
        sched=w._make_scheduler(opt,total_steps=50*5270,warmup_ratio=.02,base_lr=3e-4,minimum_lr=1e-6)
        restored = w._load_resume(cp_path,config=config,sealed=sealed,lambdas=weights,ddp=ddp,optimizer=opt,scheduler=sched,rank=rank,device=device)
        if restored[:2] != (1,2): raise RuntimeError('resume step/epoch mismatch')
        expected=torch.load(cp_path,map_location='cpu',weights_only=False)
        if not (_state_equal(model.state_dict(),expected['model_state_dict']) and _state_equal(opt.state_dict(),expected['optimizer_state_dict'])
                and _state_equal(sched.state_dict(),expected['scheduler_state_dict']) and _state_equal(w._capture_rng_state(device),expected['rng_by_rank'][rank])):
            raise RuntimeError('full strict restore mismatch')
        saved_sha=w._model_state_sha256(model);del expected
        repeated=update(ddp,opt,sched,tail)
        expected=torch.load(baseline_path,map_location='cpu',weights_only=False)
        restore_difference = {'rank':rank, 'model_equal':_state_equal(model.state_dict(),expected['model']),
            'optimizer_equal':_state_equal(opt.state_dict(),expected['optimizer']),
            'scheduler_equal':_state_equal(sched.state_dict(),expected['scheduler']),
            'rng_equal':_state_equal(w._capture_rng_state(device),baseline_rng),
            'model_max_abs':max(float((v.detach().cpu()-expected['model'][k]).abs().max()) for k,v in model.state_dict().items()),
            'baseline_loss':baseline, 'restored_loss':repeated}
        w._atomic_json(config.output_root/f'audit/smoke_restore_difference_rank{rank}.json',restore_difference)
        restore_difference['optimizer_max_tensor_abs'] = max(float((v.detach().cpu()-expected['optimizer']['state'][k][n]).abs().max())
            for k,state in opt.state_dict()['state'].items() for n,v in state.items() if torch.is_tensor(v))
        restore_difference['next_update_tolerance']={'model_atol':1e-7,'optimizer_moments_atol':1e-6,'rtol':1e-6}
        w._atomic_json(config.output_root/f'audit/smoke_restore_difference_rank{rank}.json',restore_difference)
        if not (_state_close(model.state_dict(),expected['model']) and _state_close(opt.state_dict(),expected['optimizer'],atol=1e-6)
                and _state_equal(sched.state_dict(),expected['scheduler']) and _state_equal(w._capture_rng_state(device),baseline_rng)):
            raise RuntimeError('resumed next update differs from uninterrupted update')
        peak=torch.tensor(torch.cuda.max_memory_reserved(device)/2**30,device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if float(peak)>=36.: raise RuntimeError(f'smoke peak exceeds 36GiB: {float(peak)}')
        result=dict(initial_unifrac_gradient_ratios=[r['unifrac_over_mpc_gradient_norm'] for r in initial_probe],status='PASS',manifest_sha256=sealed['manifest_sha256'],calibration_sha256=bound['calibration_sha256'],
            world_size=world,batch_size_per_rank=128,optimizer_steps=3,tail_global_rows=284,losses=losses,
            tail_baseline=baseline,tail_resumed=repeated,validation_rows=767,gradient_probe_batches=len(probe),
            gradient_sync_max_abs=max(x['gradient_sync'] for x in losses+[baseline,repeated]),
            parameter_sync_max_abs=max(x['parameter_sync'] for x in losses+[baseline,repeated]),
            restored_parameter_sync_max_abs=w._assert_parameter_sync(ddp,world),saved_model_state_sha256=saved_sha,
            strict_rng_and_next_update_restore=True,strict_model_optimizer_scheduler_rng_restore=True,
            next_update_tolerance=restore_difference['next_update_tolerance'],next_update_difference=restore_difference,
            peak_reserved_gib=float(peak),all_a100=True,finite=True)
        if rank==0:
            w._atomic_json(config.output_root/'audit/ddp_smoke.json',result)
            print(json.dumps(result),flush=True)
        dist.barrier()
        return config.output_root/'audit/ddp_smoke.json'
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
