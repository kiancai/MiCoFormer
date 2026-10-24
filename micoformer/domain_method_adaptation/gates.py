"""真实A100输入：原点等价、分支梯度、独立SDPA参照及完整断点恢复门控。"""
from __future__ import annotations

from contextlib import nullcontext
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from micoformer.domain_adaptation.prepare import load_origin
from micoformer.domain_adaptation.workflow import state_equal
from .workflow import restore
from micoformer.domain_adaptation.lora import frozen_sha256
from micoformer.domain_pretraining.input import InMemoryFullSupportRowDataset
from micoformer.mpc_unifrac.data import DualCorruptionCollator, SharedEpoch
from micoformer.mpc_unifrac import workflow as common
from micoformer.domain_bucket_adaptation.sampling import TensorCollator
from .workflow import (load_model, device_batch, split_backward, update, make_optimizer,
                       save_checkpoint, make_loader, NAMES)
from .prepare import now, RUN


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_clone(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(v) for v in value)
    return value


def compare_gradients(first, second):
    numerator=sum(float((a.double()-b.double()).square().sum()) for a,b in zip(first,second,strict=True))
    denominator=sum(float(a.double().square().sum()) for a in first)
    other=sum(float(a.double().square().sum()) for a in second)
    dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(first,second,strict=True))
    if not all(bool(torch.isfinite(v).all()) for v in first+second) or min(denominator,other)<=0:
        raise RuntimeError("invalid or empty gradient")
    return dict(relative_l2=(numerator/denominator)**.5,
                cosine=dot/(denominator*other)**.5,
                first_norm=denominator**.5,second_norm=other**.5)


def gradients(model, arguments, weights, *, split, bf16, backend=None):
    model.zero_grad(set_to_none=True)
    common._seed_all(1234)
    with sdpa_kernel(backend) if backend is not None else nullcontext():
        if split:
            output=split_backward(model,arguments,weights,bf16=bf16)
        else:
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bf16):
                output=model(**arguments)
                loss=sum(weights[n]*getattr(output,n+"_loss") for n in NAMES)
            loss.backward()
    parameters=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None for p in parameters):
        raise RuntimeError("missing LoRA gradient")
    return ([p.grad.detach().cpu().clone() for p in parameters],
            {n:float(getattr(output,n+"_loss")) for n in NAMES})


def run(manifest,sha,device):
    started=perf_counter()
    arm=next(iter(manifest["arms"].values()))
    corpus=manifest["assets"]["corpus"]["path"]
    rows=np.load(arm["data"]["paths"]["train"],allow_pickle=False)
    dataset=InMemoryFullSupportRowDataset(corpus,rows)
    ptr=dataset._h5()["X"]["indptr"]
    lengths=np.minimum(ptr[rows+1]-ptr[rows],512)
    sorted_positions=np.argsort(lengths,kind="stable")
    selected=sorted_positions[[0,len(rows)//2,9*len(rows)//10,-1]]
    native=DualCorruptionCollator(observed_genus_ids=np.load(arm["data"]["observed_genus_ids"],allow_pickle=False),
                                  seed=42,stream=91,shared_epoch=SharedEpoch(0))
    collator=TensorCollator(native)
    small=collator([dataset[int(i)] for i in selected])
    args=device_batch(small,device)
    model=load_model(arm,manifest,device).eval()
    original=load_origin(manifest["assets"]["prior"]["path"],Path(manifest["assets"]["origin"]["path"])).to(device).eval()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        left=original.unmasked_representations(args["genus_ids"],args["rclr"],args["padding_mask"])[1]
        right=model.unmasked_representations(args["genus_ids"],args["rclr"],args["padding_mask"])[1]
    zero_exact = torch.equal(left,right)
    initial_cosine = float(torch.nn.functional.cosine_similarity(left.float(),right.float(),dim=1).min())
    if initial_cosine <= .9999:
        raise RuntimeError('initial BF16 representation disagrees with origin')
    with torch.inference_mode():
        left_fp=original.unmasked_representations(args['genus_ids'],args['rclr'],args['padding_mask'])[1]
        right_fp=model.unmasked_representations(args['genus_ids'],args['rclr'],args['padding_mask'])[1]
    torch.testing.assert_close(left_fp,right_fp,atol=2e-5,rtol=2e-5)
    initial_fp32_max_abs=float((left_fp-right_fp).abs().max())
    del left_fp,right_fp
    del original,left,right
    # 非零B同时检验A/B梯度，避免零初始化A梯度掩盖错误。
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.endswith(("lora_B", "adapter.up.weight")):
                p.normal_(0,.001)
    parity={}
    for bf16 in (False,True):
        model.train(bf16)
        first,loss1=gradients(model,args,arm["weights"],split=False,bf16=bf16)
        second,loss2=gradients(model,args,arm["weights"],split=True,bf16=bf16)
        difference=compare_gradients(first,second)
        tolerance=.003 if bf16 else 2e-5
        if difference["relative_l2"]>tolerance:
            raise RuntimeError(f"split gradient mismatch: {bf16} {difference}")
        parity[str(bf16)]=dict(**difference,tolerance=tolerance,
                             loss_differences={n:abs(loss1[n]-loss2[n]) for n in NAMES})
        del first,second
    # 当前模型/当前环境单独验证，不能由另一模型的故障或finite检查推断正确性。
    model.eval()
    backend_checks={}
    for bf16 in (False,True):
        reference,_=gradients(model,args,arm["weights"],split=True,bf16=bf16,backend=SDPBackend.MATH)
        actual,_=gradients(model,args,arm["weights"],split=True,bf16=bf16)
        difference=compare_gradients(reference,actual)
        tolerance=.05 if bf16 else .01
        if difference["relative_l2"]>tolerance or difference["cosine"]<.998:
            raise RuntimeError(f"default SDPA disagrees with MATH: {bf16} {difference}")
        backend_checks[str(bf16)]=dict(**difference,tolerance=tolerance)
        del reference,actual
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            model(**args)
        torch.cuda.synchronize()
    operations=[event.key for event in profiler.key_averages() if "attention" in event.key.lower()]
    del model,args
    gc.collect();torch.cuda.empty_cache()
    print(f"{now()} numerical gradient gates PASS; testing real B128 and strict resume",flush=True)
    # 前128条中强制包含最长输入，保持真实样本/原collator。
    positions=np.r_[sorted_positions[-1],np.arange(len(rows))[np.arange(len(rows))!=sorted_positions[-1]][:127]]
    full=collator([dataset[int(i)] for i in positions])
    tail=collator([dataset[int(i)] for i in positions[:20]])
    model=load_model(arm,manifest,device)
    frozen=frozen_sha256(model)
    optimizer,scheduler=make_optimizer(model,arm)
    torch.cuda.reset_peak_memory_stats(device)
    statistics,_=update(model,optimizer,scheduler,full,arm,device)
    path=RUN/"audit/gate_resume.pt"
    save_checkpoint(model,optimizer,scheduler,arm,sha,device,0,1,1,float("inf"),statistics,path,
                    dict(samples=128,tokens=int((~full["padding_mask"]).sum()),padded_positions=full["padding_mask"].numel()))
    _,expected_record=update(model,optimizer,scheduler,tail,arm,device)
    expected=(cpu_clone(model.state_dict()),cpu_clone(optimizer.state_dict()),scheduler.state_dict())
    peak=torch.cuda.max_memory_allocated(device)/1024**3
    del model,optimizer,scheduler
    gc.collect();torch.cuda.empty_cache()
    model=load_model(arm,manifest,device)
    optimizer,scheduler=make_optimizer(model,arm)
    restored=restore(path,model,optimizer,scheduler,arm,sha,device)
    if (restored["next_batch"]!=1 or restored["global_step"]!=1
            or restored["bucket_counters"]["samples"]!=128):
        raise RuntimeError("restored sampler cursor or coverage counters changed")
    del restored
    _,record=update(model,optimizer,scheduler,tail,arm,device)
    if (not state_equal(expected[0],model.state_dict()) or not state_equal(expected[1],optimizer.state_dict())
            or expected[2]!=scheduler.state_dict() or expected_record!=record):
        raise RuntimeError("next update after full restore is not exact")
    if frozen_sha256(model)!=frozen:
        raise RuntimeError("frozen weights changed")
    result=dict(status="PASS",time=now(),manifest_sha256=sha,source_commit=manifest["source"]["commit"],
        zero_delta_embedding_exact=zero_exact,initial_bf16_min_cosine=initial_cosine,
        initial_fp32_max_abs=initial_fp32_max_abs,initial_fp32_atol_rtol=2e-5,
        frozen_weights_unchanged=True,strict_restore_next_update_exact=True,
        selected_rows=rows[selected].tolist(),selected_lengths=lengths[selected].tolist(),
        split_parity=parity,sdpa_reference=backend_checks,attention_operations=operations,
        full_batch=128,full_max_length=int(full["genus_ids"].shape[1]),tail_batch=20,
        peak_allocated_gib=peak,seconds=perf_counter()-started)
    common._atomic_json(RUN/"audit/cuda_gates.json",result)
    del model,optimizer,scheduler,expected
    gc.collect();torch.cuda.empty_cache()
    return result


def benchmark(manifest,sha,device):
    """同一A100短程校准，两臂独立重置；不把测速权重接入正式训练。"""
    results=[]
    for name in manifest["arms"]:
        arm=manifest["arms"][name]
        loader,_,_=make_loader(arm,manifest["assets"]["corpus"]["path"],0)
        model=load_model(arm,manifest,device)
        optimizer,scheduler=make_optimizer(model,arm)
        iterator=iter(loader)
        samples=tokens=padded=0
        for step in range(23):
            if step==3:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
                begin=perf_counter()
            batch=next(iterator)
            update(model,optimizer,scheduler,batch,arm,device)
            if step>=3:
                samples+=len(batch["row_ids"])
                tokens+=int((~batch["padding_mask"]).sum())
                padded+=batch["padding_mask"].numel()
        torch.cuda.synchronize()
        seconds=perf_counter()-begin
        record=dict(arm=name,updates=20,samples=samples,tokens=tokens,seconds=seconds,
            samples_per_second=samples/seconds,padding_fraction=1-tokens/padded,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
            estimated_10epoch_training_minutes=1108680/(samples/seconds)/60,
            estimate_excludes="validation, save, initialization; short run not a full epoch")
        results.append(record)
        print(json.dumps(record),flush=True)
        del model,optimizer,scheduler,loader,iterator
        gc.collect();torch.cuda.empty_cache()
    receipt=dict(status="PASS",time=now(),manifest_sha256=sha,results=results)
    common._atomic_json(RUN/"audit/throughput.json",receipt)
    return receipt
