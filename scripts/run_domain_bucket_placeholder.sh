#!/bin/bash
# 只执行allocation内的工作层进程；不管理占位循环或作业生命周期。
set -euo pipefail
worktree=${1:?experiment worktree}
action=${2:?prepare or gates or sequence or pipeline}
pair=${3:?r128_seed43 or r64_seed42}
target_gpu=${4:?verified spare GPU UUID}
shift 4
export MCF_PAIR="$pair" MCF_TARGET_GPU_UUID="$target_gpu"
export CUDA_VISIBLE_DEVICES="$target_gpu"
cd -- "$worktree"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$worktree"
driver_version="$(awk '/Kernel Module/ {for(i=1;i<NF;i++) if($i=="Module") print $(i+1)}' /proc/driver/nvidia/version)"
driver_root="/opt/app/nvidia/$driver_version"
test -x "$driver_root/bin/nvidia-smi"
export PATH="$driver_root/bin:/home/cml_lab/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="$driver_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python \
    scripts/2.train_domain_bucket.py "$action" "$@"
