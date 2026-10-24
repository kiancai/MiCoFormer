#!/bin/bash
set -euo pipefail
worktree=${1:?worktree}
family=${2:?method family}
target_gpu=${3:?allocated GPU UUID}
cpu_list=${4:?allocated CPU list}
shift 4
cd -- "$worktree"
export MCF_METHOD_FAMILY="$family" MCF_TARGET_GPU_UUID="$target_gpu" CUDA_VISIBLE_DEVICES="$target_gpu"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$worktree"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
driver_version="$(awk '/Kernel Module/ {for(i=1;i<NF;i++) if($i=="Module") print $(i+1)}' /proc/driver/nvidia/version)"
driver_root="/opt/app/nvidia/$driver_version"
test -x "$driver_root/bin/nvidia-smi"
export PATH="$driver_root/bin:/home/cml_lab/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="$driver_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec taskset -c "$cpu_list" /home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python scripts/2.train_domain_methods.py pipeline "$@"
