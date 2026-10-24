#!/bin/bash
# Work-layer launcher only. It never controls the allocation or batch script.
set -euo pipefail
worktree=${1:?source worktree}
manifest=${2:?sealed three-card manifest}
manifest_sha=${3:?manifest sha256}
cd -- "$worktree"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$worktree"
driver_version="$(awk '/Kernel Module/ {for(i=1;i<NF;i++) if($i=="Module") print $(i+1)}' /proc/driver/nvidia/version)"
driver_root="/opt/app/nvidia/$driver_version"
test -x "$driver_root/bin/nvidia-smi"
export PATH="$driver_root/bin:$PATH"
export LD_LIBRARY_PATH="$driver_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python \
    scripts/2.train_domain_lora_ddp.py sequence --manifest "$manifest" --manifest-sha256 "$manifest_sha"
