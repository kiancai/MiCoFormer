#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@180

set -euo pipefail
# Slurm copies the batch script to its spool; use the worktree bound to the
# manifest/commit instead of treating that temporary script path as a repo.
worktree=/home/cml_lab/caiqy/project/MCFProjet/exp-20261022_domain_corpus_pretraining
cd -- "$worktree"
manifest=$1
manifest_sha=$2
arm=$3
expected_commit=$4
if [[ "$(/home/cml_lab/anaconda3/bin/git rev-parse HEAD)" != "$expected_commit" ]]; then
    printf '%s\n' 'ERROR: submitted source commit changed' >&2
    exit 1
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$worktree"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm must assign one GPU}"
# gpu1 does not put NVIDIA tools in the batch PATH. Match the userspace driver
# to the running kernel, using the site's versioned installation, within this
# process only (do not modify /usr/local/nvidia or shared shell configuration).
driver_version="$(awk '/Kernel Module/ {for(i=1;i<NF;i++) if($i=="Module") print $(i+1)}' /proc/driver/nvidia/version)"
driver_root="/opt/app/nvidia/$driver_version"
if [[ -z "$driver_version" || ! -x "$driver_root/bin/nvidia-smi" || ! -d "$driver_root/lib" ]]; then
    printf 'ERROR: no userspace NVIDIA driver matching kernel %s at %s\n' "$driver_version" "$driver_root" >&2
    exit 1
fi
export PATH="$driver_root/bin:$PATH"
export LD_LIBRARY_PATH="$driver_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
hostname
date -Is
printf 'NVIDIA driver root: %s\n' "$driver_root"
scontrol show job -dd "$SLURM_JOB_ID"
nvidia-smi
exec /home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python \
    scripts/2.train_domain_mpc.py job --manifest "$manifest" \
    --manifest-sha256 "$manifest_sha" --arm "$arm" "${@:5}"
