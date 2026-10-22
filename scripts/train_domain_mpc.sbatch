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
hostname
date -Is
scontrol show job -dd "$SLURM_JOB_ID"
nvidia-smi
exec /home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python \
    scripts/2.train_domain_mpc.py job --manifest "$manifest" \
    --manifest-sha256 "$manifest_sha" --arm "$arm" "${@:5}"
