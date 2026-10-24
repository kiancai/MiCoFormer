#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=core56
set -euo pipefail
cd /hpcdisk1/limk_group/caiqy/project/260905_MCF/tasks/20261025_001_respiratory_method_screen_cpu
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /hpcdisk1/limk_group/caiqy/project/260905_MCF/envs/rm_site_cpu_v1/bin/python scripts/preflight.py
