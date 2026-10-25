#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=core56
set -euo pipefail
cd /hpcdisk1/limk_group/caiqy/project/260905_MCF/tasks/20261025_001_respiratory_method_screen_cpu
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin=/hpcdisk1/limk_group/caiqy/project/260905_MCF/envs/rm_site_cpu_v1/bin/python
"$python_bin" scripts/receive_model.py --arm "$MCF_ARM"
"$python_bin" scripts/run_cpu.py --arm "$MCF_ARM"
"$python_bin" scripts/finish.py --arm "$MCF_ARM"
