"""Frozen independent ablation arms; no chained removal or recalibration."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'tmp/20261007_fixedpp_ablation_addendum_e15/a11_stage'
BASE_INIT = ROOT / 'tmp/20260817_mpc_fixed_context_full_data_pretraining/run/epoch0.pt'
BASE_WEIGHTS = dict(mlm=1., continuous=0.534399128791011,
                   presence=0.24296011504612833, unifrac=0.00005)
ARMS = {'A11': ('joint_prior_row_shuffle', None)}
ORDER = ('A11',)
JOB_ID = os.environ.get('SLURM_JOB_ID', 'UNALLOCATED')

def weights_for(arm):
    weights = BASE_WEIGHTS.copy()
    disabled = ARMS[arm][1]
    if disabled is not None:
        weights[disabled] = 0.
    return weights
