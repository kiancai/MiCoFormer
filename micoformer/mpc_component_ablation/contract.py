"""Frozen independent ablation arms; no chained removal or recalibration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'tmp/20261007_fixedpp_ablation_addendum_e15'
BASE_INIT = ROOT / 'tmp/20260817_mpc_fixed_context_full_data_pretraining/run/epoch0.pt'
BASE_WEIGHTS = dict(mlm=1., continuous=0.534399128791011,
                   presence=0.24296011504612833, unifrac=0.00005)
ARMS = {
    'A9': ('mlm_only', ['continuous', 'presence', 'unifrac']),
    'A10': ('query_only', ['mlm', 'unifrac']),
}
ORDER = ('A9', 'A10')
JOB_ID = '657311'
GPU_UUIDS = ('GPU-3c1bc64f-868c-2fb1-72d4-862fba149f6e',
             'GPU-4ac8f658-13eb-4710-e4cf-fd0652067aa9',
             'GPU-16a868cf-b056-027c-5135-ae1acc46deb6')

def weights_for(arm):
    weights = BASE_WEIGHTS.copy()
    disabled = ARMS[arm][1]
    for component in disabled:
        weights[component] = 0.
    return weights
