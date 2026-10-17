"""Frozen independent ablation arms; no chained removal or recalibration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'tmp/20260930_fixedpp_component_ablation_e20'
BASE_INIT = ROOT / 'tmp/20260817_mpc_fixed_context_full_data_pretraining/run/epoch0.pt'
BASE_WEIGHTS = dict(mlm=1., continuous=0.534399128791011,
                   presence=0.24296011504612833, unifrac=0.00005)
ARMS = {
    'A0': ('full', None),
    'A1': ('w_o_phy', None),
    'A2': ('w_o_prot', None),
    'A3': ('w_o_both_priors', None),
    'A4': ('w_o_mlm', 'mlm'),
    'A5': ('w_o_query_continuous', 'continuous'),
    'A6': ('w_o_query_presence', 'presence'),
    'A7': ('w_o_unifrac', 'unifrac'),
    'A8': ('ntv3_identity', None),
}
ORDER = ('A0', 'A8', 'A3', 'A1', 'A2', 'A7', 'A4', 'A5', 'A6')
JOB_ID = '657311'
GPU_UUIDS = ('GPU-3c1bc64f-868c-2fb1-72d4-862fba149f6e',
             'GPU-4ac8f658-13eb-4710-e4cf-fd0652067aa9',
             'GPU-16a868cf-b056-027c-5135-ae1acc46deb6')

def weights_for(arm):
    weights = BASE_WEIGHTS.copy()
    disabled = ARMS[arm][1]
    if disabled is not None:
        weights[disabled] = 0.
    return weights
