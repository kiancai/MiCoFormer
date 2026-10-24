"""Strict method checkpoint reconstruction without optimizer/RNG reuse in inference."""
from pathlib import Path
import torch
from micoformer.domain_adaptation.prepare import load_origin, verify_file
from micoformer.mpc_unifrac import workflow as common
from .modules import inject, frozen_sha256


def load_adapted_checkpoint(path, *, manifest_sha256, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['checkpoint_kind'] != 'domain_method_full_resume' or state['manifest_sha256'] != manifest_sha256:
        raise RuntimeError('method checkpoint identity mismatch')
    verify_file(state['prior'])
    verify_file(state['origin'])
    model = load_origin(state['prior']['path'], Path(state['origin']['path']))
    inject(model, state['adaptation'])
    frozen = frozen_sha256(model)
    model.load_state_dict(state['model_state_dict'], strict=True)
    if frozen_sha256(model) != frozen or state['frozen_parameter_sha256'] != frozen:
        raise RuntimeError('checkpoint changed frozen pretrained parameters')
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise RuntimeError('nonfinite method checkpoint')
    return model.to(device).eval().requires_grad_(False)
