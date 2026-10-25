"""Strict loading for frozen downstream use of an adapted encoder."""
from pathlib import Path

import numpy as np
import torch

from micoformer.mpc_fixed_pretraining.model import MPCFixedModelConfig
from micoformer.mpc_unifrac.model import UniFracModel
from .lora import LoRAConfig, inject_lora, frozen_sha256
from .prepare import verify_file


def load_adapted_checkpoint(path: Path, *, manifest_sha256: str, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["checkpoint_kind"] != "domain_lora_full_resume"
            or state["manifest_sha256"] != manifest_sha256):
        raise RuntimeError("adapted checkpoint contract mismatch")
    verify_file(state["prior"])
    with np.load(state["prior"]["path"], allow_pickle=False) as data:
        prior = torch.from_numpy(data["pp_table"].copy()).float()
    model = UniFracModel(prior, MPCFixedModelConfig(**state["model_config"]))
    inject_lora(model, LoRAConfig(**state["lora_config"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    if frozen_sha256(model) != state["frozen_parameter_sha256"]:
        raise RuntimeError("frozen parameters differ from recorded origin")
    model.requires_grad_(False)
    return model.to(device).eval()
