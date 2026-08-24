"""Fail-closed full-data workflow for the pure fixed-PP MPC endpoint.

The training mechanics are intentionally delegated to the already validated MPC
workflow.  We replace only its model/config factories and source-seal inventory;
all calibration, DDP, validation, checkpoint and resume behavior stays identical.
"""
from __future__ import annotations

from pathlib import Path

from micoformer.mpc_pretraining import workflow as _base

from .model import MPCFixedModelConfig, MPCFixedPretrainingModel


def _source_files() -> list[Path]:
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    base_package = repo / "micoformer/mpc_pretraining"
    experiment_scripts = (
        repo.parent
        / "tmp/20260817_mpc_fixed_context_full_data_pretraining/scripts"
    )
    files = [
        base_package / "__init__.py",
        base_package / "data.py",
        base_package / "model.py",
        base_package / "workflow.py",
        package / "__init__.py",
        package / "model.py",
        package / "workflow.py",
        repo / "scripts/2.train_mpc_fixed.py",
        repo / "micoformer/fullscale_relation_pretraining/model.py",
        repo / "micoformer/models/attn_bias.py",
        repo / "micoformer/models/heads.py",
        experiment_scripts / "run_common.sh",
        experiment_scripts / "run_prepare.sh",
        experiment_scripts / "run_calibrate.sh",
        experiment_scripts / "run_smoke.sh",
        experiment_scripts / "run_train.sh",
        experiment_scripts / "run_pipeline.sh",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"fixed-context MPC source files are missing: {missing}")
    return files


# The reused functions resolve these names from the base module at call time.
# Activate the fixed endpoint once, before exposing any public entry point.
_base.MPCModelConfig = MPCFixedModelConfig
_base.MPCPretrainingModel = MPCFixedPretrainingModel
_base._source_files = _source_files

MPCRunConfig = _base.MPCRunConfig
prepare_run = _base.prepare_run
calibrate_loss_weights = _base.calibrate_loss_weights
run_ddp_smoke = _base.run_ddp_smoke
run_full_pretraining = _base.run_full_pretraining

__all__ = [
    "MPCRunConfig",
    "prepare_run",
    "calibrate_loss_weights",
    "run_ddp_smoke",
    "run_full_pretraining",
]
