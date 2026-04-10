# Changelog

All notable changes to MiCoFormer will be documented in this file.

The format is based on Keep a Changelog, adapted for the current research workflow.

## [Unreleased]

### Changed
- Reserved for post-`v0.3` development on the `dev` branch.

## [0.3.0] - 2026-04-10

### Changed
- Reworked the pretraining and fine-tuning workflows around reusable workflow primitives, thinner CLI wrappers, and shared validation utilities.
- Simplified split generation into a single filtering primitive and aligned the script layout to the current numbered entrypoints.
- Archived the legacy pretraining pipeline notebook to keep the active notebook workspace focused on the new workflow.

### Fixed
- Fixed the finetune path to inherit abundance-bin protocol settings from the pretraining checkpoint and return stable best-checkpoint metrics.
- Fixed pretraining FF parameter resolution so `ff_dim` and the default `ff_ratio` path both work consistently through the shared validation layer.
- Added split-overlap checks for pretraining, fine-tuning, and hyperparameter search to prevent silently polluted experiments.

## [0.2.0] - 2026-04-05

### Added
- Added downstream classification fine-tuning pipeline, including classification datamodule, classifier module, and finetune split generation.
- Added taxonomy-distance attention bias (`R2`) implementation and related training support.
- Added structured pretraining hyperparameter search workflow with one-at-a-time and 2D grid modes.

### Changed
- Renamed and reorganized pretraining / classification modules to use clearer `pretrain_*` and `classification_*` naming.
- Refined the pretraining script with clearer argument grouping, budget validation, and unified `rank_zero_info` logging style.
- Aligned `scripts/5.train_finetune.py` with the pretraining script style, including grouped CLI arguments, centralized validation, and clearer single-run vs. k-fold mode boundaries.
- Updated package requirements and project metadata for the current Python / PyTorch / anndata baseline.

### Fixed
- Improved dataset parsing, h5ad handling, and taxonomy vocabulary generation.
- Simplified data pipeline internals and removed redundant model code paths.

## [0.1.0] - 2026-03-30

### Added
- Initial public release of MiCoFormer with core encoder, data pipeline, and pretraining workflow.
