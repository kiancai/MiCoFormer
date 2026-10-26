# Changelog

All notable changes to MiCoFormer will be documented in this file.

The format is based on Keep a Changelog, adapted for the current research workflow.

## [0.5.0] - 2026-10-26

### Added
- Added configurable length-bucketed training batches, enabled by default, with epoch-wise shuffling and the existing distributed sample-coverage policy; validation and extraction order remain unchanged.
- Added optional normalized weighted UniFrac supervision to MiCoFormer pretraining, using full-composition tree masses and the existing Query sample representation.
- Added CE/entropy/KL diagnostics for the neighborhood objective; training with only the original three objectives remains available.

### Changed
- Switched to MCFCorpusV4.3 and its frozen 768-dimensional PHY288 + raw-PROT480 prior, with fixed centering and one RMS scale per block to equalize reference-pair squared-distance totals; input and Query use the same stored table without row normalization.
- Increased model width to 768 while retaining 12 layers, 16 attention heads, and FFN width 2048. Previous 512-wide checkpoints require their original code; they cannot resume this model.
- Made preparation select required HDF5 metadata directly, preserving the sample-level split policy without loading every observation column at once.
- Length-bucketed training changes batch-local Query-negative and UniFrac candidate pools, without changing their sampling rules, loss formulas, or weights.
- Unified objective outputs as loss/count/diagnostics and shared their training/validation statistics and logs; unmasked diagnostics remain separate from training losses, with no objective-specific loop branches or fixed statistics layouts.
- Integrated neighborhood weight and temperatures into TrainConfig, enabling weak UniFrac by default (weight 0.00005); setting the weight to 0 skips tree loading, full-composition targets, auxiliary losses, and diagnostics.
- Kept UniFrac tree assets inside the unified corpus; removed the separate tree-directory setting, keeping all UniFrac configuration out of the training CLI.
- Pinned the released corpus version and SHA256 for preparation and the default training input.
- Required complete raw protein features for every genus, removing legacy protein-feature validity masking and older corpus-version branches.
- Centralized training defaults in TrainConfig: batch size 96 per GPU, a 100-epoch limit, and fixed-prior loss weights calibrated for V4.3.
- Loaded training and validation data from one read-only CSR matrix per rank, shared with forked CPU workers; standalone embedding extraction remains disk-backed.
- Allowed training resumption after changing only the output directory or data-loading worker count, while preserving other configuration, data, loss-weight, and world-size checks.

### Removed
- Removed redundant fixed-asset and internal-interface runtime checks and the temporary test suite; retained corpus identity checks and training failure handling.

## [0.4.0] - 2026-10-16

### Added
- Added fixed phylogenetic and protein genus embeddings as the default model input, with learned genus embeddings available through configuration.
- Added joint masked-abundance reconstruction, Query presence prediction, and Query continuous-abundance prediction objectives.
- Added MCFCorpusV3.3 input support, with phylogenetic coordinates and protein features read from the same AnnData corpus.
- Added reproducible analysis_primary sample selection and train/validation split generation with corpus version and SHA256 checks.
- Added native PyTorch distributed pretraining with BF16 mixed precision, checkpoint resumption, and validation-based model selection and early stopping.
- Added self-contained checkpoints containing model parameters, the fixed prior table, and the genus vocabulary.
- Added streaming extraction of unmasked mean-pooled sample embeddings, row indices, and provenance records.

### Changed
- Consolidated the implementation into data, models, workflows, and shared IO utilities, with model and training settings defined in Python configuration classes.
- Replaced numbered scripts with prepare_data.py, train.py, and extract.py, including English command-line help.
- Removed legacy model implementations, fine-tuning and experiment-specific pipelines, PyTorch Lightning, and unused dependencies.
- Replaced legacy model, checkpoint, and command-line interfaces; older checkpoints are not automatically migrated to the new format.

### Fixed
- Replaced the additive masked-abundance input with a dedicated trainable vector, so masked positions use no abundance-MLP output.
- Fixed RNG state restoration when resuming from checkpoints loaded onto a GPU.
- Prevented overwriting existing embedding outputs or restarting a saved training run without an explicit resume request.

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
