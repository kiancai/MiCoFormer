"""Full-V3 relation-guided pretraining package.

This package is versioned separately from the frozen reduced relation and
C0/C1/C2 experiment families.  It implements the user-approved F1/F2 gate
without relaxing historical checkpoint loaders.
"""

from .losses import MultiHeadRelationLossOutput, multi_head_relation_triplet_loss
from .data import (
    CadencedRelationMLMDataset,
    DeterministicMLMBatchDataset,
    ExactTeacherBatchDataset,
    FullscaleAnnDataDataset,
    collate_fullscale_samples,
    deterministic_abundance_mask,
)
from .mining import MultiHeadRelationMiningOutput, mine_relations_by_teacher
from .fast_data import FastFullCorpusDataModule, FastScheduledRelationMLMDataset
from .fast_teacher import (
    FAST_TEACHER_KIND,
    FastFullCorpusPretrainingModule,
    FastFullCorpusTeacher,
)
from .module import (
    FullscaleRelationOptimizationConfig,
    FullscaleRelationPretrainingModule,
    build_fullscale_source_manifest,
    common_initialization_sha256,
    load_fullscale_relation_checkpoint,
)
from .model import (
    FULLSCALE_RELATION_ARMS,
    FullscaleRelationArm,
    FullscaleRelationModel,
    FullscaleRelationModelConfig,
    FullscaleRelationModelOutput,
    masked_token_mean,
)

__all__ = [
    "FULLSCALE_RELATION_ARMS",
    "FullscaleRelationArm",
    "FullscaleAnnDataDataset",
    "FullscaleRelationModel",
    "FullscaleRelationModelConfig",
    "FullscaleRelationModelOutput",
    "FullscaleRelationOptimizationConfig",
    "FullscaleRelationPretrainingModule",
    "MultiHeadRelationLossOutput",
    "MultiHeadRelationMiningOutput",
    "ExactTeacherBatchDataset",
    "DeterministicMLMBatchDataset",
    "CadencedRelationMLMDataset",
    "FAST_TEACHER_KIND",
    "FastFullCorpusDataModule",
    "FastFullCorpusPretrainingModule",
    "FastFullCorpusTeacher",
    "FastScheduledRelationMLMDataset",
    "masked_token_mean",
    "build_fullscale_source_manifest",
    "common_initialization_sha256",
    "collate_fullscale_samples",
    "deterministic_abundance_mask",
    "load_fullscale_relation_checkpoint",
    "mine_relations_by_teacher",
    "multi_head_relation_triplet_loss",
]
