from __future__ import annotations

import unittest

import torch

from micoformer.mpc_relation_adaptation.losses import (
    OrdinalLossConfig,
    cosine_anchor_loss,
    unifrac_ordinal_loss,
)


class UniFracOrdinalLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.teacher = torch.tensor(
            [
                [0.0, 0.1, 0.3, 0.8],
                [0.1, 0.0, 0.4, 0.9],
                [0.3, 0.4, 0.0, 0.5],
                [0.8, 0.9, 0.5, 0.0],
            ]
        )
        self.blocks = torch.tensor([10, 11, 12, 13])

    def test_ordered_embedding_scores_better_than_reversed_embedding(self) -> None:
        ordered = torch.tensor([[0.0], [0.1], [0.3], [0.8]])
        reversed_middle = torch.tensor([[0.0], [0.8], [0.3], [0.1]])
        settings = OrdinalLossConfig(tie_fraction=0.0, temperature=0.1)
        good = unifrac_ordinal_loss(ordered, self.teacher, self.blocks, settings)
        bad = unifrac_ordinal_loss(reversed_middle, self.teacher, self.blocks, settings)
        self.assertLess(float(good.loss), float(bad.loss))
        self.assertEqual(good.valid_anchors, 4)
        self.assertGreater(good.comparisons, 0)

    def test_same_study_candidates_are_excluded(self) -> None:
        embedding = torch.tensor([[0.0], [0.1], [0.3], [0.8]])
        all_cross = unifrac_ordinal_loss(embedding, self.teacher, self.blocks)
        blocked = unifrac_ordinal_loss(
            embedding, self.teacher, torch.tensor([10, 10, 12, 13])
        )
        self.assertLess(blocked.comparisons, all_cross.comparisons)

    def test_no_cross_study_comparison_fails_closed(self) -> None:
        embedding = torch.tensor([[0.0], [0.1], [0.3], [0.8]])
        with self.assertRaisesRegex(RuntimeError, "no valid"):
            unifrac_ordinal_loss(
                embedding, self.teacher, torch.zeros(4, dtype=torch.long)
            )

    def test_teacher_contract_rejects_asymmetry(self) -> None:
        teacher = self.teacher.clone()
        teacher[0, 1] = 0.2
        with self.assertRaisesRegex(ValueError, "symmetric"):
            unifrac_ordinal_loss(torch.randn(4, 3), teacher, self.blocks)

    def test_cosine_anchor_is_zero_at_parity(self) -> None:
        value = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        self.assertAlmostEqual(float(cosine_anchor_loss(value, value)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
