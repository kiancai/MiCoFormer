from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from micoformer.relation_pretraining.losses import (
    relation_triplet_loss,
    squared_l2_triplet_hinge,
)
from micoformer.relation_pretraining.mining import TeacherMiningResult
from micoformer.relation_pretraining.mining import MiningConfig, mine_relations
from micoformer.relation_pretraining.model import RelationModelConfig, RelationOnlyModel


def _result(name: str, positives: list[int], negatives: list[int]) -> TeacherMiningResult:
    if len(positives) != len(negatives):
        raise ValueError
    status = []
    for positive, negative in zip(positives, negatives, strict=True):
        if positive < 0:
            status.append("no_positive")
        elif negative < 0:
            status.append("no_next")
        else:
            status.append("valid_next")
    batch_size = len(positives)
    long_zeros = torch.zeros(batch_size, dtype=torch.long)
    float_nan = torch.full((batch_size,), float("nan"))
    return TeacherMiningResult(
        name=name,
        positive_index=torch.tensor(positives, dtype=torch.long),
        negative_index=torch.tensor(negatives, dtype=torch.long),
        status=tuple(status),
        positive_tie_count=torch.where(
            torch.tensor(positives) >= 0,
            torch.ones(batch_size, dtype=torch.long),
            long_zeros,
        ),
        teacher_far_before_project_count=long_zeros.clone(),
        teacher_far_after_project_count=long_zeros.clone(),
        teacher_far_after_protection_count=long_zeros.clone(),
        other_positive_protected=torch.zeros(batch_size, dtype=torch.bool),
        positive_teacher_distance=float_nan.clone(),
        negative_teacher_distance=float_nan.clone(),
        positive_student_distance=float_nan.clone(),
        negative_student_distance=float_nan.clone(),
    )


class RelationLossTest(unittest.TestCase):
    def test_model_mining_loss_integration_has_finite_backward(self) -> None:
        torch.manual_seed(17)
        model = RelationOnlyModel(
            RelationModelConfig(
                vocab_size=32,
                d_model=32,
                rclr_hidden_dim=8,
                num_layers=1,
                encoder_heads=4,
                encoder_ffn_dim=64,
                decoder_heads=4,
                decoder_ffn_dim=64,
                dropout=0.0,
                max_seq_len=8,
            )
        )
        genus_ids = torch.tensor(
            [[2, 3, 4], [5, 6, 0], [7, 8, 9], [10, 11, 0]],
            dtype=torch.long,
        )
        rclr = torch.tensor(
            [[0.3, 0.0, -0.3], [0.2, -0.2, 0.0], [0.4, -0.1, -0.3], [0.1, -0.1, 0.0]]
        )
        z = model(genus_ids, rclr).z
        teachers = {
            "protein": torch.tensor(
                [[0.0, 0.1, 0.2, 0.3], [0.1, 0.0, 0.4, 0.5],
                 [0.2, 0.4, 0.0, 0.6], [0.3, 0.5, 0.6, 0.0]],
                dtype=torch.float64,
            ),
            "unifrac": torch.tensor(
                [[0.0, 0.2, 0.1, 0.3], [0.2, 0.0, 0.4, 0.5],
                 [0.1, 0.4, 0.0, 0.6], [0.3, 0.5, 0.6, 0.0]],
                dtype=torch.float64,
            ),
        }
        mining = mine_relations(
            z,
            teachers,
            row_ids=[10, 20, 30, 40],
            project_ids=[1, 2, 3, 4],
            config=MiningConfig(no_next_mode="closest_radius_inside"),
        )
        output = relation_triplet_loss(z, mining)
        self.assertTrue(output.has_relation_update)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_squared_l2_margin_hinge_exact_values(self) -> None:
        anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        positive = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
        negative = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        hinge, positive_distance, negative_distance = squared_l2_triplet_hinge(
            anchor,
            positive,
            negative,
            margin=0.10,
        )
        torch.testing.assert_close(positive_distance, torch.tensor([0.02, 2.0]))
        torch.testing.assert_close(negative_distance, torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(hinge, torch.tensor([0.0, 2.1]))

    def test_valid_anchor_mean_includes_zero_hinge_and_both_teachers_are_half_weighted(
        self,
    ) -> None:
        raw = torch.tensor(
            [
                [1.0, 0.0],
                [0.98, 0.20],
                [-1.0, 0.0],
                [0.0, 1.0],
            ],
            requires_grad=True,
        )
        z = F.normalize(raw, dim=-1)
        protein = _result("protein", [1, 2, -1, 0], [2, 0, -1, -1])
        unifrac = _result("unifrac", [3, -1, -1, -1], [2, -1, -1, -1])
        output = relation_triplet_loss(
            z,
            {"protein": protein, "unifrac": unifrac},
            margin=0.10,
        )

        protein_hinge, _, _ = squared_l2_triplet_hinge(
            z[[0, 1]],
            z[[1, 2]],
            z[[2, 0]],
            margin=0.10,
        )
        unifrac_hinge, _, _ = squared_l2_triplet_hinge(
            z[[0]],
            z[[3]],
            z[[2]],
            margin=0.10,
        )
        self.assertEqual(protein_hinge[0].item(), 0.0)
        expected = 0.5 * protein_hinge.mean() + 0.5 * unifrac_hinge.mean()
        torch.testing.assert_close(output.loss, expected)
        self.assertEqual(output.teacher_weights, {"protein": 0.5, "unifrac": 0.5})
        self.assertEqual(output.teacher_stats["protein"].valid_count, 2)
        self.assertEqual(output.teacher_stats["protein"].active_count, 1)
        self.assertEqual(output.teacher_stats["protein"].inactive_count, 1)
        self.assertEqual(output.counters["all/valid_anchor"], 3)
        self.assertEqual(output.counters["all/skipped_anchor"], 5)
        self.assertEqual(output.counters["all/teacher_anchor_total"], 8)

        output.loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_single_present_teacher_is_renormalized_to_one(self) -> None:
        z = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ]
            ),
            dim=-1,
        )
        protein = _result("protein", [1, -1, -1], [2, -1, -1])
        unifrac = _result("unifrac", [-1, -1, -1], [-1, -1, -1])
        output = relation_triplet_loss(z, {"protein": protein, "unifrac": unifrac})
        self.assertEqual(output.teacher_weights, {"protein": 1.0, "unifrac": 0.0})
        torch.testing.assert_close(output.loss, output.teacher_stats["protein"].loss)
        self.assertTrue(output.has_relation_update)

    def test_none_present_returns_graph_connected_zero_and_conserved_counters(self) -> None:
        z = torch.randn(3, 4, requires_grad=True)
        missing_a = _result("protein", [-1, -1, -1], [-1, -1, -1])
        missing_b = _result("unifrac", [-1, -1, -1], [-1, -1, -1])
        output = relation_triplet_loss(z, {"protein": missing_a, "unifrac": missing_b})
        self.assertEqual(output.loss.item(), 0.0)
        self.assertFalse(output.has_relation_update)
        self.assertEqual(output.teacher_weights, {"protein": 0.0, "unifrac": 0.0})
        self.assertEqual(output.counters["all/valid_anchor"], 0)
        self.assertEqual(output.counters["all/skipped_anchor"], 6)
        output.loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.equal(z.grad, torch.zeros_like(z)))

    def test_invalid_margin_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "margin"):
            squared_l2_triplet_hinge(
                torch.zeros((1, 2)),
                torch.zeros((1, 2)),
                torch.zeros((1, 2)),
                margin=-0.1,
            )

    def test_half_precision_loss_geometry_fails_closed(self) -> None:
        missing = _result("protein", [-1, -1], [-1, -1])
        with self.assertRaisesRegex(ValueError, "float32"):
            relation_triplet_loss(torch.zeros((2, 3), dtype=torch.float16), {"protein": missing})

    def test_teacher_order_recovery_uses_frozen_student_tolerance(self) -> None:
        # Anchor 0 has d_pos=0 and d_neg=5e-8, which is a float32 numerical
        # tie under the frozen 1e-7 strict-order tolerance.  Anchor 1 has a
        # clearly farther negative and must still count as recovered.
        z = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [float((5e-8) ** 0.5), 0.0],
                [float((2e-7) ** 0.5), 0.0],
            ],
            dtype=torch.float32,
        )
        result = _result("protein", [1, 0, -1, -1], [2, 3, -1, -1])
        output = relation_triplet_loss(z, {"protein": result})
        self.assertEqual(
            output.teacher_stats["protein"].teacher_order_recovered_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()
