from __future__ import annotations

import unittest

import torch

from micoformer.relation_pretraining.mining import (
    MiningConfig,
    STATUS_NAMES,
    mine_relations,
    select_output_negative,
    select_positive,
)


def _zero_tolerance_config(no_next_mode: str = "skip") -> MiningConfig:
    return MiningConfig(
        no_next_mode=no_next_mode,
        teacher_rtol=0.0,
        teacher_atol=0.0,
        student_rtol=0.0,
        student_atol=0.0,
    )


def _teacher_matrices() -> dict[str, torch.Tensor]:
    return {
        "protein": torch.tensor(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.0, 0.4, 0.5],
                [0.2, 0.4, 0.0, 0.6],
                [0.3, 0.5, 0.6, 0.0],
            ],
            dtype=torch.float64,
        ),
        "unifrac": torch.tensor(
            [
                [0.0, 0.2, 0.1, 0.3],
                [0.2, 0.0, 0.4, 0.5],
                [0.1, 0.4, 0.0, 0.6],
                [0.3, 0.5, 0.6, 0.0],
            ],
            dtype=torch.float64,
        ),
    }


class RelationMiningTest(unittest.TestCase):
    def test_frozen_default_tolerances(self) -> None:
        config = MiningConfig()
        self.assertEqual(config.teacher_rtol, 1e-10)
        self.assertEqual(config.teacher_atol, 1e-12)
        self.assertEqual(config.teacher_scale_floor, 1.0)
        self.assertEqual(config.student_rtol, 0.0)
        self.assertEqual(config.student_atol, 1e-7)

        # A float32 squared-L2 delta inside 1e-7 is not strictly farther.
        index, status = select_output_negative(
            torch.tensor(0.5),
            torch.tensor([0.5 + 5e-8]),
            torch.tensor([True]),
            torch.tensor([7]),
            config=config,
        )
        self.assertEqual((index, status), (-1, "no_next"))

    def test_positive_excludes_same_project_and_ties_use_min_row_id(self) -> None:
        distance = torch.tensor(
            [
                [0.0, 0.01, 0.2, 0.2],
                [0.01, 0.0, 0.4, 0.5],
                [0.2, 0.4, 0.0, 0.6],
                [0.2, 0.5, 0.6, 0.0],
            ],
            dtype=torch.float64,
        )
        valid = torch.isfinite(distance)
        rows = torch.tensor([100, 50, 30, 20])
        projects = torch.tensor([1, 1, 2, 3])
        different_project = projects[:, None] != projects[None, :]
        index, tie_count = select_positive(
            distance,
            valid,
            rows,
            different_project,
            0,
            config=_zero_tolerance_config(),
        )
        self.assertEqual(index, 3)
        self.assertEqual(tie_count, 2)

    def test_frozen_teacher_tolerance_boundary_controls_argmin_and_strict_far(self) -> None:
        config = MiningConfig()
        distance = torch.tensor(
            [
                [0.0, 0.5, 0.5 + 5e-11, 0.5 + 2e-10],
                [0.5, 0.0, 0.7, 0.8],
                [0.5 + 5e-11, 0.7, 0.0, 0.9],
                [0.5 + 2e-10, 0.8, 0.9, 0.0],
            ],
            dtype=torch.float64,
        )
        rows = torch.tensor([100, 20, 10, 5])
        projects = torch.tensor([1, 2, 3, 4])
        different_project = projects[:, None] != projects[None, :]
        index, tie_count = select_positive(
            distance,
            torch.isfinite(distance),
            rows,
            different_project,
            0,
            config=config,
        )
        # 5e-11 is inside the max(1e-12, 1e-10*scale) tie boundary,
        # while 2e-10 is outside it.
        self.assertEqual(index, 2)
        self.assertEqual(tie_count, 2)

        z = torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0]],
            dtype=torch.float32,
        )
        output = mine_relations(
            z,
            {"protein": distance},
            row_ids=rows,
            project_ids=projects,
            config=config,
        )
        result = output.teachers["protein"]
        self.assertEqual(int(result.positive_index[0]), 2)
        # Candidate row 20 is tolerance-tied to the positive and therefore is
        # not teacher-far; only row 5 crosses the strict-far boundary.
        self.assertEqual(int(result.teacher_far_after_protection_count[0]), 1)
        self.assertEqual(int(result.negative_index[0]), 3)

    def test_output_selector_next_fallback_tie_and_tolerance_boundaries(self) -> None:
        rows = torch.tensor([10, 20, 30])
        eligible = torch.tensor([True, True, True])
        index, status = select_output_negative(
            torch.tensor(0.60),
            torch.tensor([0.55, 0.63, 1.20]),
            eligible,
            rows,
            config=_zero_tolerance_config(),
        )
        self.assertEqual((index, status), (1, "valid_next"))

        # There is deliberately no classic positive+margin upper bound.
        index, status = select_output_negative(
            torch.tensor(0.60),
            torch.tensor([1.20]),
            torch.tensor([True]),
            torch.tensor([30]),
            config=_zero_tolerance_config(),
        )
        self.assertEqual((index, status), (0, "valid_next"))

        # Equal output distances use the canonical sample row, not batch order.
        index, status = select_output_negative(
            torch.tensor(0.60),
            torch.tensor([0.63, 0.63]),
            torch.tensor([True, True]),
            torch.tensor([30, 20]),
            config=_zero_tolerance_config(),
        )
        self.assertEqual((index, status), (1, "valid_next"))

        index, status = select_output_negative(
            torch.tensor(0.60),
            torch.tensor([0.20, 0.55]),
            torch.tensor([True, True]),
            torch.tensor([10, 20]),
            config=_zero_tolerance_config("closest_radius_inside"),
        )
        self.assertEqual((index, status), (1, "valid_fallback"))

        tolerance_config = MiningConfig(
            no_next_mode="skip",
            teacher_rtol=0.0,
            teacher_atol=0.0,
            student_rtol=0.0,
            student_atol=1e-5,
        )
        index, status = select_output_negative(
            torch.tensor(0.60),
            torch.tensor([0.600001]),
            torch.tensor([True]),
            torch.tensor([10]),
            config=tolerance_config,
        )
        self.assertEqual((index, status), (-1, "no_next"))

    def test_two_teachers_are_independent_and_protect_both_positives(self) -> None:
        z = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
            ]
        )
        output = mine_relations(
            z,
            _teacher_matrices(),
            row_ids=torch.tensor([100, 200, 300, 400]),
            project_ids=torch.tensor([1, 2, 3, 4]),
            config=_zero_tolerance_config(),
        )
        protein = output.teachers["protein"]
        unifrac = output.teachers["unifrac"]
        self.assertEqual(int(protein.positive_index[0]), 1)
        self.assertEqual(int(unifrac.positive_index[0]), 2)
        self.assertTrue(bool(protein.other_positive_protected[0]))
        self.assertTrue(bool(unifrac.other_positive_protected[0]))
        self.assertEqual(int(protein.teacher_far_after_project_count[0]), 2)
        self.assertEqual(int(protein.teacher_far_after_protection_count[0]), 1)
        self.assertEqual(int(protein.negative_index[0]), 3)
        self.assertEqual(int(unifrac.negative_index[0]), 3)
        self.assertEqual(protein.status[0], "valid_next")
        self.assertEqual(unifrac.status[0], "valid_next")

        counters = output.counters()
        self.assertEqual(counters["all/teacher_anchor_total"], 8)
        self.assertEqual(counters["all/status_total"], 8)
        self.assertEqual(counters["all/valid_relation"] + counters["all/skipped_relation"], 8)
        self.assertEqual(sum(counters[f"all/status/{status}"] for status in STATUS_NAMES), 8)

    def test_no_next_modes_never_bypass_teacher_eligibility(self) -> None:
        z = torch.ones((4, 3))
        cases = (
            ("skip", "no_next", False),
            ("closest_radius_inside", "valid_fallback", True),
        )
        for mode, expected_status, expect_negative in cases:
            with self.subTest(mode=mode):
                output = mine_relations(
                    z,
                    _teacher_matrices(),
                    row_ids=[40, 10, 20, 30],
                    project_ids=["P0", "P1", "P2", "P3"],
                    config=_zero_tolerance_config(mode),
                )
                result = output.teachers["protein"]
                self.assertEqual(result.status[0], expected_status)
                self.assertEqual(bool(result.negative_index[0] >= 0), expect_negative)
                if expect_negative:
                    # Protein positive row 10 and UniFrac positive row 20 are
                    # protected, leaving canonical row 30 as fallback.
                    self.assertEqual(int(result.negative_index[0]), 3)

    def test_no_positive_and_no_teacher_far_are_distinct_and_conserved(self) -> None:
        z = torch.eye(3)
        same_project_output = mine_relations(
            z,
            {
                "protein": torch.ones((3, 3), dtype=torch.float64),
                "unifrac": torch.ones((3, 3), dtype=torch.float64),
            },
            row_ids=[1, 2, 3],
            project_ids=["same", "same", "same"],
            config=_zero_tolerance_config(),
        )
        self.assertEqual(set(same_project_output.teachers["protein"].status), {"no_positive"})

        equal_teacher = torch.ones((3, 3), dtype=torch.float64)
        equal_teacher.fill_diagonal_(0.0)
        no_far_output = mine_relations(
            z,
            {"protein": equal_teacher, "unifrac": equal_teacher.clone()},
            row_ids=[3, 2, 1],
            project_ids=["P0", "P1", "P2"],
            config=_zero_tolerance_config(),
        )
        self.assertEqual(set(no_far_output.teachers["protein"].status), {"no_teacher_far"})
        self.assertEqual(no_far_output.counters()["all/status_total"], 6)

    def test_duplicate_canonical_rows_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            mine_relations(
                torch.eye(3),
                {
                    "protein": torch.eye(3, dtype=torch.float64),
                    "unifrac": torch.eye(3, dtype=torch.float64),
                },
                row_ids=[1, 1, 2],
                project_ids=[1, 2, 3],
            )

    def test_float32_teacher_cache_fails_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "float64"):
            mine_relations(
                torch.eye(3),
                {"protein": torch.eye(3), "unifrac": torch.eye(3)},
                row_ids=[1, 2, 3],
                project_ids=[1, 2, 3],
            )

    def test_half_precision_student_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "float32"):
            mine_relations(
                torch.eye(3, dtype=torch.float16),
                {
                    "protein": torch.eye(3, dtype=torch.float64),
                    "unifrac": torch.eye(3, dtype=torch.float64),
                },
                row_ids=[1, 2, 3],
                project_ids=[1, 2, 3],
            )


if __name__ == "__main__":
    unittest.main()
