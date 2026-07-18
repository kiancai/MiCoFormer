from __future__ import annotations

import unittest

import torch

from micoformer.models.pma import PMA


class PMAMultiQueryTest(unittest.TestCase):
    def test_output_shapes_and_backward(self) -> None:
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        padding = torch.tensor(
            [[False, False, False, False, False], [False, False, False, True, True]]
        )

        legacy = PMA(d_model=8, nhead_pma=2, k=1)(hidden, key_padding_mask=padding)
        multi = PMA(d_model=8, nhead_pma=2, k=3)(hidden, key_padding_mask=padding)

        self.assertEqual(tuple(legacy.shape), (2, 8))
        self.assertEqual(tuple(multi.shape), (2, 3, 8))
        (legacy.sum() + multi.sum()).backward()
        self.assertIsNotNone(hidden.grad)

    def test_rejects_non_positive_query_count(self) -> None:
        with self.assertRaises(ValueError):
            PMA(d_model=8, nhead_pma=2, k=0)


if __name__ == "__main__":
    unittest.main()
