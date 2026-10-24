"""Regression checks for scientific semantics affected by single-GPU execution."""
from types import SimpleNamespace

import numpy as np
import unittest
import torch

from micoformer.domain_pretraining.data import (
    DomainModelConfig, ExactBatchSampler, accumulation_loss, accumulation_groups,
)
from micoformer.mpc_unifrac.support import NAMES


def check_exact_coverage_no_singleton_and_resume_suffix(size):
    sampler = ExactBatchSampler(size, seed=42, epoch=3, shuffle=True)
    batches = list(sampler)
    assert all(2 <= len(batch) <= 128 for batch in batches)
    assert sorted(i for batch in batches for i in batch) == list(range(size))
    cursor = min(3, len(batches))
    resumed = ExactBatchSampler(size, seed=42, epoch=3, shuffle=True, start_batch=cursor)
    assert list(resumed) == batches[cursor:]
    assert list(ExactBatchSampler(size, seed=42, epoch=3, shuffle=True)) == batches
    assert sum(len(group) for group in accumulation_groups(batches)) == len(batches)


def test_accumulated_gradient_matches_global_target_mean_with_uneven_counts():
    weights = dict(zip(NAMES, [1., .534399128791011, .24296011504612833, .00005]))
    counts = np.array([[37, 19, 38, 128], [41, 21, 42, 128], [3, 1, 2, 2]])
    parameter = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
    targets = [[torch.linspace(i - j - .3, i + j + 1., int(n), dtype=torch.float64)
                for j, n in enumerate(row)] for i, row in enumerate(counts)]
    for row, values in zip(counts, targets):
        fields = {}
        for name, n, data in zip(NAMES, row, values):
            fields[name + "_loss"] = (parameter - data).square().mean()
            fields[name + "_count"] = int(n)
        accumulation_loss(SimpleNamespace(**fields), weights, counts.sum(0)).backward()
    actual = parameter.grad.clone()
    parameter.grad = None
    reference = sum(weights[name] * (parameter - torch.cat([v[j] for v in targets])).square().mean()
                    for j, name in enumerate(NAMES))
    reference.backward()
    torch.testing.assert_close(actual, parameter.grad, rtol=1e-12, atol=1e-12)


def test_only_depth_changes_the_fixed_model_contract():
    six = DomainModelConfig(num_layers=6).to_dict()
    twelve = DomainModelConfig().to_dict()
    assert {key for key in six if six[key] != twelve[key]} == {"num_layers"}
    for kwargs in [dict(num_layers=3), dict(d_model=256), dict(dropout=.2), dict(grad_checkpointing=True)]:
        with unittest.TestCase().assertRaises(ValueError):
            DomainModelConfig(**kwargs)


class SingleGpuContractTests(unittest.TestCase):
    def test_exact_coverage(self):
        for size in (2, 127, 128, 129, 255, 256, 257, 383, 384, 385, 111439):
            with self.subTest(size=size):
                check_exact_coverage_no_singleton_and_resume_suffix(size)

    def test_target_count_weighted_gradient(self):
        test_accumulated_gradient_matches_global_target_mean_with_uneven_counts()

    def test_model_contract(self):
        test_only_depth_changes_the_fixed_model_contract()



if __name__ == "__main__":
    unittest.main()
