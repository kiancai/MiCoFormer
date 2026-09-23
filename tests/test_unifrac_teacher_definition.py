"""Analytical invariants for the contracted normalized weighted UniFrac."""
import numpy as np
import tempfile
import unittest
from pathlib import Path
from scipy import sparse
import torch

from micoformer.mpc_unifrac_continuation.teacher import UniFracTeacher


def make_teacher(tmp_path, scale=1.0, shared=False, chunk_size=1):
    incidence = np.array([[1, 0, 1], [0, 1, 1]]) if shared else np.eye(2)
    branches = np.array([1., 3., 2.]) if shared else np.array([1., 3.])
    sparse.save_npz(tmp_path / 'incidence.npz', sparse.csr_matrix(incidence))
    np.save(tmp_path / 'branches.npy', branches * scale)
    return UniFracTeacher(incidence_path=tmp_path / 'incidence.npz', branch_lengths_path=tmp_path / 'branches.npy', expected_n_genus=2, chunk_size=chunk_size)


class TeacherDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tmp_path = Path(self.directory.name)

    def test_analytical_disjoint_tips_and_branch_scale(self):
        for scale in (.1, 1., 10.):
            for shared, expected in ((False, 1.), (True, .5)):
                with self.subTest(scale=scale, shared=shared):
                    teacher = make_teacher(self.tmp_path, scale, shared)
                    actual = teacher(torch.tensor([[1., 0.], [0., 1.]]))
                    torch.testing.assert_close(actual, torch.tensor([[0., expected], [expected, 0.]]))

    def test_sample_depth_invariance_and_identical_samples(self):
        teacher = make_teacher(self.tmp_path)
        x = torch.tensor([[.75, .25], [.25, .75], [75., 25.]])
        expected = torch.tensor([[0., .5, 0.], [.5, 0., .5], [0., .5, 0.]])
        torch.testing.assert_close(teacher(x), expected)

    def test_teacher_stays_float32_inside_autocast(self):
        teacher = make_teacher(self.tmp_path, shared=True)
        x = torch.tensor([[.731234, .268766], [.198761, .801239]])
        expected = teacher(x)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = teacher(x)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)

    def test_empty_composition_rejected(self):
        teacher = make_teacher(self.tmp_path)
        with self.assertRaisesRegex(ValueError, 'positive finite totals'):
            teacher(torch.tensor([[0., 0.], [1., 0.]]))


if __name__ == '__main__':
    unittest.main()
