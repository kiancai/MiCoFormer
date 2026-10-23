import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from micoformer.domain_pretraining.input import InMemoryFullSupportRowDataset, _resident_csr
from micoformer.mpc_unifrac.data import FullSupportRowDataset


class ResidentInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "csr.h5ad"
        with h5py.File(self.path, "w") as h:
            x = h.create_group("X")
            x.attrs["shape"] = (3, 8114)
            # Repeated abundance/ties and unsorted indices must preserve the
            # inherited lexsort and full-support UniFrac order exactly.
            x.create_dataset("data", data=np.array([.1, .3, .2, .3, .1] * 3, dtype=np.float32), compression="gzip")
            x.create_dataset("indices", data=np.array([8, 7, 6, 5, 4] * 3, dtype=np.int32), compression="gzip")
            x.create_dataset("indptr", data=np.array([0, 5, 10, 15], dtype=np.int32))

    def tearDown(self):
        _resident_csr.cache_clear()
        self.temp.cleanup()

    def test_same_rows_and_shared_immutable_arrays(self):
        rows = np.array([2, 0], dtype=np.int64)
        old = FullSupportRowDataset(self.path, rows)
        new = InMemoryFullSupportRowDataset(self.path, rows)
        val = InMemoryFullSupportRowDataset(self.path, np.array([1]))
        self.assertIs(new._csr_arrays, val._csr_arrays)
        for i in range(len(rows)):
            actual, expected = new[i], old[i]
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
                self.assertEqual(np.asarray(actual[key]).dtype, np.asarray(expected[key]).dtype)
        self.assertIsNone(new._handle)
        for array in new._csr_arrays.values():
            self.assertFalse(array.flags.writeable)
        old._handle.close()

    def test_pickle_does_not_copy_arrays_or_open_handles(self):
        new = InMemoryFullSupportRowDataset(self.path, np.array([0]))
        payload = pickle.dumps(new)
        restored = pickle.loads(payload)
        self.assertIsNone(restored._csr_arrays)
        for key, expected in new[0].items():
            np.testing.assert_array_equal(restored[0][key], expected)
        self.assertIs(restored._csr_arrays, new._csr_arrays)

    def test_budget_fails_before_loading_arrays(self):
        with patch("micoformer.domain_pretraining.input.MAX_CSR_BYTES", 1):
            with self.assertRaises(MemoryError):
                InMemoryFullSupportRowDataset(self.path, np.array([0]))


if __name__ == "__main__":
    unittest.main()
