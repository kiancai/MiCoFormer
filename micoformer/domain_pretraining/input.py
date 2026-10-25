"""Keep the original UniFrac row transforms; remove repeated gzip decoding.

Only the domain training entrypoint opts in. One read-only CSR copy is reused
by train/validation and epochs; Linux fork workers inherit the same RAM pages.
No HDF5 handles survive initialization and no new corpus artifact is produced.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from micoformer.mpc_unifrac.data import FullSupportRowDataset as HDF5RowDataset


MAX_CSR_BYTES = 3 * 1024**3


@lru_cache(maxsize=1)
def _resident_csr(corpus: Path) -> dict[str, np.ndarray]:
    """Load original dtypes/order exactly, before starting DataLoader workers."""
    started = perf_counter()
    with h5py.File(corpus, "r") as handle:
        x = handle["X"]
        keys = ("data", "indices", "indptr")
        arrays_bytes = sum(x[key].size * x[key].dtype.itemsize for key in keys)
        if arrays_bytes > MAX_CSR_BYTES:
            raise MemoryError(f"CSR arrays need {arrays_bytes} bytes; budget is {MAX_CSR_BYTES}")
        arrays = {key: x[key][:] for key in keys}
        shape = tuple(int(v) for v in x.attrs["shape"])
    if (arrays["data"].ndim != 1 or arrays["indices"].shape != arrays["data"].shape
            or arrays["indptr"].shape != (shape[0] + 1,)
            or arrays["indptr"][0] != 0 or arrays["indptr"][-1] != len(arrays["data"])):
        raise ValueError("invalid CSR array layout")
    for array in arrays.values():
        array.flags.writeable = False
    print(f"Resident CSR: {arrays_bytes} bytes, loaded in {perf_counter()-started:.3f}s from {corpus}", flush=True)
    return arrays


class InMemoryFullSupportRowDataset(HDF5RowDataset):
    """Serve the inherited, unmodified row algorithm from immutable arrays."""

    def __init__(self, corpus: str | Path, rows: np.ndarray) -> None:
        super().__init__(corpus, rows)
        self._csr_arrays = _resident_csr(self.corpus)

    def _h5(self):
        if self._csr_arrays is None:
            self._csr_arrays = _resident_csr(self.corpus)
        return {"X": self._csr_arrays}

    def __getstate__(self):
        # Spawn/pickle must not serialize gigabytes. Fork needs no pickling and
        # shares the arrays; a spawned process loads one process-local copy.
        state = super().__getstate__()
        state["_csr_arrays"] = None
        return state

    @property
    def resident_bytes(self) -> int:
        return sum(value.nbytes for value in self._h5()["X"].values())
