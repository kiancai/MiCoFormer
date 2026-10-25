"""随机位置序列相同，只改变窗口内组批；每轮完整覆盖，断点按批重放。"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch.utils.data import Sampler


class BucketSampler(Sampler[list[int]]):
    def __init__(self, lengths, *, mode, epoch, seed=42, batch_size=128,
                 window=4096, cursor=0):
        lengths = np.asarray(lengths, dtype=np.int64)
        if lengths.ndim != 1 or len(lengths) < 2 or np.any(lengths < 2):
            raise ValueError("need at least two valid rows")
        if mode not in ("random", "window4096") or batch_size < 3:
            raise ValueError("unsupported grouping")
        if window % batch_size:
            raise ValueError("window must contain complete ordinary batches")
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(lengths), generator=generator).numpy()
        batches = []
        if mode == "random":
            batches = [
                order[i:i+batch_size].tolist() for i in range(0, len(order), batch_size)]
        else:
            for begin in range(0, len(order), window):
                positions = order[begin:begin+window]
                positions = positions[np.argsort(lengths[positions], kind="stable")]
                batches.extend(positions[i:i+batch_size].tolist()
                               for i in range(0, len(positions), batch_size))
        # 通用尾批保护：UniFrac至少需要两个样本，借一行而不漏/重复行。
        if len(batches[-1]) == 1:
            batches[-1].insert(0, batches[-2].pop())
        if mode != "random":
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in permutation]
        if not 0 <= cursor <= len(batches):
            raise ValueError("invalid resume cursor")
        self.batches, self.cursor = batches, cursor
        flat = np.concatenate(batches).astype(np.int64)
        if not np.array_equal(np.sort(flat), np.arange(len(lengths))):
            raise RuntimeError("sampler omitted or duplicated a row")
        self.order_sha256 = hashlib.sha256(flat.tobytes()).hexdigest()

    def __iter__(self):
        return iter(self.batches[self.cursor:])

    def __len__(self):
        return len(self.batches) - self.cursor


class TensorCollator:
    """保持原始mask/负例/树质量算法，仅转换输出供pin_memory使用。"""
    def __init__(self, native):
        self.native = native

    def __call__(self, samples):
        return {name: torch.as_tensor(value) for name, value in self.native(samples).items()}
