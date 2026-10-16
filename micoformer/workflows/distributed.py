"""Initialize distributed training and random streams.
初始化分布式训练与随机过程。
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch

import torch.distributed as dist


def _seed_all(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA random streams.
    设置 Python、NumPy、PyTorch 和 CUDA 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_distributed() -> tuple[int, int, int, torch.device]:
    """Return global rank, world size, local rank, and this process's device.
    返回全局进程编号、进程总数、本机进程编号及当前设备。
    """
    # This entry requires CUDA; it does not switch to CPU training.
    # 当前入口要求 CUDA，不自动切换为 CPU 训练。
    if not torch.cuda.is_available():
        raise RuntimeError("MiCoFormer training requires CUDA; no implicit CPU fallback")

    # torchrun supplies one LOCAL_RANK per local training process.
    # torchrun 为每个本机训练进程提供 LOCAL_RANK。
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun --nproc_per_node=NUM_GPUS")
    local_rank = int(os.environ["LOCAL_RANK"])

    # LOCAL_RANK indexes the GPUs visible through CUDA_VISIBLE_DEVICES.
    # LOCAL_RANK 对应 CUDA_VISIBLE_DEVICES 暴露后的逻辑 GPU 编号。
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK is outside the visible CUDA devices")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # The training loop uses BF16 autocast.
    # 训练循环使用 BF16 混合精度。
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the MiCoFormer training recipe requires BF16 support")

    # NCCL connects the ranks; DDP handles gradient synchronization separately.
    # NCCL 建立进程间通信，梯度同步由 DDP 负责。
    dist.init_process_group(backend="nccl")

    return dist.get_rank(), dist.get_world_size(), local_rank, device
