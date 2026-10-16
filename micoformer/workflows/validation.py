"""Aggregate validation losses and monitor unmasked representation covariance.
汇总验证损失，并监测无遮盖表征的协方差。
"""
from __future__ import annotations

import math
from typing import Any, Mapping
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from micoformer.data.collate import batch_to_device


def _loss_statistics_tensor(
    output: Any,
    lambdas: Mapping[str, float],
    *,
    device: torch.device,
) -> Tensor:
    """Pack loss sums/counts and batch totals into a float64 vector [8].
    将损失总和、目标数及批次总量打包为 float64 向量 [8]。
    """
    total = (
        output.mlm_loss
        + float(lambdas["continuous"]) * output.continuous_loss
        + float(lambdas["presence"]) * output.presence_loss
    )
    # Keep each task's sum/count separate; target counts can differ between batches.
    # 各任务分别累计总和与目标数，避免不同批次的目标数量影响均值。
    return torch.tensor(
        [
            float(output.mlm_loss.detach()) * output.mlm_count,
            float(output.mlm_count),
            float(output.continuous_loss.detach()) * output.continuous_count,
            float(output.continuous_count),
            float(output.presence_loss.detach()) * output.presence_count,
            float(output.presence_count),
            float(total.detach()),
            1.0,
        ],
        device=device,
        dtype=torch.float64,
    )


def _unpack_loss_statistics(
    values: Tensor, lambdas: Mapping[str, float]
) -> dict[str, float]:
    """Compute target-weighted loss means from accumulated statistics.
    按累计预测目标数计算各项损失均值。
    """
    array = values.detach().cpu().numpy()
    mlm = float(array[0] / array[1])
    continuous = float(array[2] / array[3])
    presence = float(array[4] / array[5])
    return {
        "mlm": mlm,
        "continuous": continuous,
        "presence": presence,
        "weighted_total": (
            mlm
            + float(lambdas["continuous"]) * continuous
            + float(lambdas["presence"]) * presence
        ),
        "mean_batch_weighted_total": float(array[6] / array[7]),
    }


def _spectral_summary(count: int, sum_h: Tensor, sum_outer: Tensor) -> dict[str, Any]:
    """Summarize sample covariance using its eigenvalues.
    通过特征值概括样本表征的协方差。
    """
    mean = sum_h / count
    # Recover covariance from accumulated vectors [D] and outer products [D, D].
    # 根据累计向量 [D] 和外积 [D, D] 还原协方差。
    covariance = sum_outer / count - torch.outer(mean, mean)
    # Enforce symmetry and remove negative eigenvalues caused by rounding.
    # 恢复数值对称性，并将舍入误差产生的负特征值截为零。
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = torch.linalg.eigvalsh(covariance.cpu()).double().clamp_min(0.0)
    total = float(eigenvalues.sum())
    # A finite positive total variance is required by the rank calculations below.
    # 下方秩指标要求总方差为有限正数。
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("validation representation covariance collapsed")
    probabilities = eigenvalues / total
    # Exclude zero terms from p*log(p).
    # 计算 p*log(p) 时排除零项。
    positive = probabilities > 0
    entropy_rank = float(
        torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum())
    )
    participation_ratio = float(1.0 / probabilities.square().sum())
    maximum = float(eigenvalues[-1])
    stable_rank = total / maximum
    descending = torch.flip(eigenvalues, dims=(0,))
    cumulative = torch.cumsum(descending, dim=0) / total

    # Count leading directions needed to explain the requested variance fraction.
    # 计算解释指定比例方差所需的主要方向数。
    def k_at(fraction: float) -> int:
        return int(
            torch.searchsorted(
                cumulative, torch.tensor(fraction, dtype=cumulative.dtype)
            ).item()
            + 1
        )

    return {
        "n": count,
        "dimension": int(eigenvalues.numel()),
        "total_variance": total,
        "maximum_eigenvalue": maximum,
        "effective_rank_covariance_entropy": entropy_rank,
        "participation_ratio": participation_ratio,
        "stable_rank_representation": stable_rank,
        "k50": k_at(0.50),
        "k90": k_at(0.90),
        "k95": k_at(0.95),
        "k99": k_at(0.99),
    }

@torch.inference_mode()
def _validate_epoch(
    ddp: DDP,
    loader: DataLoader[dict[str, np.ndarray]],
    *,
    lambdas: Mapping[str, float],
    rank: int,
    device: torch.device,
) -> dict[str, Any]:
    """Validate without updates; include any samples repeated by the distributed sampler.
    不更新参数地运行验证；统计包含分布式采样器补齐的重复样本。
    """
    # Disable dropout; inference_mode also disables autograd.
    # 关闭 dropout；inference_mode 同时关闭梯度记录。
    ddp.eval()
    statistics = torch.zeros(8, device=device, dtype=torch.float64)
    representation_count = torch.zeros((), device=device, dtype=torch.float64)
    # Accumulate first/second moments without storing all sample embeddings.
    # 累计一阶和二阶矩，不保存全部样本 embedding。
    dimension = ddp.module.config.d_model
    sum_h = torch.zeros(dimension, device=device, dtype=torch.float64)
    sum_outer = torch.zeros((dimension, dimension), device=device, dtype=torch.float64)
    for batch in loader:
        arguments = batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = ddp(**arguments)
            # Encode an extra unmasked view for representation monitoring, not for the loss.
            # 额外编码无遮盖视图以监测表征，不计入损失。
            _, mean = ddp.module.unmasked_representations(
                arguments["genus_ids"], arguments["abundance"], arguments["padding_mask"]
            )
        statistics += _loss_statistics_tensor(output, lambdas, device=device)
        mean64 = mean.float().double()
        representation_count += mean64.shape[0]
        sum_h += mean64.sum(0)
        sum_outer += mean64.T @ mean64
    # Sum statistics across all ranks before computing global metrics.
    # 先跨进程汇总统计，再计算全局指标。
    for tensor in (statistics, representation_count, sum_h, sum_outer):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        payload = {
            "loss": _unpack_loss_statistics(statistics, lambdas),
            "representation": _spectral_summary(
                int(representation_count.item()), sum_h.cpu(), sum_outer.cpu()
            ),
            # Count processed samples including padding repeats; prepared rows are unique.
            # 实际处理数包含采样补齐的重复样本；准备好的划分本身没有重复行号。
            "row_count": int(representation_count.item()),
            "row_unique_count": len(loader.dataset),
        }
    else:
        payload = None
    objects = [payload]
    # Send rank 0's metrics to every rank for identical checkpoint/early-stop decisions.
    # 将主进程的指标发给各进程，保持存档和早停判断一致。
    dist.broadcast_object_list(objects, src=0)
    return objects[0]
