"""Aggregate objective losses and monitor unmasked representation covariance.
汇总各项目标，并监测无遮盖表征的协方差。
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
from micoformer.models import LossResult


def _weighted_loss(output: Mapping[str, LossResult], lambdas: Mapping[str, float]) -> Tensor:
    """Combine active objectives in their declared order.
    按声明顺序组合所有启用目标，保留原系数与浮点相加顺序。
    """
    objectives = iter(output.items())
    name, result = next(objectives)
    weight = float(lambdas[name])
    total = result.loss if weight == 1.0 else weight * result.loss
    for name, result in objectives:
        total = total + float(lambdas[name]) * result.loss
    return total


class LossStatistics:
    """Accumulate losses and diagnostics by effective targets in one reduction tensor.
    各项目标统一按有效数量累计，合并成一次跨卡通信。
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: Tensor | None = None
        self.layout: tuple[tuple[str, tuple[str, ...]], ...] = ()
        self.track_total = False

    def update(self, output: Mapping[str, LossResult], total: Tensor | None = None) -> None:
        if not output:
            return
        values = []
        for result in output.values():
            values.extend([float(result.loss.detach()) * result.count, float(result.count)])
            values.extend(float(value.detach()) * result.count for value in result.diagnostics.values())
        if total is not None:
            values.extend([float(total.detach()), 1.0])
        batch = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.values is None:
            self.layout = tuple((name, tuple(result.diagnostics)) for name, result in output.items())
            self.track_total = total is not None
            self.values = torch.zeros_like(batch)
        self.values += batch

    def synchronize(self) -> None:
        if self.values is not None:
            dist.all_reduce(self.values, op=dist.ReduceOp.SUM)

    def summary(self, lambdas: Mapping[str, float] | None = None) -> dict[str, Any]:
        if self.values is None:
            return {}
        array = self.values.detach().cpu().numpy()
        objectives = {}
        index = 0
        for name, diagnostic_names in self.layout:
            count = float(array[index + 1])
            if count <= 0:
                raise RuntimeError(f"no effective targets for objective {name}")
            loss = float(array[index] / count)
            index += 2
            diagnostics = {
                key: float(array[index + offset] / count)
                for offset, key in enumerate(diagnostic_names)
            }
            index += len(diagnostic_names)
            # Distribution-matching diagnostics obey CE = entropy + KL.
            # 分布匹配诊断在汇总后计算 CE、熵与 KL 的恒等式余量。
            if "teacher_entropy" in diagnostics and "kl" in diagnostics:
                diagnostics["ce_minus_entropy_minus_kl"] = (
                    loss - diagnostics["teacher_entropy"] - diagnostics["kl"]
                )
            objectives[name] = {"loss": loss, "count": int(count), "diagnostics": diagnostics}
        result = {"objectives": objectives}
        if self.track_total:
            weighted = iter(float(lambdas[name]) * item["loss"] for name, item in objectives.items())
            total = next(weighted)
            for value in weighted:
                total += value
            result["weighted_total"] = total
            result["mean_batch_weighted_total"] = float(array[index] / array[index + 1])
        return result


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
    """Validate without updates, including distributed-sampler padding repeats.
    不更新参数地运行验证，统计包含分布式采样器补齐的重复条目。
    """
    ddp.eval()
    statistics = LossStatistics(device)
    clean_statistics = LossStatistics(device)
    representation_count = torch.zeros((), device=device, dtype=torch.float64)
    # Accumulate moments without storing all sample embeddings.
    # 累计一阶、二阶矩，不保存全部样本 embedding。
    dimension = ddp.module.config.d_model
    sum_h = torch.zeros(dimension, device=device, dtype=torch.float64)
    sum_outer = torch.zeros((dimension, dimension), device=device, dtype=torch.float64)
    for batch in loader:
        arguments = batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = ddp(**arguments)
            _, mean = ddp.module.unmasked_representations(
                arguments["genus_ids"], arguments["abundance"], arguments["padding_mask"]
            )
        statistics.update(output, _weighted_loss(output, lambdas))
        # Keep unmasked diagnostics separate from the Query-view training loss.
        # 无遮盖视图诊断与 Query 视图训练目标分开汇总。
        clean_statistics.update(ddp.module.sample_objectives(mean, arguments.get("edge_mass")))
        mean64 = mean.float().double()
        representation_count += mean64.shape[0]
        sum_h += mean64.sum(0)
        sum_outer += mean64.T @ mean64

    statistics.synchronize()
    for tensor in (representation_count, sum_h, sum_outer):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    clean_statistics.synchronize()
    # Finalize on every rank before rank 0 builds and broadcasts its report.
    # 各 rank 先完成有效计数处理，再由主进程生成并广播报告。
    loss_report = statistics.summary(lambdas)
    clean_report = clean_statistics.summary()
    if rank == 0:
        payload = {
            "loss": loss_report,
            "clean_objectives": clean_report.get("objectives", {}),
            "representation": _spectral_summary(
                int(representation_count.item()), sum_h.cpu(), sum_outer.cpu()
            ),
            "row_count": int(representation_count.item()),
            "row_unique_count": len(loader.dataset),
        }
    else:
        payload = None
    objects = [payload]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]
