"""Stream unmasked sample embeddings to disk in the requested row order.
按指定行号顺序，将无遮盖样本 embedding 流式保存到磁盘。
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import h5py
import numpy as np
import torch
from micoformer.workflows.checkpoints import load_model
from micoformer.data.loaders import TrainingData
from micoformer.utils.io import atomic_json, sha256_file


def extract_representations(
    *,
    checkpoint: Path,
    corpus: Path,
    rows_path: Path,
    output: Path,
    device: str = "cpu",
    batch_size: int = 128,
    workers: int = 0,
    precision: str = "auto",
) -> Path:
    """Load a checkpoint and export embeddings [N, D], row IDs [N], and a manifest.
    加载 checkpoint，导出 embedding [N, D]、行号 [N] 及来源记录。
    """
    output = Path(output).resolve()
    # Extraction writes a new result directory instead of overwriting an existing one.
    # 提取结果写入新目录，不覆盖已有结果。
    if output.exists():
        raise FileExistsError(f"refusing to overwrite extraction output: {output}")
    # Resolve the device and select FP32 or CUDA BF16 inference.
    # 确定设备，并选择 FP32 或 CUDA BF16 推理。
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
    if precision not in ("auto", "float32", "bf16"):
        raise ValueError("precision must be auto, float32 or bf16")
    use_bf16 = precision == "bf16" or (precision == "auto" and target.type == "cuda")
    if use_bf16 and target.type != "cuda":
        raise ValueError("BF16 extraction is supported only on CUDA in this release")
    if use_bf16:
        with torch.cuda.device(target):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("selected CUDA device does not support BF16")
    # The checkpoint contains model parameters, the prior table, and genus names.
    # checkpoint 包含模型参数、先验表和菌属名称。
    model = load_model(
        checkpoint,
        device=target,
    )

    # Match genus names/order before interpreting corpus columns as token identities.
    # 解释语料列号为菌属身份前，核对菌属名称与顺序。
    with h5py.File(corpus, "r") as source:
        names = tuple(source["var/_index"].asstr()[:])
    if names != model.vocabulary:
        raise ValueError("input genus columns do not match the checkpoint vocabulary")
    # Read corpus row IDs; output follows this exact order.
    # 读取语料行号，输出严格沿用这个顺序。
    rows = np.load(rows_path, allow_pickle=False)
    # Use ordered batches with padding, without MLM/Query corruption.
    # 按顺序组批并补齐，不施加 MLM/Query 扰动。
    loader = TrainingData.prediction_loader(
        corpus, rows, max_seq_len=model.config.max_seq_len, batch_size=batch_size, workers=workers
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Stage the files beside the destination; remove temporary contents on exit.
        # 先在目标旁的临时目录中写入，退出时清理临时内容。
        with tempfile.TemporaryDirectory(
            prefix=".micoformer-extract-", dir=output.parent
        ) as scratch:
            stage = Path(scratch)
            # Write a float32 [N, D] array without collecting every embedding in RAM.
            # 按批写入 float32 数组 [N, D]，不在内存中积累全部 embedding。
            embeddings = np.lib.format.open_memmap(
                stage / "embeddings.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), model.config.d_model),
            )
            position = 0
            with torch.inference_mode():
                for batch in loader:
                    args = {
                        key: torch.as_tensor(batch[key], device=target)
                        for key in ("genus_ids", "abundance", "padding_mask")
                    }
                    with torch.autocast(
                        device_type=target.type, dtype=torch.bfloat16, enabled=use_bf16
                    ):
                        # Mean-pool valid tokens to one vector per sample [B, D].
                        # 对有效 token 求均值，得到每个样本一个向量 [B, D]。
                        _, mean = model.unmasked_representations(**args)
                    array = mean.float().cpu().numpy()
                    if not np.isfinite(array).all():
                        raise RuntimeError("nonfinite sample embeddings")
                    end = position + array.shape[0]
                    embeddings[position:end] = array
                    position = end
            # Flush the mapped array before hashing and moving the generated files.
            # 先刷写映射数组，再计算指纹并移动生成文件。
            embeddings.flush()
            del embeddings
            np.save(stage / "rows.npy", rows, allow_pickle=False)
            # Record source paths, hashes, and the actual inference precision.
            # 记录来源路径、指纹与实际推理精度。
            metadata = {
                "representation": "unmasked_valid_token_mean",
                "model_config": model.config.to_dict(),
                "rows": len(rows),
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "corpus": str(Path(corpus).resolve()),
                "corpus_size": Path(corpus).stat().st_size,
                "rows_sha256": sha256_file(rows_path),
                "embedding_sha256": sha256_file(stage / "embeddings.npy"),
                "device": str(target),
                "precision": "bf16" if use_bf16 else "float32",
            }
            atomic_json(stage / "manifest.json", metadata)
            # Create the destination exclusively, then move the three completed files into it.
            # 独占创建目标目录，再逐个移入三个已写好的文件。
            output.mkdir()
            for name in ("embeddings.npy", "rows.npy", "manifest.json"):
                os.replace(stage / name, output / name)
    finally:
        # Close this process's corpus file handle on both success and failure.
        # 正常或异常退出时，都关闭当前进程的语料文件句柄。
        loader.dataset.close()
    return output
