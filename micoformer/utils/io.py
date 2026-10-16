"""Shared file hashing, serialization, and atomic replacement.
共用的文件指纹、序列化与原子替换操作。
"""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
import torch


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks without loading it all into memory.
    分块计算文件指纹，不将整个文件载入内存。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialize a JSON record, then replace its destination after writing.
    将记录转成 JSON，写完后替换目标文件。
    """
    # Reject NaN/Inf values, which are not standard JSON numbers.
    # 拒绝 NaN/Inf，它们不是标准 JSON 数字。
    _atomic_write(
        path,
        lambda handle: handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode()
            + b"\n"
        ),
    )


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    """Save a PyTorch payload, then replace its destination after writing.
    保存 PyTorch 对象，写完后替换目标文件。
    """
    _atomic_write(path, lambda handle: torch.save(payload, handle))


def _atomic_write(path: Path, writer) -> None:
    """Write through a temporary file in the destination directory.
    通过目标目录中的临时文件完成写入。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            name = handle.name
            writer(handle)
        # Replace after closing the temporary file; failed writes leave the old file intact.
        # 临时文件关闭后才替换目标；写入失败时保留原有内容。
        os.replace(name, path)
    finally:
        # Remove a leftover temporary file if writing or replacement failed.
        # 写入或替换失败时，清理残留的临时文件。
        if name is not None and os.path.exists(name):
            os.unlink(name)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one JSON record per line.
    以每行一条 JSON 的形式追加记录。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
