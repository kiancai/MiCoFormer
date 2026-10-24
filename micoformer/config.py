from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, Mapping

REFERENCE_LAMBDAS = {
    # fixed：Encoder 的菌属身份输入取自固定 PHY32 + PROT480 表。
    "fixed": {
        "mlm": 1.0,
        "continuous": 0.534399128791011,
        "presence": 0.24296011504612833,
    },
    # learned：Encoder 的菌属身份输入使用可训练的 embedding。
    # 当前实现的 Query 候选端仍使用固定先验，不代表完全取消先验。
    "learned": {
        "mlm": 1.0,
        "continuous": 1.099683411444247,
        "presence": 0.3055661297159814,
    },
}

@dataclass(frozen=True)
class ModelConfig:
    identity_type: Literal["fixed", "learned"] = "fixed"
    vocab_size: int = field(default=8116, init=False)
    d_model: int = field(default=512, init=False)
    abundance_hidden_dim: int = 64
    num_layers: int = 12
    encoder_heads: int = 16
    encoder_ffn_dim: int = 2048
    mlm_head_hidden_dim: int = 64
    dropout: float = 0.1
    max_seq_len: int = field(default=512, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_checkpoint_dict(cls, payload: Mapping[str, Any]) -> ModelConfig:
        # Fixed fields are recorded but cannot be constructor arguments.
        # 固定字段仍保存在权重记录中，恢复时核对后不传给构造函数。
        for item in fields(cls):
            if not item.init and payload[item.name] != item.default:
                raise ValueError(f"checkpoint {item.name} does not match the model structure")
        return cls(**{item.name: payload[item.name] for item in fields(cls) if item.init})


@dataclass(frozen=True)
class TrainConfig:
    # Per-GPU batch size supplied by train.py. / 由 train.py 提供的每卡批大小。
    batch_size_per_rank: int
    # CPU data-loading subprocesses per training process (one process per GPU).
    # 每个训练进程的 CPU 数据加载子进程数；每张 GPU 对应一个训练进程。
    # Zero loads data in the training process; this is not a CPU core count.
    # 0 表示由训练进程自己加载数据；这个数不代表 CPU 核数。
    workers_per_rank: int = 2
    seed: int = 42
    # Group similar input lengths to reduce padding; False keeps random batches.
    # 按相近输入长度组批以减少 padding；设为 False 保留随机组批。
    length_bucketing: bool = True
    # Per-token selection probabilities for the MLM and Query tasks.
    # MLM 丰度遮盖与 Query 菌属选择的逐 token 抽样概率。
    mlm_mask_probability: float = 0.15
    query_mask_probability: float = 0.10
    # Weak neighborhood supervision by default; 0 skips tree targets and computation.
    # 默认弱邻域监督；设为 0 时跳过树目标准备和相关计算。
    unifrac_weight: float = 0.00005
    unifrac_teacher_temperature: float = 0.03
    unifrac_student_temperature: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    max_epochs: int = 50
    min_epochs: int = 10
    warmup_ratio: float = 0.02
    minimum_lr: float = 1e-6
    early_stopping_patience: int = 5
    relative_min_improvement: float = 0.001


@dataclass(frozen=True)
class DataConfig:
    corpus: Path
    train_rows: Path
    val_rows: Path
    preparation_manifest: Path


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    output_root: Path
    training: TrainConfig
    model: ModelConfig

    # Return a copy with absolute paths. / 返回路径转为绝对路径的副本。
    def normalized(self) -> RunConfig:
        return replace(
            self,
            data=replace(
                self.data,
                corpus=Path(self.data.corpus).resolve(),
                train_rows=Path(self.data.train_rows).resolve(),
                val_rows=Path(self.data.val_rows).resolve(),
                preparation_manifest=Path(self.data.preparation_manifest).resolve(),
            ),
            output_root=Path(self.output_root).resolve(),
        )

    # Convert configs and paths for JSON serialization. / 将配置和路径转换为可保存到 JSON 的形式。
    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["output_root"] = str(value["output_root"])
        for key in ("corpus", "train_rows", "val_rows", "preparation_manifest"):
            value["data"][key] = str(value["data"][key])
        return value
