from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AbundanceBinHead(nn.Module):

    def __init__(self, d_model: int, num_bins: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_bins)

    def forward(self, token_repr: torch.Tensor) -> torch.Tensor:
        return self.proj(token_repr)
