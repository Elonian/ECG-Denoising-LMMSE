from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        length = values.size(1)
        if length > self.encoding.size(1):
            raise ValueError(f"Sequence length {length} exceeds max positional length {self.encoding.size(1)}")
        return values + self.encoding[:, :length, :]

