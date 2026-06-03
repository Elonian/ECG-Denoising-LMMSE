from __future__ import annotations

import torch
import torch.nn as nn

from modules.positional_encoding import SinusoidalPositionalEncoding
from modules.transformer_blocks import TransformerEncoderStack


class ECGTransformerDenoiser(nn.Module):
    def __init__(
        self,
        window_size: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.scalar_projection = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = SinusoidalPositionalEncoding(d_model=d_model, max_length=window_size + 1)
        self.encoder = TransformerEncoderStack(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        windows: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if windows.ndim != 2:
            raise ValueError(f"Expected windows with shape [batch, P], got {tuple(windows.shape)}")
        if windows.size(1) != self.window_size:
            raise ValueError(f"Expected window size {self.window_size}, got {windows.size(1)}")
        tokens = self.scalar_projection(windows.unsqueeze(-1))
        cls = self.cls_token.expand(windows.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.position(tokens)
        encoded, attentions = self.encoder(tokens, return_attention=return_attention)
        prediction = self.regression_head(encoded[:, 0, :]).squeeze(-1)
        if return_attention:
            return prediction, attentions
        return prediction

    def config_dict(self) -> dict[str, int | float]:
        first_layer = self.encoder.layers[0]
        return {
            "window_size": self.window_size,
            "d_model": self.d_model,
            "nhead": first_layer.self_attention.num_heads,
            "num_layers": len(self.encoder.layers),
            "dim_feedforward": first_layer.feedforward[0].out_features,
            "dropout": first_layer.dropout.p,
        }

