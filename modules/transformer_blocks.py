from __future__ import annotations

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_output, attention_weights = self.self_attention(
            values,
            values,
            values,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        values = self.norm_attention(values + self.dropout(attention_output))
        feedforward_output = self.feedforward(values)
        values = self.norm_feedforward(values + self.dropout(feedforward_output))
        return values, attention_weights if return_attention else None


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, values: torch.Tensor, return_attention: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attentions: list[torch.Tensor] = []
        for layer in self.layers:
            values, attention = layer(values, return_attention=return_attention)
            if attention is not None:
                attentions.append(attention)
        return values, attentions

