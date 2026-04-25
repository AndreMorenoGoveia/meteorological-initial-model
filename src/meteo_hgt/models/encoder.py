"""Per-type recurrent encoder.

Consumes the (RevIN-normalized) context features stacked with optional spatiotemporal
positional embeddings. Returns the final hidden state per instance — the per-instance
"summary" embedding used as the seed of stage 2 (graph) and stage 3 (decoder).
"""

from __future__ import annotations

import torch
from torch import nn


class PerTypeEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, T_c, in_dim) -> (B*N, hidden_dim)."""
        _, h = self.gru(x)
        return h[-1]
