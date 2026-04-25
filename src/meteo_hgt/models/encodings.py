"""Fixed sinusoidal positional encoding for scalar coordinates."""

from __future__ import annotations

import math

import torch


def sinusoidal_encoding(values: torch.Tensor, dim: int, scale: float = 1.0) -> torch.Tensor:
    """Map a scalar tensor of shape (...,) to a sinusoidal embedding of shape (..., dim).

    ``scale`` controls the base period (larger = lower-frequency basis).
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=values.device, dtype=torch.float32) / half
    )
    angles = (values.float().unsqueeze(-1) / scale) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
