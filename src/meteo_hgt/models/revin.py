"""Reversible Instance Normalization (RevIN).

Computes per-(batch, instance, feature) mean/std on the context window and applies
the inverse on the model output. Mask-aware so missing entries don't bias the stats.

Kim et al., ICLR 2022.
"""

from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.weight = None
            self.bias = None

    def fit(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-instance mean/std along the time axis.

        Args:
            x:    (..., T, F) values; positions where mask is False are ignored.
            mask: (..., T, F) bool, True where ``x`` is finite.

        Returns:
            mean, std with shape (..., 1, F)
        """
        m = mask.to(x.dtype)
        count = m.sum(dim=-2, keepdim=True).clamp(min=1.0)
        mean = (x * m).sum(dim=-2, keepdim=True) / count
        var = ((x - mean) ** 2 * m).sum(dim=-2, keepdim=True) / count
        std = torch.sqrt(var + self.eps)
        return mean, std

    def normalize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        out = (x - mean) / std
        if self.weight is not None:
            out = out * self.weight + self.bias
        return out

    def denormalize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        if self.weight is not None:
            x = (x - self.bias) / self.weight
        return x * std + mean
