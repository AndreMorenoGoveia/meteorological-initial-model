"""Missing Completely At Random (MCAR) augmentations.

Two flavors mirror the spec:
- ``mcar_instances``: drop a random fraction of instance IDs across the batch
- ``mcar_timestamps``: drop a random fraction of timesteps within each sequence

Both ops set the corresponding entries of ``valid_ctx`` to False (and zero the
features), so downstream RevIN / loss masking handle the holes naturally.

Uses the global torch RNG; seed it via ``meteo_hgt.utils.seed.set_seed``.
"""

from __future__ import annotations

import torch


def _sample_rate(lo: float, hi: float) -> float:
    if hi <= 0.0:
        return 0.0
    return float(torch.empty(()).uniform_(lo, hi).item())


def mcar_instances(
    features: torch.Tensor,    # (B, N, T, F)
    valid: torch.Tensor,       # (B, N, T, F)
    rate_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    rate = _sample_rate(*rate_range)
    if rate <= 0.0:
        return features, valid
    B, N = features.shape[:2]
    keep_prob = 1.0 - rate
    keep = torch.rand((B, N), device=features.device) < keep_prob   # (B, N)
    keep_full = keep[:, :, None, None]
    return features * keep_full, valid & keep_full


def mcar_timestamps(
    features: torch.Tensor,    # (B, N, T, F)
    valid: torch.Tensor,       # (B, N, T, F)
    rate_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(batch, instance) drop rate sampled independently."""
    if rate_range[1] <= 0.0:
        return features, valid
    B, N, T, _ = features.shape
    lo, hi = rate_range
    rates = torch.empty((B, N, 1), device=features.device).uniform_(lo, hi)
    keep_prob = 1.0 - rates
    keep = torch.rand((B, N, T), device=features.device) < keep_prob   # (B, N, T)
    keep_full = keep[..., None]
    return features * keep_full, valid & keep_full
