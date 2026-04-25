"""Index-of-agreement loss (Willmott 1981).

IoA = 1 - Σ (y - ŷ)^2 / Σ (|ŷ - ȳ| + |y - ȳ|)^2

Loss = 1 - IoA, computed per (instance, feature) over the forecast time axis,
then averaged within type and finally across target types — matching §3.4 of the
paper / spec, so that dense gridded sources do not dominate purely by count.
"""

from __future__ import annotations

import torch


def _per_instance_ioa(
    y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """y_pred/y_true: (B, N, T, F). mask: same shape, True where valid.

    Returns (B, N, F) with the IoA per (instance, feature). Instances/features whose
    mask is all-False contribute nan; the caller is responsible for filtering them.
    """
    m = mask.to(y_pred.dtype)
    count = m.sum(dim=2, keepdim=True).clamp(min=1.0)
    y_bar = (y_true * m).sum(dim=2, keepdim=True) / count

    num = (((y_true - y_pred) ** 2) * m).sum(dim=2)
    denom = (((y_pred - y_bar).abs() + (y_true - y_bar).abs()) ** 2 * m).sum(dim=2)
    ioa = 1.0 - num / (denom + eps)
    # If a series had no valid observations, mask its IoA contribution to NaN.
    has_obs = m.sum(dim=2) > 0
    ioa = torch.where(has_obs, ioa, torch.full_like(ioa, float("nan")))
    return ioa


def ioa_complement_loss(
    pred_by_type: dict, target_by_type: dict, mask_by_type: dict
) -> torch.Tensor:
    """Compute average (1 - IoA) loss aggregated as in the spec.

    Each dict is keyed by InstanceType and maps to tensors:
    - pred / target: (B, N, T, F)
    - mask:          (B, N, T, F) bool
    """
    per_type_losses = []
    for t, y_pred in pred_by_type.items():
        y_true = target_by_type[t]
        m = mask_by_type[t]
        ioa = _per_instance_ioa(y_pred, y_true, m)            # (B, N, F)
        # Drop NaN entries (no observations) when averaging.
        finite = torch.isfinite(ioa)
        if not finite.any():
            continue
        loss_per = 1.0 - ioa
        loss_per = torch.where(finite, loss_per, torch.zeros_like(loss_per))
        denom = finite.to(loss_per.dtype).sum().clamp(min=1.0)
        per_type_losses.append(loss_per.sum() / denom)

    if not per_type_losses:
        # Should not happen in normal training, but keep the loss differentiable.
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(per_type_losses).mean()
