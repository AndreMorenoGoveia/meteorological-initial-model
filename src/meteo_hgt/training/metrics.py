"""Forecast quality metrics, mask-aware.

All metrics expect tensors shaped (B, N, T, F) and a boolean mask of the same shape.
Each function returns a scalar (averaged over the valid entries).
"""

from __future__ import annotations

from typing import Callable

import torch

from .losses import _per_instance_ioa


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    denom = m.sum().clamp(min=1.0)
    return (x * m).sum() / denom


def mae(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean((y_pred - y_true).abs(), mask)


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_masked_mean((y_pred - y_true) ** 2, mask))


def corr(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pearson correlation over the masked entries (pooled across all dims)."""
    m = mask.to(y_pred.dtype)
    denom = m.sum().clamp(min=1.0)
    mp = (y_pred * m).sum() / denom
    mt = (y_true * m).sum() / denom
    cov = ((y_pred - mp) * (y_true - mt) * m).sum() / denom
    sp = torch.sqrt(((y_pred - mp) ** 2 * m).sum() / denom + 1e-12)
    st = torch.sqrt(((y_true - mt) ** 2 * m).sum() / denom + 1e-12)
    return cov / (sp * st)


def ioa(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = _per_instance_ioa(y_pred, y_true, mask)         # (B, N, F)
    finite = torch.isfinite(per)
    if not finite.any():
        return torch.tensor(float("nan"))
    return per[finite].mean()


_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "ioa": ioa,
    "mae": mae,
    "rmse": rmse,
    "corr": corr,
}


def compute_metrics(
    pred_by_type: dict,
    target_by_type: dict,
    mask_by_type: dict,
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """Returns ``{type_name: {metric_name: scalar}}``."""
    out: dict[str, dict[str, float]] = {}
    for t, y_pred in pred_by_type.items():
        y_true = target_by_type[t]
        m = mask_by_type[t]
        out[t.value] = {}
        for name in metrics:
            fn = _REGISTRY[name]
            out[t.value][name] = float(fn(y_pred, y_true, m).item())
    return out


def metrics_per_variable(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    feature_names: list[str],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """Per-feature metrics. Returns ``{feature_name: {metric_name: scalar}}``."""
    out: dict[str, dict[str, float]] = {}
    for fi, name in enumerate(feature_names):
        p = y_pred[..., fi : fi + 1]
        t = y_true[..., fi : fi + 1]
        m = mask[..., fi : fi + 1]
        out[name] = {}
        for mname in metrics:
            fn = _REGISTRY[mname]
            out[name][mname] = float(fn(p, t, m).item())
    return out


def per_leadtime_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "mae",
) -> torch.Tensor:
    """Returns shape (T_f, F) — error per forecast step and per variable.

    ``y_pred`` etc. have shape (W, N, T_f, F). Reduction is over W and N (the
    instances and the windows), keeping T_f and F.
    """
    if kind not in ("mae", "rmse"):
        raise ValueError(kind)
    m = mask.to(y_pred.dtype)
    diff = (y_pred - y_true) * m
    if kind == "mae":
        num = diff.abs().sum(dim=(0, 1))           # (T_f, F)
    else:
        num = (diff ** 2).sum(dim=(0, 1))
    denom = m.sum(dim=(0, 1)).clamp(min=1.0)
    val = num / denom
    return val.sqrt() if kind == "rmse" else val
