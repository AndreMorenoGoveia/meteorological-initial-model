"""Observer-time windowing.

For each batch element we pick an observer time ``t_phi`` and slice each instance
trajectory into a context window ``[t_phi - T_c, t_phi)`` and a forecast window
``[t_phi, t_phi + T_f]``.

Observer times are sampled on a fixed grid within each partition's valid range
(stride configurable). This keeps val/test reproducible across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObserverWindow:
    t_phi: int           # unix seconds
    t_ctx_lo: int
    t_ctx_hi: int        # exclusive
    t_fcst_lo: int       # equals t_phi
    t_fcst_hi: int       # exclusive


def build_observer_times(
    partition_start_unix: int,
    partition_end_unix: int,
    context_hours: int,
    forecast_hours: int,
    stride_hours: int,
    dataset_t_lo_unix: int,
    dataset_t_hi_unix: int,
) -> list[ObserverWindow]:
    """Enumerate valid observer times in ``[partition_start, partition_end)``.

    A ``t_phi`` is valid only if both context and forecast windows fit within both the
    partition and the underlying dataset coverage.
    """
    H = 3600
    ctx_s = context_hours * H
    fcst_s = forecast_hours * H
    stride_s = stride_hours * H

    t_min = max(partition_start_unix + ctx_s, dataset_t_lo_unix + ctx_s)
    t_max = min(partition_end_unix - fcst_s, dataset_t_hi_unix - fcst_s)

    if t_min >= t_max:
        return []

    # Snap to the stride grid (relative to the partition start).
    first = partition_start_unix + ((t_min - partition_start_unix + stride_s - 1) // stride_s) * stride_s
    times = np.arange(first, t_max + 1, stride_s, dtype=np.int64)

    return [
        ObserverWindow(
            t_phi=int(t),
            t_ctx_lo=int(t) - ctx_s,
            t_ctx_hi=int(t),
            t_fcst_lo=int(t),
            t_fcst_hi=int(t) + fcst_s,
        )
        for t in times
    ]
