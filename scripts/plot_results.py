#!/usr/bin/env python
"""Generate diagnostic figures from a trained checkpoint.

Produces two PNGs in ``<run_output>/plots/``:
  1. error_by_leadtime.png    - IoA per forecast hour, per variable, per type
  2. timeseries_examples.png  - context + forecast + prediction at IAG, sampling
                                windows spread across the full validation period.

Usage:
    python3 scripts/plot_results.py \
        --config configs/variant3_hgt.yaml \
        --checkpoint runs/hgt/best.pt \
        --partition val
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch

from meteo_hgt.data.unified import InstanceType
from meteo_hgt.training.losses import _per_instance_ioa
from meteo_hgt.training.metrics import per_leadtime_ioa
from meteo_hgt.viz import CollectedRun, collect_predictions


# ---- units / display ---------------------------------------------------------

UNITS: dict[str, str] = {
    "air_temperature_c": "°C",
    "dew_point_temperature_c": "°C",
    "wind_u_ms": "m/s",
    "wind_v_ms": "m/s",
    "wind_speed_ms": "m/s",
    "wind_direction_deg": "°",
}

PRETTY: dict[str, str] = {
    "air_temperature_c": "Air temp.",
    "dew_point_temperature_c": "Dew point",
    "wind_u_ms": "Wind u",
    "wind_v_ms": "Wind v",
    "wind_speed_ms": "Wind speed",
    "wind_direction_deg": "Wind dir.",
}


def _label(v: str) -> str:
    return f"{PRETTY.get(v, v)} [{UNITS.get(v, '')}]" if v in UNITS else v


def _to_dt(ts_seconds: np.ndarray) -> np.ndarray:
    return np.array(
        [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_seconds]
    )


# ---- Figure 1: IoA by lead time ---------------------------------------------

def fig_error_by_leadtime(run: CollectedRun, out: Path) -> None:
    n_types = sum(1 for t in run.target_types if t in run.per_type)
    fig, axes = plt.subplots(1, n_types, figsize=(5.5 * n_types, 4), sharex=True)
    if n_types == 1:
        axes = [axes]

    ax_i = 0
    for t in run.target_types:
        if t not in run.per_type:
            continue
        d = run.per_type[t]
        ioa_lt = per_leadtime_ioa(
            torch.from_numpy(d["pred"]),
            torch.from_numpy(np.nan_to_num(d["target"], nan=0.0)),
            torch.from_numpy(d["mask"]),
        ).numpy()                                           # (T_f, F)
        T_f = ioa_lt.shape[0]
        hours = np.arange(1, T_f + 1)
        ax = axes[ax_i]; ax_i += 1
        for fi, name in enumerate(run.feature_names):
            ax.plot(hours, ioa_lt[:, fi], marker="o", markersize=3,
                    label=PRETTY.get(name, name))
        ax.set_title(f"{t.value} — IoA vs forecast lead time")
        ax.set_xlabel("forecast hour")
        ax.set_ylabel("IoA  (higher is better, 1 = perfect)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(1.0, color="0.7", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Forecast skill over the 24h window (Willmott IoA)", y=1.04, fontsize=12)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---- Figure 2: time-series examples at IAG ----------------------------------

def _window_ioa(d: dict) -> np.ndarray:
    """Return per-window mean IoA (W,). NaN for windows with no valid obs."""
    pred = torch.from_numpy(d["pred"])
    target = torch.from_numpy(np.nan_to_num(d["target"], nan=0.0))
    mask = torch.from_numpy(d["mask"])
    ioa = _per_instance_ioa(pred, target, mask)             # (W, N, F)
    finite = torch.isfinite(ioa)
    ioa_z = torch.where(finite, ioa, torch.zeros_like(ioa))
    denom = finite.float().sum(dim=(1, 2)).clamp(min=1.0)
    return (ioa_z.sum(dim=(1, 2)) / denom).numpy()          # (W,)


def fig_timeseries_examples(run: CollectedRun, out: Path, n_examples: int = 3) -> None:
    if InstanceType.IAG not in run.per_type:
        target_t = next(iter(run.per_type))
        n_idx = 0
    else:
        target_t = InstanceType.IAG
        n_idx = 0
    d = run.per_type[target_t]

    qual = _window_ioa(d)
    valid = np.isfinite(qual)
    if not valid.any():
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "no validation windows had observed targets",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out, bbox_inches="tight", dpi=140)
        plt.close(fig)
        return

    # Select windows evenly spread across the validation period.
    valid_idx = np.flatnonzero(valid)
    temporal_order = np.argsort(d["t_phi"][valid_idx])
    temporal_idx = valid_idx[temporal_order]

    if len(temporal_idx) >= n_examples:
        picks = np.round(np.linspace(0, len(temporal_idx) - 1, n_examples)).astype(int)
        chosen = [temporal_idx[p] for p in picks]
    else:
        chosen = list(temporal_idx)

    F = len(run.feature_names)
    fig, axes = plt.subplots(F, len(chosen), figsize=(4.6 * len(chosen), 2.6 * F), sharex="col")
    if F == 1:
        axes = axes[None, :]
    if len(chosen) == 1:
        axes = axes[:, None]

    for col, w in enumerate(chosen):
        t_ctx = _to_dt(d["times_ctx"][w])
        t_fcst = _to_dt(d["times_fcst"][w])
        t_phi_dt = datetime.fromtimestamp(int(d["t_phi"][w]), tz=timezone.utc)
        date_str = t_phi_dt.strftime("%Y-%m-%d %H:%M UTC")
        for fi, name in enumerate(run.feature_names):
            ax = axes[fi, col]
            obs_ctx = d["ctx_obs"][w, n_idx, :, fi]
            obs_fcst = d["target"][w, n_idx, :, fi]
            yhat = d["pred"][w, n_idx, :, fi]
            ax.plot(t_ctx, obs_ctx, color="gray", linewidth=1.0, label="observed (history)")
            ax.plot(t_fcst, obs_fcst, color="black", linewidth=1.6, label="observed (future)")
            ax.plot(t_fcst, yhat, color="C3", linewidth=1.6, label="prediction")
            ax.axvline(t_fcst[0], color="0.5", linestyle="--", linewidth=0.8)
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel(_label(name))
            if fi == 0:
                ax.set_title(f"{date_str}\nIoA={qual[w]:.2f}", fontsize=9)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(20)
                lbl.set_ha("right")

    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Sample forecasts at {target_t.value} (instance {d['instance_ids'][n_idx]})"
        " — windows spread across validation period",
        y=1.005, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---- main -------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition", default="val", choices=["train", "val", "test"])
    p.add_argument("--out", default=None, help="Output dir; defaults to <run_dir>/plots/")
    args = p.parse_args()

    run = collect_predictions(args.config, args.checkpoint, partition=args.partition)
    out_dir = Path(args.out) if args.out else (run.output_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_error_by_leadtime(run, out_dir / "error_by_leadtime.png")
    fig_timeseries_examples(run, out_dir / "timeseries_examples.png")

    print(f"wrote 2 figures to {out_dir}/")


if __name__ == "__main__":
    main()
