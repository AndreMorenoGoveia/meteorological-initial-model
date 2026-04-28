#!/usr/bin/env python
"""Generate diagnostic figures from a trained checkpoint.

Produces four PNGs in ``<run_output>/plots/``:
  1. learning_curves.png       - train / val loss + IoA across epochs
  2. metrics_by_variable.png   - MAE / RMSE / IoA / corr per variable, per type
  3. error_by_leadtime.png     - MAE per forecast hour, per variable, per type
  4. timeseries_examples.png   - context + forecast + prediction at IAG, on
                                 the best / median / worst observer windows.

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
from meteo_hgt.training.metrics import (
    metrics_per_variable,
    per_leadtime_error,
)
from meteo_hgt.training.losses import _per_instance_ioa
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


# ---- Figure 1: learning curves ----------------------------------------------

def fig_learning_curves(run: CollectedRun, out: Path) -> None:
    h = run.history
    if h is None or not h.get("train_loss"):
        # Still produce an empty figure so the user knows the file lives here.
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "history.json not found in run directory",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out, bbox_inches="tight", dpi=140)
        plt.close(fig)
        return

    epochs = np.arange(len(h["train_loss"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, h["train_loss"], label="train", linewidth=1.6)
    axes[0].plot(epochs, h["val_loss"], label="val", linewidth=1.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("1 - IoA  (lower is better)")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Per-type IoA over epochs.
    type_names = sorted({k for d in h["val_metrics"] for k in d.keys()})
    for tn in type_names:
        ioa = [d.get(tn, {}).get("ioa", float("nan")) for d in h["val_metrics"]]
        axes[1].plot(epochs, ioa, label=tn, linewidth=1.6)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("IoA  (higher is better, 1 = perfect)")
    axes[1].set_title("Validation IoA per target type")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Learning curves", y=1.02, fontsize=12)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---- Figure 2: metrics per variable -----------------------------------------

def fig_metrics_by_variable(run: CollectedRun, out: Path) -> None:
    metric_list = ["mae", "rmse", "ioa", "corr"]
    type_names = [t.value for t in run.target_types if t in run.per_type]

    fig, axes = plt.subplots(1, len(metric_list), figsize=(4 * len(metric_list), 4), sharey=False)
    width = 0.36
    x = np.arange(len(run.feature_names))

    for ax, mname in zip(axes, metric_list):
        for i, t in enumerate(run.target_types):
            if t not in run.per_type:
                continue
            d = run.per_type[t]
            per_var = metrics_per_variable(
                torch.from_numpy(d["pred"]),
                torch.from_numpy(np.nan_to_num(d["target"], nan=0.0)),
                torch.from_numpy(d["mask"]),
                run.feature_names,
                [mname],
            )
            vals = [per_var[v][mname] for v in run.feature_names]
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=t.value)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(v, v) for v in run.feature_names], rotation=20, ha="right")
        ax.set_title(mname.upper())
        ax.grid(alpha=0.25, axis="y")
        if mname in ("mae", "rmse"):
            # show units of the first variable as a hint (not strictly homogeneous)
            ax.set_ylabel("error  (variable units)")
        else:
            ax.set_ylabel(mname)
    axes[0].legend(title="target type")
    fig.suptitle("Held-out metrics, per variable, per target type", y=1.04, fontsize=12)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---- Figure 3: error by lead time -------------------------------------------

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
        mae = per_leadtime_error(
            torch.from_numpy(d["pred"]),
            torch.from_numpy(np.nan_to_num(d["target"], nan=0.0)),
            torch.from_numpy(d["mask"]),
            kind="mae",
        ).numpy()                                        # (T_f, F)
        T_f = mae.shape[0]
        hours = np.arange(1, T_f + 1)                    # leads 1..T_f hours
        ax = axes[ax_i]; ax_i += 1
        for fi, name in enumerate(run.feature_names):
            ax.plot(hours, mae[:, fi], marker="o", markersize=3, label=PRETTY.get(name, name))
        ax.set_title(f"{t.value} — MAE vs forecast lead time")
        ax.set_xlabel("forecast hour")
        ax.set_ylabel("MAE  (variable units)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("How error grows over the 24h forecast window", y=1.04, fontsize=12)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)


# ---- Figure 4: time-series examples at IAG ----------------------------------

def _window_quality(d: dict) -> np.ndarray:
    """Return per-window mean IoA (over instances and features).

    NaN where a window has no valid forecast obs at all.
    """
    pred = torch.from_numpy(d["pred"])
    target = torch.from_numpy(np.nan_to_num(d["target"], nan=0.0))
    mask = torch.from_numpy(d["mask"])
    ioa = _per_instance_ioa(pred, target, mask)          # (W, N, F)
    finite = torch.isfinite(ioa)
    ioa_z = torch.where(finite, ioa, torch.zeros_like(ioa))
    denom = finite.float().sum(dim=(1, 2)).clamp(min=1.0)
    return (ioa_z.sum(dim=(1, 2)) / denom).numpy()       # (W,)


def fig_timeseries_examples(run: CollectedRun, out: Path, n_examples: int = 3) -> None:
    if InstanceType.IAG not in run.per_type:
        # Fallback: plot first available type
        target_t = next(iter(run.per_type))
        n_idx = 0
    else:
        target_t = InstanceType.IAG
        n_idx = 0
    d = run.per_type[target_t]

    qual = _window_quality(d)
    valid = np.isfinite(qual)
    if not valid.any():
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "no validation windows had observed targets",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out, bbox_inches="tight", dpi=140)
        plt.close(fig)
        return

    valid_idx = np.flatnonzero(valid)
    sorted_idx = valid_idx[np.argsort(qual[valid_idx])]
    if len(sorted_idx) >= n_examples:
        chosen = [sorted_idx[-1], sorted_idx[len(sorted_idx) // 2], sorted_idx[0]]
        labels = ["best", "median", "worst"]
    else:
        chosen = list(sorted_idx[: n_examples])
        labels = [f"sample {i}" for i in range(len(chosen))]

    F = len(run.feature_names)
    fig, axes = plt.subplots(F, len(chosen), figsize=(4.6 * len(chosen), 2.6 * F), sharex="col")
    if F == 1:
        axes = axes[None, :]
    if len(chosen) == 1:
        axes = axes[:, None]

    for col, (w, lab) in enumerate(zip(chosen, labels)):
        t_ctx = _to_dt(d["times_ctx"][w])
        t_fcst = _to_dt(d["times_fcst"][w])
        t_phi_str = datetime.fromtimestamp(int(d["t_phi"][w]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
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
                ax.set_title(f"{lab}  (IoA={qual[w]:.2f})\nt_phi={t_phi_str}", fontsize=9)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(20)
                lbl.set_ha("right")

    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Sample forecasts at {target_t.value} (instance {d['instance_ids'][n_idx]})",
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

    fig_learning_curves(run, out_dir / "learning_curves.png")
    fig_metrics_by_variable(run, out_dir / "metrics_by_variable.png")
    fig_error_by_leadtime(run, out_dir / "error_by_leadtime.png")
    fig_timeseries_examples(run, out_dir / "timeseries_examples.png")

    print(f"wrote 4 figures to {out_dir}/")


if __name__ == "__main__":
    main()
