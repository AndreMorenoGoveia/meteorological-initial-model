"""High-level entry points for training and evaluation.

These are functions, not CLI binaries — the thin scripts/ wrappers parse argv and
delegate here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import (
    MeteoDataset,
    UnifiedStore,
    build_observer_times,
    collate_batch,
)
from .data.unified import InstanceType
from .models.build import build_model
from .training.metrics import compute_metrics
from .training.trainer import Trainer
from .utils.logging import get_logger
from .utils.seed import set_seed


log = get_logger()


def _make_store(cfg) -> UnifiedStore:
    bbox = (
        float(cfg.data.era5_region.lat_min),
        float(cfg.data.era5_region.lat_max),
        float(cfg.data.era5_region.lon_min),
        float(cfg.data.era5_region.lon_max),
    )
    iag_center = (float(cfg.data.iag_center[0]), float(cfg.data.iag_center[1]))
    return UnifiedStore(
        netcdf_path=str(cfg.data.netcdf_path),
        variables=list(cfg.data.variables),
        era5_bbox=bbox,
        inmet_max_distance_km=float(cfg.data.inmet_max_distance_km),
        iag_center=iag_center,
    )


def _make_dataset(cfg, store: UnifiedStore, partition: str) -> MeteoDataset:
    start, end = cfg.data.splits[partition]
    p_start, p_end = store.partition_unix_range(start, end)
    ds_lo = int(store.timestamps[0])
    ds_hi = int(store.timestamps[-1]) + 3600  # +1 hour, exclusive
    windows = build_observer_times(
        partition_start_unix=p_start,
        partition_end_unix=p_end,
        context_hours=int(cfg.windows.context_hours),
        forecast_hours=int(cfg.windows.forecast_hours),
        stride_hours=int(cfg.windows.observer_stride_hours),
        dataset_t_lo_unix=ds_lo,
        dataset_t_hi_unix=ds_hi,
    )
    log.info("partition=%s observer_windows=%d", partition, len(windows))
    return MeteoDataset(store, windows, decompose_wind=bool(cfg.data.decompose_wind))


def _make_loader(cfg, ds: MeteoDataset, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.training.num_workers),
        collate_fn=collate_batch,
        pin_memory=False,
    )


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        log.info(
            "using device=cuda:%d  name=%s  mem=%.1f GB",
            idx, props.name, props.total_memory / 1e9,
        )
    else:
        log.info("using device=%s", device)
    return device


def _output_dir(cfg) -> Path:
    base = Path(str(cfg.training.output_dir))
    return base / cfg.model.variant


def train(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.seed))

    store = _make_store(cfg)
    counts = store.counts_by_type()
    log.info(
        "selected instances: ERA5=%d  IAG=%d  INMET=%d",
        counts[InstanceType.ERA5],
        counts[InstanceType.IAG],
        counts[InstanceType.INMET],
    )

    train_ds = _make_dataset(cfg, store, "train")
    val_ds = _make_dataset(cfg, store, "val")

    feature_dim = len(train_ds.feature_names())
    log.info("features: %s (dim=%d)", train_ds.feature_names(), feature_dim)

    device = _resolve_device(str(cfg.training.device))
    model = build_model(cfg, feature_dim=feature_dim).to(device)
    log.info("model variant=%s  params=%d", cfg.model.variant, sum(p.numel() for p in model.parameters()))

    out_dir = _output_dir(cfg)
    trainer = Trainer(model=model, cfg=cfg, device=device, output_dir=out_dir)
    trainer.fit(_make_loader(cfg, train_ds, shuffle=True), _make_loader(cfg, val_ds, shuffle=False))


def evaluate(config_path: str | Path, checkpoint_path: str | Path, partition: str = "test") -> dict:
    cfg = load_config(config_path)
    set_seed(int(cfg.seed))

    store = _make_store(cfg)
    ds = _make_dataset(cfg, store, partition)
    feature_dim = len(ds.feature_names())

    device = _resolve_device(str(cfg.training.device))
    model = build_model(cfg, feature_dim=feature_dim).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state["model_state"])

    out_dir = _output_dir(cfg)
    trainer = Trainer(model=model, cfg=cfg, device=device, output_dir=out_dir)
    loss, metrics = trainer.eval_epoch(_make_loader(cfg, ds, shuffle=False), epoch=-1, tag=partition)

    log.info("%s loss=%.4f metrics=%s", partition, loss, metrics)
    out_file = out_dir / f"eval_{partition}.json"
    with open(out_file, "w") as f:
        json.dump({"loss": loss, "metrics": metrics}, f, indent=2)
    return {"loss": loss, "metrics": metrics}
