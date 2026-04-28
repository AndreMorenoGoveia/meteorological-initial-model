"""Helpers for offline visualization.

The plotting script in ``scripts/plot_results.py`` calls into this module so the
notebook-style logic (running inference, collecting per-window arrays) stays
testable and out of the script file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import collate_batch
from .data.unified import InstanceType
from .models.build import build_model
from .runners import (
    _make_dataset,
    _make_loader,
    _make_store,
    _output_dir,
    _resolve_device,
)
from .training.trainer import _build_edges, _to_inputs


@dataclass
class CollectedRun:
    """Concatenated predictions + targets for one held-out partition."""

    feature_names: list[str]
    target_types: list[InstanceType]
    per_type: dict[InstanceType, dict]                # see fields below
    history: dict | None                              # parsed history.json, if present
    output_dir: Path
    config_path: Path

    # per_type entries:
    #   pred         np.ndarray (W, N, T_f, F)  — predictions in physical units
    #   target       np.ndarray (W, N, T_f, F)
    #   mask         np.ndarray (W, N, T_f, F)  bool
    #   ctx_obs      np.ndarray (W, N, T_c, F)  — observed context values (NaN where missing)
    #   t_phi        np.ndarray (W,)           int64 unix seconds
    #   times_ctx    np.ndarray (W, T_c)        int64 unix seconds
    #   times_fcst   np.ndarray (W, T_f)        int64 unix seconds
    #   lat          np.ndarray (N,) float
    #   lon          np.ndarray (N,) float
    #   instance_ids list[str]
    #   instance_names list[str]


def _to_numpy_with_nan(features: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Restore NaN in ``features`` wherever ``mask`` is False (the dataset zeros NaNs)."""
    arr = features.numpy().astype("float32").copy()
    arr[~mask.numpy().astype(bool)] = np.nan
    return arr


@torch.no_grad()
def collect_predictions(
    config_path: str | Path,
    checkpoint_path: str | Path,
    partition: str = "val",
) -> CollectedRun:
    cfg = load_config(config_path)
    cfg_path = Path(config_path)

    store = _make_store(cfg)
    ds = _make_dataset(cfg, store, partition)
    feature_names = ds.feature_names()

    device = _resolve_device(str(cfg.training.device))
    model = build_model(cfg, feature_dim=len(feature_names)).to(device)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    relations = model.relations
    knn_cfg = dict(cfg.graph.knn)
    use_hgt = model.variant == "hgt"
    target_types = list(model.target_types)

    loader = DataLoader(
        ds, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=0, collate_fn=collate_batch,
    )

    # Buckets per type — accumulate as Python lists, concat at the end.
    buckets: dict[InstanceType, dict[str, list]] = {
        t: {k: [] for k in (
            "pred", "target", "mask", "ctx", "ctx_mask",
            "t_phi", "times_ctx", "times_fcst",
        )} for t in target_types
    }

    for batch in loader:
        per_type = batch["per_type"]
        t_phi = batch["t_phi"].to(device)
        inputs = _to_inputs(per_type, device)
        edges = (
            _build_edges(per_type, relations, knn_cfg, device) if use_hgt else None
        )
        preds = model(inputs, t_phi, edges)
        for t in target_types:
            if t not in preds:
                continue
            tt = per_type[t]
            buckets[t]["pred"].append(preds[t].detach().cpu())
            buckets[t]["target"].append(tt.features_fcst)
            buckets[t]["mask"].append(tt.valid_fcst)
            buckets[t]["ctx"].append(tt.features_ctx)
            buckets[t]["ctx_mask"].append(tt.valid_ctx)
            buckets[t]["t_phi"].append(batch["t_phi"])
            buckets[t]["times_ctx"].append(tt.times_ctx)
            buckets[t]["times_fcst"].append(tt.times_fcst)

    out_per_type: dict[InstanceType, dict] = {}
    for t in target_types:
        b = buckets[t]
        if not b["pred"]:
            continue
        pred = torch.cat(b["pred"], dim=0)            # (W, N, T_f, F)
        target = torch.cat(b["target"], dim=0)
        mask = torch.cat(b["mask"], dim=0)
        ctx = torch.cat(b["ctx"], dim=0)
        ctx_mask = torch.cat(b["ctx_mask"], dim=0)

        metas = ds.metas_for_type(t)
        out_per_type[t] = {
            "pred": pred.numpy().astype("float32"),
            "target": _to_numpy_with_nan(target, mask),
            "mask": mask.numpy().astype(bool),
            "ctx_obs": _to_numpy_with_nan(ctx, ctx_mask),
            "t_phi": torch.cat(b["t_phi"], dim=0).numpy().astype("int64"),
            "times_ctx": torch.cat(b["times_ctx"], dim=0).numpy().astype("int64"),
            "times_fcst": torch.cat(b["times_fcst"], dim=0).numpy().astype("int64"),
            "lat": np.array([m.latitude for m in metas], dtype="float32"),
            "lon": np.array([m.longitude for m in metas], dtype="float32"),
            "instance_ids": [m.instance_id for m in metas],
            "instance_names": [m.instance_name for m in metas],
        }

    out_dir = _output_dir(cfg)
    history = None
    h_path = out_dir / "history.json"
    if h_path.exists():
        import json

        history = json.loads(h_path.read_text())

    return CollectedRun(
        feature_names=feature_names,
        target_types=target_types,
        per_type=out_per_type,
        history=history,
        output_dir=out_dir,
        config_path=cfg_path,
    )
