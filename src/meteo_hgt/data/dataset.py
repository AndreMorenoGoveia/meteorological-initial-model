"""PyTorch Dataset and collate function.

A sample is one observer time. Within each sample, we group instances by type
(ERA5/IAG/INMET) so the model can apply per-type encoders cleanly. We also expose
absolute timestamps (used to compute relative time inside the model) and lat/lon
(used both as positional features and to build the kNN graph).

Wind decomposition (dir, speed) -> (u, v) is done here when ``decompose_wind``
is set, since the loss is computed in the same space the network outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .unified import InstanceMeta, InstanceType, UnifiedStore
from .windows import ObserverWindow


_WIND_DIR = "wind_direction_deg"
_WIND_SPD = "wind_speed_ms"


def _decompose_wind(features: np.ndarray, var_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Replace (dir_deg, speed_ms) with (u_ms, v_ms) using meteorological convention.

    u = -speed * sin(dir_rad), v = -speed * cos(dir_rad)  (wind 'from' direction)
    """
    if _WIND_DIR not in var_names or _WIND_SPD not in var_names:
        return features, var_names
    di = var_names.index(_WIND_DIR)
    si = var_names.index(_WIND_SPD)
    direction = np.deg2rad(features[..., di])
    speed = features[..., si]
    u = -speed * np.sin(direction)
    v = -speed * np.cos(direction)
    out = features.copy()
    out[..., di] = u
    out[..., si] = v
    new_names = list(var_names)
    new_names[di] = "wind_u_ms"
    new_names[si] = "wind_v_ms"
    return out, new_names


@dataclass
class TypeTensors:
    """Per-type stack of instance trajectories at one observer time."""

    type: InstanceType
    features_ctx: torch.Tensor   # (N, T_c, F)  may contain NaN
    features_fcst: torch.Tensor  # (N, T_f, F)  ground-truth target
    times_ctx: torch.Tensor      # (T_c,) int64 unix seconds
    times_fcst: torch.Tensor     # (T_f,) int64 unix seconds
    lat: torch.Tensor            # (N,) float32
    lon: torch.Tensor            # (N,) float32
    valid_ctx: torch.Tensor      # (N, T_c, F) bool, True where finite
    valid_fcst: torch.Tensor     # (N, T_f, F) bool


class MeteoDataset(Dataset):
    """One sample == one observer time. Returns a dict[InstanceType -> TypeTensors]."""

    def __init__(
        self,
        store: UnifiedStore,
        observer_times: list[ObserverWindow],
        decompose_wind: bool = True,
    ):
        self.store = store
        self.observer_times = observer_times
        self.decompose_wind = decompose_wind
        self.var_names = list(store.variables)

        # Pre-group instances by type (positions in NetCDF instance dim).
        self._by_type: dict[InstanceType, list[InstanceMeta]] = {t: [] for t in InstanceType}
        for meta in store.instances:
            self._by_type[meta.type].append(meta)

    def __len__(self) -> int:
        return len(self.observer_times)

    def types_present(self) -> list[InstanceType]:
        return [t for t, members in self._by_type.items() if members]

    def feature_names(self) -> list[str]:
        if not self.decompose_wind:
            return list(self.var_names)
        # Return the post-decomposition names.
        names = list(self.var_names)
        if _WIND_DIR in names and _WIND_SPD in names:
            names[names.index(_WIND_DIR)] = "wind_u_ms"
            names[names.index(_WIND_SPD)] = "wind_v_ms"
        return names

    def metas_for_type(self, t: InstanceType) -> list[InstanceMeta]:
        return list(self._by_type[t])

    def _slice_one_type(
        self, t: InstanceType, win: ObserverWindow
    ) -> TypeTensors | None:
        metas = self._by_type[t]
        if not metas:
            return None
        idx = np.array([m.index for m in metas], dtype=np.int64)

        c_lo, c_hi = self.store.time_range_indices(win.t_ctx_lo, win.t_ctx_hi)
        f_lo, f_hi = self.store.time_range_indices(win.t_fcst_lo, win.t_fcst_hi)

        ctx = self.store.slice_features(idx, c_lo, c_hi)
        fcst = self.store.slice_features(idx, f_lo, f_hi)
        t_ctx = self.store.slice_timestamps(c_lo, c_hi)
        t_fcst = self.store.slice_timestamps(f_lo, f_hi)

        if self.decompose_wind:
            ctx, _ = _decompose_wind(ctx, self.var_names)
            fcst, _ = _decompose_wind(fcst, self.var_names)

        valid_ctx = np.isfinite(ctx)
        valid_fcst = np.isfinite(fcst)
        # Replace NaN with 0 so tensor ops don't propagate NaN; the mask carries the info.
        ctx = np.where(valid_ctx, ctx, 0.0).astype("float32")
        fcst = np.where(valid_fcst, fcst, 0.0).astype("float32")

        lat = np.array([m.latitude for m in metas], dtype="float32")
        lon = np.array([m.longitude for m in metas], dtype="float32")

        return TypeTensors(
            type=t,
            features_ctx=torch.from_numpy(ctx),
            features_fcst=torch.from_numpy(fcst),
            times_ctx=torch.from_numpy(t_ctx),
            times_fcst=torch.from_numpy(t_fcst),
            lat=torch.from_numpy(lat),
            lon=torch.from_numpy(lon),
            valid_ctx=torch.from_numpy(valid_ctx),
            valid_fcst=torch.from_numpy(valid_fcst),
        )

    def __getitem__(self, idx: int) -> dict:
        win = self.observer_times[idx]
        per_type: dict[InstanceType, TypeTensors] = {}
        for t in InstanceType:
            tt = self._slice_one_type(t, win)
            if tt is not None:
                per_type[t] = tt
        return {
            "t_phi": int(win.t_phi),
            "per_type": per_type,
        }


def collate_batch(samples: list[dict]) -> dict:
    """Stack samples along a batch dim. Per-type instance counts are constant across
    samples (instances are determined by the store), so we can stack cleanly.
    """
    if not samples:
        return {}
    types = list(samples[0]["per_type"].keys())
    out = {
        "t_phi": torch.tensor([s["t_phi"] for s in samples], dtype=torch.int64),
        "per_type": {},
    }
    for t in types:
        items = [s["per_type"][t] for s in samples]
        out["per_type"][t] = TypeTensors(
            type=t,
            features_ctx=torch.stack([x.features_ctx for x in items], dim=0),
            features_fcst=torch.stack([x.features_fcst for x in items], dim=0),
            times_ctx=torch.stack([x.times_ctx for x in items], dim=0),
            times_fcst=torch.stack([x.times_fcst for x in items], dim=0),
            lat=items[0].lat,           # static across batch
            lon=items[0].lon,
            valid_ctx=torch.stack([x.valid_ctx for x in items], dim=0),
            valid_fcst=torch.stack([x.valid_fcst for x in items], dim=0),
        )
    return out
