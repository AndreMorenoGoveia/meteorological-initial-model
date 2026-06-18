"""Loader for the unified ERA5 + IAG + INMET NetCDF store.

The on-disk format is described in ``data/data_format.md``: a single NetCDF with
dimensions ``(instance, time)`` and per-instance metadata (``source_type``,
``latitude``, ``longitude``, etc.).

This module exposes a thin wrapper that:
- selects the instance subset of interest (ERA5 bbox, INMET radius, all IAG)
- exposes per-instance metadata as a typed list
- provides slicing into the (instance, time) feature array by absolute timestamp
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr


class InstanceType(str, Enum):
    ERA5 = "ERA5"
    IAG = "IAG"
    INMET = "INMET"

    @classmethod
    def from_raw(cls, raw: str) -> "InstanceType":
        return cls(str(raw).strip().upper())


@dataclass(frozen=True)
class InstanceMeta:
    """Stable per-instance attributes (one row of the ``instance`` dimension)."""

    index: int           # position in the underlying NetCDF (instance dim)
    instance_id: str
    instance_name: str
    type: InstanceType
    latitude: float
    longitude: float


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance from a single point to many points (km)."""
    R = 6371.0088
    lat1r = math.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _to_unix_seconds(s: str | datetime) -> int:
    if isinstance(s, str):
        s = datetime.fromisoformat(s)
    if s.tzinfo is None:
        s = s.replace(tzinfo=timezone.utc)
    return int(s.timestamp())


class UnifiedStore:
    """Lazy view over the unified NetCDF.

    Variables are loaded as float32 arrays sliced on demand by absolute timestamp.
    """

    def __init__(
        self,
        netcdf_path: str | Path,
        variables: Sequence[str],
        era5_bbox: tuple[float, float, float, float] | None = None,
        inmet_max_distance_km: float | None = None,
        iag_center: tuple[float, float] | None = None,
    ):
        self.path = Path(netcdf_path)
        self.variables = list(variables)
        self.era5_bbox = era5_bbox
        self.inmet_max_distance_km = inmet_max_distance_km
        self.iag_center = iag_center

        self.ds = xr.open_dataset(self.path, engine="netcdf4", chunks=None)

        self.timestamps: np.ndarray = self._decode_time(self.ds["time"])  # int64 unix seconds
        # Map from unix timestamp -> position in the time dimension.
        self._t_to_idx: dict[int, int] = {int(t): i for i, t in enumerate(self.timestamps)}

        self.instances: list[InstanceMeta] = self._select_instances()
        self.instance_idx_array: np.ndarray = np.array(
            [m.index for m in self.instances], dtype=np.int64
        )

    @staticmethod
    def _decode_time(time_var: xr.DataArray) -> np.ndarray:
        """Return unix seconds (int64), regardless of whether xarray CF-decoded
        the time axis to datetime64[ns] or left it as numeric seconds-since-1970."""
        vals = time_var.values
        if np.issubdtype(vals.dtype, np.datetime64):
            return vals.astype("datetime64[s]").astype("int64")
        return vals.astype("int64")

    # ------------------------------------------------------------------ instances

    def _select_instances(self) -> list[InstanceMeta]:
        n = int(self.ds.sizes["instance"])
        # Keep raw strings in numpy form so that ``arr == 'IAG'`` is a proper
        # boolean mask. Casting to a numpy array of StrEnum members goes via
        # ``str(member)`` and silently truncates to e.g. ``'Insta'`` on Py3.12+.
        types_raw = np.array(
            [str(s).strip().upper() for s in self.ds["source_type"].values]
        )
        lats = self.ds["latitude"].values.astype("float64")
        lons = self.ds["longitude"].values.astype("float64")
        ids = self.ds["instance_id"].values
        names = self.ds["instance_name"].values

        keep = np.zeros(n, dtype=bool)

        keep |= types_raw == InstanceType.IAG.value

        inmet_mask = types_raw == InstanceType.INMET.value
        if self.inmet_max_distance_km is not None and self.iag_center is not None:
            d = _haversine_km(self.iag_center[0], self.iag_center[1], lats, lons)
            inmet_mask = inmet_mask & (d <= self.inmet_max_distance_km)
        keep |= inmet_mask

        era5_mask = types_raw == InstanceType.ERA5.value
        if self.era5_bbox is not None:
            lat_min, lat_max, lon_min, lon_max = self.era5_bbox
            era5_mask = era5_mask & (lats >= lat_min) & (lats <= lat_max)
            era5_mask = era5_mask & (lons >= lon_min) & (lons <= lon_max)
        keep |= era5_mask

        out: list[InstanceMeta] = []
        for i in np.flatnonzero(keep):
            out.append(
                InstanceMeta(
                    index=int(i),
                    instance_id=str(ids[i]),
                    instance_name=str(names[i]),
                    type=InstanceType(str(types_raw[i])),
                    latitude=float(lats[i]),
                    longitude=float(lons[i]),
                )
            )
        return out

    # ------------------------------------------------------------------ time index

    def time_index(self, ts_unix: int) -> int:
        idx = self._t_to_idx.get(int(ts_unix))
        if idx is None:
            raise KeyError(f"timestamp {ts_unix} not in dataset (covers "
                           f"{self.timestamps[0]}..{self.timestamps[-1]})")
        return idx

    def time_range_indices(self, t_start_unix: int, t_end_unix: int) -> tuple[int, int]:
        """Half-open interval [t_start, t_end) in unix seconds → array slice (lo, hi)."""
        lo = int(np.searchsorted(self.timestamps, t_start_unix, side="left"))
        hi = int(np.searchsorted(self.timestamps, t_end_unix, side="left"))
        return lo, hi

    # ------------------------------------------------------------------ slicing

    def slice_features(
        self,
        instance_indices: Iterable[int],
        t_lo: int,
        t_hi: int,
    ) -> np.ndarray:
        """Return shape ``(I, T, F)`` of float32 features.

        ``instance_indices`` are positions in the underlying NetCDF instance dim.
        ``t_lo:t_hi`` is a half-open slice along the time dim.
        Missing values stay as NaN; downstream code handles them.
        """
        idx = np.fromiter(instance_indices, dtype=np.int64)
        arrays = []
        for var in self.variables:
            a = self.ds[var].isel(instance=xr.DataArray(idx, dims="i"), time=slice(t_lo, t_hi))
            arrays.append(a.values.astype("float32"))
        return np.stack(arrays, axis=-1)  # (I, T, F)

    def slice_timestamps(self, t_lo: int, t_hi: int) -> np.ndarray:
        return self.timestamps[t_lo:t_hi].astype("int64")

    # ------------------------------------------------------------------ partitioning

    def partition_unix_range(self, start: str, end: str) -> tuple[int, int]:
        return _to_unix_seconds(start), _to_unix_seconds(end)

    def close(self) -> None:
        self.ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ summary

    def counts_by_type(self) -> dict[InstanceType, int]:
        out: dict[InstanceType, int] = {t: 0 for t in InstanceType}
        for m in self.instances:
            out[m.type] += 1
        return out
