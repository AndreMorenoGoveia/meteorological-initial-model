#!/usr/bin/env python
"""Quick inspection of the unified NetCDF: instance counts per type after filtering,
time coverage, and a sample feature slice. Useful before committing to a full training run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from meteo_hgt.config import load_config
from meteo_hgt.data.unified import InstanceType, UnifiedStore


def _fmt(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    bbox = (
        float(cfg.data.era5_region.lat_min),
        float(cfg.data.era5_region.lat_max),
        float(cfg.data.era5_region.lon_min),
        float(cfg.data.era5_region.lon_max),
    )
    iag_center = (float(cfg.data.iag_center[0]), float(cfg.data.iag_center[1]))
    store = UnifiedStore(
        netcdf_path=str(cfg.data.netcdf_path),
        variables=list(cfg.data.variables),
        era5_bbox=bbox,
        inmet_max_distance_km=float(cfg.data.inmet_max_distance_km),
        iag_center=iag_center,
    )

    print(f"file: {store.path}")
    print(f"time coverage: {_fmt(int(store.timestamps[0]))}  ->  {_fmt(int(store.timestamps[-1]))}")
    print(f"timesteps:    {len(store.timestamps)}")
    counts = store.counts_by_type()
    print("instance counts (after filtering):")
    for t in InstanceType:
        print(f"  {t.value:6s}  {counts[t]}")

    print()
    print("first IAG instance:")
    iag = next((m for m in store.instances if m.type == InstanceType.IAG), None)
    if iag:
        print(f"  id={iag.instance_id}")
        print(f"  name={iag.instance_name}")
        print(f"  lat={iag.latitude:.4f}  lon={iag.longitude:.4f}")
    print("first INMET instances:")
    for m in [m for m in store.instances if m.type == InstanceType.INMET][:5]:
        print(f"  {m.instance_id}  ({m.latitude:.3f}, {m.longitude:.3f})  {m.instance_name}")


if __name__ == "__main__":
    main()
