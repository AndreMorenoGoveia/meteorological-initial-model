"""End-to-end smoke tests on synthetic tensors.

These do not touch the NetCDF file. They verify:
- shapes flow through the three model variants
- IoA-complement loss returns a finite scalar
- mask-aware metrics handle a partly-missing target
"""

from __future__ import annotations

import torch

from meteo_hgt.data.unified import InstanceType
from meteo_hgt.models.forecast import (
    DEFAULT_RELATIONS,
    ForecastInputs,
    ForecastModel,
)
from meteo_hgt.models.graph import knn_edges
from meteo_hgt.training.losses import ioa_complement_loss
from meteo_hgt.training.metrics import compute_metrics


B = 2
T_C = 24
T_F = 12
F = 4
N_BY_TYPE = {InstanceType.ERA5: 9, InstanceType.IAG: 1, InstanceType.INMET: 4}


def _make_inputs(N: int, lat0: float, lon0: float) -> ForecastInputs:
    g = torch.Generator().manual_seed(0)
    lat = lat0 + torch.randn(N, generator=g) * 0.1
    lon = lon0 + torch.randn(N, generator=g) * 0.1
    times_ctx = torch.arange(T_C).unsqueeze(0).expand(B, -1).long() * 3600
    times_fcst = (T_C + torch.arange(T_F)).unsqueeze(0).expand(B, -1).long() * 3600
    return ForecastInputs(
        features_ctx=torch.randn(B, N, T_C, F, generator=g),
        valid_ctx=torch.ones(B, N, T_C, F, dtype=torch.bool),
        times_ctx=times_ctx,
        times_fcst=times_fcst,
        lat=lat,
        lon=lon,
    )


def _build_inputs() -> dict[InstanceType, ForecastInputs]:
    return {
        InstanceType.ERA5: _make_inputs(N_BY_TYPE[InstanceType.ERA5], -23.55, -46.65),
        InstanceType.IAG: _make_inputs(N_BY_TYPE[InstanceType.IAG], -23.65, -46.62),
        InstanceType.INMET: _make_inputs(N_BY_TYPE[InstanceType.INMET], -23.50, -46.60),
    }


def _build_edges(inputs: dict[InstanceType, ForecastInputs]) -> dict:
    edges = {}
    for src, dst in DEFAULT_RELATIONS:
        ei = knn_edges(
            inputs[src].lat.numpy(), inputs[src].lon.numpy(),
            inputs[dst].lat.numpy(), inputs[dst].lon.numpy(),
            k=3,
        )
        edges[(src, dst)] = ei
    return edges


def _check_predictions(preds: dict[InstanceType, torch.Tensor]) -> None:
    for t, p in preds.items():
        assert p.shape == (B, N_BY_TYPE[t], T_F, F), f"bad shape for {t}: {p.shape}"
        assert torch.isfinite(p).all(), f"non-finite outputs for {t}"


def test_variant_gru():
    model = ForecastModel(
        variant="gru",
        feature_dim=F,
        target_types=[InstanceType.IAG, InstanceType.INMET],
        hidden_dim=32, pe_time_dim=8, pe_space_dim=8,
        hgt_num_layers=1,
    )
    inputs = _build_inputs()
    t_phi = torch.tensor([T_C, T_C]) * 3600
    preds = model(inputs, t_phi)
    _check_predictions(preds)


def test_variant_st_mcar():
    model = ForecastModel(
        variant="st_mcar",
        feature_dim=F,
        target_types=[InstanceType.IAG, InstanceType.INMET],
        hidden_dim=32, pe_time_dim=8, pe_space_dim=8,
        hgt_num_layers=1,
    )
    inputs = _build_inputs()
    t_phi = torch.tensor([T_C, T_C]) * 3600
    preds = model(inputs, t_phi)
    _check_predictions(preds)


def test_variant_hgt():
    model = ForecastModel(
        variant="hgt",
        feature_dim=F,
        target_types=[InstanceType.IAG, InstanceType.INMET],
        hidden_dim=32, pe_time_dim=8, pe_space_dim=8,
        hgt_num_heads=2, hgt_num_layers=2,
    )
    inputs = _build_inputs()
    edges = _build_edges(inputs)
    t_phi = torch.tensor([T_C, T_C]) * 3600
    preds = model(inputs, t_phi, edges)
    _check_predictions(preds)


def test_loss_and_metrics_with_missing_target():
    inputs = _build_inputs()
    targets = {t: inputs[t].features_ctx[:, :, :T_F, :] for t in inputs}     # reuse for shape
    preds = {t: targets[t] + 0.1 * torch.randn_like(targets[t]) for t in targets}
    masks = {t: torch.ones_like(targets[t], dtype=torch.bool) for t in targets}
    masks[InstanceType.IAG][:, :, ::2, :] = False                            # half missing

    loss = ioa_complement_loss(preds, targets, masks)
    assert torch.isfinite(loss)

    m = compute_metrics(preds, targets, masks, ["ioa", "mae", "rmse", "corr"])
    for t_name, vals in m.items():
        for k, v in vals.items():
            assert v == v, f"NaN metric {k} for {t_name}"     # NaN check
