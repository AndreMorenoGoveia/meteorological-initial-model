"""Factory: turn a config + dataset feature info into a ForecastModel."""

from __future__ import annotations

from typing import Any

from ..data.unified import InstanceType
from .forecast import DEFAULT_RELATIONS, ForecastModel


def build_model(cfg: Any, feature_dim: int) -> ForecastModel:
    target_types = [InstanceType(t) for t in cfg.target_types]
    return ForecastModel(
        variant=cfg.model.variant,
        feature_dim=feature_dim,
        target_types=target_types,
        hidden_dim=int(cfg.model.hidden_dim),
        pe_time_dim=int(cfg.model.pe_time_dim),
        pe_space_dim=int(cfg.model.pe_space_dim),
        hgt_num_heads=int(cfg.model.hgt.num_heads),
        hgt_num_layers=int(cfg.graph.num_layers),
        hgt_dropout=float(cfg.model.hgt.dropout),
        relations=list(DEFAULT_RELATIONS),
    )
