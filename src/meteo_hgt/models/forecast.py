"""End-to-end forecast model with three variants.

variants:
- ``gru``     : per-type GRU encoder/decoder + RevIN. No spatiotemporal encoding,
                no graph fusion. Decoder is conditioned on a learned step token.
- ``st_mcar`` : adds sinusoidal time/lat/lon encodings on encoder *and* decoder.
                Still no graph fusion. (MCAR is applied in the training loop, not here.)
- ``hgt``     : full model — same as ``st_mcar`` plus stage-2 heterogeneous
                message passing over a kNN graph.

Conventions:
- Each instance type ``k`` carries a per-type encoder, decoder, and (in HGT) a
  type-specific node embedding inside HGTConv.
- Inputs are dicts keyed by ``InstanceType``.
- Outputs are predictions in *physical units* (RevIN inverse already applied).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from ..data.unified import InstanceType
from .decoder import PerTypeDecoder
from .encoder import PerTypeEncoder
from .encodings import sinusoidal_encoding
from .hgt import HGTStack
from .revin import RevIN


# --- shared constants ---------------------------------------------------------

# Order of types is fixed for reproducibility; this is the universe the model knows.
ALL_TYPES: tuple[InstanceType, ...] = (
    InstanceType.ERA5,
    InstanceType.IAG,
    InstanceType.INMET,
)

# Edges follow the spec: ERA5 -> {INMET, IAG}, INMET -> IAG.
DEFAULT_RELATIONS: tuple[tuple[InstanceType, InstanceType], ...] = (
    (InstanceType.ERA5, InstanceType.INMET),
    (InstanceType.ERA5, InstanceType.IAG),
    (InstanceType.INMET, InstanceType.IAG),
)


@dataclass
class ForecastInputs:
    """One per type. Tensors batch-first: (B, N, T, F) etc."""

    features_ctx: torch.Tensor
    valid_ctx: torch.Tensor
    times_ctx: torch.Tensor       # (B, T_c) absolute unix seconds
    times_fcst: torch.Tensor      # (B, T_f)
    lat: torch.Tensor             # (N,)
    lon: torch.Tensor             # (N,)


# --- positional encoding helpers ---------------------------------------------

# Time scale ~ a year in seconds; spatial scale ~ a degree.
_T_SCALE = 86400.0 * 365.25
_S_SCALE = 1.0


def _time_pe(times_rel: torch.Tensor, dim: int) -> torch.Tensor:
    """times_rel: (..., T) seconds -> (..., T, dim)."""
    return sinusoidal_encoding(times_rel, dim, scale=_T_SCALE)


def _space_pe(lat: torch.Tensor, lon: torch.Tensor, dim_each: int) -> torch.Tensor:
    """lat, lon: (N,) -> (N, 2 * dim_each)."""
    return torch.cat([
        sinusoidal_encoding(lat, dim_each, scale=_S_SCALE),
        sinusoidal_encoding(lon, dim_each, scale=_S_SCALE),
    ], dim=-1)


# --- forecast model ----------------------------------------------------------


class ForecastModel(nn.Module):
    def __init__(
        self,
        variant: str,
        feature_dim: int,
        target_types: list[InstanceType],
        hidden_dim: int = 128,
        pe_time_dim: int = 32,
        pe_space_dim: int = 32,
        hgt_num_heads: int = 4,
        hgt_num_layers: int = 2,
        hgt_dropout: float = 0.1,
        relations: list[tuple[InstanceType, InstanceType]] | None = None,
    ):
        super().__init__()
        if variant not in ("gru", "st_mcar", "hgt"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.pe_time_dim = pe_time_dim
        self.pe_space_dim = pe_space_dim
        self.target_types = list(target_types)
        self.relations = list(relations or DEFAULT_RELATIONS)

        use_st = variant in ("st_mcar", "hgt")
        self.use_st = use_st
        st_dim = pe_time_dim + 2 * pe_space_dim if use_st else 0

        self.revin = nn.ModuleDict(
            {t.value: RevIN(num_features=feature_dim) for t in ALL_TYPES}
        )
        self.encoders = nn.ModuleDict(
            {
                t.value: PerTypeEncoder(
                    in_dim=feature_dim + st_dim,
                    hidden_dim=hidden_dim,
                )
                for t in ALL_TYPES
            }
        )
        # Decoder conditioning: either a learned step token (gru) or sinusoidal (t,s) (others).
        if use_st:
            cond_dim = st_dim
            self.dec_step_emb = None
        else:
            cond_dim = hidden_dim
            self.dec_step_emb = nn.Embedding(1, cond_dim)  # repeated per step

        self.decoders = nn.ModuleDict(
            {
                t.value: PerTypeDecoder(
                    cond_dim=cond_dim,
                    hidden_dim=hidden_dim,
                    out_dim=feature_dim,
                )
                for t in self.target_types
            }
        )

        self.hgt: HGTStack | None
        if variant == "hgt":
            self.hgt = HGTStack(
                hidden_dim=hidden_dim,
                types=list(ALL_TYPES),
                relations=self.relations,
                num_layers=hgt_num_layers,
                num_heads=hgt_num_heads,
                dropout=hgt_dropout,
            )
        else:
            self.hgt = None

    # ------------------------------------------------------------------ helpers

    def _build_encoder_input(
        self, t: InstanceType, inputs: ForecastInputs, t_phi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (encoder_input, mean, std) for one type.

        encoder_input shape: (B*N, T_c, feature_dim + st_dim)
        mean/std shape:      (B, N, 1, F) — kept for the inverse normalization later.
        """
        x = inputs.features_ctx                  # (B, N, T_c, F)
        mask = inputs.valid_ctx                  # (B, N, T_c, F)
        revin = self.revin[t.value]
        mean, std = revin.fit(x, mask)
        x_norm = revin.normalize(x, mean, std) * mask.to(x.dtype)
        # Re-zero positions that were missing (so the encoder doesn't see norm artifacts).
        x_norm = x_norm * mask.to(x_norm.dtype)

        if self.use_st:
            # Relative time per batch element.
            t_rel = inputs.times_ctx - t_phi.unsqueeze(-1)             # (B, T_c)
            t_pe = _time_pe(t_rel.float(), self.pe_time_dim)            # (B, T_c, pe_t)
            s_pe = _space_pe(inputs.lat, inputs.lon, self.pe_space_dim) # (N, 2*pe_s)

            B, N, T_c, _ = x_norm.shape
            t_pe_exp = t_pe.unsqueeze(1).expand(B, N, T_c, self.pe_time_dim)
            s_pe_exp = s_pe.unsqueeze(0).unsqueeze(2).expand(B, N, T_c, s_pe.shape[-1])
            x_full = torch.cat([x_norm, t_pe_exp, s_pe_exp], dim=-1)
        else:
            x_full = x_norm

        B, N, T_c, D = x_full.shape
        return x_full.reshape(B * N, T_c, D), mean, std

    def _build_decoder_cond(
        self, inputs: ForecastInputs, t_phi: torch.Tensor
    ) -> torch.Tensor:
        """Returns (B*N, T_f, cond_dim) decoder conditioning."""
        B = inputs.times_fcst.shape[0]
        N = inputs.lat.shape[0]
        T_f = inputs.times_fcst.shape[1]

        if self.use_st:
            t_rel = inputs.times_fcst - t_phi.unsqueeze(-1)
            t_pe = _time_pe(t_rel.float(), self.pe_time_dim)              # (B, T_f, pe_t)
            s_pe = _space_pe(inputs.lat, inputs.lon, self.pe_space_dim)   # (N, 2*pe_s)
            t_pe_exp = t_pe.unsqueeze(1).expand(B, N, T_f, self.pe_time_dim)
            s_pe_exp = s_pe.unsqueeze(0).unsqueeze(2).expand(B, N, T_f, s_pe.shape[-1])
            cond = torch.cat([t_pe_exp, s_pe_exp], dim=-1)                # (B, N, T_f, st_dim)
            return cond.reshape(B * N, T_f, cond.shape[-1])

        # gru variant: a single learned token repeated across steps.
        token = self.dec_step_emb.weight[0]                               # (cond_dim,)
        return token.unsqueeze(0).unsqueeze(0).expand(B * N, T_f, token.shape[0])

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        inputs_by_type: Mapping[InstanceType, ForecastInputs],
        t_phi: torch.Tensor,                                 # (B,) unix seconds
        edges_by_relation: Mapping[tuple[InstanceType, InstanceType], torch.Tensor] | None = None,
    ) -> dict[InstanceType, torch.Tensor]:
        """Returns predictions in physical units, per target type, shape (B, N_k, T_f, F)."""

        # Stage 1 — per-type encoding.
        h_by_type: dict[InstanceType, torch.Tensor] = {}
        norm_stats: dict[InstanceType, tuple[torch.Tensor, torch.Tensor]] = {}
        sizes: dict[InstanceType, int] = {}

        for t, inp in inputs_by_type.items():
            enc_in, mean, std = self._build_encoder_input(t, inp, t_phi)
            h_flat = self.encoders[t.value](enc_in)                       # (B*N, H)
            B = inp.features_ctx.shape[0]
            N = inp.features_ctx.shape[1]
            h_by_type[t] = h_flat.reshape(B, N, self.hidden_dim)
            norm_stats[t] = (mean, std)
            sizes[t] = N

        # Stage 2 — HGT fusion (only for variant 3).
        if self.hgt is not None:
            assert edges_by_relation is not None, "hgt variant requires edges_by_relation"
            h_by_type = self.hgt(h_by_type, dict(edges_by_relation))

        # Stage 3 — per-type decoding for target types.
        out: dict[InstanceType, torch.Tensor] = {}
        for t in self.target_types:
            if t not in h_by_type:
                continue
            inp = inputs_by_type[t]
            B = inp.features_ctx.shape[0]
            N = inp.features_ctx.shape[1]
            cond = self._build_decoder_cond(inp, t_phi)
            h0 = h_by_type[t].reshape(B * N, self.hidden_dim)
            y_norm = self.decoders[t.value](h0, cond)                     # (B*N, T_f, F)
            T_f = y_norm.shape[1]
            y_norm = y_norm.reshape(B, N, T_f, self.feature_dim)

            mean, std = norm_stats[t]
            y_phys = self.revin[t.value].denormalize(y_norm, mean, std)
            out[t] = y_phys

        return out
