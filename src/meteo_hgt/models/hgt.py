"""Heterogeneous Graph Transformer fusion stage.

Wraps PyTorch Geometric's ``HGTConv`` so the model file stays small. Operates on a
``HeteroData`` object built from the (B, N_k, H) per-type embeddings — we flatten
the batch dim into the node dim, run L stacked HGT layers, then unflatten.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv

from ..data.unified import InstanceType


class HGTStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        types: list[InstanceType],
        relations: list[tuple[InstanceType, InstanceType]],
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.types = list(types)
        self.relations = list(relations)
        self.hidden_dim = hidden_dim

        node_types = [t.value for t in self.types]
        edge_types = [(s.value, "to", d.value) for (s, d) in self.relations]
        self.metadata = (node_types, edge_types)

        self.layers = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=self.metadata,
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_by_type: dict[InstanceType, torch.Tensor],         # each (B, N_k, H)
        edges_by_relation: dict[tuple[InstanceType, InstanceType], torch.Tensor],
        # each edge_index (2, E_k) in *per-type* node coords
    ) -> dict[InstanceType, torch.Tensor]:
        # Flatten batch into node dim per type, and offset edges per batch element.
        sizes = {t: int(h.shape[1]) for t, h in h_by_type.items()}
        any_t = next(iter(h_by_type))
        B = h_by_type[any_t].shape[0]

        x_dict = {t.value: h.reshape(B * sizes[t], self.hidden_dim) for t, h in h_by_type.items()}

        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        for (s, d), ei in edges_by_relation.items():
            if ei.numel() == 0:
                edge_index_dict[(s.value, "to", d.value)] = torch.empty((2, 0), dtype=torch.long, device=ei.device)
                continue
            S, D = sizes[s], sizes[d]
            # Replicate the per-sample edges across the batch with per-batch offsets.
            off_src = (torch.arange(B, device=ei.device) * S).repeat_interleave(ei.shape[1])
            off_dst = (torch.arange(B, device=ei.device) * D).repeat_interleave(ei.shape[1])
            ei_b = ei.repeat(1, B)
            ei_b = torch.stack([ei_b[0] + off_src, ei_b[1] + off_dst], dim=0)
            edge_index_dict[(s.value, "to", d.value)] = ei_b

        for layer in self.layers:
            out_dict = layer(x_dict, edge_index_dict)
            new_x: dict[str, torch.Tensor] = {}
            for k in x_dict:
                if k in out_dict:
                    new_x[k] = self.dropout(torch.relu(out_dict[k]))
                else:
                    new_x[k] = x_dict[k]
            x_dict = new_x

        return {
            InstanceType(k): v.reshape(B, sizes[InstanceType(k)], self.hidden_dim)
            for k, v in x_dict.items()
        }
