"""kNN graph construction for the heterogeneous spatial fusion stage.

For each directed relation (k_src -> k_dst), we pick, for every dst node, its
``k`` nearest src nodes in great-circle distance. The result is a per-relation
edge_index tensor in PyG format ``(2, E)`` with rows ``[src, dst]``.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _haversine_matrix(
    lat_src: np.ndarray, lon_src: np.ndarray,
    lat_dst: np.ndarray, lon_dst: np.ndarray,
) -> np.ndarray:
    R = 6371.0088
    lat1 = np.radians(lat_src)[:, None]    # (S, 1)
    lat2 = np.radians(lat_dst)[None, :]    # (1, D)
    dlat = lat2 - lat1
    dlon = np.radians(lon_dst)[None, :] - np.radians(lon_src)[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))   # (S, D)


def knn_edges(
    lat_src: np.ndarray, lon_src: np.ndarray,
    lat_dst: np.ndarray, lon_dst: np.ndarray,
    k: int,
) -> torch.Tensor:
    """Return PyG edge_index ``(2, E)`` with row 0 = src, row 1 = dst (positions
    within their respective node sets, *not* global indices).
    """
    S = lat_src.shape[0]
    D = lat_dst.shape[0]
    if S == 0 or D == 0:
        return torch.empty((2, 0), dtype=torch.long)
    k_eff = min(k, S)
    dist = _haversine_matrix(lat_src, lon_src, lat_dst, lon_dst)   # (S, D)
    # For each dst column, k smallest srcs.
    src_idx = np.argpartition(dist, kth=k_eff - 1, axis=0)[:k_eff, :]   # (k_eff, D)
    dst_idx = np.broadcast_to(np.arange(D)[None, :], src_idx.shape)
    edges = np.stack([src_idx.reshape(-1), dst_idx.reshape(-1)], axis=0).astype("int64")
    return torch.from_numpy(edges)
