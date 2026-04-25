"""Training loop.

This is intentionally simple: one optimizer, IoA-complement loss, masked metrics
on a held-out validation split. Best checkpoint is saved by validation loss.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import TypeTensors
from ..data.unified import InstanceType
from ..models.forecast import ForecastInputs, ForecastModel
from ..models.graph import knn_edges
from ..models.mcar import mcar_instances, mcar_timestamps
from ..utils.logging import get_logger
from .losses import ioa_complement_loss
from .metrics import compute_metrics


log = get_logger()


def _to_inputs(
    per_type: Mapping[InstanceType, TypeTensors], device: torch.device
) -> dict[InstanceType, ForecastInputs]:
    return {
        t: ForecastInputs(
            features_ctx=tt.features_ctx.to(device),
            valid_ctx=tt.valid_ctx.to(device),
            times_ctx=tt.times_ctx.to(device),
            times_fcst=tt.times_fcst.to(device),
            lat=tt.lat.to(device),
            lon=tt.lon.to(device),
        )
        for t, tt in per_type.items()
    }


def _build_edges(
    per_type: Mapping[InstanceType, TypeTensors],
    relations: list[tuple[InstanceType, InstanceType]],
    knn_cfg: Mapping[str, int],
    device: torch.device,
) -> dict[tuple[InstanceType, InstanceType], torch.Tensor]:
    edges: dict[tuple[InstanceType, InstanceType], torch.Tensor] = {}
    for src, dst in relations:
        if src not in per_type or dst not in per_type:
            edges[(src, dst)] = torch.empty((2, 0), dtype=torch.long, device=device)
            continue
        key = f"{src.value.lower()}_to_{dst.value.lower()}"
        k = int(knn_cfg.get(key, 8))
        ei = knn_edges(
            per_type[src].lat.cpu().numpy(),
            per_type[src].lon.cpu().numpy(),
            per_type[dst].lat.cpu().numpy(),
            per_type[dst].lon.cpu().numpy(),
            k=k,
        ).to(device)
        edges[(src, dst)] = ei
    return edges


def _apply_mcar(
    per_type: dict[InstanceType, ForecastInputs],
    inst_range: tuple[float, float],
    ts_range: tuple[float, float],
) -> dict[InstanceType, ForecastInputs]:
    if max(inst_range[1], ts_range[1]) <= 0.0:
        return per_type
    out = {}
    for t, inp in per_type.items():
        f, v = inp.features_ctx, inp.valid_ctx
        f, v = mcar_instances(f, v, inst_range)
        f, v = mcar_timestamps(f, v, ts_range)
        out[t] = ForecastInputs(
            features_ctx=f,
            valid_ctx=v,
            times_ctx=inp.times_ctx,
            times_fcst=inp.times_fcst,
            lat=inp.lat,
            lon=inp.lon,
        )
    return out


@dataclass
class Trainer:
    model: ForecastModel
    cfg: Any
    device: torch.device
    output_dir: Path

    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        self.relations = self.model.relations
        self.knn_cfg = dict(self.cfg.graph.knn)
        self.target_types = self.model.target_types
        self.metrics = list(self.cfg.eval.metrics)
        self.use_hgt = self.model.variant == "hgt"
        self.use_mcar = self.model.variant in ("st_mcar", "hgt")

    # ------------------------------------------------------------------ step

    def _forward_batch(self, batch: dict, training: bool) -> tuple[torch.Tensor, dict]:
        per_type = batch["per_type"]
        t_phi = batch["t_phi"].to(self.device)
        inputs = _to_inputs(per_type, self.device)

        if training and self.use_mcar:
            inputs = _apply_mcar(
                inputs,
                tuple(self.cfg.transforms.mcar_instance_drop_range),
                tuple(self.cfg.transforms.mcar_timestamp_drop_range),
            )

        edges = (
            _build_edges(per_type, self.relations, self.knn_cfg, self.device)
            if self.use_hgt
            else None
        )

        preds = self.model(inputs, t_phi, edges)

        # Build target / mask dicts in physical units.
        targets = {t: per_type[t].features_fcst.to(self.device) for t in preds}
        masks = {t: per_type[t].valid_fcst.to(self.device) for t in preds}
        loss = ioa_complement_loss(preds, targets, masks)
        return loss, {"preds": preds, "targets": targets, "masks": masks}

    # ------------------------------------------------------------------ loops

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
        for step, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            loss, _ = self._forward_batch(batch, training=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.grad_clip))
            self.optimizer.step()
            total += float(loss.item())
            n += 1
            if step % int(self.cfg.training.log_every) == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total / max(n, 1)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader, epoch: int, tag: str = "val") -> tuple[float, dict]:
        self.model.eval()
        total = 0.0
        n = 0
        # Accumulate predictions/targets concatenated along the batch dim per type.
        accum: dict[InstanceType, dict[str, list[torch.Tensor]]] = {
            t: {"pred": [], "target": [], "mask": []} for t in self.target_types
        }
        for batch in tqdm(loader, desc=f"epoch {epoch} {tag}", leave=False):
            loss, parts = self._forward_batch(batch, training=False)
            total += float(loss.item())
            n += 1
            for t, p in parts["preds"].items():
                accum[t]["pred"].append(p.cpu())
                accum[t]["target"].append(parts["targets"][t].cpu())
                accum[t]["mask"].append(parts["masks"][t].cpu())

        preds = {t: torch.cat(a["pred"], dim=0) for t, a in accum.items() if a["pred"]}
        targets = {t: torch.cat(a["target"], dim=0) for t, a in accum.items() if a["target"]}
        masks = {t: torch.cat(a["mask"], dim=0) for t, a in accum.items() if a["mask"]}
        metrics = compute_metrics(preds, targets, masks, self.metrics)
        return total / max(n, 1), metrics

    # ------------------------------------------------------------------ fit

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        best_val = float("inf")
        for epoch in range(int(self.cfg.training.epochs)):
            t0 = time.time()
            tr_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics = self.eval_epoch(val_loader, epoch, "val")
            dt = time.time() - t0
            log.info(
                "epoch %d  train_loss=%.4f  val_loss=%.4f  (%.1fs)  metrics=%s",
                epoch, tr_loss, val_loss, dt, val_metrics,
            )
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            if val_loss < best_val:
                best_val = val_loss
                self.save("best.pt", val_loss=val_loss, val_metrics=val_metrics, epoch=epoch)
            self.save("last.pt", val_loss=val_loss, val_metrics=val_metrics, epoch=epoch)

        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        return history

    def save(self, name: str, **extra: Any) -> Path:
        path = self.output_dir / name
        torch.save(
            {"model_state": self.model.state_dict(), "extra": extra},
            path,
        )
        return path
