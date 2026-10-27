"""Support for the subspace-decomposed noise prior (V1/V2/V3).

Two pieces live here, both of which read ONLY the fitting folds and only ever
compute moments -- no Jensen-Shannon distance is reachable from this module.

* `internal_rmsf_torch` -- the per-example internal RMSF of a training target
  block, with each frame superposed on the block's own mean.  It supplies the
  regression target for V3's per-molecule internal-scale head.
* `OnlineScaleCalibrator` -- V2's learning rule.  A plain gradient on the two
  log-scales is degenerate: the conditional flow-matching loss is monotonically
  decreasing in the prior's width (shrinking the source makes the regression
  target easier), so a gradient-learned scale collapses toward zero rather than
  matching the data.  V2 therefore learns the two scales during training by
  repeated moment matching against the FITTING folds' own generated dispersion,
  interleaved with the weight updates so the network adapts to the prior it
  ends up with.  The updates are frozen partway through the run so that the
  `dev_loss` used for checkpoint selection is comparable across the epochs it
  is allowed to choose between (pre-registration guard 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def internal_rmsf_torch(
    coords: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """(B,) mean per-atom RMSF with translation and rotation removed.

    Mirrors `mdwm.eval.ensemble_metrics.rmsf_profile`: every frame is Kabsch
    superposed on the ensemble's own mean before the fluctuation is taken.
    """
    batch, frames, atoms, _ = coords.shape
    weights = (
        torch.ones(batch, atoms, device=coords.device, dtype=coords.dtype)
        if mask is None
        else mask.to(coords.dtype)
    )
    w = weights[:, None, :, None]
    counts = weights.sum(dim=1).clamp_min(1.0)
    centred = coords - (coords * w).sum(dim=2, keepdim=True) / counts[:, None, None, None]
    centred = centred * w
    target = centred.mean(dim=1, keepdim=True)
    target = target - (target * w).sum(dim=2, keepdim=True) / counts[:, None, None, None]
    target = target * w
    covariance = centred.transpose(-2, -1) @ target.expand_as(centred)
    u, _, vt = torch.linalg.svd(covariance.reshape(-1, 3, 3).double())
    sign = torch.sign(torch.linalg.det(u @ vt))
    flip = torch.diag_embed(
        torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1)
    )
    rotation = (u @ flip @ vt).reshape(batch, frames, 3, 3).to(coords.dtype)
    aligned = centred @ rotation
    deviation = aligned - aligned.mean(dim=1, keepdim=True)
    per_atom = torch.sqrt(deviation.square().sum(dim=-1).mean(dim=1) + 1.0e-12)
    return (per_atom * weights).sum(dim=1) / counts


class OnlineScaleCalibrator:
    """V2: moment-match the two subspace scales while the model trains."""

    def __init__(
        self,
        examples: pd.DataFrame,
        *,
        store: str | Path,
        mappings: dict[str, Any],
        frame_alignment: str,
        product_init_mode: str,
        max_pocket_atoms: int | None,
        disable_mapping: bool,
        seed: int,
        steps: int = 16,
        batch_size: int = 4,
        n_examples: int = 48,
        damping: float = 0.8,
        last_update_epoch: int = 4,
        limit: tuple[float, float] = (0.1, 20.0),
    ) -> None:
        from mdwm.eval.subspace_dispersion import reference_moments

        self.kwargs = dict(
            store=store, mappings=mappings, frame_alignment=frame_alignment,
            product_init_mode=product_init_mode, max_pocket_atoms=max_pocket_atoms,
            disable_mapping=disable_mapping, seed=int(seed), steps=int(steps),
            batch_size=int(batch_size),
        )
        self.damping = float(damping)
        self.last_update_epoch = int(last_update_epoch)
        self.limit = limit
        chosen = examples.head(int(n_examples)).reset_index(drop=True)
        self.reference, _, self.usable = reference_moments(
            chosen, store=store, mappings=mappings,
            frame_alignment=frame_alignment, seed=int(seed),
        )
        self.history: list[dict[str, float]] = []

    def update(self, model, device: torch.device, epoch: int) -> dict[str, float]:
        """One moment-matching step on the two scales; returns what it saw."""
        from mdwm.eval.subspace_dispersion import measure_dispersion

        if epoch > self.last_update_epoch:
            # The scales are frozen from here on, so there is nothing to learn
            # and no reason to pay for another sampling pass.  The remaining
            # epochs are the ones `dev_loss` is allowed to choose between.
            record = dict(self.history[-1]) if self.history else {}
            record.update({
                "epoch": float(epoch),
                "updated": 0.0,
                "rigid_noise_scale_after": float(model.rigid_noise_scale),
                "internal_noise_scale_after": float(model.internal_noise_scale),
                "measured": 0.0,
            })
            self.history.append(record)
            return record

        was_training = model.training
        model.eval()
        with torch.no_grad():
            _, summary = measure_dispersion(
                model, self.usable, device=device,
                reference=self.reference, usable=self.usable, **self.kwargs,
            )
        if was_training:
            model.train()
        internal = float(summary["internal_rmsf_ratio_median"])
        centroid = float(summary["centroid_dispersion_ratio_median"])
        record = {
            "epoch": float(epoch),
            "internal_rmsf_ratio_median": internal,
            "centroid_dispersion_ratio_median": centroid,
            "orientation_dispersion_ratio_median": float(
                summary["orientation_dispersion_ratio_median"]
            ),
            "rigid_noise_scale_before": float(model.rigid_noise_scale),
            "internal_noise_scale_before": float(model.internal_noise_scale),
            "n_calibration_examples": int(summary["n_examples"]),
            "n_calibration_targets": int(summary["n_targets"]),
        }
        if epoch <= self.last_update_epoch:
            low, high = self.limit
            model.internal_noise_scale = float(
                np.clip(
                    model.internal_noise_scale
                    * max(internal, 1.0e-6) ** (-self.damping),
                    low, high,
                )
            )
            model.rigid_noise_scale = float(
                np.clip(
                    model.rigid_noise_scale
                    * max(centroid, 1.0e-6) ** (-self.damping),
                    low, high,
                )
            )
        record["rigid_noise_scale_after"] = float(model.rigid_noise_scale)
        record["internal_noise_scale_after"] = float(model.internal_noise_scale)
        record["updated"] = float(epoch <= self.last_update_epoch)
        record["measured"] = 1.0
        self.history.append(record)
        return record
