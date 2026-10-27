"""Rigid / internal dispersion measurement for the subspace-prior gate.

`outputs/subspace_prior_v1/PREREGISTRATION.md` defines two dispersion ratios,
generated over reference, on held-out grouped folds inside `train`:

* **internal RMSF ratio** -- per-atom RMSF after superposing every frame onto
  its OWN ensemble mean, so rigid-body motion is removed entirely.  This is
  exactly `mdwm.eval.ensemble_metrics.rmsf_profile`, which is imported here
  UNMODIFIED; the separation between the two subspaces is the whole point of
  the gate and re-implementing the superposition would put it at risk.
* **rigid-body dispersion ratio**, reported as two numbers: the spread of the
  ligand centroid (Angstrom) and the spread of the orientation (radian).

Nothing in this module computes a Jensen-Shannon distance.  It is allowed to
run on the fitting folds for calibration, and on the dev fold for reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mdwm.eval.ensemble_metrics import rmsf_profile


# --------------------------------------------------------------------------
# the three scalar moments of one (frames, atoms, 3) block
# --------------------------------------------------------------------------
def internal_rmsf_mean(block: np.ndarray) -> float:
    """Mean per-atom RMSF with global translation AND rotation removed."""
    return float(rmsf_profile(np.asarray(block, dtype=np.float64)).mean())


def centroid_dispersion_angstrom(block: np.ndarray) -> float:
    """RMS 3-D deviation of the ligand centroid about its own mean."""
    centroids = np.asarray(block, dtype=np.float64).mean(axis=1)
    deviation = centroids - centroids.mean(axis=0, keepdims=True)
    return float(np.sqrt((deviation**2).sum(axis=-1).mean()))


def orientation_dispersion_radian(block: np.ndarray) -> float:
    """RMS rotation angle taking each frame onto the ensemble mean shape.

    The principal axes of a ligand carry a sign ambiguity and can swap order
    when two inertia eigenvalues are close, so the orientation spread is read
    from the Kabsch rotation that superposes a frame onto the ensemble mean --
    the same rotation `rmsf_profile` divides out -- and summarised by its
    angle.  Zero means every frame already shares the mean orientation.
    """
    frames = np.asarray(block, dtype=np.float64)
    target = frames.mean(axis=0)
    target = target - target.mean(axis=0)
    angles = np.empty(len(frames))
    for index, frame in enumerate(frames):
        centred = frame - frame.mean(axis=0)
        u, _, vt = np.linalg.svd(centred.T @ target)
        flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
        rotation = u @ flip @ vt
        cosine = (float(np.trace(rotation)) - 1.0) / 2.0
        angles[index] = np.arccos(float(np.clip(cosine, -1.0, 1.0)))
    return float(np.sqrt((angles**2).mean()))


def block_moments(block: np.ndarray) -> dict[str, float]:
    return {
        "internal_rmsf": internal_rmsf_mean(block),
        "centroid_dispersion": centroid_dispersion_angstrom(block),
        "orientation_dispersion": orientation_dispersion_radian(block),
    }


# --------------------------------------------------------------------------
# reference moments for a set of directional examples
# --------------------------------------------------------------------------
def reference_moments(
    examples: pd.DataFrame,
    *,
    store: str | Path,
    mappings: dict[str, Any],
    frame_alignment: str,
    seed: int,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], pd.DataFrame]:
    """Moments of each example's REFERENCE product ensemble.

    Returns the moments keyed by ``parent->product``, the cached geometries,
    and the subset of `examples` whose geometry is usable.
    """
    from mdwm.eval.mmp_transfer_eval import draw_frames, example_geometry

    moments: dict[str, dict[str, float]] = {}
    geometries: dict[str, Any] = {}
    keep: list[int] = []
    for position, row in enumerate(examples.itertuples()):
        geometry = example_geometry(
            str(row.parent), str(row.product), str(row.pair_key), str(row.direction),
            store=store, mappings=mappings, frame_alignment=frame_alignment,
        )
        if geometry is None:
            continue
        key = f"{row.parent}->{row.product}"
        frames = draw_frames(
            str(row.pair_key), str(row.direction),
            geometry["product"]["n_frames"], seed,
        )
        block = geometry["product"]["coords"].astype(np.float64)[frames["reference"]]
        moments[key] = block_moments(block)
        geometries[key] = geometry
        keep.append(position)
    return moments, geometries, examples.iloc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------
def measure_dispersion(
    model,
    examples: pd.DataFrame,
    *,
    device: torch.device,
    store: str | Path,
    mappings: dict[str, Any],
    frame_alignment: str,
    product_init_mode: str,
    max_pocket_atoms: int | None,
    seed: int,
    steps: int,
    batch_size: int,
    disable_mapping: bool = False,
    pocket_selection: str = "min_over_frames",
    reference: dict[str, dict[str, float]] | None = None,
    usable: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-example generated/reference ratios for the three moments."""
    from mdwm.eval.mmp_transfer_eval import sample_product_ensembles

    if reference is None or usable is None:
        reference, _, usable = reference_moments(
            examples, store=store, mappings=mappings,
            frame_alignment=frame_alignment, seed=seed,
        )
    ensembles, _ = sample_product_ensembles(
        model, usable, device=device, store=store, mappings=mappings,
        frame_alignment=frame_alignment, product_init_mode=product_init_mode,
        max_pocket_atoms=max_pocket_atoms, pocket_selection=pocket_selection,
        seed=seed, steps=int(steps),
        batch_size=int(batch_size), disable_mapping=bool(disable_mapping),
    )
    rows: list[dict[str, Any]] = []
    for row in usable.itertuples():
        key = f"{row.parent}->{row.product}"
        if key not in ensembles:
            continue
        generated = block_moments(ensembles[key])
        record: dict[str, Any] = {
            "key": key,
            "uniprot": str(row.uniprot),
            "pair_key": str(row.pair_key),
            "direction": str(row.direction),
            "n_frames_generated": int(len(ensembles[key])),
        }
        for name, value in generated.items():
            record[f"gen_{name}"] = value
            record[f"ref_{name}"] = reference[key][name]
            record[f"ratio_{name}"] = value / max(reference[key][name], 1.0e-9)
        rows.append(record)
    frame = pd.DataFrame(rows)
    summary = {
        "n_examples": int(len(frame)),
        "n_targets": int(frame["uniprot"].nunique()) if len(frame) else 0,
        "n_samples_per_side": int(frame["n_frames_generated"].iloc[0]) if len(frame) else 0,
    }
    for name in ("internal_rmsf", "centroid_dispersion", "orientation_dispersion"):
        summary[f"{name}_ratio_median"] = float(np.median(frame[f"ratio_{name}"]))
        summary[f"{name}_ratio_iqr"] = [
            float(np.percentile(frame[f"ratio_{name}"], 25)),
            float(np.percentile(frame[f"ratio_{name}"], 75)),
        ]
        summary[f"{name}_generated_median"] = float(np.median(frame[f"gen_{name}"]))
        summary[f"{name}_reference_median"] = float(np.median(frame[f"ref_{name}"]))
    return frame, summary
