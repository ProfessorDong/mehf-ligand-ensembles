"""Does a real Cartesian flow-matching model show the predicted geometry spectrum?

`outputs/isotropic_noise_cost_v1/FINDING.md` measures, with no model involved,
what isotropic per-atom Cartesian noise costs in internal coordinates at matched
coordinate-level dispersion: bond length 15.3x, bond angle 9.8x, dihedral 3.5x.

That is a statement about the noise. The sharper, falsifiable claim is that a
network TRAINED under an isotropic Cartesian objective inherits the same
ordering in its samples, because an isotropic squared-error objective applies no
directional preference and its residual is therefore isotropic.

MEHF is exactly such a model: SE(3)-equivariant conditional flow matching on
atom coordinates, isotropic Gaussian path noise, squared-error velocity loss.
This script samples it and measures the same three spreads against the reference
MD ensemble, heavy atoms only, on the FITTING folds.

Prediction under test, stated before running:
    bond length ratio > bond angle ratio > dihedral ratio, all above 1,
    with the bond length ratio the largest by a wide margin.

Only spreads are computed. No Jensen-Shannon distance is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from isotropic_noise_geometry_cost import (  # noqa: E402
    bond_angles_degrees, bond_lengths, circular_std_degrees, dihedrals_degrees,
    enumerate_internals, internal_rmsf,
)
from mdwm.data.mmp_transfer import (  # noqa: E402
    DEFAULT_MAPPINGS, DEFAULT_STORE, load_mappings, load_system,
)
from mdwm.eval.mmp_transfer_eval import draw_frames  # noqa: E402
from mdwm.training.mehf_transfer import build_model  # noqa: E402
from run_subspace_plane import (  # noqa: E402
    build_batches, sample_grid_point, select_examples,
)


def ring_atoms(bond_index: np.ndarray, n_atoms: int) -> set[int]:
    """Atoms lying on a cycle of length 5 or 6 in the heavy-atom bond graph."""
    neighbours: list[set[int]] = [set() for _ in range(n_atoms)]
    for a, b in np.asarray(bond_index, dtype=np.int64).reshape(-1, 2):
        a, b = int(a), int(b)
        if a != b and 0 <= a < n_atoms and 0 <= b < n_atoms:
            neighbours[a].add(b)
            neighbours[b].add(a)
    found: set[int] = set()
    for start in range(n_atoms):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > 6:
                continue
            for nxt in neighbours[node]:
                if nxt == start and 5 <= len(path) <= 6:
                    found.update(path)
                elif nxt not in path and len(path) < 6:
                    stack.append((nxt, path + [nxt]))
    return found


def improper_centres(
    bond_index: np.ndarray, n_atoms: int
) -> list[tuple[int, int, int, int]]:
    """Ring atoms with exactly three heavy neighbours, as (j, a, b, c).

    These are the candidate planar sp2 / aromatic centres.  Whether a candidate
    really is planar is decided later, from the REFERENCE ensemble only.
    """
    neighbours: list[set[int]] = [set() for _ in range(n_atoms)]
    for a, b in np.asarray(bond_index, dtype=np.int64).reshape(-1, 2):
        a, b = int(a), int(b)
        if a != b and 0 <= a < n_atoms and 0 <= b < n_atoms:
            neighbours[a].add(b)
            neighbours[b].add(a)
    in_ring = ring_atoms(bond_index, n_atoms)
    centres = []
    for j in sorted(in_ring):
        near = sorted(neighbours[j])
        if len(near) == 3:
            centres.append((j, near[0], near[1], near[2]))
    return centres


def out_of_plane_degrees(coords: np.ndarray, centres) -> np.ndarray:
    """Signed out-of-plane angle of j->a against the b-j-c plane, in degrees."""
    if not centres:
        return np.zeros((len(coords), 0))
    j = np.array([c[0] for c in centres]); a = np.array([c[1] for c in centres])
    b = np.array([c[2] for c in centres]); c_ = np.array([c[3] for c in centres])
    u = coords[:, a, :] - coords[:, j, :]
    v = coords[:, b, :] - coords[:, j, :]
    w = coords[:, c_, :] - coords[:, j, :]
    normal = np.cross(v, w)
    normal = normal / np.clip(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-9, None)
    unit_u = u / np.clip(np.linalg.norm(u, axis=-1, keepdims=True), 1.0e-9, None)
    return np.degrees(np.arcsin(np.clip((unit_u * normal).sum(-1), -1.0, 1.0)))


def system_internals(system: dict[str, Any]) -> dict[str, Any] | None:
    """Bonds, angles and dihedrals of a loaded ligand.

    `load_system` already drops hydrogens and remaps `bond_index` into
    heavy-atom indexing, which is the space MEHF itself works in, so the
    heavy-atom restriction that the noise measurement had to impose by hand is
    automatic here.
    """
    n_atoms = int(np.asarray(system["coords"]).shape[1])
    bonds, angles, quads = enumerate_internals(
        np.asarray(system["bond_index"]), n_atoms
    )
    if not bonds or not angles or not quads:
        return None
    return {
        "bonds": bonds, "angles": angles, "dihedrals": quads, "n_atoms": n_atoms,
        "impropers": improper_centres(np.asarray(system["bond_index"]), n_atoms),
    }


CHANNELS = ("bond_length", "bond_angle", "dihedral", "internal_rmsf")


def spreads(
    block: np.ndarray, internals: dict[str, Any], *, impropers: bool = False
) -> dict[str, float]:
    out = {
        "bond_length": float(np.mean(bond_lengths(block, internals["bonds"]).std(axis=0))),
        "bond_angle": float(np.mean(
            bond_angles_degrees(block, internals["angles"]).std(axis=0))),
        "dihedral": float(np.mean(circular_std_degrees(
            dihedrals_degrees(block, internals["dihedrals"])))),
        "internal_rmsf": internal_rmsf(block),
    }
    if impropers:
        values = out_of_plane_degrees(block, internals["impropers"])
        out["planar_improper"] = (
            float(np.mean(values.std(axis=0))) if values.shape[1] else float("nan")
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rigid", type=float, required=True)
    parser.add_argument("--internal", type=float, required=True)
    parser.add_argument("--n-examples", type=int, default=160)
    parser.add_argument("--examples-csv", type=Path, default=None,
                        help="evaluate on this example list instead of fit_examples.csv")
    parser.add_argument("--all-examples", action="store_true",
                        help="use every example rather than a balanced sample")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, required=True)
    # Two OPT-IN switches.  With neither passed the script does exactly what it
    # did before: same model configuration, same four channels, same numbers.
    parser.add_argument(
        "--angle-projection", choices=("keep", "on", "off"), default="keep",
        help="override the checkpoint's inference-time angle projection",
    )
    parser.add_argument(
        "--restore", choices=("keep", "on", "off"), default="keep",
        help="override the checkpoint's post-projection amplitude restore",
    )
    parser.add_argument(
        "--angle-target-width", type=float, default=None,
        help=(
            "radians; drive angles to reference + N(0, width^2) instead of to "
            "reference exactly.  The corpus-measured physical angle width is "
            "0.048766 rad (2.794 deg).  Omit to keep the checkpoint's value."
        ),
    )
    parser.add_argument(
        "--projection-iterations", type=int, default=None,
        help="override the checkpoint's projection sweep count",
    )
    parser.add_argument(
        "--projection-angle-relaxation", type=float, default=None,
        help="override the angle projection's relaxation factor",
    )
    parser.add_argument(
        "--impropers", action="store_true",
        help="also measure out-of-plane spread at planar ring centres",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_cfg = dict(checkpoint.get("data_cfg", {}))
    store = data_cfg.get("store", str(DEFAULT_STORE))
    mappings = load_mappings(data_cfg.get("mappings_path", str(DEFAULT_MAPPINGS)))
    frame_alignment = str(data_cfg.get("frame_alignment", "pocket"))
    product_init_mode = str(data_cfg.get("product_init_mode", "product_frame0"))
    max_pocket_atoms = data_cfg.get("max_pocket_atoms", 96)
    disable_mapping = bool(data_cfg.get("disable_mapping", False))

    directory = args.checkpoint.parent
    if args.examples_csv is not None:
        # An explicit example list (used for the held-out `val` confirmation).
        # It must share no UniProt target with the checkpoint's training data.
        fit = pd.read_csv(args.examples_csv)
        trained_on = set(pd.read_csv(directory / "fit_examples.csv")["uniprot"]) | set(
            pd.read_csv(directory / "dev_examples.csv")["uniprot"]
        )
        if set(fit["uniprot"]) & trained_on:
            raise SystemExit("examples-csv shares a UniProt target with the training split")
    else:
        fit = pd.read_csv(directory / "fit_examples.csv")
    if args.all_examples:
        # Every example, in file order: no sampling, so no slice bias.
        chosen = fit.reset_index(drop=True)
    else:
        chosen = select_examples(fit, int(args.n_examples), "balanced", int(args.seed))

    # Reference ensembles and heavy-atom internals, per product system.
    reference: dict[str, dict[str, float]] = {}
    internals: dict[str, dict[str, Any]] = {}
    keep: list[int] = []
    for position, row in enumerate(chosen.itertuples()):
        product_id = str(row.product)
        try:
            system = load_system(product_id, store, frame_alignment)
        except Exception:
            continue
        geometry = system_internals(system)
        if geometry is None:
            continue
        coords = np.asarray(system["coords"], dtype=np.float64)
        frames = draw_frames(
            str(row.pair_key), str(row.direction), coords.shape[0], int(args.seed)
        )
        key = f"{row.parent}->{row.product}"
        reference_block = coords[frames["reference"]]
        if args.impropers and geometry["impropers"]:
            # Keep only the centres that are actually planar in the REFERENCE
            # ensemble; this filter never sees a generated coordinate.  It
            # removes sp3 centres, which are heavy-atom-trivalent too once the
            # hydrogens are gone but sit ~35 deg out of plane.
            deviation = np.abs(
                out_of_plane_degrees(reference_block, geometry["impropers"])
            ).mean(axis=0)
            geometry["impropers"] = [
                centre for centre, value in zip(geometry["impropers"], deviation)
                if value < 10.0
            ]
        reference[key] = spreads(
            reference_block, geometry, impropers=bool(args.impropers)
        )
        internals[key] = geometry
        keep.append(position)
    usable = chosen.iloc[keep].reset_index(drop=True)
    if usable.empty:
        raise SystemExit("no usable examples")

    batches = build_batches(
        usable, store=store, mappings=mappings, frame_alignment=frame_alignment,
        product_init_mode=product_init_mode, max_pocket_atoms=max_pocket_atoms,
        seed=int(args.seed), batch_size=int(args.batch_size),
        disable_mapping=disable_mapping,
    )
    dummy = {
        "product_features": torch.zeros(1, 1, int(checkpoint["atom_feature_dim"])),
        "edit_features": torch.zeros(1, 1, int(checkpoint["edit_feature_dim"])),
        "pocket_features": torch.zeros(1, 1, int(checkpoint["pocket_feature_dim"])),
    }
    model = build_model(dict(checkpoint.get("model_cfg", {})), dummy)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model.rigid_noise_scale = float(args.rigid)
    model.internal_noise_scale = float(args.internal)
    if args.angle_projection != "keep":
        # `_project_geometry` is reached only from `sample()` and has no learned
        # parameter, so this switch is purely an inference-time operator and one
        # checkpoint serves both levels of the factorial.
        model.angle_projection = args.angle_projection == "on"
    if args.restore != "keep":
        # Also inference-only, and also parameter-free.
        model.restore_projection_amplitude = args.restore == "on"
    if args.angle_target_width is not None:
        # Soft angle target: drive each angle to reference + N(0, width^2)
        # rather than to reference exactly.  Inference-only and parameter-free
        # like the other two.  0.0 is the hard constraint.
        model.angle_target_width_radian = float(args.angle_target_width)
    if args.projection_iterations is not None:
        model.projection_iterations = int(args.projection_iterations)
    if args.projection_angle_relaxation is not None:
        model.projection_angle_relaxation = float(args.projection_angle_relaxation)

    ensembles = sample_grid_point(model, batches, device=device, steps=int(args.steps))

    rows: list[dict[str, Any]] = []
    for key, block in ensembles.items():
        if key not in reference:
            continue
        generated = spreads(
            np.asarray(block, dtype=np.float64), internals[key],
            impropers=bool(args.impropers),
        )
        row = {"key": key}
        channels = CHANNELS + (("planar_improper",) if args.impropers else ())
        for name in channels:
            row[f"{name}_reference"] = reference[key][name]
            row[f"{name}_generated"] = generated[name]
            row[f"{name}_ratio"] = generated[name] / max(reference[key][name], 1.0e-9)
        rows.append(row)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "rigid_scale": float(args.rigid),
        "internal_scale": float(args.internal),
        "n_examples_sampled": len(rows),
        "steps": int(args.steps),
        "heavy_atoms_only": True,
        "prediction": "bond_length > bond_angle > dihedral, all above 1",
        "angle_loss_weight": float(
            dict(checkpoint.get("model_cfg", {})).get("angle_loss_weight", 0.0)
        ),
        "angle_projection": bool(model.angle_projection),
        "restore_projection_amplitude": bool(model.restore_projection_amplitude),
        "angle_target_width_radian": float(model.angle_target_width_radian),
        "projection_iterations": int(model.projection_iterations),
        "projection_angle_relaxation": float(model.projection_angle_relaxation),
    }
    for name in CHANNELS + (("planar_improper",) if args.impropers else ()):
        values = np.array([r[f"{name}_ratio"] for r in rows])
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        usable_rows = [
            r for r in rows
            if np.isfinite(r[f"{name}_reference"]) and np.isfinite(r[f"{name}_generated"])
        ]
        summary[name] = {
            "ratio_median": float(np.median(values)),
            "ratio_q25": float(np.percentile(values, 25)),
            "ratio_q75": float(np.percentile(values, 75)),
            "reference_median": float(
                np.median([r[f"{name}_reference"] for r in usable_rows])
            ),
            "generated_median": float(
                np.median([r[f"{name}_generated"] for r in usable_rows])
            ),
            "n_examples": len(usable_rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".per_example.csv"), index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
