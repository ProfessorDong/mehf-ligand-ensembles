"""Control: does `_structure_local_noise` actually structure the noise?

The anisotropic-prior model generates samples whose bond ANGLES are worse than
the isotropic baseline's, while its aromatic out-of-plane channel is much
better.  Before that can be read as "the prior is right but does not propagate
to the output", the prior itself has to be shown to be right.  This draws the
model's own `_structured_noise` and measures its internal-coordinate spreads
directly, with the anisotropic path on and off, on the same systems and the
same seed.

If the anisotropic draw does NOT have tighter bond and angle components than
the isotropic draw, the implementation is broken and every conclusion built on
it is void.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from isotropic_noise_geometry_cost import (  # noqa: E402
    bond_angles_degrees, bond_lengths, circular_std_degrees, dihedrals_degrees,
)
from mdwm.data.mmp_transfer import (  # noqa: E402
    DEFAULT_MAPPINGS, DEFAULT_STORE, load_mappings,
)
from mdwm.training.mehf_transfer import build_condition, build_model  # noqa: E402
from mehf_internal_geometry_spectrum import system_internals  # noqa: E402
from run_subspace_plane import build_batches, select_examples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--tau-absolute", action="store_true")
    parser.add_argument("--no-bonds", action="store_true")
    parser.add_argument("--sweeps", type=int, default=0)
    parser.add_argument("--internal-scale", type=float, default=None,
                        help="override the checkpoint's internal noise scale")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_cfg = dict(checkpoint.get("data_cfg", {}))
    store = data_cfg.get("store", str(DEFAULT_STORE))
    mappings = load_mappings(data_cfg.get("mappings_path", str(DEFAULT_MAPPINGS)))
    directory = args.checkpoint.parent
    fit = pd.read_csv(directory / "fit_examples.csv")
    chosen = select_examples(fit, int(args.n_examples), "balanced", int(args.seed))

    batches = build_batches(
        chosen, store=store, mappings=mappings,
        frame_alignment=str(data_cfg.get("frame_alignment", "pocket")),
        product_init_mode=str(data_cfg.get("product_init_mode", "product_frame0")),
        max_pocket_atoms=data_cfg.get("max_pocket_atoms", 96),
        seed=int(args.seed), batch_size=4,
        disable_mapping=bool(data_cfg.get("disable_mapping", False)),
    )
    dummy = {
        "product_features": torch.zeros(1, 1, int(checkpoint["atom_feature_dim"])),
        "edit_features": torch.zeros(1, 1, int(checkpoint["edit_feature_dim"])),
        "pocket_features": torch.zeros(1, 1, int(checkpoint["pocket_feature_dim"])),
    }
    model = build_model(dict(checkpoint.get("model_cfg", {})), dummy).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.anisotropic_tau_absolute = bool(args.tau_absolute)
    model.anisotropic_structure_bonds = not args.no_bonds
    model.anisotropic_sweeps = int(args.sweeps)
    if args.internal_scale is not None:
        model.internal_noise_scale = float(args.internal_scale)

    rows: dict[str, list[dict[str, float]]] = {"isotropic": [], "anisotropic": []}
    for record in batches:
        condition, _ = build_condition(record["batch"], device)
        for arm in ("isotropic", "anisotropic"):
            model.anisotropic_local_noise = arm == "anisotropic"
            generator = torch.Generator(device=device).manual_seed(int(args.seed))
            with torch.no_grad():
                noise = model._structured_noise(
                    condition, int(args.frames), multiplier=1.0, generator=generator
                )
            base = condition.product_init[:, None] + noise
            for index in range(base.shape[0]):
                live = (
                    condition.atom_mask[index].bool().cpu().numpy()
                    if condition.atom_mask is not None
                    else np.ones(base.shape[2], dtype=bool)
                )
                n_live = int(record["n_atoms"][index])
                coords = base[index][:, :n_live].double().cpu().numpy()
                bond_index = (
                    condition.bond_index[index].cpu().numpy()
                    if condition.bond_index is not None else None
                )
                if bond_index is None:
                    continue
                keep = (bond_index >= 0).all(axis=-1) & (bond_index < n_live).all(axis=-1)
                bond_index = bond_index[keep]
                geometry = system_internals(
                    {"coords": coords, "bond_index": bond_index}
                )
                if geometry is None:
                    continue
                rows[arm].append({
                    "bond_length": float(np.mean(
                        bond_lengths(coords, geometry["bonds"]).std(axis=0))),
                    "bond_angle": float(np.mean(
                        bond_angles_degrees(coords, geometry["angles"]).std(axis=0))),
                    "dihedral": float(np.mean(circular_std_degrees(
                        dihedrals_degrees(coords, geometry["dihedrals"])))),
                })

    summary = {"n_systems": len(rows["isotropic"]), "frames": int(args.frames)}
    for channel in ("bond_length", "bond_angle", "dihedral"):
        iso = np.array([r[channel] for r in rows["isotropic"]])
        ani = np.array([r[channel] for r in rows["anisotropic"]])
        summary[channel] = {
            "isotropic_noise_spread_median": float(np.median(iso)),
            "anisotropic_noise_spread_median": float(np.median(ani)),
            "aniso_over_iso": float(np.median(ani) / max(np.median(iso), 1e-12)),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
