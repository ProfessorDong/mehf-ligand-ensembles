"""Is the tiny physical bond-length spread an artefact of MD bond constraints?

MD engines usually freeze bonds involving hydrogen (SHAKE/RATTLE) so they can
take a 2 fs step.  A frozen bond has exactly zero length fluctuation, so if the
ligands carry hydrogens the measured physical bond-length spread -- and with it
the isotropic-noise ratio -- could be an artefact of the simulation protocol
rather than a statement about molecular physics.

This splits the bonds into three groups and reports each separately:
  * any-hydrogen bonds, which a constraint algorithm would freeze
  * heavy-heavy bonds, which are integrated normally and are the honest test
  * all bonds, the number the main script reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isotropic_noise_geometry_cost import (  # noqa: E402
    bond_lengths, enumerate_internals, internal_rmsf, noise_arm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path,
                        default=ROOT / "data/MISATO/extracted/frames100_all_v1")
    parser.add_argument("--n-systems", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/isotropic_noise_cost_v1/hydrogen_check.json")
    args = parser.parse_args()

    paths = sorted(args.store.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(paths)

    groups: dict[str, list[float]] = {"any_hydrogen": [], "heavy_heavy": [], "all": []}
    physical: dict[str, list[float]] = {k: [] for k in groups}
    synthetic: dict[str, list[float]] = {k: [] for k in groups}
    n_with_hydrogen = 0
    used = 0

    for path in paths:
        if used >= int(args.n_systems):
            break
        with np.load(path, allow_pickle=True) as store:
            coords = store["ligand_coords"].astype(np.float64)
            bond_index = store["bond_index"]
            elements = store["ligand_atomic_numbers"].astype(int)
        n_frames, n_atoms, _ = coords.shape
        if n_atoms < 4 or n_frames < 10:
            continue
        bonds, _, _ = enumerate_internals(bond_index, n_atoms)
        if not bonds:
            continue
        target = internal_rmsf(coords)
        if not np.isfinite(target) or target <= 1.0e-6:
            continue
        noisy = noise_arm(coords[0], n_frames, target, rng)
        has_hydrogen = bool((elements == 1).any())
        n_with_hydrogen += int(has_hydrogen)

        subsets = {
            "all": bonds,
            "any_hydrogen": [b for b in bonds if elements[b[0]] == 1 or elements[b[1]] == 1],
            "heavy_heavy": [b for b in bonds if elements[b[0]] != 1 and elements[b[1]] != 1],
        }
        for name, subset in subsets.items():
            if not subset:
                continue
            p = float(np.mean(bond_lengths(coords, subset).std(axis=0)))
            s = float(np.mean(bond_lengths(noisy, subset).std(axis=0)))
            physical[name].append(p)
            synthetic[name].append(s)
            groups[name].append(s / max(p, 1.0e-9))
        used += 1

    summary = {
        "n_systems": used,
        "n_systems_containing_hydrogen": n_with_hydrogen,
        "note": (
            "heavy_heavy is the confound-free number: those bonds are never "
            "frozen by SHAKE/RATTLE, so their fluctuation is physical"
        ),
    }
    for name, values in groups.items():
        if not values:
            summary[name] = {"n": 0}
            continue
        summary[name] = {
            "n_systems_with_this_bond_type": len(values),
            "ratio_median": float(np.median(values)),
            "ratio_q25": float(np.percentile(values, 25)),
            "ratio_q75": float(np.percentile(values, 75)),
            "physical_spread_median_angstrom": float(np.median(physical[name])),
            "synthetic_spread_median_angstrom": float(np.median(synthetic[name])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
