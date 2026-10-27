"""(a, continued) Is the exchange rate really just sigma?

ratio_k = sigma ||B_k|| / tau_k is a FIRST-ORDER statement.  If it is exact then
dividing the noise-induced internal spread by the injected sigma returns ||B_k||
and nothing else -- no protocol, no corpus, no matching rule.  It can only fail
by finite-amplitude curvature.  This sweeps sigma directly, with no matching at
all, and reports the implied ||B_k|| at each amplitude against the values the
Wilson B-matrix predicts:  ||B_bond|| = sqrt(2) = 1.41421 exactly,
||B_angle|| = 1.6353 rad/A and ||B_torsion|| = 2.1213 rad/A (medians measured in
outputs/theory_modules_v1).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from isotropic_noise_geometry_cost import (  # noqa: E402
    bond_angles_degrees, bond_lengths, circular_std_degrees, dihedrals_degrees,
    enumerate_internals,
)

STORE = ROOT / "data/MISATO/extracted/frames100_all_v1"
SIGMAS = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.2823, 0.40]
N_SYSTEMS = 400
SEED = 20260925
TAU = {"bond_length": 0.0326, "bond_angle": np.radians(2.774),
       "dihedral": np.radians(14.6)}
BNORM_PREDICTED = {"bond_length": np.sqrt(2.0), "bond_angle": np.sqrt(2.674),
                   "dihedral": np.sqrt(4.5)}


def main() -> None:
    excluded: set[str] = set()
    with (ROOT / "outputs/mmp_transfer_corpus_v1/pairs.csv").open() as handle:
        for row in csv.DictReader(handle):
            for side in ("a", "b"):
                if str(row.get(f"split_{side}", "")).strip().lower() == "test":
                    excluded.add(str(row[f"pdb_{side}"]).strip().upper())
    paths = sorted(STORE.glob("*.npz"))
    rng = np.random.default_rng(SEED)
    rng.shuffle(paths)

    acc = {s: {k: [] for k in TAU} for s in SIGMAS}
    used = 0
    for path in paths:
        if used >= N_SYSTEMS:
            break
        if path.stem.upper() in excluded:
            continue
        with np.load(path, allow_pickle=True) as store:
            coords = store["ligand_coords"].astype(np.float64)
            bond_index = store["bond_index"]
            elements = store["ligand_atomic_numbers"].astype(int)
        if coords.shape[0] < 10:
            continue
        heavy = elements != 1
        if heavy.sum() < 8 or heavy.sum() > 40:
            continue
        bonds, angles, quads = enumerate_internals(bond_index, len(elements))
        bonds = [b for b in bonds if heavy[b[0]] and heavy[b[1]]]
        angles = [a for a in angles if all(heavy[i] for i in a)]
        quads = [q for q in quads if all(heavy[i] for i in q)]
        if not bonds or not angles or not quads:
            continue
        remap = -np.ones(len(elements), dtype=int)
        remap[heavy] = np.arange(int(heavy.sum()))
        bonds = [(int(remap[a]), int(remap[b])) for a, b in bonds]
        angles = [tuple(int(remap[i]) for i in a) for a in angles]
        quads = [tuple(int(remap[i]) for i in q) for q in quads]
        frame = coords[0, heavy, :]
        draw = rng.standard_normal((100, *frame.shape))
        for sigma in SIGMAS:
            block = frame[None] + sigma * draw
            acc[sigma]["bond_length"].append(
                float(np.mean(bond_lengths(block, bonds).std(axis=0))) / sigma)
            acc[sigma]["bond_angle"].append(
                np.radians(float(np.mean(bond_angles_degrees(block, angles).std(axis=0)))) / sigma)
            acc[sigma]["dihedral"].append(
                np.radians(float(np.mean(circular_std_degrees(dihedrals_degrees(block, quads))))) / sigma)
        used += 1

    out = {"n_systems": used, "Bnorm_predicted": {k: float(v) for k, v in BNORM_PREDICTED.items()},
           "tau": {k: float(v) for k, v in TAU.items()}, "by_sigma": {}}
    for sigma in SIGMAS:
        entry = {}
        for k in TAU:
            implied = float(np.median(acc[sigma][k]))
            entry[k] = {
                "implied_Bnorm": implied,
                "error_vs_predicted_pct": 100.0 * (implied / BNORM_PREDICTED[k] - 1.0),
                "ratio_to_thermal_width": sigma * implied / TAU[k],
                "first_order_ratio": sigma * float(BNORM_PREDICTED[k]) / TAU[k],
            }
        entry["bond_over_angle_measured"] = (
            entry["bond_length"]["ratio_to_thermal_width"]
            / entry["bond_angle"]["ratio_to_thermal_width"])
        out["by_sigma"][str(sigma)] = entry
    here = Path(__file__).resolve().parent
    (here / "a2_blaw_amplitude.json").write_text(json.dumps(out, indent=2))
    print(f"{'sigma':>7} {'B_bond':>8} {'err%':>7} {'B_ang':>8} {'err%':>7} {'B_dih':>8} {'err%':>7} "
          f"{'r_bond':>8} {'r_ang':>8} {'r_dih':>8} {'b/a':>6}")
    for sigma in SIGMAS:
        e = out["by_sigma"][str(sigma)]
        print(f"{sigma:7.4f} "
              f"{e['bond_length']['implied_Bnorm']:8.4f} {e['bond_length']['error_vs_predicted_pct']:7.2f} "
              f"{e['bond_angle']['implied_Bnorm']:8.4f} {e['bond_angle']['error_vs_predicted_pct']:7.2f} "
              f"{e['dihedral']['implied_Bnorm']:8.4f} {e['dihedral']['error_vs_predicted_pct']:7.2f} "
              f"{e['bond_length']['ratio_to_thermal_width']:8.2f} {e['bond_angle']['ratio_to_thermal_width']:8.2f} "
              f"{e['dihedral']['ratio_to_thermal_width']:8.2f} {e['bond_over_angle_measured']:6.3f}")


if __name__ == "__main__":
    main()
