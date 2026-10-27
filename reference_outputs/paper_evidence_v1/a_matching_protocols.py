"""(a) Is there ONE defensible way to state the cost of isotropic noise?

The published number moved 15.27 -> 11.55 -> ~7.02 as confounds were removed and
as the matching statistic changed.  This script separates the two things that
are being conflated:

  1. the PHYSICS, ratio_k = sigma * ||B_k|| / tau_k, which has one free
     parameter and is a property of the Wilson B-matrix only;
  2. the PROTOCOL, which is nothing but a rule for choosing sigma.

If (1) is right then, within every matching protocol, dividing the measured
synthetic spread by the injected sigma must return the same ||B_k|| -- and every
protocol difference collapses into a single number, sigma.

Four matching statistics, all heavy-atom, all on the same systems and the same
random draws:

  kabsch_rmsf     mean per-atom RMSF after superposition to the mean structure
                  (the published protocol; rotation-invariant)
  pairdist        mean over heavy-atom pairs of the SD of the interatomic
                  distance (rotation-invariant, superposition-free)
  centroid_radius mean over atoms of the SD of |x_i - centroid|
                  (rotation-invariant, superposition-free, different weighting)
  centroid_rmsf   mean per-atom SD about the per-frame centroid: translation
                  removed but NOT rotation.  Ruled out on principle (rigid
                  tumbling contributes dispersion and zero internal distortion);
                  reported only to bound the range.

Heavy atoms only everywhere: internals, the match target, and the noise
amplitude solve.  PDB ids carrying a `test` label in the transfer corpus are
excluded.
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

from mdwm.eval.ensemble_metrics import rmsf_profile  # noqa: E402
from isotropic_noise_geometry_cost import (  # noqa: E402
    bond_angles_degrees, bond_lengths, circular_std_degrees, dihedrals_degrees,
    enumerate_internals,
)

STORE = ROOT / "data/MISATO/extracted/frames100_all_v1"
N_SYSTEMS = 800
SEED = 20260923


# ------------------------------------------------------------ match statistics
def stat_kabsch_rmsf(block: np.ndarray) -> float:
    return float(np.mean(rmsf_profile(block)))


def stat_pairdist(block: np.ndarray) -> float:
    n = block.shape[1]
    iu, ju = np.triu_indices(n, k=1)
    d = np.linalg.norm(block[:, iu, :] - block[:, ju, :], axis=-1)
    return float(np.mean(d.std(axis=0)))


def stat_centroid_radius(block: np.ndarray) -> float:
    c = block.mean(axis=1, keepdims=True)
    r = np.linalg.norm(block - c, axis=-1)
    return float(np.mean(r.std(axis=0)))


def stat_centroid_rmsf(block: np.ndarray) -> float:
    c = block.mean(axis=1, keepdims=True)
    centred = block - c
    return float(np.mean(np.sqrt(((centred - centred.mean(0)) ** 2).sum(-1).mean(0))))


STATS = {
    "kabsch_rmsf": stat_kabsch_rmsf,
    "pairdist": stat_pairdist,
    "centroid_radius": stat_centroid_radius,
    "centroid_rmsf": stat_centroid_rmsf,
}
ROTATION_INVARIANT = ("kabsch_rmsf", "pairdist", "centroid_radius")


def solve_sigma(frame, draw, target, fn, lo=1e-4, hi=10.0, iters=60):
    """Bisection on sigma so that fn(frame + sigma*draw) == target, exactly."""
    def f(s):
        return fn(frame[None, :, :] + s * draw)
    if f(hi) < target:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spreads(block, bonds, angles, quads):
    return {
        "bond_length": float(np.mean(bond_lengths(block, bonds).std(axis=0))),
        "bond_angle": float(np.mean(bond_angles_degrees(block, angles).std(axis=0))),
        "dihedral": float(np.mean(circular_std_degrees(dihedrals_degrees(block, quads)))),
    }


def main() -> None:
    excluded: set[str] = set()
    pairs = ROOT / "outputs/mmp_transfer_corpus_v1/pairs.csv"
    if pairs.exists():
        with pairs.open() as handle:
            for row in csv.DictReader(handle):
                for side in ("a", "b"):
                    if str(row.get(f"split_{side}", "")).strip().lower() == "test":
                        excluded.add(str(row[f"pdb_{side}"]).strip().upper())
    paths = sorted(STORE.glob("*.npz"))
    rng = np.random.default_rng(SEED)
    rng.shuffle(paths)

    rows = []
    for path in paths:
        if len(rows) >= N_SYSTEMS:
            break
        if path.stem.upper() in excluded:
            continue
        with np.load(path, allow_pickle=True) as store:
            coords = store["ligand_coords"].astype(np.float64)
            bond_index = store["bond_index"]
            elements = store["ligand_atomic_numbers"].astype(int)
        n_frames, n_atoms, _ = coords.shape
        if n_atoms < 4 or n_frames < 10:
            continue
        heavy = elements != 1
        if heavy.sum() < 4:
            continue
        bonds, angles, quads = enumerate_internals(bond_index, n_atoms)
        bonds = [b for b in bonds if heavy[b[0]] and heavy[b[1]]]
        angles = [a for a in angles if all(heavy[i] for i in a)]
        quads = [q for q in quads if all(heavy[i] for i in q)]
        if not bonds or not angles or not quads:
            continue
        # everything in the heavy-atom index space
        remap = -np.ones(len(elements), dtype=int)
        remap[heavy] = np.arange(int(heavy.sum()))
        bonds = [(int(remap[a]), int(remap[b])) for a, b in bonds]
        angles = [tuple(int(remap[i]) for i in a) for a in angles]
        quads = [tuple(int(remap[i]) for i in q) for q in quads]
        xyz = coords[:, heavy, :]

        physical = spreads(xyz, bonds, angles, quads)
        if physical["bond_length"] <= 1e-6:
            continue
        draw = rng.standard_normal(xyz.shape)
        row = {"system": path.stem, "n_heavy": int(heavy.sum()),
               "n_bonds": len(bonds), "n_angles": len(angles), "n_dih": len(quads)}
        for name, value in physical.items():
            row[f"phys_{name}"] = value
        ok = True
        for pname, fn in STATS.items():
            target = fn(xyz)
            row[f"target_{pname}"] = target
            sigma = solve_sigma(xyz[0], draw, target, fn)
            if sigma is None:
                ok = False
                break
            synth = xyz[0][None, :, :] + sigma * draw
            s = spreads(synth, bonds, angles, quads)
            row[f"sigma_{pname}"] = sigma
            row[f"check_{pname}"] = fn(synth) / target
            for name in physical:
                row[f"{pname}_{name}_ratio"] = s[name] / physical[name]
            # implied ||B_k||: synthetic spread per unit sigma, radians for angles
            row[f"{pname}_Bnorm_bond"] = s["bond_length"] / sigma
            row[f"{pname}_Bnorm_angle"] = np.radians(s["bond_angle"]) / sigma
            row[f"{pname}_Bnorm_dih"] = np.radians(s["dihedral"]) / sigma
        if ok:
            rows.append(row)

    out: dict = {"n_systems": len(rows), "store": str(STORE), "seed": SEED,
                 "n_test_pdb_excluded": len(excluded)}
    med = lambda key: float(np.median([r[key] for r in rows]))
    q = lambda key: [float(np.percentile([r[key] for r in rows], 25)),
                     float(np.percentile([r[key] for r in rows], 75))]
    out["physical_widths"] = {k: med(f"phys_{k}")
                              for k in ("bond_length", "bond_angle", "dihedral")}
    for pname in STATS:
        out[pname] = {
            "match_check_median": med(f"check_{pname}"),
            "sigma_median_A": med(f"sigma_{pname}"),
            "sigma_iqr": q(f"sigma_{pname}"),
            "ratios": {k: {"median": med(f"{pname}_{k}_ratio"),
                           "iqr": q(f"{pname}_{k}_ratio")}
                       for k in ("bond_length", "bond_angle", "dihedral")},
            "implied_Bnorm": {k: med(f"{pname}_Bnorm_{k}")
                              for k in ("bond", "angle", "dih")},
            "bond_over_angle": med(f"{pname}_bond_length_ratio") /
                               med(f"{pname}_bond_angle_ratio"),
            "angle_over_dih": med(f"{pname}_bond_angle_ratio") /
                              med(f"{pname}_dihedral_ratio"),
        }
    here = Path(__file__).resolve().parent
    (here / "a_matching_protocols.json").write_text(json.dumps(out, indent=2))
    with (here / "a_matching_protocols.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
