"""Exact harmonic prior Sigma* = (B^T F B)^+ for ligands, and its control.

PREDICTIONS ARE STATED HERE, BEFORE ANY MEASUREMENT.

The anisotropic prior currently structures noise with Gauss-Seidel sweeps that
drive each stiff internal coordinate to an INDEPENDENT draw at its physical
width. `outputs/theory_modules_v1/RESULT.md` 3.5 measured what that costs: at
matched torsional width the sampler reaches 31.5% of the physical internal
RMSF while the exact harmonic covariance reaches 125%. The defect is not
correlation between internal coordinates (measured median |corr| 0.041) but the
COMPLEMENT: Sigma*'s soft subspace is collective, with the top mode carrying
45.4% of its trace, and the sweeps leave that subspace white.

That report treats Sigma* as unusable at O(N^3). That is true for proteins and
false for ligands: a 30-heavy-atom ligand gives a 90x90 matrix, whose dense
eigendecomposition is microseconds, and the reference geometry is fixed for a
given system so it is computed once and cached.

The internal coordinate set must be COMPLETE. With only bonds and angles, the
torsional directions lie in the null space of H = B^T F B and the pseudo-inverse
would hand them unbounded amplitude. Torsions and out-of-plane coordinates are
therefore included with their own measured widths, so every non-rigid direction
is finite:

    tau_bond 0.0326 A | tau_angle 2.774 deg | tau_torsion 14.6 deg | tau_oop 6.48 deg
    F = diag(1/tau_k^2),  H = B^T F B,  Sigma* = H^+ over the non-rigid subspace

PREDICTIONS, against the same MISATO systems and the same protocol the sampler
was measured on.

P1  Sampling from Sigma* reproduces each channel's PHYSICAL width to within 35%:
    every one of bond, angle, torsion and out-of-plane ratio lands in
    [0.65, 1.35]. This is the property the Gauss-Seidel sampler does not have,
    where the same four ratios span 2.28 to 5.57.
P2  The channel spread (max ratio minus min ratio) is below 1.0, against the
    Gauss-Seidel sampler's 3.295.
P3  At matched torsional width, internal RMSF relative to physical lands in
    [0.9, 1.6], against the sampler's 0.315.
P4  H has exactly 6 eigenvalues below 1e-6 of the largest, the rigid-body modes,
    for essentially every system. A system with more is disconnected or has a
    degenerate internal set and is reported, not silently dropped.

P1 to P3 all failing would mean the harmonic model itself is wrong for these
molecules, not that the implementation is. P4 failing means the internal set is
incomplete and nothing else in the file can be read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from isotropic_noise_geometry_cost import (  # noqa: E402
    bond_angles_degrees, bond_lengths, circular_std_degrees, dihedrals_degrees,
    enumerate_internals, internal_rmsf,
)

TAU = {"bond": 0.032637, "angle": math.radians(2.774),
       "torsion": math.radians(14.6), "oop": math.radians(6.48)}


def improper_centres(bond_index, n_atoms):
    near = defaultdict(list)
    for a, b in np.asarray(bond_index).reshape(-1, 2):
        a, b = int(a), int(b)
        if a != b:
            near[a].append(b)
            near[b].append(a)
    return [(j, *sorted(v)[:3]) for j, v in near.items() if len(v) == 3]


def b_rows(coords, bonds, angles, torsions, centres, n_atoms):
    """Wilson B rows by central finite difference, one row per coordinate.

    Finite difference rather than the analytic forms deliberately: the analytic
    angle and out-of-plane vectors are already verified elsewhere, and using an
    independent route here means a mistake in either does not cancel.
    """
    rows, taus = [], []
    step = 1.0e-5

    def measure(p):
        out = []
        for a, b in bonds:
            out.append(float(np.linalg.norm(p[a] - p[b])))
        for i, j, k in angles:
            u, v = p[i] - p[j], p[k] - p[j]
            c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
            out.append(math.acos(max(-1.0, min(1.0, c))))
        for quad in torsions:
            out.append(math.radians(float(dihedrals_degrees(p[None], [quad])[0, 0])))
        for j, a, b, c in centres:
            va, vb, vc = p[a] - p[j], p[b] - p[j], p[c] - p[j]
            n = np.cross(vb, vc)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                out.append(0.0)
                continue
            out.append(math.asin(max(-1.0, min(1.0, float(
                np.dot(va / (np.linalg.norm(va) + 1e-12), n / nn))))))
        return np.asarray(out)

    base = measure(coords)
    matrix = np.zeros((len(base), 3 * n_atoms))
    for atom in range(n_atoms):
        for axis in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            delta = measure(plus) - measure(minus)
            start = len(bonds) + len(angles)
            stop = start + len(torsions)
            delta[start:stop] = (delta[start:stop] + math.pi) % (2 * math.pi) - math.pi
            matrix[:, 3 * atom + axis] = delta / (2 * step)
    taus = ([TAU["bond"]] * len(bonds) + [TAU["angle"]] * len(angles)
            + [TAU["torsion"]] * len(torsions) + [TAU["oop"]] * len(centres))
    return matrix, np.asarray(taus)


def sigma_star_sample(coords, bonds, angles, torsions, centres, n_frames, rng):
    """Draw from N(0, Sigma*) with Sigma* = (B^T F B)^+ on the non-rigid subspace."""
    n_atoms = coords.shape[0]
    matrix, taus = b_rows(coords, bonds, angles, torsions, centres, n_atoms)
    weights = 1.0 / np.square(taus)
    hessian = matrix.T @ (weights[:, None] * matrix)
    values, vectors = np.linalg.eigh(hessian)
    cut = max(values.max(), 1e-30) * 1e-6
    keep = values > cut
    n_rigid = int((~keep).sum())
    amplitude = 1.0 / np.sqrt(values[keep])
    draw = rng.standard_normal((n_frames, int(keep.sum())))
    flat = (draw * amplitude[None, :]) @ vectors[:, keep].T
    return flat.reshape(n_frames, n_atoms, 3), n_rigid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path,
                        default=ROOT / "data/MISATO/extracted/frames100_all_v1")
    parser.add_argument("--n-systems", type=int, default=120)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/sigma_star_v1/control.json")
    args = parser.parse_args()

    paths = sorted(args.store.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(paths)

    ratios = defaultdict(list)
    rigid_counts = []
    used = 0
    for path in paths:
        if used >= args.n_systems:
            break
        with np.load(path, allow_pickle=True) as store:
            coords = store["ligand_coords"].astype(np.float64)
            bond_index = store["bond_index"]
            elements = store["ligand_atomic_numbers"].astype(int)
        heavy = np.where(elements != 1)[0]
        if not (8 <= len(heavy) <= args.max_atoms):
            continue
        remap = {int(a): i for i, a in enumerate(heavy)}
        pairs = [(remap[int(a)], remap[int(b)])
                 for a, b in np.asarray(bond_index).reshape(-1, 2)
                 if int(a) in remap and int(b) in remap and int(a) != int(b)]
        if not pairs:
            continue
        block = coords[:, heavy, :]
        n_atoms = len(heavy)
        bonds, angles, torsions = enumerate_internals(np.asarray(pairs), n_atoms)
        centres = improper_centres(np.asarray(pairs), n_atoms)
        if not bonds or not angles or not torsions:
            continue
        reference = block.mean(axis=0)
        try:
            noise, n_rigid = sigma_star_sample(
                reference, bonds, angles, torsions, centres, block.shape[0], rng
            )
        except np.linalg.LinAlgError:
            continue
        rigid_counts.append(n_rigid)
        generated = reference[None] + noise

        def widths(b):
            out = {
                "bond": float(np.mean(bond_lengths(b, bonds).std(axis=0))),
                "angle": float(np.mean(bond_angles_degrees(b, angles).std(axis=0))),
                "torsion": float(np.mean(circular_std_degrees(
                    dihedrals_degrees(b, torsions)))),
                "rmsf": internal_rmsf(b),
            }
            if centres:
                quads = [(c[1], c[0], c[2], c[3]) for c in centres]
                out["oop"] = float(np.mean(circular_std_degrees(
                    dihedrals_degrees(b, quads))))
            return out

        physical, synthetic = widths(block), widths(generated)
        for key in physical:
            if key in synthetic and physical[key] > 1e-9:
                ratios[key].append(synthetic[key] / physical[key])
        used += 1

    summary = {"n_systems": used, "seed": args.seed,
               "rigid_modes_median": float(np.median(rigid_counts)) if rigid_counts else None,
               "rigid_modes_is_six_fraction": float(np.mean(np.array(rigid_counts) == 6))
               if rigid_counts else None}
    channel = {}
    for key, values in ratios.items():
        channel[key] = {"median": float(np.median(values)),
                        "q25": float(np.percentile(values, 25)),
                        "q75": float(np.percentile(values, 75))}
    summary["ratios"] = channel
    named = [channel[k]["median"] for k in ("bond", "angle", "torsion", "oop")
             if k in channel]
    summary["channel_spread"] = float(max(named) - min(named)) if named else None
    summary["verdict"] = {
        "P1_all_channels_in_0.65_1.35": bool(all(0.65 <= v <= 1.35 for v in named)),
        "P2_spread_below_1.0": bool(summary["channel_spread"] is not None
                                    and summary["channel_spread"] < 1.0),
        "P4_six_rigid_modes": bool(summary["rigid_modes_is_six_fraction"] is not None
                                   and summary["rigid_modes_is_six_fraction"] > 0.95),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
