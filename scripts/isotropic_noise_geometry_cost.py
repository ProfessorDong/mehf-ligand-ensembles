"""How much internal geometry does isotropic Cartesian noise destroy?

Every 3D molecular generative model in wide use -- diffusion or flow matching --
perturbs ATOM COORDINATES with isotropic, per-atom Gaussian noise, and is then
judged on INTERNAL-COORDINATE statistics: bond length, bond angle and dihedral
distributions, and geometric validity checks built from them.

This script measures the exchange rate between the two, with no model involved.
For each ligand it takes the real MD ensemble as the physical arm, then builds a
synthetic arm by perturbing a single frame with isotropic per-atom noise whose
amplitude is tuned so that the two arms have the SAME internal RMSF, i.e. the
same coordinate-level dispersion.  It then compares how wide the internal
coordinates became in each arm.

The prediction under test is a SPECTRUM, not a single number.  Physical thermal
motion is soft in torsions and near rigid in bond lengths and angles, so it buys
its coordinate dispersion almost entirely with torsions.  Isotropic noise has no
such preference and spends equally in every direction, so at matched RMSF it
should distort the stiff coordinates far more than the soft ones:

    bond length ratio  >>  bond angle ratio  >  torsion ratio

Nothing here is selected on any reported metric; it is a property of the noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdwm.eval.ensemble_metrics import rmsf_profile  # noqa: E402


def bond_graph(bond_index: np.ndarray, n_atoms: int) -> list[list[int]]:
    neighbours: list[list[int]] = [[] for _ in range(n_atoms)]
    for a, b in bond_index:
        a, b = int(a), int(b)
        if a == b or not (0 <= a < n_atoms and 0 <= b < n_atoms):
            continue
        if b not in neighbours[a]:
            neighbours[a].append(b)
        if a not in neighbours[b]:
            neighbours[b].append(a)
    return neighbours


def enumerate_internals(
    bond_index: np.ndarray, n_atoms: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    """Unique bonds, angles (j at the vertex) and proper dihedrals."""
    neighbours = bond_graph(bond_index, n_atoms)
    bonds = sorted({(min(int(a), int(b)), max(int(a), int(b)))
                    for a, b in bond_index if int(a) != int(b)})
    angles: list[tuple[int, int, int]] = []
    for j in range(n_atoms):
        near = sorted(neighbours[j])
        for x in range(len(near)):
            for y in range(x + 1, len(near)):
                angles.append((near[x], j, near[y]))
    dihedrals: list[tuple[int, int, int, int]] = []
    for j, k in bonds:
        for i in neighbours[j]:
            if i == k:
                continue
            for l in neighbours[k]:
                if l == j or l == i:
                    continue
                dihedrals.append((i, j, k, l))
    return bonds, angles, dihedrals


def bond_lengths(coords: np.ndarray, bonds) -> np.ndarray:
    if not bonds:
        return np.zeros((len(coords), 0))
    a = np.array([b[0] for b in bonds]); b = np.array([x[1] for x in bonds])
    return np.linalg.norm(coords[:, a, :] - coords[:, b, :], axis=-1)


def bond_angles_degrees(coords: np.ndarray, angles) -> np.ndarray:
    if not angles:
        return np.zeros((len(coords), 0))
    i = np.array([a[0] for a in angles]); j = np.array([a[1] for a in angles])
    k = np.array([a[2] for a in angles])
    u = coords[:, i, :] - coords[:, j, :]
    v = coords[:, k, :] - coords[:, j, :]
    cosine = (u * v).sum(-1) / np.clip(
        np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1.0e-9, None
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def dihedrals_degrees(coords: np.ndarray, quads) -> np.ndarray:
    if not quads:
        return np.zeros((len(coords), 0))
    i = np.array([q[0] for q in quads]); j = np.array([q[1] for q in quads])
    k = np.array([q[2] for q in quads]); l = np.array([q[3] for q in quads])
    b0 = coords[:, i, :] - coords[:, j, :]
    b1 = coords[:, k, :] - coords[:, j, :]
    b2 = coords[:, l, :] - coords[:, k, :]
    b1n = b1 / np.clip(np.linalg.norm(b1, axis=-1, keepdims=True), 1.0e-9, None)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def circular_std_degrees(angles: np.ndarray) -> np.ndarray:
    """Per-column circular standard deviation, in degrees."""
    radians = np.radians(angles)
    resultant = np.sqrt(
        np.cos(radians).mean(axis=0) ** 2 + np.sin(radians).mean(axis=0) ** 2
    )
    resultant = np.clip(resultant, 1.0e-12, 1.0)
    return np.degrees(np.sqrt(-2.0 * np.log(resultant)))


def internal_rmsf(block: np.ndarray) -> float:
    """Mean per-atom RMSF after the frozen suite's superposition."""
    return float(np.mean(rmsf_profile(block)))


def noise_arm(
    frame: np.ndarray, n_frames: int, target_rmsf: float, rng: np.random.Generator
) -> np.ndarray:
    """Isotropic per-atom Gaussian noise, amplitude solved to match `target_rmsf`.

    The superposition in `rmsf_profile` removes the rigid component of the draw,
    so the achieved RMSF is not sigma*sqrt(3); it is solved for numerically by a
    single secant-free rescale, which is exact because the achieved RMSF is
    linear in sigma for a fixed random draw.
    """
    draw = rng.standard_normal((n_frames, *frame.shape))
    trial = frame[None, :, :] + draw
    achieved = internal_rmsf(trial)
    if achieved <= 0.0:
        return np.repeat(frame[None, :, :], n_frames, axis=0)
    return frame[None, :, :] + draw * (target_rmsf / achieved)


def analyse(path: Path, rng: np.random.Generator) -> dict | None:
    with np.load(path, allow_pickle=True) as store:
        coords = store["ligand_coords"].astype(np.float64)
        bond_index = store["bond_index"]
        elements = store["ligand_atomic_numbers"].astype(int)
    n_frames, n_atoms, _ = coords.shape
    if n_atoms < 4 or n_frames < 10:
        return None
    bonds, angles, quads = enumerate_internals(bond_index, n_atoms)
    # MISATO's MD constrains bonds to hydrogen (measured spread 2e-6 Angstrom,
    # i.e. exactly frozen), so any internal coordinate touching a hydrogen
    # reports the integrator's constraint, not molecular physics.  The headline
    # numbers are therefore heavy-atom only.
    heavy = elements != 1
    bonds = [b for b in bonds if heavy[b[0]] and heavy[b[1]]]
    angles = [a for a in angles if all(heavy[i] for i in a)]
    quads = [q for q in quads if all(heavy[i] for i in q)]
    if not bonds or not angles or not quads:
        return None

    physical_rmsf = internal_rmsf(coords)
    if not np.isfinite(physical_rmsf) or physical_rmsf <= 1.0e-6:
        return None
    synthetic = noise_arm(coords[0], n_frames, physical_rmsf, rng)

    out = {
        "system": path.stem,
        "n_atoms": int(n_atoms),
        "n_bonds": len(bonds),
        "n_angles": len(angles),
        "n_dihedrals": len(quads),
        "internal_rmsf_physical": physical_rmsf,
        "internal_rmsf_synthetic": internal_rmsf(synthetic),
    }
    for name, physical, artificial in (
        ("bond_length", bond_lengths(coords, bonds), bond_lengths(synthetic, bonds)),
        ("bond_angle", bond_angles_degrees(coords, angles),
         bond_angles_degrees(synthetic, angles)),
        ("dihedral", dihedrals_degrees(coords, quads),
         dihedrals_degrees(synthetic, quads)),
    ):
        if name == "dihedral":
            spread_p = float(np.mean(circular_std_degrees(physical)))
            spread_s = float(np.mean(circular_std_degrees(artificial)))
        else:
            spread_p = float(np.mean(physical.std(axis=0)))
            spread_s = float(np.mean(artificial.std(axis=0)))
        out[f"{name}_spread_physical"] = spread_p
        out[f"{name}_spread_synthetic"] = spread_s
        out[f"{name}_ratio"] = spread_s / max(spread_p, 1.0e-9)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path,
                        default=ROOT / "data/MISATO/extracted/frames100_all_v1")
    parser.add_argument("--n-systems", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/isotropic_noise_cost_v1/summary.json")
    args = parser.parse_args()

    paths = sorted(args.store.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(paths)
    paths = paths[: int(args.n_systems)]

    rows = []
    for path in paths:
        try:
            row = analyse(path, rng)
        except Exception:
            continue
        if row is not None:
            rows.append(row)

    summary = {
        "n_systems_analysed": len(rows),
        "store": str(args.store),
        "seed": int(args.seed),
        "design": (
            "physical arm = real MD ensemble; synthetic arm = one frame plus "
            "isotropic per-atom Gaussian noise rescaled to the SAME internal "
            "RMSF; ratio = synthetic internal-coordinate spread / physical"
        ),
    }
    for name in ("bond_length", "bond_angle", "dihedral"):
        values = np.array([r[f"{name}_ratio"] for r in rows])
        summary[name] = {
            "ratio_median": float(np.median(values)),
            "ratio_q25": float(np.percentile(values, 25)),
            "ratio_q75": float(np.percentile(values, 75)),
            "physical_spread_median": float(
                np.median([r[f"{name}_spread_physical"] for r in rows])),
            "synthetic_spread_median": float(
                np.median([r[f"{name}_spread_synthetic"] for r in rows])),
        }
    summary["internal_rmsf_median_angstrom"] = float(
        np.median([r["internal_rmsf_physical"] for r in rows]))
    summary["rmsf_match_check_median_ratio"] = float(np.median(
        [r["internal_rmsf_synthetic"] / max(r["internal_rmsf_physical"], 1e-9)
         for r in rows]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    import csv
    with (args.output.parent / "per_system.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
