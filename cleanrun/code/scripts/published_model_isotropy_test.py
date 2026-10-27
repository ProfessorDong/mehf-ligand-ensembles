"""Is a published 3D generative model's geometric error isotropic in Cartesian space?

PREDICTIONS ARE STATED HERE, BEFORE ANY GENERATED MOLECULE WAS MEASURED.

Background (outputs/isotropic_noise_cost_v1/FINDING.md). For internal coordinate
q_k the Wilson B-vector gives, under isotropic per-atom Cartesian error of
standard deviation sigma,

    Var(q_k) = sigma^2 ||B_k||^2                    (first order, exact)

with ||B_bond|| = sqrt(2) exactly, and ||B_angle||, ||B_dihedral|| computable
from geometry alone. Diffusion and flow models for 3D molecules perturb atom
COORDINATES under an isotropic Gaussian process and an isotropic squared-error
objective, so their residual error should be isotropic and the three
coordinate classes' errors should be locked to each other by geometry, with ONE
free parameter for all three. Three measurements, one parameter: two degrees of
freedom to falsify.

TWO ARMS, because they answer different questions.

  ARM A, "raw vs MMFF-relaxed, same molecules".  Each generated molecule is
  copied, relaxed with MMFF94 keeping its graph fixed, and the per-coordinate
  CHANGE is measured. Composition is identical by construction, so this isolates
  the model's geometric error from any difference in what molecules it makes.
  This is the primary arm.

  ARM B, "generated vs CrossDocked reference".  Class-wise spreads across
  different molecule sets, the comparison the SBDD literature reports. Its
  composition is not matched, so it is secondary and is reported with that
  caveat.

PREDICTIONS.

P1 (PRIMARY, arm A).  A single sigma fits all three classes. Concretely, with
   sigma estimated from the bond channel alone as e_bond/sqrt(2), the predicted
   angle and dihedral changes are within 35% of the measured ones.
   FALSIFIED if either is off by more than 35%.
P2.  The implied sigma lies in 0.02-0.40 Angstrom. Below that the model would
   be essentially exact; above it the molecules would not be recognisable.
P3.  Relative to physical thermal widths (bond 0.033 A, angle 2.8 deg, torsion
   14.6 deg), the damage is ordered bond > angle > torsion. So a relaxation
   moves bond lengths by many physical widths and torsions by a fraction of one.
P4.  Aromatic rings come out non-planar: the out-of-plane deviation at aromatic
   centres in the RAW output is well above the relaxed value.
P5 (arm B).  Generated class-wise spreads exceed reference spreads for every
   class, with the bond excess largest relative to the reference spread.

If P1 holds, the isotropy claim is demonstrated for a published model and is not
a property of our own architecture. If P1 fails, the residual error of a trained
diffusion model is NOT isotropic and the mechanism, as applied to published
models, is refuted; that must be reported first and plainly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

PHYSICAL = {"bond": 0.0326, "angle": 2.774, "dihedral": 14.6}


def internals(mol: Chem.Mol):
    """Bonds, angles and non-ring proper dihedrals over heavy atoms."""
    neighbours = defaultdict(list)
    bonds = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((a, b))
        neighbours[a].append(b)
        neighbours[b].append(a)
    angles = []
    for j, near in neighbours.items():
        for x in range(len(near)):
            for y in range(x + 1, len(near)):
                angles.append((near[x], j, near[y]))
    ring = mol.GetRingInfo()
    dihedrals = []
    for j, k in bonds:
        if ring.NumBondRings(mol.GetBondBetweenAtoms(j, k).GetIdx()):
            continue
        for i in neighbours[j]:
            if i == k:
                continue
            for l in neighbours[k]:
                if l in (j, i):
                    continue
                dihedrals.append((i, j, k, l))
    return bonds, angles, dihedrals


def aromatic_centres(mol: Chem.Mol):
    out = []
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic():
            continue
        near = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(near) == 3:
            out.append((atom.GetIdx(), *near))
    return out


def measure(conf, bonds, angles, dihedrals, centres):
    p = conf.GetPositions()
    def length(i, j):
        return float(np.linalg.norm(p[i] - p[j]))
    def angle(i, j, k):
        u, v = p[i] - p[j], p[k] - p[j]
        c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))
    def dihedral(i, j, k, l):
        b0, b1, b2 = p[i] - p[j], p[k] - p[j], p[l] - p[k]
        b1n = b1 / (np.linalg.norm(b1) + 1e-12)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        return math.degrees(math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))
    def improper(j, a, b, c):
        u = p[a] - p[j]
        normal = np.cross(p[b] - p[j], p[c] - p[j])
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        u = u / (np.linalg.norm(u) + 1e-12)
        return math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(u, normal))))))
    return (
        [length(*b) for b in bonds],
        [angle(*a) for a in angles],
        [dihedral(*d) for d in dihedrals],
        [improper(*c) for c in centres],
    )


def b_norms(conf, angles, dihedrals):
    """||B_k|| per coordinate: analytic for angles, finite difference for torsions."""
    p = conf.GetPositions()
    angle_norms = []
    for i, j, k in angles:
        u, v = p[i] - p[j], p[k] - p[j]
        r1, r2 = np.linalg.norm(u), np.linalg.norm(v)
        c = float(np.dot(u, v) / (r1 * r2 + 1e-12))
        value = 2.0 / r1**2 + 2.0 / r2**2 - 2.0 * c / (r1 * r2)
        angle_norms.append(math.degrees(math.sqrt(max(value, 1e-12))))
    dihedral_norms = []
    step = 1.0e-4
    for quad in dihedrals:
        total = 0.0
        for atom in quad:
            for axis in range(3):
                shifted = p.copy()
                shifted[atom, axis] += step
                b0, b1, b2 = (
                    shifted[quad[0]] - shifted[quad[1]],
                    shifted[quad[2]] - shifted[quad[1]],
                    shifted[quad[3]] - shifted[quad[2]],
                )
                b1n = b1 / (np.linalg.norm(b1) + 1e-12)
                v = b0 - np.dot(b0, b1n) * b1n
                w = b2 - np.dot(b2, b1n) * b1n
                plus = math.degrees(math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))
                b0, b1, b2 = (
                    p[quad[0]] - p[quad[1]], p[quad[2]] - p[quad[1]], p[quad[3]] - p[quad[2]]
                )
                b1n = b1 / (np.linalg.norm(b1) + 1e-12)
                v = b0 - np.dot(b0, b1n) * b1n
                w = b2 - np.dot(b2, b1n) * b1n
                base = math.degrees(math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))
                delta = (plus - base + 180.0) % 360.0 - 180.0
                total += (delta / step) ** 2
        dihedral_norms.append(math.sqrt(total))
    return angle_norms, dihedral_norms


def circular_rms(values):
    values = np.asarray(values, dtype=float)
    wrapped = (values + 180.0) % 360.0 - 180.0
    return float(np.sqrt(np.mean(wrapped**2))) if len(values) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=2000)
    parser.add_argument("--max-per-pocket", type=int, default=20)
    parser.add_argument("--restraint", type=float, default=0.0,
                        help="harmonic positional restraint, kcal/mol/A^2; 0 = unrestrained")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.generated_dir.glob("*.sdf"))
    bond_d, angle_d, dihedral_d = [], [], []
    improper_raw, improper_relaxed = [], []
    angle_bnorm, dihedral_bnorm = [], []
    n_mols = n_relax_fail = n_read_fail = 0

    for path in files:
        if n_mols >= args.max_molecules:
            break
        kept = 0
        for mol in Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True):
            if mol is None:
                n_read_fail += 1
                continue
            if kept >= args.max_per_pocket or n_mols >= args.max_molecules:
                break
            try:
                relaxed = Chem.Mol(mol)
                properties = AllChem.MMFFGetMoleculeProperties(relaxed)
                if properties is None:
                    n_relax_fail += 1
                    continue
                field = AllChem.MMFFGetMoleculeForceField(relaxed, properties)
                if field is None:
                    n_relax_fail += 1
                    continue
                if args.restraint > 0.0:
                    # UNRESTRAINED minimisation does not only repair local
                    # geometry, it rotates torsions to find a nearby minimum, so
                    # the dihedral change would measure conformational search
                    # rather than the model's error.  A harmonic positional
                    # restraint lets bonds and angles relax (~0.1 A moves) while
                    # holding the conformer.
                    for index in range(relaxed.GetNumAtoms()):
                        field.MMFFAddPositionConstraint(index, 0.0, args.restraint)
                field.Minimize(maxIts=2000)
            except Exception:
                n_relax_fail += 1
                continue
            bonds, angles, dihedrals = internals(mol)
            centres = aromatic_centres(mol)
            if not bonds or not angles:
                continue
            raw = measure(mol.GetConformer(), bonds, angles, dihedrals, centres)
            rel = measure(relaxed.GetConformer(), bonds, angles, dihedrals, centres)
            bond_d += list(np.array(raw[0]) - np.array(rel[0]))
            angle_d += list(np.array(raw[1]) - np.array(rel[1]))
            if dihedrals:
                diff = (np.array(raw[2]) - np.array(rel[2]) + 180.0) % 360.0 - 180.0
                dihedral_d += list(diff)
            improper_raw += [abs(v) for v in raw[3]]
            improper_relaxed += [abs(v) for v in rel[3]]
            an, dn = b_norms(relaxed.GetConformer(), angles, dihedrals)
            angle_bnorm += an
            dihedral_bnorm += dn
            kept += 1
            n_mols += 1

    e_bond = float(np.sqrt(np.mean(np.square(bond_d))))
    e_angle = circular_rms(angle_d)
    e_dihedral = circular_rms(dihedral_d)
    mean_angle_b = float(np.mean(angle_bnorm))
    mean_dihedral_b = float(np.mean(dihedral_bnorm))

    sigma = e_bond / math.sqrt(2.0)
    predicted_angle = sigma * mean_angle_b
    predicted_dihedral = sigma * mean_dihedral_b

    report = {
        "generated_dir": str(args.generated_dir),
        "n_molecules": n_mols,
        "n_read_failures": n_read_fail,
        "n_relax_failures": n_relax_fail,
        "arm": "A: raw vs MMFF-relaxed, identical molecules",
        "restraint_kcal_per_mol_A2": float(args.restraint),
        "measured_excess": {
            "bond_angstrom": e_bond,
            "angle_degrees": e_angle,
            "dihedral_degrees": e_dihedral,
        },
        "mean_B_norm": {
            "bond_per_angstrom": math.sqrt(2.0),
            "angle_deg_per_angstrom": mean_angle_b,
            "dihedral_deg_per_angstrom": mean_dihedral_b,
        },
        "implied_sigma_angstrom": sigma,
        "predicted_from_bond_channel": {
            "angle_degrees": predicted_angle,
            "dihedral_degrees": predicted_dihedral,
        },
        "relative_error": {
            "angle": predicted_angle / e_angle - 1.0 if e_angle else None,
            "dihedral": predicted_dihedral / e_dihedral - 1.0 if e_dihedral else None,
        },
        "in_physical_widths": {
            "bond": e_bond / PHYSICAL["bond"],
            "angle": e_angle / PHYSICAL["angle"],
            "dihedral": e_dihedral / PHYSICAL["dihedral"],
        },
        "aromatic_out_of_plane_mean_abs_degrees": {
            "raw": float(np.mean(improper_raw)) if improper_raw else None,
            "relaxed": float(np.mean(improper_relaxed)) if improper_relaxed else None,
            "n": len(improper_raw),
        },
    }
    verdict = {}
    for key in ("angle", "dihedral"):
        error = report["relative_error"][key]
        verdict[f"P1_{key}_within_35pct"] = bool(error is not None and abs(error) <= 0.35)
    verdict["P2_sigma_in_range"] = bool(0.02 <= sigma <= 0.40)
    widths = report["in_physical_widths"]
    verdict["P3_ordering_bond_gt_angle_gt_torsion"] = bool(
        widths["bond"] > widths["angle"] > widths["dihedral"]
    )
    report["verdict"] = verdict
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
