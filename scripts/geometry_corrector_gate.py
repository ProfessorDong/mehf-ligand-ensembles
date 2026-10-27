"""GATE: does the equivariant geometric projection do anything a force field does not?

PREDICTIONS ARE STATED HERE, BEFORE ANY ARM WAS MEASURED.

Why this gate exists. The method's confirmed result is on conformational
ensembles of GIVEN molecules (MISATO MMP transfer), a task the molecular
generation field does not work on. To be an advance IN that field the operator
has to do something useful to that field's own models, on that field's own
metrics. The obvious objection is that a bonded-geometry projection is just a
force-field minimisation with fewer terms. This gate tests exactly that
objection, and it is designed so the method can lose.

FOUR ARMS on the same generated molecules (DiffSBDD released CrossDocked
samples), identical molecule set, identical metrics:
  A  raw model output
  B  MMFF94 unrestrained  -- what "relax the output" usually means
  C  MMFF94 with a 100 kcal/mol/A^2 positional restraint -- the careful version,
     and THE REAL COMPARATOR
  D  ours: SE(3)-equivariant Gauss-Seidel projection of bonds and angles onto
     class-ideal values, under-relaxation 0.5, no nonbonded term, no energy

Class-ideal bond lengths and angles are estimated from the 100 REAL CrossDocked
test ligands, keyed by (elements, bond order, aromaticity) for bonds and by
(elements, aromaticity of the vertex) for angles, with a minimum class count.
Deriving the targets from data rather than from a force field is deliberate: it
is the same choice the prior makes for its tau values.

METRICS, per molecule:
  bonded geometry   RMS deviation of bonds (A) and angles (deg) from class ideal
  planarity         mean |out-of-plane| at aromatic centres (deg)
  pose preservation heavy-atom RMSD from arm A, no realignment
  clash             count of ligand-receptor heavy-atom pairs below 2.0 A

PREDICTIONS.

P1  D repairs bonded geometry comparably to C: D's bond and angle RMS deviations
    are within 1.5x of C's, and both are at least 30% below A's.
P2  D preserves the pose far better than B: RMSD(D from A) < 0.25 A while
    RMSD(B from A) > 0.5 A. This is the claimed advantage and the reason a
    pose-conditioned generator cannot simply relax its output.
P3  D LOSES to C and B on planarity. The stiff set is bonds and angles, and
    bond+angle B-vectors were derived to be first-order blind to aromatic
    out-of-plane distortion, whereas MMFF carries an explicit out-of-plane term.
    Predicted: D's aromatic deviation is within 20% of A's, i.e. essentially
    unchanged, while B and C reduce it substantially.
P4  D LOSES to B and C on clashes, or is neutral, because it has no nonbonded
    term. Predicted: D's clash count is within 10% of A's.

THE GATE. The method survives only if P1 AND P2 both hold. P1 alone would make
it a worse force field. P2 alone would make it a no-op that preserves the pose.
P3 and P4 are predicted LOSSES and are stated in advance so that the method's
scope is not overclaimed afterwards; if D unexpectedly wins them, the
implementation is doing something it was not designed to do and must be checked
before anything is reported.

If C matches D on both P1 and P2, the contribution collapses to "a restrained
force-field minimisation" and that must be reported as the outcome.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import wasserstein_distance

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
MIN_CLASS = 20


def bond_key(mol, bond):
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    elements = tuple(sorted((a.GetSymbol(), b.GetSymbol())))
    return (*elements, str(bond.GetBondType()), bond.GetIsAromatic())


def angle_key(mol, i, j, k):
    """Angle class.  RING SIZE IS PART OF THE KEY, and must be.

    A six-membered aromatic internal angle is about 120 degrees and a
    five-membered one about 108. Pooling them gives a median near 115, and
    three such targets at a trivalent centre sum to 345, below the 360 that a
    PLANAR centre requires. The only way to satisfy three too-small angles is to
    pyramidalise, so a ring-size-blind key does not merely fail to protect
    aromatic planarity, it actively destroys it: measured, the projection took
    the aromatic out-of-plane deviation from 4.14 to 13.36 degrees before this
    key was fixed.
    """
    atoms = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j), mol.GetAtomWithIdx(k)
    ends = tuple(sorted((atoms[0].GetSymbol(), atoms[2].GetSymbol())))
    ring = mol.GetRingInfo()
    smallest = 0
    for size in (3, 4, 5, 6, 7):
        if ring.IsAtomInRingOfSize(j, size):
            # the ring that carries BOTH ends, so an exocyclic angle at a ring
            # atom is not labelled with the ring it is not part of
            if ring.IsAtomInRingOfSize(i, size) and ring.IsAtomInRingOfSize(k, size):
                smallest = size
                break
    return (ends[0], atoms[1].GetSymbol(), ends[1], atoms[1].GetIsAromatic(),
            atoms[1].GetDegree(), smallest)


def enumerate_terms(mol):
    neighbours = defaultdict(list)
    bonds = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((a, b, bond_key(mol, bond)))
        neighbours[a].append(b)
        neighbours[b].append(a)
    angles = []
    for j, near in neighbours.items():
        for x in range(len(near)):
            for y in range(x + 1, len(near)):
                i, k = near[x], near[y]
                angles.append((i, j, k, angle_key(mol, i, j, k)))
    centres = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            near = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(near) == 3:
                centres.append((atom.GetIdx(), *near))
    return bonds, angles, centres


def geometry(p, bonds, angles, centres):
    lengths = [float(np.linalg.norm(p[a] - p[b])) for a, b, _ in bonds]
    degs = []
    for i, j, k, _ in angles:
        u, v = p[i] - p[j], p[k] - p[j]
        c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        degs.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    out = []
    for j, a, b, c in centres:
        u = p[a] - p[j]
        n = np.cross(p[b] - p[j], p[c] - p[j])
        n = n / (np.linalg.norm(n) + 1e-12)
        u = u / (np.linalg.norm(u) + 1e-12)
        out.append(abs(math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(u, n))))))))
    return lengths, degs, out



def improper_gradient(p, j, a, b, c):
    """Wilson out-of-plane B-vector at centre j, atom a out of the b-j-c plane.

    Derivation and verification in outputs/theory_modules_v1/RESULT.md 3.7:
    checked against central finite differences of the out-of-plane angle itself,
    median relative error 6.3e-11.  At planarity the tan terms vanish and all
    four vectors are parallel to the plane normal, so the row is the
    centroid-preserving umbrella mode and the update is SE(3)-equivariant
    exactly as the bond and angle updates are.

    Returns (chi_degrees, s_a, s_b, s_c, s_j) or None where it is degenerate.
    """
    va, vb, vc = p[a] - p[j], p[b] - p[j], p[c] - p[j]
    ra, rb, rc = (float(np.linalg.norm(v)) for v in (va, vb, vc))
    if min(ra, rb, rc) < 1e-8:
        return None
    ua, ub, uc = va / ra, vb / rb, vc / rc
    cross_bc = np.cross(ub, uc)
    sin_phi = float(np.linalg.norm(cross_bc))
    if sin_phi < 1e-6:
        return None
    cos_phi = float(np.clip(np.dot(ub, uc), -1.0, 1.0))
    sin_chi = float(np.clip(np.dot(cross_bc, ua) / sin_phi, -1.0, 1.0))
    chi = math.asin(sin_chi)
    cos_chi = math.cos(chi)
    if abs(cos_chi) < 1e-6:
        return None
    tan_chi = sin_chi / cos_chi
    denom = cos_chi * sin_phi
    s_a = (np.cross(ub, uc) / denom - tan_chi * ua) / ra
    s_b = (np.cross(uc, ua) / denom
           - (tan_chi / (sin_phi ** 2)) * (ub - cos_phi * uc)) / rb
    s_c = (np.cross(ua, ub) / denom
           - (tan_chi / (sin_phi ** 2)) * (uc - cos_phi * ub)) / rc
    s_j = -(s_a + s_b + s_c)
    return math.degrees(chi), s_a, s_b, s_c, s_j

def project(p, bonds, angles, ideal_bond, ideal_angle, sweeps=32, relax=0.5,
            centres=None, improper=False, soft=None, rng=None):
    """SE(3)-equivariant Gauss-Seidel projection of bonds and angles onto ideals.

    Every update moves atoms along the analytic gradient of an INVARIANT
    (a length or an angle), built from unit vectors, and each term's updates sum
    to zero over its own atoms, so the operator commutes with rotation and
    translation and does not move the centroid. Under-relaxation is required:
    bonds and angles are a redundant system and un-relaxed Gauss-Seidel on
    redundant constraints diverges.
    """
    p = p.copy()
    for _ in range(sweeps):
        for a, b, key in bonds:
            target = ideal_bond.get(key)
            if target is None:
                continue
            if soft is not None:
                # SOFT target.  Driving every instance of a class to its median
                # collapses the class distribution, which the deviation-from-
                # median score cannot see but a distributional score can.  The
                # target is drawn once per term at the class's PHYSICAL width,
                # so the operator reproduces the spread instead of erasing it.
                target = target + rng.normal(0.0, soft[0].get(key, 0.0))
            d = p[a] - p[b]
            length = float(np.linalg.norm(d))
            if length < 1e-8:
                continue
            step = relax * 0.5 * (target - length) * (d / length)
            p[a] += step
            p[b] -= step
        for i, j, k, key in angles:
            target = ideal_angle.get(key)
            if target is None:
                continue
            if soft is not None:
                target = target + rng.normal(0.0, soft[1].get(key, 0.0))
            u, v = p[i] - p[j], p[k] - p[j]
            ru, rv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
            if ru < 1e-8 or rv < 1e-8:
                continue
            uu, vv = u / ru, v / rv
            cosine = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
            sine = max(math.sqrt(max(1.0 - cosine * cosine, 1e-12)), 1e-6)
            current = math.degrees(math.acos(cosine))
            gi = -(vv - cosine * uu) / (ru * sine)
            gk = -(uu - cosine * vv) / (rv * sine)
            gj = -(gi + gk)
            norm2 = float(np.dot(gi, gi) + np.dot(gj, gj) + np.dot(gk, gk))
            if norm2 < 1e-12:
                continue
            scale = relax * math.radians(current - target) / norm2
            p[i] -= scale * gi
            p[j] -= scale * gj
            p[k] -= scale * gk
        if improper and centres:
            # Target is planarity, chi = 0.  B_oop is exactly orthogonal to every
            # bond and angle row at planarity, so this step is the one direction
            # the existing stiff set cannot see -- and, as its own control, the
            # bond and angle channels must barely move when it is switched on.
            for j, a, b, c in centres:
                built = improper_gradient(p, j, a, b, c)
                if built is None:
                    continue
                chi, s_a, s_b, s_c, s_j = built
                norm2 = float(sum(np.dot(v, v) for v in (s_a, s_b, s_c, s_j)))
                if norm2 < 1e-12:
                    continue
                scale = relax * math.radians(chi) / norm2
                p[a] -= scale * s_a
                p[b] -= scale * s_b
                p[c] -= scale * s_c
                p[j] -= scale * s_j
    return p


def build_widths(raw):
    """Per-class physical spread, the target WIDTH rather than a target point."""
    return {k: float(np.std(v)) for k, v in raw.items() if len(v) >= MIN_CLASS}


def build_ideals(paths):
    lengths, degs = defaultdict(list), defaultdict(list)
    for path in paths:
        for mol in Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True):
            if mol is None or mol.GetNumConformers() == 0:
                continue
            p = mol.GetConformer().GetPositions()
            bonds, angles, _ = enumerate_terms(mol)
            for a, b, key in bonds:
                lengths[key].append(float(np.linalg.norm(p[a] - p[b])))
            for i, j, k, key in angles:
                u, v = p[i] - p[j], p[k] - p[j]
                c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
                degs[key].append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    ideal_bond = {k: float(np.median(v)) for k, v in lengths.items() if len(v) >= MIN_CLASS}
    ideal_angle = {k: float(np.median(v)) for k, v in degs.items() if len(v) >= MIN_CLASS}
    return ideal_bond, ideal_angle, lengths, degs


def rms_deviation(values, keys, ideal):
    errors = [v - ideal[k] for v, k in zip(values, keys) if k in ideal]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")


def receptor_atoms(path):
    coords = []
    with open(path) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[76:78].strip() != "H":
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(coords) if coords else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--receptor-root", type=Path, default=None)
    parser.add_argument("--max-molecules", type=int, default=600)
    parser.add_argument("--max-per-pocket", type=int, default=6)
    parser.add_argument("--sweeps", type=int, default=32)
    parser.add_argument("--holdout", action="store_true",
                        help="fit targets on half the reference ligands, score on the other half")
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # rglob: the CrossDocked test ligands live one directory down, per target.
    reference_files = sorted(args.reference_dir.rglob("*.sdf"))
    if not reference_files:
        raise SystemExit(f"no reference .sdf under {args.reference_dir}")
    if args.holdout:
        # The operator's targets are ESTIMATED from reference ligands and the
        # Wasserstein score is computed AGAINST reference ligands.  Using one
        # sample for both fits the operator to the very distribution it is
        # scored on, which MMFF does not get, so the margin would be optimistic.
        # Split the ligands: fit on one half, score on the other.
        rng = np.random.default_rng(args.split_seed)
        order = rng.permutation(len(reference_files))
        half = len(reference_files) // 2
        fit_files = [reference_files[i] for i in order[:half]]
        score_files = [reference_files[i] for i in order[half:]]
    else:
        fit_files = score_files = reference_files
    ideal_bond, ideal_angle, raw_b_fit, raw_a_fit = build_ideals(fit_files)
    width_bond, width_angle = build_widths(raw_b_fit), build_widths(raw_a_fit)
    _, _, raw_b, raw_a = build_ideals(score_files)
    print(f"reference ligands: fit {len(fit_files)} | score {len(score_files)} | "
          f"holdout={bool(args.holdout)}")
    if not ideal_bond or not ideal_angle:
        raise SystemExit("ideal lookup is empty; nothing can be projected")
    print(f"reference files {len(reference_files)} | "
          f"bond classes {len(ideal_bond)} | angle classes {len(ideal_angle)}")
    rows = {arm: defaultdict(list) for arm in "ABCDEFG"}
    pooled = {arm: {"bond": defaultdict(list), "angle": defaultdict(list)}
              for arm in "ABCDEFG"}
    n_mols = 0
    receptors: dict[str, np.ndarray | None] = {}

    for path in sorted(args.generated_dir.glob("*.sdf")):
        if n_mols >= args.max_molecules:
            break
        pocket = path.name.split("-")[0]
        if args.receptor_root is not None and pocket not in receptors:
            hits = list(args.receptor_root.glob(f"*/{pocket}_*_rec.pdb"))
            receptors[pocket] = receptor_atoms(hits[0]) if hits else None
        rec = receptors.get(pocket)
        kept = 0
        for mol in Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True):
            if mol is None or mol.GetNumConformers() == 0:
                continue
            if kept >= args.max_per_pocket or n_mols >= args.max_molecules:
                break
            bonds, angles, centres = enumerate_terms(mol)
            if not bonds or not angles:
                continue
            base = mol.GetConformer().GetPositions()
            arms = {"A": base}
            # Restraint 5 added after the theory pass showed the sweep has a
            # sweet spot the original 0-and-100 comparison missed entirely.
            for arm, restraint in (("B", 0.0), ("C", 100.0), ("G", 5.0)):
                copy = Chem.Mol(mol)
                properties = AllChem.MMFFGetMoleculeProperties(copy)
                if properties is None:
                    break
                field = AllChem.MMFFGetMoleculeForceField(copy, properties)
                if field is None:
                    break
                if restraint > 0.0:
                    for index in range(copy.GetNumAtoms()):
                        field.MMFFAddPositionConstraint(index, 0.0, restraint)
                field.Minimize(maxIts=2000)
                arms[arm] = copy.GetConformer().GetPositions()
            if len(arms) < 4:
                continue
            arms["D"] = project(base, bonds, angles, ideal_bond, ideal_angle,
                                sweeps=args.sweeps)
            arms["E"] = project(base, bonds, angles, ideal_bond, ideal_angle,
                                sweeps=args.sweeps, centres=centres, improper=True)
            arms["F"] = project(base, bonds, angles, ideal_bond, ideal_angle,
                                sweeps=args.sweeps, centres=centres, improper=True,
                                soft=(width_bond, width_angle),
                                rng=np.random.default_rng(20260924))
            bkeys = [k for _, _, k in bonds]
            akeys = [k for *_, k in angles]
            for arm, p in arms.items():
                lengths, degs, out = geometry(p, bonds, angles, centres)
                for value, key in zip(lengths, [k for _, _, k in bonds]):
                    pooled[arm]["bond"][key].append(value)
                for value, key in zip(degs, [k for *_, k in angles]):
                    pooled[arm]["angle"][key].append(value)
                rows[arm]["bond_rms"].append(rms_deviation(lengths, bkeys, ideal_bond))
                rows[arm]["angle_rms"].append(rms_deviation(degs, akeys, ideal_angle))
                rows[arm]["planarity"].append(float(np.mean(out)) if out else np.nan)
                rows[arm]["rmsd_from_raw"].append(
                    float(np.sqrt(np.mean(np.sum((p - base) ** 2, axis=1))))
                )
                if rec is not None:
                    d = np.linalg.norm(p[:, None, :] - rec[None, :, :], axis=-1)
                    rows[arm]["clashes"].append(float((d < 2.0).sum()))
            kept += 1
            n_mols += 1

    # DISTRIBUTIONAL metric, which NEITHER arm optimises.  The deviation-from-
    # ideal score is the projection's own objective, so winning it proves
    # nothing; worse, an operator that collapses every bond onto the class
    # median would score perfectly there while destroying the distribution.
    # Wasserstein distance to the REAL ligand distribution, per class, weighted
    # by class count, catches exactly that failure.
    def distributional(arm_values):
        total = weight = 0.0
        for key, observed in arm_values.items():
            reference = raw_b.get(key) or raw_a.get(key)
            if reference is None or len(reference) < MIN_CLASS or len(observed) < 5:
                continue
            total += len(observed) * wasserstein_distance(observed, reference)
            weight += len(observed)
        return total / weight if weight else None

    def summarise(arm):
        out = {}
        for key, values in rows[arm].items():
            clean = [v for v in values if np.isfinite(v)]
            out[key] = float(np.median(clean)) if clean else None
        out["wasserstein_bond"] = distributional(pooled[arm]["bond"])
        out["wasserstein_angle"] = distributional(pooled[arm]["angle"])
        return out

    report = {
        "n_molecules": n_mols,
        "sweeps": args.sweeps,
        "ideal_classes": {"bond": len(ideal_bond), "angle": len(ideal_angle)},
        "arms": {
            "A_raw": summarise("A"),
            "B_mmff_free": summarise("B"),
            "C_mmff_restrained": summarise("C"),
            "D_projection": summarise("D"),
            "E_projection_plus_improper": summarise("E"),
            "F_soft_target": summarise("F"),
            "G_mmff_restraint5": summarise("G"),
        },
    }
    print(json.dumps(report["arms"], indent=2))
    a, b, c, d, e = (report["arms"][k] for k in
                     ("A_raw", "B_mmff_free", "C_mmff_restrained", "D_projection",
                      "E_projection_plus_improper"))
    if any(v is None for arm in (a, b, c, d) for v in arm.values()):
        args.output.write_text(json.dumps(report, indent=2))
        raise SystemExit("an arm produced no finite values; see the dump above")
    report["gate"] = {
        "P1_bond_within_1.5x_of_C": bool(d["bond_rms"] <= 1.5 * c["bond_rms"]),
        "P1_angle_within_1.5x_of_C": bool(d["angle_rms"] <= 1.5 * c["angle_rms"]),
        "P1_bond_30pct_below_A": bool(d["bond_rms"] <= 0.7 * a["bond_rms"]),
        "P1_angle_30pct_below_A": bool(d["angle_rms"] <= 0.7 * a["angle_rms"]),
        "P2_D_rmsd_below_0.25": bool(d["rmsd_from_raw"] < 0.25),
        "P2_B_rmsd_above_0.5": bool(b["rmsd_from_raw"] > 0.5),
        "P3_D_planarity_unchanged": bool(
            abs(d["planarity"] - a["planarity"]) <= 0.2 * a["planarity"]),
        "E_planarity_below_raw": bool(e["planarity"] < a["planarity"]),
        "E_planarity_beats_C": bool(e["planarity"] < c["planarity"]),
        "E_control_angle_unmoved_vs_D": bool(
            abs(e["angle_rms"] - d["angle_rms"]) <= 0.25 * d["angle_rms"]),
        "E_control_bond_unmoved_vs_D": bool(
            abs(e["bond_rms"] - d["bond_rms"]) <= 0.25 * d["bond_rms"]),
        "C_matches_D_on_both": bool(
            c["bond_rms"] <= d["bond_rms"] * 1.1
            and c["angle_rms"] <= d["angle_rms"] * 1.1
            and c["rmsd_from_raw"] <= d["rmsd_from_raw"] * 1.1
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
