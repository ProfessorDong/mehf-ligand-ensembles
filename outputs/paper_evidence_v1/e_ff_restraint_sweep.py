"""(e) Is force-field relaxation destructive?  Against the STRONGEST version.

`corrector_gate_v1` measured 1.675 A of displacement and 2 receptor clashes per
molecule for UNRESTRAINED MMFF.  That is the weakest form of the practice.
Published pipelines commonly use a restrained or local minimisation, and the
same directory already contains one restrained arm (100 kcal/mol/A^2) that moves
0.153 A and introduces no clashes.  A claim that "force-field relaxation is
destructive" therefore has to be tested against the whole restraint family, not
against its worst member.

This sweeps the positional restraint and reports, on the same molecules and the
same metrics:

  W(bond), W(angle)   class-count-weighted Wasserstein distance to HELD-OUT real
                      CrossDocked ligands -- a metric no arm optimises
  planarity           mean |out-of-plane| at aromatic centres
  rmsd_from_raw       heavy-atom RMSD from the model's own output, no realignment
                      (this is the pose displacement)
  rmsd_internal       the same after optimal superposition.  This decomposes the
                      displacement: whatever survives alignment is a change of
                      the molecule's own shape, which no receptor term and no
                      in-pocket minimisation can prevent.  If most of the 1.675 A
                      survives alignment, "just minimise it inside the pocket"
                      does not rescue the practice.
  clashes             ligand-receptor heavy-atom pairs below 2.0 A
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from geometry_corrector_gate import (  # noqa: E402
    MIN_CLASS, build_ideals, enumerate_terms, geometry, receptor_atoms,
)

RDLogger.DisableLog("rdApp.*")
RESTRAINTS = [0.0, 0.2, 1.0, 5.0, 20.0, 100.0, 1000.0]


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    pc = p - p.mean(0)
    qc = q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = pc @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - qc) ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path,
                        default=ROOT / "data/external/diffsbdd/crossdocked_fullatom_cond")
    parser.add_argument("--reference-dir", type=Path,
                        default=ROOT / "data/crossdocked/crossdocked_pocket10")
    parser.add_argument("--receptor-root", type=Path,
                        default=ROOT / "data/crossdocked/crossdocked_pocket10")
    parser.add_argument("--max-molecules", type=int, default=200)
    parser.add_argument("--max-per-pocket", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "e_ff_restraint_sweep.json")
    args = parser.parse_args()

    reference_files = sorted(args.reference_dir.rglob("*.sdf"))
    if not reference_files:
        raise SystemExit(f"no reference .sdf under {args.reference_dir}")
    rng = np.random.default_rng(args.split_seed)
    order = rng.permutation(len(reference_files))
    half = len(reference_files) // 2
    score_files = [reference_files[i] for i in order[half:]]
    _, _, raw_b, raw_a = build_ideals(score_files)
    print(f"scoring reference ligands: {len(score_files)}", flush=True)

    arms = ["raw"] + [f"mmff_{r:g}" for r in RESTRAINTS]
    rows = {a: defaultdict(list) for a in arms}
    pooled = {a: {"bond": defaultdict(list), "angle": defaultdict(list)} for a in arms}
    receptors: dict[str, np.ndarray | None] = {}
    n_mols = 0

    for path in sorted(args.generated_dir.glob("*.sdf")):
        if n_mols >= args.max_molecules:
            break
        pocket = path.name.split("-")[0]
        if pocket not in receptors:
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
            positions = {"raw": base}
            failed = False
            for restraint in RESTRAINTS:
                copy = Chem.Mol(mol)
                properties = AllChem.MMFFGetMoleculeProperties(copy)
                if properties is None:
                    failed = True
                    break
                field = AllChem.MMFFGetMoleculeForceField(copy, properties)
                if field is None:
                    failed = True
                    break
                if restraint > 0.0:
                    for index in range(copy.GetNumAtoms()):
                        field.MMFFAddPositionConstraint(index, 0.0, restraint)
                field.Minimize(maxIts=2000)
                positions[f"mmff_{restraint:g}"] = copy.GetConformer().GetPositions()
            if failed:
                continue
            for arm, p in positions.items():
                lengths, degs, out = geometry(p, bonds, angles, centres)
                for value, key in zip(lengths, [k for _, _, k in bonds]):
                    pooled[arm]["bond"][key].append(value)
                for value, key in zip(degs, [k for *_, k in angles]):
                    pooled[arm]["angle"][key].append(value)
                rows[arm]["planarity"].append(float(np.mean(out)) if out else np.nan)
                rows[arm]["rmsd_from_raw"].append(
                    float(np.sqrt(np.mean(np.sum((p - base) ** 2, axis=1)))))
                rows[arm]["rmsd_internal"].append(kabsch_rmsd(p, base))
                if rec is not None:
                    d = np.linalg.norm(p[:, None, :] - rec[None, :, :], axis=-1)
                    rows[arm]["clashes"].append(float((d < 2.0).sum()))
                    rows[arm]["min_receptor_distance"].append(float(d.min()))
            kept += 1
            n_mols += 1

    def distributional(values, reference_pool):
        total = weight = 0.0
        for key, observed in values.items():
            reference = reference_pool.get(key)
            if reference is None or len(reference) < MIN_CLASS or len(observed) < 5:
                continue
            total += len(observed) * wasserstein_distance(observed, reference)
            weight += len(observed)
        return total / weight if weight else None

    report = {"n_molecules": n_mols, "restraints": RESTRAINTS,
              "reference_score_files": len(score_files), "arms": {}}
    for arm in arms:
        entry = {}
        for key, values in rows[arm].items():
            clean = [v for v in values if np.isfinite(v)]
            entry[key] = float(np.median(clean)) if clean else None
        entry["clash_mean"] = (float(np.mean(rows[arm]["clashes"]))
                               if rows[arm]["clashes"] else None)
        entry["frac_with_clash"] = (float(np.mean(np.array(rows[arm]["clashes"]) > 0))
                                    if rows[arm]["clashes"] else None)
        entry["wasserstein_bond"] = distributional(pooled[arm]["bond"], raw_b)
        entry["wasserstein_angle"] = distributional(pooled[arm]["angle"], raw_a)
        report["arms"][arm] = entry
    args.output.write_text(json.dumps(report, indent=2))
    print(f"{'arm':12s} {'W(bond)':>9s} {'W(angle)':>9s} {'planar':>8s} "
          f"{'RMSD':>7s} {'RMSDint':>8s} {'clash':>7s} {'clash%':>7s}")
    for arm in arms:
        e = report["arms"][arm]
        fmt = lambda v, w, p: (f"{v:{w}.{p}f}" if v is not None else " " * w)
        print(f"{arm:12s} {fmt(e['wasserstein_bond'],9,4)} {fmt(e['wasserstein_angle'],9,3)} "
              f"{fmt(e['planarity'],8,3)} {fmt(e['rmsd_from_raw'],7,3)} "
              f"{fmt(e['rmsd_internal'],8,3)} {fmt(e['clash_mean'],7,2)} "
              f"{fmt(e['frac_with_clash'],7,2)}")


if __name__ == "__main__":
    main()
