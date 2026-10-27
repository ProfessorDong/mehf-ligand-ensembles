"""tab:ff restraint sweep, original heavy-atom MMFF arms AND corrected explicit-H MMFF
arms on the identical molecule set, with identical metric code (copied verbatim from
e_ff_restraint_sweep.py). Molecule selection is the original one (incl. heavy-atom
MMFF setup success), so the 'raw' and 'mmff_*' arms must reproduce the original JSON."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "orig" / "scripts"))
from geometry_corrector_gate import (  # noqa: E402  (byte-identical copy of MEHF/scripts)
    MIN_CLASS, build_ideals, enumerate_terms, geometry, receptor_atoms,
)
from mmffh import relax_explicit_h, relax_heavy_only  # noqa: E402

RDLogger.DisableLog("rdApp.*")
RESTRAINTS = [0.0, 0.2, 1.0, 5.0, 20.0, 100.0, 1000.0]


def kabsch_rmsd(p, q):
    pc = p - p.mean(0)
    qc = q - q.mean(0)
    u, _, vt = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = pc @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - qc) ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--receptor-root", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=200)
    parser.add_argument("--max-per-pocket", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--extra-maxits", type=int, default=0,
                        help="if >0, add explicit-H arms k=0 and k=5 at this maxIts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_files = sorted(args.reference_dir.rglob("*.sdf"))
    rng = np.random.default_rng(args.split_seed)
    order = rng.permutation(len(reference_files))
    half = len(reference_files) // 2
    score_files = [reference_files[i] for i in order[half:]]
    _, _, raw_b, raw_a = build_ideals(score_files)

    h_arms = [f"mmffH_{r:g}" for r in RESTRAINTS]
    extra = []
    if args.extra_maxits:
        extra = [(f"mmffH_{r:g}_its{args.extra_maxits}", r) for r in (0.0, 5.0)]
    arms = ["raw"] + [f"mmff_{r:g}" for r in RESTRAINTS] + h_arms + [a for a, _ in extra]
    rows = {a: defaultdict(list) for a in arms}
    pooled = {a: {"bond": defaultdict(list), "angle": defaultdict(list)} for a in arms}
    status = {a: defaultdict(int) for a in arms if a != "raw"}
    per_molecule = []
    receptors = {}
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
        for mol_index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)):
            if mol is None or mol.GetNumConformers() == 0:
                continue
            if kept >= args.max_per_pocket or n_mols >= args.max_molecules:
                break
            bonds, angles, centres = enumerate_terms(mol)
            if not bonds or not angles:
                continue
            base = mol.GetConformer().GetPositions()
            positions = {"raw": base}
            codes = {}
            failed = False
            for restraint in RESTRAINTS:            # ORIGINAL arms + original selection rule
                p, code = relax_heavy_only(mol, restraint)
                if p is None:
                    failed = True
                    break
                positions[f"mmff_{restraint:g}"] = p
                codes[f"mmff_{restraint:g}"] = code
            if failed:
                continue
            for restraint in RESTRAINTS:            # CORRECTED arms
                p, code = relax_explicit_h(mol, restraint)
                codes[f"mmffH_{restraint:g}"] = code
                if p is not None:
                    positions[f"mmffH_{restraint:g}"] = p
            for name, restraint in extra:
                p, code = relax_explicit_h(mol, restraint, max_its=args.extra_maxits)
                codes[name] = code
                if p is not None:
                    positions[name] = p
            for arm, code in codes.items():
                status[arm][str(code)] += 1
            per_molecule.append({"file": path.name, "index": mol_index,
                                 "n_heavy": mol.GetNumAtoms(), "minimize_code": codes})
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
              "reference_score_files": len(score_files), "arms": {},
              "minimize_status": {a: dict(v) for a, v in status.items()}}
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
        entry["n_evaluated"] = len(rows[arm]["planarity"])
        entry["n_with_receptor"] = len(rows[arm]["clashes"])
        report["arms"][arm] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_name(args.output.stem + "_per_molecule.json").write_text(
        json.dumps(per_molecule))
    for arm in arms:
        e = report["arms"][arm]
        st = report["minimize_status"].get(arm, {})
        print(f"{arm:22s} n={e['n_evaluated']:4d} W(b)={e['wasserstein_bond']:.4f} "
              f"W(a)={e['wasserstein_angle']:.3f} plan={e['planarity']:.3f} "
              f"RMSD={e['rmsd_from_raw']:.3f} clash%={100*e['frac_with_clash']:.1f} status={st}")


if __name__ == "__main__":
    main()
