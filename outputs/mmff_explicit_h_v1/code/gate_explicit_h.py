"""tab:gates, per split seed: original arms A-G (logic verbatim from
MEHF/scripts/geometry_corrector_gate.py, final version) PLUS corrected explicit-H MMFF
arms HB (free), HC (k=100), HG (k=5) on the identical molecule set with the identical
summarise() code. Molecule selection is the original one (heavy-atom MMFF setup must
succeed), so arms A-G must reproduce the reference records exactly."""
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
from geometry_corrector_gate import (  # noqa: E402  (byte-identical copy)
    MIN_CLASS, build_ideals, build_widths, enumerate_terms, geometry, project,
    receptor_atoms, rms_deviation,
)
from mmffh import relax_explicit_h, relax_heavy_only  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ORIG_MMFF = (("B", 0.0), ("C", 100.0), ("G", 5.0))
H_MMFF = (("HB", 0.0), ("HC", 100.0), ("HG", 5.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--receptor-root", type=Path, default=None)
    parser.add_argument("--max-molecules", type=int, default=2000)
    parser.add_argument("--max-per-pocket", type=int, default=20)
    parser.add_argument("--sweeps", type=int, default=32)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--extra-maxits", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_files = sorted(args.reference_dir.rglob("*.sdf"))
    rng = np.random.default_rng(args.split_seed)          # --holdout, as in 05 script
    order = rng.permutation(len(reference_files))
    half = len(reference_files) // 2
    fit_files = [reference_files[i] for i in order[:half]]
    score_files = [reference_files[i] for i in order[half:]]
    ideal_bond, ideal_angle, raw_b_fit, raw_a_fit = build_ideals(fit_files)
    width_bond, width_angle = build_widths(raw_b_fit), build_widths(raw_a_fit)
    _, _, raw_b, raw_a = build_ideals(score_files)

    extra = []
    if args.extra_maxits:
        extra = [(f"HB_its{args.extra_maxits}", 0.0), (f"HG_its{args.extra_maxits}", 5.0)]
    arm_names = list("ABCDEFG") + [a for a, _ in H_MMFF] + [a for a, _ in extra]
    rows = {arm: defaultdict(list) for arm in arm_names}
    pooled = {arm: {"bond": defaultdict(list), "angle": defaultdict(list)} for arm in arm_names}
    status = {arm: defaultdict(int) for arm in ("B", "C", "G") + tuple(a for a, _ in H_MMFF)
              + tuple(a for a, _ in extra)}
    per_molecule = []
    n_mols = 0
    receptors = {}

    for path in sorted(args.generated_dir.glob("*.sdf")):
        if n_mols >= args.max_molecules:
            break
        pocket = path.name.split("-")[0]
        if args.receptor_root is not None and pocket not in receptors:
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
            arms = {"A": base}
            codes = {}
            for arm, restraint in ORIG_MMFF:
                p, code = relax_heavy_only(mol, restraint)
                if p is None:
                    break
                arms[arm] = p
                codes[arm] = code
            if len(arms) < 4:
                continue
            for arm, restraint in H_MMFF:
                p, code = relax_explicit_h(mol, restraint)
                codes[arm] = code
                if p is not None:
                    arms[arm] = p
            for arm, restraint in extra:
                p, code = relax_explicit_h(mol, restraint, max_its=args.extra_maxits)
                codes[arm] = code
                if p is not None:
                    arms[arm] = p
            arms["D"] = project(base, bonds, angles, ideal_bond, ideal_angle,
                                sweeps=args.sweeps)
            arms["E"] = project(base, bonds, angles, ideal_bond, ideal_angle,
                                sweeps=args.sweeps, centres=centres, improper=True)
            arms["F"] = project(base, bonds, angles, ideal_bond, ideal_angle,
                                sweeps=args.sweeps, centres=centres, improper=True,
                                soft=(width_bond, width_angle),
                                rng=np.random.default_rng(20260924))
            for arm, code in codes.items():
                status[arm][str(code)] += 1
            per_molecule.append({"file": path.name, "index": mol_index,
                                 "minimize_code": codes})
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
                    float(np.sqrt(np.mean(np.sum((p - base) ** 2, axis=1)))))
                if rec is not None:
                    d = np.linalg.norm(p[:, None, :] - rec[None, :, :], axis=-1)
                    rows[arm]["clashes"].append(float((d < 2.0).sum()))
            kept += 1
            n_mols += 1

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

    def extras(arm):   # additional fields, NOT in the original records
        c = rows[arm]["clashes"]
        return {"n_evaluated": len(rows[arm]["angle_rms"]), "n_with_receptor": len(c),
                "clash_mean": float(np.mean(c)) if c else None,
                "frac_with_clash": float(np.mean(np.array(c) > 0)) if c else None}

    names = {"A": "A_raw", "B": "B_mmff_free", "C": "C_mmff_restrained", "D": "D_projection",
             "E": "E_projection_plus_improper", "F": "F_soft_target", "G": "G_mmff_restraint5",
             "HB": "HB_mmffH_free", "HC": "HC_mmffH_restraint100", "HG": "HG_mmffH_restraint5"}
    for arm, _ in extra:
        names[arm] = arm
    report = {"n_molecules": n_mols, "sweeps": args.sweeps, "split_seed": args.split_seed,
              "ideal_classes": {"bond": len(ideal_bond), "angle": len(ideal_angle)},
              "arms": {names[a]: summarise(a) for a in arm_names},
              "extra_fields": {names[a]: extras(a) for a in arm_names},
              "minimize_status": {names[a]: dict(v) for a, v in status.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_name(args.output.stem + "_per_molecule.json").write_text(
        json.dumps(per_molecule))
    print(json.dumps({"n": n_mols, "status": report["minimize_status"]}))


if __name__ == "__main__":
    main()
