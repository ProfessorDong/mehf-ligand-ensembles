"""MMFF94 relaxation helpers: the ORIGINAL heavy-atom-only protocol (verbatim logic)
and the CORRECTED explicit-hydrogen protocol. Both return heavy-atom coordinates in
the input heavy-atom order, plus the Minimize() return code (0 = converged)."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def relax_heavy_only(mol, restraint, max_its=2000):
    """Original protocol, unchanged: MMFF on the H-stripped molecule, restraint on every atom."""
    copy = Chem.Mol(mol)
    properties = AllChem.MMFFGetMoleculeProperties(copy)
    if properties is None:
        return None, "setup_failed_properties"
    field = AllChem.MMFFGetMoleculeForceField(copy, properties)
    if field is None:
        return None, "setup_failed_field"
    if restraint > 0.0:
        for index in range(copy.GetNumAtoms()):
            field.MMFFAddPositionConstraint(index, 0.0, restraint)
    code = field.Minimize(maxIts=max_its)
    return copy.GetConformer().GetPositions(), int(code)


def relax_explicit_h(mol, restraint, max_its=2000):
    """Corrected protocol: AddHs(addCoords=True); MMFF94 on the H-complete molecule;
    harmonic position restraint k on HEAVY atoms only (H free); Minimize(maxIts);
    RemoveHs; return heavy coordinates (order checked against the input)."""
    n_heavy = mol.GetNumAtoms()
    heavy_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    try:
        withh = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    except Exception as exc:  # pragma: no cover - recorded as a failure
        return None, f"addhs_failed:{type(exc).__name__}"
    properties = AllChem.MMFFGetMoleculeProperties(withh)
    if properties is None:
        return None, "setup_failed_properties"
    field = AllChem.MMFFGetMoleculeForceField(withh, properties)
    if field is None:
        return None, "setup_failed_field"
    if restraint > 0.0:
        for atom in withh.GetAtoms():
            if atom.GetAtomicNum() > 1:
                field.MMFFAddPositionConstraint(atom.GetIdx(), 0.0, restraint)
    code = field.Minimize(maxIts=max_its)
    stripped = Chem.RemoveHs(withh)
    if (stripped.GetNumAtoms() != n_heavy
            or [a.GetSymbol() for a in stripped.GetAtoms()] != heavy_symbols):
        return None, "removehs_order_mismatch"
    return stripped.GetConformer().GetPositions(), int(code)


def h_counts(mol):
    withh = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    return withh.GetNumAtoms() - mol.GetNumAtoms()
