"""Build the parent->product ensemble-transfer corpus from MISATO matched pairs.

For every matched molecular pair whose two sides are both clean, extracted
systems, compute the heavy-atom atom mapping directly by MCS on the graphs
inferred from the stored coordinates, then cross-check it against the
independently derived MMP transformation recorded by the census: the net
heavy-atom change implied by the mapping must equal the net change implied by
the transformation SMARTS. Pairs that agree form the high-confidence corpus.

The census transformation is used ONLY as a check, never to build the mapping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

from mdwm.data.misato_atomic import infer_covalent_bonds

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data/MISATO/extracted/frames100_all_v1"
MANIFEST = ROOT / "data/MISATO/manifest/frames100_all_v1/manifest.jsonl"
CENSUS = ROOT / "outputs/misato_mmp_census_v1/mmp_pairs.csv"
OUT = ROOT / "outputs/mmp_transfer_corpus_v1"
MCS_TIMEOUT = 10


def heavy_graph(pdb: str) -> tuple[Chem.Mol, np.ndarray]:
    """Heavy-atom RDKit graph from the stored frame 0, plus the heavy indices."""
    with np.load(STORE / f"{pdb}.npz") as archive:
        numbers = archive["ligand_atomic_numbers"]
        coordinates = archive["ligand_coords"][0].astype(np.float64)
    heavy = np.flatnonzero(numbers > 1)
    position = {int(a): i for i, a in enumerate(heavy)}
    editable = Chem.RWMol()
    for number in numbers[heavy]:
        editable.AddAtom(Chem.Atom(int(number)))
    for a, b in infer_covalent_bonds(coordinates, numbers):
        a, b = int(a), int(b)
        if a in position and b in position:
            editable.AddBond(position[a], position[b], Chem.BondType.SINGLE)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule,
                     Chem.SANITIZE_SYMMRINGS | Chem.SANITIZE_ADJUSTHS)
    return molecule, heavy


def fragment_heavy_atoms(smarts: str) -> int | None:
    fragment = Chem.MolFromSmarts(smarts)
    if fragment is None:
        return None
    return sum(1 for atom in fragment.GetAtoms() if atom.GetAtomicNum() > 1)


def main() -> None:
    clean: dict[str, str] = {}
    for line in MANIFEST.open():
        record = json.loads(line)
        if record.get("status") != "written":
            continue
        blob = json.dumps(record)
        if ('"ligand_looks_peptidic": true' in blob
                or '"periodic_image_jump_suspected": true' in blob):
            continue
        clean[record["system_id"].upper()] = record.get("split", "?")

    census = pd.read_csv(CENSUS)
    census["A"] = census["pdb_a"].astype(str).str.upper()
    census["B"] = census["pdb_b"].astype(str).str.upper()
    usable = census[census["A"].isin(clean) & census["B"].isin(clean)]
    groups = list(usable.groupby(["A", "B"]))
    print(f"{len(usable)} rows -> {len(groups)} unique clean pairs", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, mappings = [], {}
    for index, ((a_id, b_id), group) in enumerate(groups):
        row = {"pdb_a": a_id, "pdb_b": b_id,
               "uniprot": group["uniprot"].iloc[0],
               "split_a": clean[a_id], "split_b": clean[b_id],
               "n_transformations": len(group),
               "transformations": "|".join(group["transformation"].astype(str))}
        try:
            mol_a, heavy_a = heavy_graph(a_id)
            mol_b, heavy_b = heavy_graph(b_id)
            result = rdFMCS.FindMCS(
                [mol_a, mol_b], timeout=MCS_TIMEOUT, ringMatchesRingOnly=True,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareAny)
            if result.canceled or result.numAtoms == 0:
                row["status"] = "mcs_failed"
                rows.append(row)
                continue
            pattern = Chem.MolFromSmarts(result.smartsString)
            match_a = mol_a.GetSubstructMatch(pattern)
            match_b = mol_b.GetSubstructMatch(pattern)
            if len(match_a) != len(match_b) or not match_a:
                row["status"] = "substructure_mismatch"
                rows.append(row)
                continue
            unmapped_a = mol_a.GetNumAtoms() - len(match_a)
            unmapped_b = mol_b.GetNumAtoms() - len(match_b)
            agree = False
            for transformation in group["transformation"].astype(str):
                if ">>" not in transformation:
                    continue
                left, right = transformation.split(">>")
                size_a = fragment_heavy_atoms(left)
                size_b = fragment_heavy_atoms(right)
                if size_a is None or size_b is None:
                    continue
                if (unmapped_a - unmapped_b) == (size_a - size_b):
                    agree = True
                    break
            row.update({
                "status": "ok",
                "heavy_a": mol_a.GetNumAtoms(), "heavy_b": mol_b.GetNumAtoms(),
                "n_mapped": len(match_a),
                "unmapped_a": unmapped_a, "unmapped_b": unmapped_b,
                "mcs_fraction": len(match_a) / max(mol_a.GetNumAtoms(),
                                                   mol_b.GetNumAtoms()),
                "transformation_agrees": agree})
            # store the mapping in ORIGINAL npz atom indices, both directions
            mappings[f"{a_id}|{b_id}"] = {
                "a_atom_indices": [int(heavy_a[i]) for i in match_a],
                "b_atom_indices": [int(heavy_b[i]) for i in match_b]}
        except Exception as error:                       # noqa: BLE001
            row["status"] = f"error:{type(error).__name__}"
        rows.append(row)
        if index % 250 == 0:
            print(f"  {index}/{len(groups)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pairs.csv", index=False)
    (OUT / "atom_mappings.json").write_text(json.dumps(mappings))

    good = frame[(frame["status"] == "ok") & frame["transformation_agrees"]]
    same_split = good[good["split_a"] == good["split_b"]]
    summary = {
        "n_unique_clean_pairs": len(groups),
        "status_counts": frame["status"].value_counts().to_dict(),
        "n_mcs_ok": int((frame["status"] == "ok").sum()),
        "n_transformation_agrees": int(len(good)),
        "agreement_rate_among_ok": float(
            good.shape[0] / max((frame["status"] == "ok").sum(), 1)),
        "n_same_split": int(len(same_split)),
        "split_counts": same_split["split_a"].value_counts().to_dict(),
        "n_uniprot_targets": int(same_split["uniprot"].nunique()),
        "n_unique_systems": int(len(set(same_split["pdb_a"])
                                    | set(same_split["pdb_b"]))),
        "mcs_fraction_median": float(good["mcs_fraction"].median()),
        "mapped_atoms_median": float(good["n_mapped"].median()),
        "unmapped_a_median": float(good["unmapped_a"].median()),
        "unmapped_b_median": float(good["unmapped_b"].median()),
        "note": "transformation_agrees is a CROSS-CHECK: the mapping comes from "
                "MCS on the stored coordinates only. The census transformation "
                "never enters the mapping.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
