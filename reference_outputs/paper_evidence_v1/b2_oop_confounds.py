"""(b, control) Is the 6.4 deg out-of-plane MD width an artifact?

Two confounds are checked, both of which would inflate it:

  1. PERIODIC-IMAGE JUMPS.  `project_misato_store` records that 10.2% of MISATO
     systems have frames in which the ligand is split across the periodic
     boundary, and MISATO's own rmsd_ligand does not flag them.  A split frame
     makes every internal coordinate of the affected atoms nonsense.  Detector:
     any bonded heavy-heavy pair longer than 2.5 A in any frame.
  2. FRAME COUNT / trajectory drift: width computed on the first 50 frames vs
     the last 50.

Also reports the aromatic width per element and per ring membership, so a single
odd class cannot be carrying it.
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

from mehf_internal_geometry_spectrum import (  # noqa: E402
    improper_centres, out_of_plane_degrees,
)
from rdkit import Chem, RDLogger  # noqa: E402
import networkx as nx  # noqa: E402

RDLogger.DisableLog("rdApp.*")
STORE = ROOT / "data/MISATO/extracted/frames100_all_v1"
CENSUS = ROOT / "outputs/misato_mmp_census_v1/systems_resolved.csv"
N_SYSTEMS = 400
SEED = 20260925


def main() -> None:
    smiles_by_pdb = {}
    with CENSUS.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("ligand_smiles"):
                smiles_by_pdb[row["pdb_id"].strip().upper()] = row["ligand_smiles"].strip()
    excluded = set()
    with (ROOT / "outputs/mmp_transfer_corpus_v1/pairs.csv").open() as handle:
        for row in csv.DictReader(handle):
            for side in ("a", "b"):
                if str(row.get(f"split_{side}", "")).strip().lower() == "test":
                    excluded.add(str(row[f"pdb_{side}"]).strip().upper())

    paths = sorted(STORE.glob("*.npz"))
    rng = np.random.default_rng(SEED)
    rng.shuffle(paths)

    rows = []
    n_broken = 0
    used = 0
    for path in paths:
        if used >= N_SYSTEMS:
            break
        pdb = path.stem.upper()
        if pdb in excluded:
            continue
        with np.load(path, allow_pickle=True) as store:
            coords = store["ligand_coords"].astype(np.float64)
            bond_index = np.asarray(store["bond_index"]).reshape(-1, 2)
            elements = store["ligand_atomic_numbers"].astype(int)
        if coords.shape[0] < 10:
            continue
        heavy = elements != 1
        if heavy.sum() < 8 or heavy.sum() > 40:
            continue
        full_degree = np.zeros(len(elements), dtype=int)
        seen = set()
        for a, b in bond_index:
            a, b = int(a), int(b)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            full_degree[a] += 1
            full_degree[b] += 1
        remap = -np.ones(len(elements), dtype=int)
        remap[heavy] = np.arange(int(heavy.sum()))
        hb = sorted({(min(int(remap[a]), int(remap[b])), max(int(remap[a]), int(remap[b])))
                     for a, b in bond_index if a != b and heavy[a] and heavy[b]})
        if not hb:
            continue
        hb = np.asarray(hb)
        xyz = coords[:, heavy, :]
        n = xyz.shape[1]
        used += 1
        # periodic-image detector
        d = np.linalg.norm(xyz[:, hb[:, 0], :] - xyz[:, hb[:, 1], :], axis=-1)
        broken = bool(d.max() > 2.5)
        if broken:
            n_broken += 1
        centres = improper_centres(hb, n)
        if not centres:
            continue
        aromatic_flag = None
        smiles = smiles_by_pdb.get(pdb)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and mol.GetNumAtoms() == n:
                sg = nx.Graph()
                for atom in mol.GetAtoms():
                    sg.add_node(atom.GetIdx(), z=atom.GetAtomicNum())
                for bond in mol.GetBonds():
                    sg.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                store_g = nx.Graph()
                zz = elements[heavy]
                for i in range(n):
                    store_g.add_node(i, z=int(zz[i]))
                for a, b in hb:
                    store_g.add_edge(int(a), int(b))
                if store_g.number_of_edges() == sg.number_of_edges():
                    m = nx.algorithms.isomorphism.GraphMatcher(
                        store_g, sg, node_match=lambda x, y: x["z"] == y["z"])
                    if m.is_isomorphic():
                        aromatic_flag = {i: bool(mol.GetAtomWithIdx(m.mapping[i]).GetIsAromatic())
                                         for i in range(n)}
        chi = out_of_plane_degrees(xyz, centres)
        half = chi.shape[0] // 2
        zz = elements[heavy]
        for idx, centre in enumerate(centres):
            j = centre[0]
            rows.append({
                "system": pdb, "broken": broken, "z": int(zz[j]),
                "sp2": bool(full_degree[heavy][j] == 3),
                "aromatic": (None if aromatic_flag is None else aromatic_flag[j]),
                "sd_deg": float(chi[:, idx].std()),
                "mad_deg": float(np.abs(chi[:, idx]).mean()),
                "sd_first_half": float(chi[:half, idx].std()),
                "sd_second_half": float(chi[half:, idx].std()),
            })

    def stat(sel):
        if not sel:
            return None
        return {"n": len(sel),
                "sd_median": float(np.median([r["sd_deg"] for r in sel])),
                "mad_median": float(np.median([r["mad_deg"] for r in sel])),
                "sd_first_half": float(np.median([r["sd_first_half"] for r in sel])),
                "sd_second_half": float(np.median([r["sd_second_half"] for r in sel]))}

    planar = [r for r in rows if r["mad_deg"] < 10.0]
    arom = [r for r in planar if r["aromatic"] is True]
    out = {
        "n_systems": used,
        "n_systems_with_periodic_break": n_broken,
        "fraction_broken": n_broken / max(used, 1),
        "aromatic_all": stat(arom),
        "aromatic_clean_systems_only": stat([r for r in arom if not r["broken"]]),
        "aromatic_broken_systems_only": stat([r for r in arom if r["broken"]]),
        "aromatic_carbon": stat([r for r in arom if r["z"] == 6]),
        "aromatic_nitrogen": stat([r for r in arom if r["z"] == 7]),
    }
    here = Path(__file__).resolve().parent
    (here / "b2_oop_confounds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
