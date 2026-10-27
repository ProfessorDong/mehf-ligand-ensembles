"""(b) The out-of-plane thermal width, AROMATIC centres only.

`outputs/theory_modules_v1/RESULT.md` R1 measured tau_oop = 6.48 deg and said so
itself: `improper_centres` selects any RING atom with three heavy neighbours and
is NOT filtered for aromaticity, so a saturated ring carbon carrying a hydrogen
and three heavy neighbours -- i.e. ring pucker -- is inside the number.

Three nested centre definitions, same systems, same coordinate, same estimator:

  published  ring atom with exactly three HEAVY neighbours (what R1 measured)
  sp2        + exactly three neighbours in the FULL graph, so the centre carries
             no hydrogen.  Purely a valence criterion, no geometry.
  aromatic   + RDKit's aromatic flag, transferred from the ligand's deposited
             SMILES by element-labelled graph isomorphism of the heavy-atom
             graph.  Chemical, and independent of the coordinates.

Two estimators, because the published comparison mixes them up.  DiffSBDD's
5.09 deg and the crystal reference's 1.18 deg are MEAN ABSOLUTE deviations from
planarity of a SINGLE conformer.  The thermal quantity that is like-for-like
with them is E|chi| over MD frames, not the frame-to-frame standard deviation.
Both are reported.
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


def load_smiles() -> dict[str, str]:
    out = {}
    with CENSUS.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("ligand_smiles"):
                out[row["pdb_id"].strip().upper()] = row["ligand_smiles"].strip()
    return out


def smiles_aromatic_map(smiles: str) -> tuple[nx.Graph, dict[int, bool]] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), z=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    aromatic = {atom.GetIdx(): bool(atom.GetIsAromatic()) for atom in mol.GetAtoms()}
    return graph, aromatic


def main() -> None:
    smiles_by_pdb = load_smiles()
    paths = sorted(STORE.glob("*.npz"))
    rng = np.random.default_rng(SEED)
    rng.shuffle(paths)

    excluded: set[str] = set()
    pairs = ROOT / "outputs/mmp_transfer_corpus_v1/pairs.csv"
    with pairs.open() as handle:
        for row in csv.DictReader(handle):
            for side in ("a", "b"):
                if str(row.get(f"split_{side}", "")).strip().lower() == "test":
                    excluded.add(str(row[f"pdb_{side}"]).strip().upper())

    centres_rows = []
    system_rows = []
    n_iso_ok = n_iso_fail = n_no_smiles = 0
    for path in paths:
        if len(system_rows) >= N_SYSTEMS:
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
        # full-graph degree, for the sp2 test
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
                     for a, b in bond_index
                     if a != b and heavy[a] and heavy[b]})
        if not hb:
            continue
        hb = np.asarray(hb)
        xyz = coords[:, heavy, :]
        n = xyz.shape[1]
        centres = improper_centres(hb, n)
        if not centres:
            continue
        heavy_degree_full = full_degree[heavy]
        heavy_z = elements[heavy]

        # aromaticity by graph isomorphism against the deposited SMILES
        aromatic_flag: dict[int, bool] | None = None
        smiles = smiles_by_pdb.get(pdb)
        if smiles is None:
            n_no_smiles += 1
        else:
            built = smiles_aromatic_map(smiles)
            if built is not None:
                sg, arom = built
                store_g = nx.Graph()
                for i in range(n):
                    store_g.add_node(i, z=int(heavy_z[i]))
                for a, b in hb:
                    store_g.add_edge(int(a), int(b))
                if store_g.number_of_nodes() == sg.number_of_nodes() and \
                   store_g.number_of_edges() == sg.number_of_edges():
                    matcher = nx.algorithms.isomorphism.GraphMatcher(
                        store_g, sg,
                        node_match=lambda x, y: x["z"] == y["z"])
                    if matcher.is_isomorphic():
                        mapping = matcher.mapping
                        aromatic_flag = {i: arom[mapping[i]] for i in range(n)}
                        n_iso_ok += 1
                    else:
                        n_iso_fail += 1
                else:
                    n_iso_fail += 1
            else:
                n_iso_fail += 1

        chi = out_of_plane_degrees(xyz, centres)          # (frames, centres)
        sd = chi.std(axis=0)
        mad = np.abs(chi).mean(axis=0)
        absmean = np.abs(chi.mean(axis=0))
        for idx, centre in enumerate(centres):
            j = centre[0]
            centres_rows.append({
                "system": pdb, "centre": j,
                "z": int(heavy_z[j]),
                "sp2": bool(heavy_degree_full[j] == 3),
                "aromatic": (None if aromatic_flag is None
                             else bool(aromatic_flag[j])),
                "sd_deg": float(sd[idx]),
                "mad_deg": float(mad[idx]),
                "absmean_deg": float(absmean[idx]),
            })
        system_rows.append(pdb)

    here = Path(__file__).resolve().parent
    with (here / "b_aromatic_oop_centres.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(centres_rows[0]))
        writer.writeheader()
        writer.writerows(centres_rows)

    def summarise(rows, label):
        if not rows:
            return {"n_centres": 0}
        sd = np.array([r["sd_deg"] for r in rows])
        mad = np.array([r["mad_deg"] for r in rows])
        am = np.array([r["absmean_deg"] for r in rows])
        # per-system mean over centres then median over systems (published protocol)
        by_sys: dict[str, list[float]] = {}
        by_sys_mad: dict[str, list[float]] = {}
        for r in rows:
            by_sys.setdefault(r["system"], []).append(r["sd_deg"])
            by_sys_mad.setdefault(r["system"], []).append(r["mad_deg"])
        sys_sd = np.array([np.mean(v) for v in by_sys.values()])
        sys_mad = np.array([np.mean(v) for v in by_sys_mad.values()])
        return {
            "label": label,
            "n_centres": len(rows),
            "n_systems": len(by_sys),
            "tau_sd_published_protocol_deg": float(np.median(sys_sd)),
            "tau_sd_iqr": [float(np.percentile(sys_sd, 25)),
                           float(np.percentile(sys_sd, 75))],
            "mad_published_protocol_deg": float(np.median(sys_mad)),
            "mad_iqr": [float(np.percentile(sys_mad, 25)),
                        float(np.percentile(sys_mad, 75))],
            "per_centre_sd_median": float(np.median(sd)),
            "per_centre_mad_median": float(np.median(mad)),
            "per_centre_absmean_median": float(np.median(am)),
        }

    out = {
        "n_systems": len(system_rows),
        "n_centres_total": len(centres_rows),
        "isomorphism_ok": n_iso_ok,
        "isomorphism_failed": n_iso_fail,
        "no_smiles": n_no_smiles,
    }
    for flt_name, flt in (
        ("no_planarity_prefilter", lambda r: True),
        ("planarity_prefilter_10deg", lambda r: r["mad_deg"] < 10.0),
    ):
        base = [r for r in centres_rows if flt(r)]
        out[flt_name] = {
            "published": summarise(base, "ring atom, 3 heavy neighbours"),
            "sp2": summarise([r for r in base if r["sp2"]], "+ no hydrogen on centre"),
            "not_sp2": summarise([r for r in base if not r["sp2"]],
                                 "centres carrying a hydrogen (the contaminant)"),
            "aromatic": summarise([r for r in base if r["aromatic"] is True],
                                  "+ RDKit aromatic"),
            "sp2_nonaromatic": summarise(
                [r for r in base if r["sp2"] and r["aromatic"] is False],
                "sp2 but not aromatic"),
        }
    (here / "b_aromatic_oop.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
