"""Step 3 of the MISATO MMP feasibility census: ligand selection, target grouping,
MMP enumeration, same-context groups, and cross-target transformation recurrence.

Ligand selection
----------------
A PDB entry lists every nonpolymer component, including buffers, cryoprotectants
and ions. MISATO/PDBbind simulate one binding ligand per entry. We pick it with a
blacklist-plus-largest rule, and then VALIDATE that rule against ground truth: for
the 2,980 systems that have cached real coordinates, product_init.shape[0] is the
true heavy+H atom count of the ligand MISATO actually simulated, so we can check
whether the selected component's atom count matches. The measured agreement rate
is reported rather than assumed.

MMP enumeration
---------------
mmpdb is NOT installed in this environment and no installs are permitted, so we
use RDKit's own rdMMPA single-cut fragmentation, which is the same algorithm
mmpdb's fragment step is built on. Pairs are formed within a protein target only.

Transformation key
------------------
Canonical "variableA>>variableB" with attachment points retained, orientation
normalised so A<B lexicographically. Identity transformations are dropped.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/misato_mmp_census_v1"
CACHE = ROOT / "data/processed/misato_atomic_window_cache_3k_v1"

# Components that are crystallisation additives, ions, solvents or cofactor-like
# ubiquitous species rather than the entry's binding ligand.
BLACKLIST = {
    "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "2PE",
    "MPD", "MES", "TRS", "HEPES", "EPE", "BTB", "BIS", "CIT", "FLC", "TLA",
    "ACT", "ACY", "FMT", "OXL", "MLI", "MLA", "SCN", "AZI", "NO3", "CO3", "BCT",
    "NH4", "CL", "BR", "IOD", "F", "NA", "K", "MG", "CA", "ZN", "MN", "FE",
    "FE2", "CU", "CU1", "NI", "CO", "CD", "HG", "AU", "AG", "PT", "PB", "BA",
    "SR", "CS", "RB", "LI", "AL", "GA", "YB", "SM", "EU", "LU", "W", "MO",
    "DMS", "DMF", "ACN", "EOH", "IPA", "MOH", "BME", "DTT", "DTU", "TCE",
    "IMD", "IMZ", "BEN", "PHE", "GLY", "ALA", "SER", "URE", "SIN", "SUC",
    "GLC", "BGC", "GAL", "MAN", "NAG", "NDG", "BMA", "FUC", "XYP", "SIA",
    "P6G", "PE4", "PE8", "XPE", "12P", "15P", "7PE", "M2M", "OCT", "D10",
    "LDA", "C8E", "BOG", "LMT", "DDQ", "UNX", "UNL", "UNK", "PCA", "MYR",
    "PLM", "STE", "OLA", "OLC", "PGO", "PDO", "BU3", "BU1", "IPH", "TFA",
    "NH2", "NO", "CN", "CMO", "OXY", "PER", "H2S", "SF4", "FES", "HEM", "HEC",
}

ATTACH = re.compile(r"\[\*(:\d+)?\]")


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def heavy_and_h(smi: str) -> tuple[int, int] | None:
    """(heavy atoms, heavy+implicit/explicit H) for a SMILES."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    heavy = m.GetNumHeavyAtoms()
    total = Chem.AddHs(m).GetNumAtoms()
    return heavy, total


# --------------------------------------------------------------------------
# Step A: parse RCSB payloads -> one receptor UniProt + one ligand per entry
# --------------------------------------------------------------------------
def parse_entries() -> pd.DataFrame:
    rows = []
    seen = set()
    for line in (OUT / "rcsb_entries.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        pid = e["rcsb_id"].upper()
        if pid in seen:
            continue
        seen.add(pid)

        # receptor: longest polymer entity carrying a UniProt accession
        best_len, uni, desc = -1, None, None
        n_poly = 0
        for pe in (e.get("polymer_entities") or []):
            n_poly += 1
            ids = ((pe.get("rcsb_polymer_entity_container_identifiers") or {})
                   .get("uniprot_ids") or [])
            ln = ((pe.get("entity_poly") or {})
                  .get("rcsb_sample_sequence_length") or 0)
            if ids and ln > best_len:
                best_len, uni = ln, ids[0]
                desc = (pe.get("rcsb_polymer_entity") or {}).get("pdbx_description")

        # ligand candidates
        cands = []
        for ne in (e.get("nonpolymer_entities") or []):
            comp = (ne.get("nonpolymer_comp") or {})
            cc = comp.get("chem_comp") or {}
            dsc = comp.get("rcsb_chem_comp_descriptor") or {}
            cid = (cc.get("id") or "").upper()
            smi = dsc.get("SMILES_stereo") or dsc.get("SMILES")
            if not cid or not smi:
                continue
            hh = heavy_and_h(smi)
            if hh is None:
                continue
            cands.append({"comp_id": cid, "smiles": smi,
                          "heavy": hh[0], "total": hh[1],
                          "mw": cc.get("formula_weight"),
                          "blacklisted": cid in BLACKLIST})
        rows.append({
            "pdb_id": pid, "uniprot": uni, "protein_desc": desc,
            "receptor_len": best_len if best_len > 0 else None,
            "n_polymer_entities": n_poly,
            "n_nonpolymer_candidates": len(cands),
            "candidates": cands,
        })
    return pd.DataFrame(rows)


def select_ligand(cands: list[dict], true_atoms: int | None) -> dict:
    """Pick the binding ligand. If the true MISATO atom count is known, prefer an
    exact atom-count match; otherwise fall back to largest non-blacklisted."""
    usable = [c for c in cands if not c["blacklisted"] and c["heavy"] >= 6]
    pool = usable or [c for c in cands if c["heavy"] >= 6]
    if not pool:
        return {"ligand_smiles": None, "ligand_comp_id": None,
                "select_rule": "none", "atomcount_match": None}
    largest = max(pool, key=lambda c: (c["heavy"], c["total"]))
    if true_atoms is not None:
        exact = [c for c in pool if c["total"] == true_atoms]
        if len(exact) >= 1:
            pick = max(exact, key=lambda c: c["heavy"])
            return {"ligand_smiles": pick["smiles"], "ligand_comp_id": pick["comp_id"],
                    "select_rule": "atom_count_exact",
                    "atomcount_match": pick["comp_id"] == largest["comp_id"]}
        return {"ligand_smiles": largest["smiles"], "ligand_comp_id": largest["comp_id"],
                "select_rule": "largest_nonbuffer", "atomcount_match": False}
    return {"ligand_smiles": largest["smiles"], "ligand_comp_id": largest["comp_id"],
            "select_rule": "largest_nonbuffer", "atomcount_match": None}


# --------------------------------------------------------------------------
# Step B: MMP fragmentation
# --------------------------------------------------------------------------
def fragment(smi: str) -> list[tuple[str, str]]:
    """Single-cut MMP fragments -> list of (constant, variable) canonical pairs."""
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumHeavyAtoms() > 70:
        return []
    out = []
    try:
        frags = rdMMPA.FragmentMol(m, maxCuts=1, resultsAsMols=False,
                                   maxCutBonds=40)
    except Exception:
        return []
    for core, chains in frags:
        if not chains:
            continue
        parts = chains.split(".")
        if len(parts) != 2:
            continue
        a, b = parts
        na = Chem.MolFromSmiles(a)
        nb = Chem.MolFromSmiles(b)
        if na is None or nb is None:
            continue
        # constant = larger half, variable = smaller half
        if na.GetNumHeavyAtoms() >= nb.GetNumHeavyAtoms():
            const, var = a, b
        else:
            const, var = b, a
        cv = Chem.MolFromSmiles(var)
        # require the variable part to be a real, small R-group
        if cv is None or cv.GetNumHeavyAtoms() > 15:
            continue
        out.append((Chem.MolToSmiles(Chem.MolFromSmiles(const)),
                    Chem.MolToSmiles(cv)))
    return list(set(out))


def main() -> None:
    df = parse_entries()
    print(f"parsed {len(df)} RCSB entries", flush=True)

    # ground-truth ligand atom counts from the coordinate cache
    idx = pd.read_csv(CACHE / "atomic_windows_cached.csv")
    true_atoms: dict[str, int] = {}
    for mid, g in idx.groupby("molecule_id"):
        p = ROOT / g.iloc[0]["cache_path"]
        if p.exists():
            try:
                true_atoms[str(mid)] = int(np.load(p)["product_init"].shape[0])
            except Exception:
                pass
    print(f"ground-truth atom counts for {len(true_atoms)} cached systems", flush=True)

    sel = [select_ligand(r["candidates"], true_atoms.get(r["pdb_id"]))
           for _, r in df.iterrows()]
    for k in ("ligand_smiles", "ligand_comp_id", "select_rule", "atomcount_match"):
        df[k] = [s[k] for s in sel]
    df["has_true_atoms"] = df["pdb_id"].isin(true_atoms)

    # validation of the largest-non-buffer heuristic against ground truth
    val = df[df["has_true_atoms"] & df["ligand_smiles"].notna()]
    agree = val["atomcount_match"]
    validation = {
        "n_cached_systems_with_ligand": int(len(val)),
        "n_resolved_by_exact_atom_count": int((val["select_rule"] == "atom_count_exact").sum()),
        "heuristic_agrees_with_atom_count_match": int(agree.sum()),
        "heuristic_agreement_rate": float(agree.mean()) if len(agree) else None,
        "note": ("agreement rate = fraction of cached systems where the "
                 "largest-non-buffer pick is the same component as the one whose "
                 "heavy+H atom count exactly equals MISATO's simulated ligand"),
    }
    print("ligand-selection validation:", json.dumps(validation), flush=True)

    # coverage accounting
    coverage = {
        "n_entries": int(len(df)),
        "n_with_uniprot": int(df["uniprot"].notna().sum()),
        "n_with_ligand_smiles": int(df["ligand_smiles"].notna().sum()),
        "n_with_both": int((df["uniprot"].notna() & df["ligand_smiles"].notna()).sum()),
        "n_no_nonpolymer_candidate": int((df["n_nonpolymer_candidates"] == 0).sum()),
        "n_all_candidates_blacklisted_or_tiny": int(
            ((df["n_nonpolymer_candidates"] > 0) & df["ligand_smiles"].isna()).sum()),
    }
    print("coverage:", json.dumps(coverage), flush=True)

    work = df[df["uniprot"].notna() & df["ligand_smiles"].notna()].copy()
    work["ligand_canon"] = [canon(s) for s in work["ligand_smiles"]]
    work = work[work["ligand_canon"].notna()]

    # availability flags
    trace_ids = set((OUT / "ids_traces_all.txt").read_text().split())
    coord_ids = set((OUT / "ids_coord_cache.txt").read_text().split())
    work["has_trace"] = work["pdb_id"].isin(trace_ids)
    work["has_coords"] = work["pdb_id"].isin(coord_ids)

    work.drop(columns=["candidates"]).to_csv(OUT / "systems_resolved.csv", index=False)

    # ---------------- MMP enumeration, within target ----------------
    frag_cache: dict[str, list[tuple[str, str]]] = {}
    for s in work["ligand_canon"].unique():
        frag_cache[s] = fragment(s)
    print(f"fragmented {len(frag_cache)} distinct ligands", flush=True)

    pairs = []
    for uni, g in work.groupby("uniprot"):
        # one row per distinct ligand per target (dedupe repeated PDB entries)
        by_lig: dict[str, list[str]] = defaultdict(list)
        for _, r in g.iterrows():
            by_lig[r["ligand_canon"]].append(r["pdb_id"])
        if len(by_lig) < 2:
            continue
        const_map: dict[str, list[str]] = defaultdict(list)
        for lig in by_lig:
            for const, var in frag_cache.get(lig, []):
                const_map[const].append(lig)
        for const, ligs in const_map.items():
            ligs = sorted(set(ligs))
            if len(ligs) < 2:
                continue
            vmap = {}
            for lig in ligs:
                for c, v in frag_cache[lig]:
                    if c == const:
                        vmap[lig] = v
                        break
            for i in range(len(ligs)):
                for j in range(i + 1, len(ligs)):
                    la, lb = ligs[i], ligs[j]
                    va, vb = vmap.get(la), vmap.get(lb)
                    if va is None or vb is None or va == vb:
                        continue
                    lo, hi = (va, vb) if va < vb else (vb, va)
                    pairs.append({
                        "uniprot": uni, "constant": const,
                        "lig_a": la, "lig_b": lb,
                        "transformation": f"{lo}>>{hi}",
                        "pdb_a": by_lig[la][0], "pdb_b": by_lig[lb][0],
                        "a_trace": by_lig[la][0] in trace_ids,
                        "b_trace": by_lig[lb][0] in trace_ids,
                        "a_coords": any(p in coord_ids for p in by_lig[la]),
                        "b_coords": any(p in coord_ids for p in by_lig[lb]),
                    })
    pdf = pd.DataFrame(pairs)
    if pdf.empty:
        print("NO PAIRS FOUND")
        (OUT / "mmp_summary.json").write_text(json.dumps(
            {"validation": validation, "coverage": coverage,
             "n_pairs": 0}, indent=2))
        return
    pdf = pdf.drop_duplicates(subset=["uniprot", "lig_a", "lig_b", "transformation"])
    pdf["both_trace"] = pdf["a_trace"] & pdf["b_trace"]
    pdf["both_coords"] = pdf["a_coords"] & pdf["b_coords"]
    pdf.to_csv(OUT / "mmp_pairs.csv", index=False)

    # ---------------- same-context groups ----------------
    # parent with >=3 distinct MMP partners on the same target and same constant
    nbr: dict[tuple, set] = defaultdict(set)
    for _, r in pdf.iterrows():
        nbr[(r["uniprot"], r["constant"], r["lig_a"])].add(r["lig_b"])
        nbr[(r["uniprot"], r["constant"], r["lig_b"])].add(r["lig_a"])
    grp_sizes = [len(v) for v in nbr.values() if len(v) >= 3]
    # also the looser definition: parent with >=3 partners on a target, any constant
    nbr_t: dict[tuple, set] = defaultdict(set)
    for _, r in pdf.iterrows():
        nbr_t[(r["uniprot"], r["lig_a"])].add(r["lig_b"])
        nbr_t[(r["uniprot"], r["lig_b"])].add(r["lig_a"])
    grp_sizes_t = [len(v) for v in nbr_t.values() if len(v) >= 3]

    # MD-available versions
    md = pdf[pdf["both_trace"]]
    nbr_md: dict[tuple, set] = defaultdict(set)
    for _, r in md.iterrows():
        nbr_md[(r["uniprot"], r["constant"], r["lig_a"])].add(r["lig_b"])
        nbr_md[(r["uniprot"], r["constant"], r["lig_b"])].add(r["lig_a"])
    grp_md = [len(v) for v in nbr_md.values() if len(v) >= 3]

    # ---------------- cross-target / cross-scaffold recurrence ----------------
    tx_targets: dict[str, set] = defaultdict(set)
    tx_scaffolds: dict[str, set] = defaultdict(set)
    scaf_cache: dict[str, str] = {}
    for _, r in pdf.iterrows():
        tx = r["transformation"]
        tx_targets[tx].add(r["uniprot"])
        c = r["constant"]
        if c not in scaf_cache:
            m = Chem.MolFromSmiles(ATTACH.sub("[H]", c))
            try:
                scaf_cache[c] = (Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
                                 if m is not None else "")
            except Exception:
                scaf_cache[c] = ""
        tx_scaffolds[tx].add(scaf_cache[c])

    n_tx = len(tx_targets)
    tgt_dist = Counter(len(v) for v in tx_targets.values())
    scaf_dist = Counter(len(v - {""}) for v in tx_scaffolds.values())
    n_multi_target = sum(1 for v in tx_targets.values() if len(v) >= 2)
    n_multi_scaf = sum(1 for v in tx_scaffolds.values() if len(v - {""}) >= 2)

    pairs_per_target = pdf.groupby("uniprot").size().sort_values(ascending=False)

    summary = {
        "validation_of_ligand_selection": validation,
        "coverage": coverage,
        "mmp": {
            "n_pairs_total": int(len(pdf)),
            "n_pairs_both_endpoints_have_trace_md": int(pdf["both_trace"].sum()),
            "n_pairs_both_endpoints_have_cached_coords": int(pdf["both_coords"].sum()),
            "n_distinct_targets_with_pairs": int(pdf["uniprot"].nunique()),
            "n_distinct_ligands_in_pairs": int(
                len(set(pdf["lig_a"]) | set(pdf["lig_b"]))),
            "n_distinct_transformations": n_tx,
            "pairs_per_target": {
                "max": int(pairs_per_target.max()),
                "median": float(pairs_per_target.median()),
                "mean": float(pairs_per_target.mean()),
                "n_targets_ge_10_pairs": int((pairs_per_target >= 10).sum()),
                "n_targets_ge_50_pairs": int((pairs_per_target >= 50).sum()),
                "top_20": pairs_per_target.head(20).to_dict(),
            },
        },
        "same_context_groups": {
            "definition_strict": "parent with >=3 distinct MMP partners at the SAME constant (core) on the same target",
            "n_groups_strict": len(grp_sizes),
            "size_distribution_strict": dict(Counter(grp_sizes)),
            "median_size_strict": float(np.median(grp_sizes)) if grp_sizes else None,
            "definition_loose": "parent with >=3 distinct MMP partners on the same target, any core",
            "n_groups_loose": len(grp_sizes_t),
            "median_size_loose": float(np.median(grp_sizes_t)) if grp_sizes_t else None,
            "n_groups_strict_with_trace_md_on_all_members": len(grp_md),
            "size_distribution_strict_md": dict(Counter(grp_md)),
        },
        "cross_target_recurrence": {
            "n_transformations": n_tx,
            "n_on_ge2_targets": n_multi_target,
            "fraction_on_ge2_targets": n_multi_target / n_tx if n_tx else None,
            "targets_per_transformation_distribution": dict(sorted(tgt_dist.items())),
            "n_on_ge2_scaffolds": n_multi_scaf,
            "fraction_on_ge2_scaffolds": n_multi_scaf / n_tx if n_tx else None,
            "scaffolds_per_transformation_distribution": dict(sorted(scaf_dist.items())),
        },
    }
    (OUT / "mmp_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
