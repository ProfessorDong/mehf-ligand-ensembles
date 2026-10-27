"""Leakage audit of the train / val / test protocol (REPRODUCIBILITY.md, section 2).

Writes outputs/leakage_audit_v1/audit.json with:
  - UniProt and PDB overlap between the three splits;
  - for every released checkpoint: training split, fit/dev containment in train,
    fit/dev target overlap, overlap with val and test targets;
  - contamination of the four physical widths tau by val/test systems, and the
    widths re-estimated without them;
  - chemical-identity overlap (heavy-atom graph, Weisfeiler-Lehman hash) of
    val/test ligands with train ligands, and the test decision recomputed with
    every test example whose parent or product ligand also occurs in train removed.
Reads only files shipped in this repository.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from mdwm.data.mmp_transfer import (  # noqa: E402
    DEFAULT_PAIRS, DEFAULT_STORE, load_usable_pairs, target_disjoint_split, usable_examples,
)

SPLITS = ("train", "val", "test")
CHECKPOINTS = [f"outputs/{d}/full_f{k}" for d in ("internal_calibration_v1", "aniso_prior_v3",
               "sigma_star_v1", "angle_term_v1") for k in (1, 2, 3)] + [
    "outputs/aniso_prior_v2/full_f3", "outputs/angle_term_v1/retrain_control_f2"] + [
    f"outputs/laplacian_prior_v1/full_f{k}" for k in (1, 2, 3)]


def key(frame: pd.DataFrame) -> set[str]:
    return set(frame["parent"].astype(str) + "->" + frame["product"].astype(str))


def graph_hash(npz: Path) -> str:
    with np.load(npz, allow_pickle=True) as z:
        el = z["ligand_atomic_numbers"].astype(int)
        bonds = np.asarray(z["bond_index"]).reshape(-1, 2)
    heavy = {i for i, e in enumerate(el) if e != 1}
    g = nx.Graph()
    g.add_nodes_from((i, {"e": int(el[i])}) for i in heavy)
    g.add_edges_from((int(a), int(b)) for a, b in bonds if int(a) in heavy and int(b) in heavy and a != b)
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="e", iterations=4)


def main() -> None:
    import torch
    import sigma_star_test_decision as decision

    pairs = target_disjoint_split(load_usable_pairs(DEFAULT_PAIRS))
    uni = {s: set(pairs[pairs.primary_split == s].uniprot) for s in SPLITS}
    pdb = {s: set(pairs[pairs.primary_split == s].pdb_a.str.upper())
              | set(pairs[pairs.primary_split == s].pdb_b.str.upper()) for s in SPLITS}
    out: dict = {"split_counts": {s: {"pairs": int((pairs.primary_split == s).sum()),
                                      "targets": len(uni[s]), "pdb_entries": len(pdb[s])} for s in SPLITS}}
    out["uniprot_overlap"] = {f"{a}&{b}": len(uni[a] & uni[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    out["pdb_overlap"] = {f"{a}&{b}": len(pdb[a] & pdb[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}

    train_keys = key(usable_examples(split="train"))
    rows = []
    for d in CHECKPOINTS:
        ck = torch.load(Path(d) / "checkpoint.pt", map_location="cpu", weights_only=False)
        fit, dev = pd.read_csv(Path(d) / "fit_examples.csv"), pd.read_csv(Path(d) / "dev_examples.csv")
        summary = json.loads((Path(d) / "summary.json").read_text())
        seen = set(fit.uniprot) | set(dev.uniprot)
        rows.append({"checkpoint": d, "split": ck["data_cfg"]["split"],
                     "fit_in_train": key(fit) <= train_keys, "dev_in_train": key(dev) <= train_keys,
                     "fit_dev_target_overlap": len(set(fit.uniprot) & set(dev.uniprot)),
                     "val_target_overlap": len(seen & uni["val"]), "test_target_overlap": len(seen & uni["test"]),
                     "selection_metric": summary.get("selection_metric")})
    out["checkpoints"] = rows

    label = lambda p: next((s for s in SPLITS if p in pdb[s]), "not_in_corpus")  # noqa: E731
    tau = {}
    ps = pd.read_csv("reference_outputs/isotropic_noise_cost_v1/per_system.csv")
    ps["split"] = ps.system.astype(str).str.upper().str.replace(".NPZ", "", regex=False).map(label)
    wb = pd.read_csv("reference_outputs/wilson_b_theory_v1/wilson_b.csv")
    wb["split"] = wb.system.astype(str).str.split("_").str[0].str.upper().map(label)
    tm = pd.DataFrame(json.loads(Path("reference_outputs/theory_modules_v1/results.json").read_text())["rows"])
    tm["split"] = tm.system.astype(str).str.split("_").str[0].str.upper().map(label)
    tm = tm.dropna(subset=["tau_oop_deg"])
    for name, frame, col in (("bond_angstrom", wb, "bond_length_spread_physical"),
                             ("angle_deg", ps, "bond_angle_spread_physical"),
                             ("torsion_deg", ps, "dihedral_spread_physical"),
                             ("oop_deg", tm, "tau_oop_deg")):
        clean = frame[~frame.split.isin(["val", "test"])]
        tau[name] = {"n_systems": int(len(frame)), "n_val": int((frame.split == "val").sum()),
                     "n_test": int((frame.split == "test").sum()),
                     "median_all": float(frame[col].median()), "median_without_val_test": float(clean[col].median()),
                     "relative_change": float(clean[col].median() / frame[col].median() - 1)}
    out["tau_contamination"] = tau

    store = Path(DEFAULT_STORE)
    files = {p.stem.split("_")[0].upper(): p for p in store.glob("*.npz")}
    hashes = {s: {p: graph_hash(files[p]) for p in pdb[s]} for s in SPLITS}
    train_hashes = set(hashes["train"].values())
    out["chemical_overlap"] = {s: {"ligand_systems": len(hashes[s]),
                                   "identical_ligand_in_train": sum(h in train_hashes for h in hashes[s].values()),
                                   "distinct_molecules": len({h for h in hashes[s].values() if h in train_hashes})}
                               for s in ("val", "test")}

    decision.OUT = ROOT / "reference_outputs/sigma_star_test_v1"
    examples = pd.read_csv(decision.OUT / "test_examples.csv")
    uniprot = dict(zip(examples.example_id, examples.uniprot))
    arms = {a: decision.load_arm(a)[0] for a in decision.ARMS}
    common = arms["ISO"].index.intersection(arms["ANISO_v3"].index).intersection(arms["SIGMA_star"].index)
    hp = lambda p: hashes["test"].get(p.upper()) or graph_hash(files[p.upper()])  # noqa: E731
    seen = {k: (hp(k.split("->")[0]) in train_hashes) or (hp(k.split("->")[1]) in train_hashes) for k in common}
    sens = {}
    for label_, keep in (("all", list(common)), ("chemically_unseen", [k for k in common if not seen[k]])):
        diff = (arms["SIGMA_star"].loc[keep, "spread"] - arms["ISO"].loc[keep, "spread"]).to_numpy()
        lo, hi = decision.cluster_bootstrap(diff, np.array([uniprot[k] for k in keep]))
        sens[label_] = {"n_examples": len(keep),
                        "median_spread": {a: float(arms[a].loc[keep, "spread"].median()) for a in decision.ARMS},
                        "paired_vs_ISO": {"mean": float(diff.mean()), "ci95": [lo, hi]}}
    out["test_sensitivity_chemical_overlap"] = sens

    dest = ROOT / "outputs/leakage_audit_v1/audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("uniprot_overlap", "pdb_overlap", "chemical_overlap",
                                          "test_sensitivity_chemical_overlap")}, indent=1))


if __name__ == "__main__":
    main()
