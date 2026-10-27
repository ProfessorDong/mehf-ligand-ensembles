"""Task B: re-estimate the four physical widths tau on a LARGE set of TRAIN-target systems.

Estimator is the original one, applied to a larger system set:
  per system, on the real MD ensemble (all stored frames), heavy atoms only:
    bond   = mean over heavy bonds of std(length)              [Angstrom]
    angle  = mean over heavy angles of std(angle)               [deg]
    torsion= mean over heavy proper torsions of circular std    [deg]
    oop    = mean over planar centres of std(out-of-plane)      [deg], only centres whose mean
             |deviation| in the reference trajectory is < 10 deg; systems with 8..40 heavy atoms
             (the filter of theory_modules_v1/analysis.py that produced the original 6.48)
  tau = median over systems.  The functions are imported, not re-implemented.
Population: every store system whose PDB belongs to a UniProt target assigned to the TRAIN split
(target_disjoint_split, seed 20260918) and is not one of the excluded systems.  Target membership
is read from all status=='ok' rows of pairs.csv; split from the usable-pair target assignment.
Reads only store npz of train-target systems.
"""
from __future__ import annotations
import json, math, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); os.chdir(ROOT)
from isotropic_noise_geometry_cost import (enumerate_internals, bond_lengths, bond_angles_degrees,  # noqa: E402
                                           dihedrals_degrees, circular_std_degrees, internal_rmsf)
from mehf_internal_geometry_spectrum import improper_centres, out_of_plane_degrees  # noqa: E402
from mdwm.data.mmp_transfer import (DEFAULT_STORE, load_usable_pairs, target_disjoint_split,  # noqa: E402
                                    load_excluded_systems)
from mdwm.models import harmonic_prior as hp  # noqa: E402

OUT = ROOT / "outputs/clean_protocol_v1"

def measure(pdb):
    with np.load(Path(DEFAULT_STORE) / f"{pdb}.npz", allow_pickle=True) as z:
        coords = z["ligand_coords"].astype(np.float64)
        bond_index = z["bond_index"]
        elements = z["ligand_atomic_numbers"].astype(int)
    n_frames, n_atoms, _ = coords.shape
    if n_atoms < 4 or n_frames < 10:
        return dict(pdb=pdb, reject="n_atoms<4 or n_frames<10")
    bonds, angles, quads = enumerate_internals(bond_index, n_atoms)
    heavy = elements != 1
    bonds = [b for b in bonds if heavy[b[0]] and heavy[b[1]]]
    angles = [a for a in angles if all(heavy[i] for i in a)]
    quads = [q for q in quads if all(heavy[i] for i in q)]
    if not bonds or not angles or not quads:
        return dict(pdb=pdb, reject="no heavy bonds/angles/torsions")
    rmsf = internal_rmsf(coords)
    if not np.isfinite(rmsf) or rmsf <= 1e-6:
        return dict(pdb=pdb, reject="rmsf<=1e-6")
    row = dict(pdb=pdb, n_heavy=int(heavy.sum()),
               bond=float(np.mean(bond_lengths(coords, bonds).std(axis=0))),
               angle=float(np.mean(bond_angles_degrees(coords, angles).std(axis=0))),
               torsion=float(np.mean(circular_std_degrees(dihedrals_degrees(coords, quads)))),
               oop=float("nan"))
    nh = int(heavy.sum())
    if 8 <= nh <= 40:
        remap = -np.ones(len(elements), dtype=int); remap[heavy] = np.arange(nh)
        hb = sorted({(min(int(remap[a]), int(remap[b])), max(int(remap[a]), int(remap[b])))
                     for a, b in np.asarray(bond_index).reshape(-1, 2)
                     if a != b and heavy[a] and heavy[b]})
        xyz = coords[:, heavy, :]
        centres = improper_centres(np.asarray(hb), nh)
        if centres:
            dev = np.abs(out_of_plane_degrees(xyz, centres)).mean(axis=0)
            centres = [c for c, d in zip(centres, dev) if d < 10.0]
        if centres:
            row["oop"] = float(np.mean(out_of_plane_degrees(xyz, centres).std(axis=0)))
    return row

def boot(values, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values); n_sys = len(v)
    sys_med = [np.median(v[rng.integers(0, n_sys, n_sys)]) for _ in range(n)]
    uniq = np.unique(groups); idx = {u: np.flatnonzero(groups == u) for u in uniq}
    cl_med = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        cl_med.append(np.median(v[np.concatenate([idx[uniq[i]] for i in pick])]))
    return float(np.std(sys_med)), float(np.std(cl_med))

def main():
    t0 = time.time()
    usable = target_disjoint_split(load_usable_pairs())
    train_targets = set(usable[usable.primary_split == "train"].uniprot)
    other_targets = set(usable[usable.primary_split != "train"].uniprot)
    raw = pd.read_csv(ROOT / "outputs/mmp_transfer_corpus_v1/pairs.csv")
    raw = raw[raw.status == "ok"]
    pu = {}
    for a, b, u in zip(raw.pdb_a.astype(str).str.upper(), raw.pdb_b.astype(str).str.upper(), raw.uniprot):
        for p in (a, b):
            assert pu.setdefault(p, u) == u, p
    excluded = load_excluded_systems(OUT / "excluded_systems.json")
    pop = sorted(p for p, u in pu.items() if u in train_targets and p not in excluded)
    assert not (set(pop) & {p for p, u in pu.items() if u in other_targets}), "train/val-test overlap"
    pop = [p for p in pop if (Path(DEFAULT_STORE) / f"{p}.npz").exists()]
    print(f"train-target systems: {len(pop)} (excluded removed: {sum(1 for p,u in pu.items() if u in train_targets and p in excluded)})", flush=True)
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        rows = pool.map(measure, pop, chunksize=8)
    df = pd.DataFrame(rows); df["uniprot"] = df.pdb.map(pu)
    rej = df[df.get("reject").notna()] if "reject" in df else df.iloc[0:0]
    ok = df[df.get("reject").isna()] if "reject" in df else df
    df.to_csv(OUT / "tau_train_only_large.per_system.csv", index=False)
    cur = {"bond": hp.TAU_BOND_ANGSTROM, "angle": math.degrees(hp.TAU_ANGLE_RADIAN),
           "torsion": math.degrees(hp.TAU_TORSION_RADIAN), "oop": math.degrees(hp.TAU_OOP_RADIAN)}
    small = json.loads((OUT / "tau_train_only.json").read_text())["widths"]
    small = {"bond": small["bond_angstrom"], "angle": small["angle_deg"], "torsion": small["torsion_deg"], "oop": small["oop_deg"]}
    res = {}
    for k in ("bond", "angle", "torsion", "oop"):
        sub = ok.dropna(subset=[k])
        se_s, se_c = boot(sub[k].to_numpy(), sub.uniprot.to_numpy())
        v = float(sub[k].median())
        res[k] = dict(n_systems=int(len(sub)), n_targets=int(sub.uniprot.nunique()), tau_large=v,
                      se_system_bootstrap=se_s, se_target_cluster_bootstrap=se_c,
                      current_constant=cur[k], rel_change_vs_current=v / cur[k] - 1,
                      earlier_small_sample=dict(n=small[k]["n_train"], value=small[k]["train_only"],
                                                se=small[k]["bootstrap_se_of_median"]),
                      rel_change_vs_small=v / small[k]["train_only"] - 1)
    out = dict(population=f"{len(pop)} store systems of train-split UniProt targets minus 28 excluded; "
                          f"{len(rej)} rejected by the estimator's own filters",
               rejected=rej[["pdb", "reject"]].to_dict("records") if len(rej) else [],
               units=dict(bond="Angstrom", angle="deg", torsion="deg", oop="deg"), widths=res,
               n_bootstrap=2000, wall_seconds=time.time() - t0)
    (OUT / "tau_train_only_large.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(res, indent=1)); print("wall", time.time() - t0)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        # reproduce stored per-system values of the original runs
        ps = pd.read_csv("reference_outputs/isotropic_noise_cost_v1/per_system.csv").set_index("system")
        for p in ("4DBN", "3F3T"):
            r = measure(p)
            print(p, r["bond"], ps.loc[p, "bond_length_spread_physical"], r["angle"], ps.loc[p, "bond_angle_spread_physical"],
                  r["torsion"], ps.loc[p, "dihedral_spread_physical"])
        tm = pd.DataFrame(json.loads(Path("reference_outputs/theory_modules_v1/results.json").read_text())["rows"]).dropna(subset=["tau_oop_deg"]).head(8)
        for s, t in zip(tm.system, tm.tau_oop_deg):
            r = measure(str(s).split("_")[0]); print(s, r.get("oop"), t)
    else:
        main()
