"""Post-hoc: overlap between trajectory-derived and first-frame-only 96-atom pockets (see PLAN.md)."""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
store = ROOT / "data/MISATO/extracted/frames100_all_v1"
ex = list(csv.DictReader(open(ROOT / "reference_outputs/sigma_star_test_v1/test_examples.csv")))
systems = sorted({r["pdb_a"] for r in ex} | {r["pdb_b"] for r in ex})
K = 96
rows = []
for pdb in systems:
    z = np.load(store / f"{pdb}.npz")
    pc0 = z["pocket_coords"][0].astype(np.float64); lig0 = z["ligand_coords"][0].astype(np.float64)
    dmin = z["pocket_min_distance_over_frames"]
    d0 = np.sqrt(((pc0[:, None, :] - lig0[None, :, :]) ** 2).sum(-1)).min(1)
    n = len(dmin)
    if n <= K:
        rows.append({"pdb": pdb, "n_candidates": n, "cropped": False, "overlap": 1.0}); continue
    U = set(np.argsort(dmin, kind="stable")[:K]); order0 = np.argsort(d0, kind="stable"); F = set(order0[:K])
    rank0 = np.empty(n, int); rank0[order0] = np.arange(n)
    diff = sorted(U - F)
    rows.append({"pdb": pdb, "n_candidates": n, "cropped": True, "overlap": len(U & F) / K,
                 "d0_96th": float(np.sort(d0)[K - 1]),
                 "diff_d0": [float(d0[i]) for i in diff], "diff_rank0": [int(rank0[i]) + 1 for i in diff]})
ov = np.array([r["overlap"] for r in rows]); cr = [r for r in rows if r["cropped"]]
dd = np.concatenate([r["diff_d0"] for r in cr if r["diff_d0"]]) if any(r["diff_d0"] for r in cr) else np.array([])
res = {"n_systems": len(rows), "n_cropped": len(cr), "n_uncropped": len(rows) - len(cr),
       "overlap_median": float(np.median(ov)), "overlap_q25": float(np.quantile(ov, .25)),
       "overlap_min": float(ov.min()), "frac_full_overlap": float((ov == 1).mean()),
       "max_d0_96th": float(max(r["d0_96th"] for r in cr)),
       "n_atoms_differing_total": int(sum(len(r["diff_d0"]) for r in cr)),
       "differing_atoms_frame0_distance_median": float(np.median(dd)) if dd.size else None,
       "differing_atoms_frame0_distance_range": [float(dd.min()), float(dd.max())] if dd.size else None,
       "differing_atoms_frame0_rank_median": float(np.median(np.concatenate([r["diff_rank0"] for r in cr if r["diff_rank0"]]))) if dd.size else None}
(OUT / "RESULT.json").write_text(json.dumps({"summary": res, "per_system": rows}, indent=1))
print(json.dumps(res, indent=1))
