"""ref_contact_v1 (post hoc). Usage: python -I ref_contact.py <MEHF root> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
root = Path(sys.argv[1]); pairs = pd.read_csv(root / "outputs/mmp_transfer_corpus_v1/pairs.csv")
pairs = pairs[pairs.status == "ok"]; systems = sorted(set(pairs.pdb_a) | set(pairs.pdb_b))
store = root / "data/MISATO/extracted/frames100_all_v1"; mins = {}
for pdb in systems:
    with np.load(store / f"{pdb}.npz", allow_pickle=True) as d:
        L = d["ligand_coords"][:, d["ligand_atomic_numbers"] != 1]
        P = d["pocket_coords"][:, d["pocket_atomic_numbers"] != 1]
    dist = np.sqrt(((L[:, :, None, :] - P[:, None, :, :]) ** 2).sum(-1))
    mins[pdb] = float(dist.min())
v = np.array(list(mins.values()))
out = {"n_systems": len(v), "median": float(np.median(v)), "p1": float(np.percentile(v, 1)), "p5": float(np.percentile(v, 5)),
       "min": float(v.min()), "argmin": min(mins, key=mins.get), "n_below_2.0": int((v < 2.0).sum()), "n_below_1.2": int((v < 1.2).sum())}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1)); print(out)
