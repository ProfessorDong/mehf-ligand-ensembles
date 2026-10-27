"""For the systems with more than six near-zero modes, the largest heavy-atom bond angle in frame 0 (post hoc)."""
import json, sys
from pathlib import Path
import numpy as np
store = Path(sys.argv[1]); res = json.loads(Path(sys.argv[2]).read_text())
out = {}
for pdb, nr, nh in res["systems_more_than_six"]:
    with np.load(store / f"{pdb}.npz", allow_pickle=True) as d:
        z = d["ligand_atomic_numbers"].astype(int); heavy = set(np.where(z != 1)[0].tolist())
        x = d["ligand_coords"][0]; nb = {}
        for a, b in np.asarray(d["bond_index"]).reshape(-1, 2):
            a, b = int(a), int(b)
            if a in heavy and b in heavy and a != b:
                nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
    best = 0.0
    for j, ns in nb.items():
        ns = sorted(ns)
        for i in range(len(ns)):
            for k in range(i + 1, len(ns)):
                u, v = x[ns[i]] - x[j], x[ns[k]] - x[j]
                best = max(best, float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))))
    out[pdb] = round(best, 1)
print(out)
res["max_heavy_angle_frame0_deg"] = out; Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
