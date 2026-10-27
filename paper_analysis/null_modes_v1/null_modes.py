"""null_modes_v1 (post hoc; plan sha256 83126af8...). Usage: python null_modes.py <MEHF root> <out.json>"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
root = Path(sys.argv[1]); sys.path.insert(0, str(root / "src"))
from mdwm.models.harmonic_prior import sigma_star_factor
pairs = pd.read_csv(root / "outputs/mmp_transfer_corpus_v1/pairs.csv")
pairs = pairs[pairs.status == "ok"]
systems = sorted(set(pairs.pdb_a) | set(pairs.pdb_b))
store = root / "data/MISATO/extracted/frames100_all_v1"
counts, extra, skipped = {}, [], 0
for pdb in systems:
    f = store / f"{pdb}.npz"
    with np.load(f, allow_pickle=True) as d:
        z = d["ligand_atomic_numbers"].astype(int); heavy = np.where(z != 1)[0]
        remap = {int(a): i for i, a in enumerate(heavy)}
        bi = [(remap[int(a)], remap[int(b)]) for a, b in np.asarray(d["bond_index"]).reshape(-1, 2)
              if int(a) in remap and int(b) in remap and int(a) != int(b)]
        x = torch.tensor(d["ligand_coords"][0, heavy], dtype=torch.float64)
    if len(heavy) < 4 or not bi:
        skipped += 1; continue
    with torch.enable_grad():
        fac, n_rigid = sigma_star_factor(x, torch.tensor(bi, dtype=torch.long))
    if fac is None:
        skipped += 1; continue
    counts[pdb] = n_rigid
    if n_rigid > 6: extra.append((pdb, n_rigid, len(heavy)))
n = len(counts); k = len(extra)
out = {"n_systems": n, "n_skipped": skipped, "n_more_than_six": k, "fraction": k / n,
       "histogram": dict(Counter(counts.values())), "systems_more_than_six": extra}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1)); print({kk: v for kk, v in out.items() if kk != "systems_more_than_six"}); print(extra[:20])
