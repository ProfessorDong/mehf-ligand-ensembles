"""Post-hoc matched-aggregation aromatic planarity reference. See PLAN.md."""
import json, math, sys
from pathlib import Path
import numpy as np
from rdkit import Chem

ref_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
pooled, per_mol, n_files, n_read = [], [], 0, 0
for path in sorted(ref_dir.glob("*.sdf")):
    n_files += 1
    mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)), None)
    if mol is None:
        continue
    n_read += 1
    p = mol.GetConformer().GetPositions()
    vals = []
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic():
            continue
        near = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(near) != 3:
            continue
        j, (a, b, c) = atom.GetIdx(), near
        u = p[a] - p[j]
        n = np.cross(p[b] - p[j], p[c] - p[j])
        n = n / (np.linalg.norm(n) + 1e-12)
        u = u / (np.linalg.norm(u) + 1e-12)
        vals.append(abs(math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(u, n))))))))
    pooled += vals
    if vals:
        per_mol.append(float(np.mean(vals)))
res = {
    "status": "post-hoc; plan PLAN.md",
    "n_files": n_files, "n_read": n_read,
    "pooled_mean_deg": float(np.mean(pooled)), "n_centres": len(pooled),
    "median_of_molecule_means_deg": float(np.median(per_mol)),
    "iqr_of_molecule_means_deg": [float(np.percentile(per_mol, 25)), float(np.percentile(per_mol, 75))],
    "n_molecules_with_centres": len(per_mol),
}
(out_dir / "RESULT.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
