"""Independent check: does harmonic_prior.py's autograd B match exact_sigma_star_noise.py's
finite-difference B, on the SAME systems, at the level of the actual Sigma* factor
(not just the aggregated channel ratios)?  Also profile the systems whose Hessian
does not have exactly 6 near-zero eigenvalues (the 4.7% reported but never inspected).
"""
import sys, math, json, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mdwm.models.harmonic_prior import sigma_star_factor, topology  # noqa: E402
import exact_sigma_star_noise as ex  # noqa: E402
from isotropic_noise_geometry_cost import enumerate_internals  # noqa: E402

store = ROOT / "data/MISATO/extracted/frames100_all_v1"
paths = sorted(store.glob("*.npz"))
rng = np.random.default_rng(20260924)
rng.shuffle(paths)

n_target = 60
max_atoms = 40
compared = []
rigid_report = []
used = 0
t0 = time.time()
for path in paths:
    if used >= n_target:
        break
    with np.load(path, allow_pickle=True) as st:
        coords = st["ligand_coords"].astype(np.float64)
        bond_index = st["bond_index"]
        elements = st["ligand_atomic_numbers"].astype(int)
    heavy = np.where(elements != 1)[0]
    if not (8 <= len(heavy) <= max_atoms):
        continue
    remap = {int(a): i for i, a in enumerate(heavy)}
    pairs = [(remap[int(a)], remap[int(b)])
             for a, b in np.asarray(bond_index).reshape(-1, 2)
             if int(a) in remap and int(b) in remap and int(a) != int(b)]
    if not pairs:
        continue
    block = coords[:, heavy, :]
    n_atoms = len(heavy)
    bonds, angles, torsions = enumerate_internals(np.asarray(pairs), n_atoms)
    centres = ex.improper_centres(np.asarray(pairs), n_atoms)
    if not bonds or not angles or not torsions:
        continue
    reference = block.mean(axis=0)

    # -- finite-difference route (exact_sigma_star_noise.py) --
    try:
        matrix, taus = ex.b_rows(reference, bonds, angles, torsions, centres, n_atoms)
    except Exception as e:
        continue
    weights = 1.0 / np.square(taus)
    hessian_fd = matrix.T @ (weights[:, None] * matrix)
    vals_fd, _ = np.linalg.eigh(hessian_fd)
    cut_fd = max(vals_fd.max(), 1e-30) * 1e-6
    keep_fd = vals_fd > cut_fd
    n_rigid_fd = int((~keep_fd).sum())

    # -- autograd route (harmonic_prior.py) --
    bt = torch.as_tensor(np.asarray(pairs), dtype=torch.long)
    ct = torch.as_tensor(reference, dtype=torch.float64)
    factor, n_rigid_ag = sigma_star_factor(ct, bt)
    if factor is None:
        continue

    # Compare eigenvalue spectra of H = B^T F B (rotation/translation invariant
    # quantity, so a permutation of atom order would still agree here).
    nonrigid_fd = np.sort(vals_fd[keep_fd])
    # reconstruct autograd hessian eigenvalues from the factor: factor columns
    # have amplitude = eig^{-1/2}, so eig = amplitude^{-2}.
    amp_ag = 1.0 / factor.norm(dim=0)  # not exactly the eigenvalue in general basis,
    # instead rebuild H_ag directly and eigendecompose for a clean comparison.
    pairs_t, angles_t, torsions_t, centres_t = topology(bt, n_atoms)
    from mdwm.models.harmonic_prior import _b_rows as _b_rows_ag
    rows_ag, w_ag = _b_rows_ag(ct, pairs_t, angles_t, torsions_t, centres_t)
    hessian_ag = (rows_ag.T @ (w_ag[:, None] * rows_ag)).numpy()
    vals_ag, _ = np.linalg.eigh(hessian_ag)
    cut_ag = max(vals_ag.max(), 1e-30) * 1e-6
    keep_ag = vals_ag > cut_ag
    n_rigid_ag2 = int((~keep_ag).sum())
    nonrigid_ag = np.sort(vals_ag[keep_ag])

    n = min(len(nonrigid_fd), len(nonrigid_ag))
    if n > 0 and len(nonrigid_fd) == len(nonrigid_ag):
        rel_err = np.abs(nonrigid_fd - nonrigid_ag) / np.clip(np.abs(nonrigid_fd), 1e-8, None)
        max_rel_err = float(rel_err.max())
        median_rel_err = float(np.median(rel_err))
    else:
        max_rel_err = None
        median_rel_err = None

    compared.append({
        "path": path.name, "n_atoms": n_atoms,
        "n_bonds": len(bonds), "n_angles": len(angles),
        "n_torsions": len(torsions), "n_centres": len(centres),
        "n_rigid_fd": n_rigid_fd, "n_rigid_ag": n_rigid_ag2,
        "n_rigid_ag_reported": n_rigid_ag,
        "n_dof_match": len(nonrigid_fd) == len(nonrigid_ag),
        "max_rel_eig_err": max_rel_err,
        "median_rel_eig_err": median_rel_err,
    })
    rigid_report.append({"path": path.name, "n_atoms": n_atoms, "n_rigid_fd": n_rigid_fd,
                          "n_rigid_ag": n_rigid_ag2})
    used += 1
    if used % 10 == 0:
        print(f"...{used} done, {time.time()-t0:.1f}s", flush=True)

print(f"compared {used} systems", flush=True)
mismatches = [c for c in compared if c["n_rigid_fd"] != c["n_rigid_ag"]]
print(f"n_rigid mismatches between FD and autograd routes: {len(mismatches)}")
for m in mismatches[:20]:
    print(" MISMATCH", m)

dof_mismatches = [c for c in compared if not c["n_dof_match"]]
print(f"non-rigid DOF count mismatches: {len(dof_mismatches)}")

errs = [c["max_rel_eig_err"] for c in compared if c["max_rel_eig_err"] is not None]
if errs:
    print("max relative eigenvalue error across all compared systems:", max(errs))
    print("median of per-system max relative eigenvalue error:", float(np.median(errs)))

not_six = [c for c in compared if c["n_rigid_fd"] != 6]
print(f"\nsystems with n_rigid != 6 (FD route): {len(not_six)} / {used} = {len(not_six)/used:.3%}")
for c in not_six:
    print(" NOT-SIX", c)

(ROOT / "outputs/sigma_star_v1/b_row_verification.json").write_text(
    json.dumps({"compared": compared, "n_used": used}, indent=2)
)
