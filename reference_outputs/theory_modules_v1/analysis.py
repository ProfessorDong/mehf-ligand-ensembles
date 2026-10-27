"""Theory-module analysis for section 4.  Predictions are pre-registered in
`outputs/theory_modules_v1/PREDICTIONS.md` (sha256 in PREDICTIONS.sha256) and
were written before any number in this directory was computed.

No model is involved anywhere in this script.  Everything is a property of real
ligand geometry, of the Wilson B-matrix built from it, of the harmonic
covariance Sigma* = (B^T F B)^+ implied by the corpus-measured thermal widths,
and of a numpy re-implementation of `_structure_local_noise`'s Gauss-Seidel
sweep (same sweep count, same under-relaxation, same targets).

Systems whose PDB id is in the corpus `test` split are excluded up front.

REVISION NOTE, recorded rather than hidden: a first version of this script
measured P-T1d as the squared projection of the out-of-plane B-vector onto
rowspace(B_bond, B_angle).  That statistic came back at 1.000 and is reported as
such, but it is the WRONG statistic: rowspace(B_bond,B_angle) already covers
~85% of the non-rigid space, so its orthogonal complement is only ~19-dimensional
and "is this direction in the span" is uninformative.  It is replaced, and both
results are reported, by (i) the exact orthogonality check at perfect planarity
and (ii) the operational quantity -- the out-of-plane width the actual sampler
produces, against the physical one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mdwm.eval.ensemble_metrics import rmsf_profile  # noqa: E402
from isotropic_noise_geometry_cost import (  # noqa: E402
    enumerate_internals, bond_lengths, bond_angles_degrees,
    dihedrals_degrees, circular_std_degrees,
)
from mehf_internal_geometry_spectrum import (  # noqa: E402
    improper_centres, out_of_plane_degrees,
)

TAU_BOND_A = 0.0326
TAU_ANGLE_RAD = np.radians(2.774)
TAU_TORSION_RAD = np.radians(14.6)
SWEEPS = 32                 # aniso_v3 config: anisotropic_sweeps = 32
RELAX = 0.5                 # projection_relaxation / projection_angle_relaxation

COVALENT_RADIUS = {
    5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 33: 1.19, 34: 1.20, 35: 1.20, 53: 1.39,
    11: 1.66, 12: 1.41, 19: 2.03, 20: 1.76, 25: 1.39, 26: 1.32,
    27: 1.26, 29: 1.32, 30: 1.22, 44: 1.46, 78: 1.36,
}


# ---------------------------------------------------------------- B-vectors
def b_bond(x, i, j, n):
    row = np.zeros(3 * n)
    d = x[i] - x[j]
    u = d / np.linalg.norm(d)
    row[3 * i:3 * i + 3] = u
    row[3 * j:3 * j + 3] = -u
    return row


def b_angle(x, i, j, k, n):
    row = np.zeros(3 * n)
    a, b = x[i] - x[j], x[k] - x[j]
    ra, rb = np.linalg.norm(a), np.linalg.norm(b)
    u, v = a / ra, b / rb
    cos = float(np.clip(u @ v, -1.0, 1.0))
    sin = max(np.sqrt(max(1.0 - cos * cos, 0.0)), 1e-8)
    si = (cos * u - v) / (ra * sin)
    sk = (cos * v - u) / (rb * sin)
    row[3 * i:3 * i + 3] = si
    row[3 * k:3 * k + 3] = sk
    row[3 * j:3 * j + 3] = -(si + sk)
    return row


def b_torsion(x, i, j, k, l, n):
    row = np.zeros(3 * n)
    f, g, h = x[i] - x[j], x[j] - x[k], x[l] - x[k]
    a, b = np.cross(f, g), np.cross(h, g)
    ga, gb, gn = a @ a, b @ b, np.linalg.norm(g)
    if ga < 1e-10 or gb < 1e-10 or gn < 1e-8:
        return row
    si = -gn / ga * a
    sl = gn / gb * b
    sj = (f @ g) / (ga * gn) * a - (h @ g) / (gb * gn) * b - si
    row[3 * i:3 * i + 3] = si
    row[3 * j:3 * j + 3] = sj
    row[3 * l:3 * l + 3] = sl
    row[3 * k:3 * k + 3] = -(si + sj + sl)
    return row


def b_out_of_plane(x, j, a, b, c, n):
    """dchi/dx for Wilson's out-of-plane bend at centre j, radians per Angstrom.

        sin(chi) = (u_b x u_c) . u_a / sin(phi_bc)

    the SAME coordinate `out_of_plane_degrees` measures.  Wilson/Decius/Cross:

        s_a = (1/r_a)[ (u_b x u_c)/(cos chi sin phi) - tan(chi) u_a ]
        s_b = (1/r_b)[ (u_c x u_a)/(cos chi sin phi)
                       - (tan chi / sin^2 phi)(u_b - cos phi u_c) ]
        s_c = (1/r_c)[ (u_a x u_b)/(cos chi sin phi)
                       - (tan chi / sin^2 phi)(u_c - cos phi u_b) ]
        s_j = -(s_a + s_b + s_c)

    At chi = 0 the tan terms vanish, all four are parallel to the plane normal
    with magnitudes 1/(r sin phi), and the row is the centroid-preserving
    umbrella mode.
    """
    row = np.zeros(3 * n)
    va, vb, vc = x[a] - x[j], x[b] - x[j], x[c] - x[j]
    ra, rb, rc = np.linalg.norm(va), np.linalg.norm(vb), np.linalg.norm(vc)
    ua, ub, uc = va / ra, vb / rb, vc / rc
    cos_phi = float(np.clip(ub @ uc, -1.0, 1.0))
    sin_phi = max(np.sqrt(max(1.0 - cos_phi ** 2, 0.0)), 1e-8)
    chi = np.arcsin(float(np.clip(np.cross(ub, uc) @ ua / sin_phi, -1.0, 1.0)))
    cos_chi = max(np.cos(chi), 1e-8)
    tan_chi = np.tan(chi)
    den = cos_chi * sin_phi
    sa = (np.cross(ub, uc) / den - tan_chi * ua) / ra
    sb = (np.cross(uc, ua) / den - (tan_chi / sin_phi ** 2) * (ub - cos_phi * uc)) / rb
    sc = (np.cross(ua, ub) / den - (tan_chi / sin_phi ** 2) * (uc - cos_phi * ub)) / rc
    row[3 * a:3 * a + 3] = sa
    row[3 * b:3 * b + 3] = sb
    row[3 * c:3 * c + 3] = sc
    row[3 * j:3 * j + 3] = -(sa + sb + sc)
    return row


def rigid_basis(x):
    n = len(x)
    c = x - x.mean(axis=0)
    cols = []
    for axis in range(3):
        v = np.zeros((n, 3)); v[:, axis] = 1.0; cols.append(v.ravel())
    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        cols.append(np.cross(np.broadcast_to(e, c.shape), c).ravel())
    q, _ = np.linalg.qr(np.stack(cols, axis=1))
    return q.T


def psd_pinv(M, rank):
    """Inverse on the top-`rank` eigenspace; the rest is set to zero."""
    w, v = np.linalg.eigh(M)
    order = np.argsort(w)[::-1][:rank]
    return (v[:, order] / np.clip(w[order], 1e-12, None)) @ v[:, order].T


# ------------------------------------------------- Gauss-Seidel, as in mehf.py
def gauss_seidel_noise(ref, bonds, angles, sigma, n_draws, rng,
                       tau_b=TAU_BOND_A, tau_a=TAU_ANGLE_RAD,
                       sweeps=SWEEPS, relax=RELAX,
                       structure_bonds=True, extra=None):
    """Numpy re-implementation of `_structure_local_noise`.

    `extra` is an optional list of (B-row, tau) pairs added to the stiff set --
    used to test the improper without touching src/.
    """
    n = len(ref)
    eps = sigma * rng.standard_normal((n_draws, n, 3))
    bi = np.array([b[0] for b in bonds]); bj = np.array([b[1] for b in bonds])
    u = ref[bi] - ref[bj]
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    wb = tau_b * rng.standard_normal((n_draws, len(bonds)))

    A = np.stack([b_angle(ref, i, j, k, n) for i, j, k in angles])
    A3 = A.reshape(len(angles), n, 3)
    a_norm2 = (A ** 2).sum(1)
    wa = tau_a * rng.standard_normal((n_draws, len(angles)))

    if extra:
        E = np.stack([e[0] for e in extra])
        E3 = E.reshape(len(extra), n, 3)
        e_norm2 = (E ** 2).sum(1)
        we = np.stack([e[1] for e in extra])[None, :] * rng.standard_normal(
            (n_draws, len(extra)))

    for _ in range(sweeps):
        if structure_bonds:
            comp = ((eps[:, bi] - eps[:, bj]) * u[None]).sum(-1)
            delta = ((wb - comp) * (0.5 * relax))[..., None] * u[None]
            np.add.at(eps, (slice(None), bi), delta)
            np.add.at(eps, (slice(None), bj), -delta)
        comp = np.einsum("dna,mna->dm", eps, A3)
        step = relax * (wa - comp) / a_norm2[None]
        eps = eps + np.einsum("dm,mna->dna", step, A3)
        if extra:
            comp = np.einsum("dna,mna->dm", eps, E3)
            step = relax * (we - comp) / e_norm2[None]
            eps = eps + np.einsum("dm,mna->dna", step, E3)
    return eps


def measure(block, bonds, angles, quads, centres):
    out = {
        "rmsf": float(np.mean(rmsf_profile(block))),
        "bond": float(np.mean(bond_lengths(block, bonds).std(axis=0))),
        "angle": float(np.mean(bond_angles_degrees(block, angles).std(axis=0))),
        "torsion": float(np.mean(circular_std_degrees(dihedrals_degrees(block, quads)))),
    }
    if centres:
        out["oop"] = float(np.mean(out_of_plane_degrees(block, centres).std(axis=0)))
    return out


# --------------------------------------------------------------- per system
def analyse_system(path, rng, n_draws=200):
    with np.load(path, allow_pickle=True) as store:
        coords = store["ligand_coords"].astype(np.float64)
        bond_index = np.asarray(store["bond_index"])
        elements = store["ligand_atomic_numbers"].astype(int)

    heavy = elements != 1
    if heavy.sum() < 8 or heavy.sum() > 40 or coords.shape[0] < 10:
        return None
    remap = -np.ones(len(elements), dtype=int)
    remap[heavy] = np.arange(int(heavy.sum()))
    hb = sorted({(min(int(remap[a]), int(remap[b])), max(int(remap[a]), int(remap[b])))
                 for a, b in np.asarray(bond_index).reshape(-1, 2)
                 if a != b and heavy[a] and heavy[b]})
    if not hb:
        return None
    hb = np.asarray(hb)
    xyz = coords[:, heavy, :]
    z = elements[heavy]
    n = xyz.shape[1]
    dof = 3 * n - 6

    bonds, angles, quads = enumerate_internals(hb, n)
    if not bonds or not angles or not quads:
        return None
    ref = xyz.mean(axis=0)

    centres = improper_centres(hb, n)
    if centres:
        dev = np.abs(out_of_plane_degrees(xyz, centres)).mean(axis=0)
        centres = [c for c, d in zip(centres, dev) if d < 10.0]

    out = {"system": path.stem, "n_atoms": n, "dof": dof,
           "n_bonds": len(bonds), "n_angles": len(angles),
           "n_torsions": len(quads), "n_planar_centres": len(centres)}

    B_bond = np.stack([b_bond(ref, i, j, n) for i, j in bonds])
    B_ang = np.stack([b_angle(ref, i, j, k, n) for i, j, k in angles])
    B_tor = np.stack([b_torsion(ref, *q, n) for q in quads])
    out["Bnorm2_bond"] = float(np.median((B_bond ** 2).sum(1)))
    out["Bnorm2_angle"] = float(np.median((B_ang ** 2).sum(1)))
    out["Bnorm2_torsion"] = float(np.median((B_tor ** 2).sum(1)))

    B_oop = None
    if centres:
        B_oop = np.stack([b_out_of_plane(ref, *c, n) for c in centres])
        out["Bnorm2_oop"] = float(np.median((B_oop ** 2).sum(1)))
        # finite-difference control on the analytic B-vector
        errs = []
        for row, centre in zip(B_oop[:3], centres[:3]):
            g = np.zeros(3 * n)
            for atom in centre:
                for ax in range(3):
                    p = ref.copy(); p[atom, ax] += 1e-5
                    hi = np.radians(out_of_plane_degrees(p[None], [centre])[0, 0])
                    p[atom, ax] -= 2e-5
                    lo = np.radians(out_of_plane_degrees(p[None], [centre])[0, 0])
                    g[3 * atom + ax] = (hi - lo) / 2e-5
            errs.append(float(np.linalg.norm(g - row) / max(np.linalg.norm(g), 1e-12)))
        out["oop_bvector_fd_relerr"] = float(np.median(errs))
        # pre-registered (wrong) statistic, reported anyway
        BA = np.concatenate([B_bond, B_ang])
        u_, s_, _ = np.linalg.svd(BA.T, full_matrices=False)
        Q = u_[:, s_ > s_.max() * 1e-8]
        out["rowspace_dim_BA_over_dof"] = float(Q.shape[1] / dof)
        proj, rnd = [], []
        R = rigid_basis(ref)
        for row in B_oop:
            e = row / np.linalg.norm(row)
            proj.append(float((Q.T @ e) @ (Q.T @ e)))
        for _ in range(20):
            e = rng.standard_normal(3 * n); e -= R.T @ (R @ e); e /= np.linalg.norm(e)
            rnd.append(float((Q.T @ e) @ (Q.T @ e)))
        out["oop_rowspace_projection"] = float(np.median(proj))
        out["random_rowspace_projection"] = float(np.median(rnd))
        # per-row overlap of the OOP direction with individual stiff coordinates
        ehat = B_oop / np.linalg.norm(B_oop, axis=1, keepdims=True)
        out["oop_vs_bond_cos_max"] = float(np.median(
            np.abs(ehat @ B_bond.T / np.linalg.norm(B_bond, axis=1)).max(axis=1)))
        out["oop_vs_angle_cos_max"] = float(np.median(
            np.abs(ehat @ B_ang.T / np.linalg.norm(B_ang, axis=1)).max(axis=1)))
        out["oop_dof_share"] = float(len(centres) / dof)
        out["tau_oop_deg"] = float(np.mean(out_of_plane_degrees(xyz, centres).std(axis=0)))
        # second-order scaling of bond/angle under a pure umbrella displacement
        sc = []
        for centre in centres[:4]:
            row = b_out_of_plane(ref, *centre, n).reshape(n, 3)
            row = row / np.linalg.norm(row)
            base_b = bond_lengths(ref[None], bonds)[0]
            base_a = bond_angles_degrees(ref[None], angles)[0]
            d = {}
            for delta in (0.1, 0.2):
                p = ref + delta * row
                d[delta] = (float(np.abs(bond_lengths(p[None], bonds)[0] - base_b).max()),
                            float(np.abs(bond_angles_degrees(p[None], angles)[0] - base_a).max()))
            sc.append((d[0.2][0] / max(d[0.1][0], 1e-12),
                       d[0.2][1] / max(d[0.1][1], 1e-12)))
        out["oop_secondorder_bond_ratio"] = float(np.median([s[0] for s in sc]))
        out["oop_secondorder_angle_ratio"] = float(np.median([s[1] for s in sc]))

    # ---------------- physical reference widths of this system --------------
    out["phys"] = measure(xyz, bonds, angles, quads, centres)

    # ---------------- harmonic metric and Sigma* ----------------------------
    f_b, f_a, f_t = 1 / TAU_BOND_A ** 2, 1 / TAU_ANGLE_RAD ** 2, 1 / TAU_TORSION_RAD ** 2
    K = f_b * B_bond.T @ B_bond + f_a * B_ang.T @ B_ang + f_t * B_tor.T @ B_tor
    R = rigid_basis(ref)
    P = np.eye(3 * n) - R.T @ R
    Kp = P @ K @ P
    Sigma = P @ psd_pinv(Kp, dof) @ P

    # T2: residual-variance allocation implied by an objective W is W^+.
    # Statistic: variance along a typical non-rigid direction, divided by
    # variance along a bond-stretch direction.  W = I gives exactly 1.
    def allocation_anisotropy(W):
        Winv = P @ psd_pinv(P @ W @ P, dof) @ P
        rq_bond = []
        for i, j in bonds:
            e = np.zeros(3 * n)
            uu = ref[i] - ref[j]; uu = uu / np.linalg.norm(uu)
            e[3 * i:3 * i + 3] = uu / np.sqrt(2)
            e[3 * j:3 * j + 3] = -uu / np.sqrt(2)
            e = P @ e; e /= np.linalg.norm(e)
            rq_bond.append(float(e @ Winv @ e))
        rq_rand = []
        for _ in range(64):
            e = P @ rng.standard_normal(3 * n); e /= np.linalg.norm(e)
            rq_rand.append(float(e @ Winv @ e))
        return float(np.median(rq_rand) / max(np.median(rq_bond), 1e-18))

    out["A_var_L2"] = allocation_anisotropy(np.eye(3 * n))
    out["A_var_cellB"] = allocation_anisotropy(np.eye(3 * n) + 0.448 * (B_ang.T @ B_ang))
    out["A_var_mahalanobis"] = allocation_anisotropy(K)

    # T3a: low-rank structure of Sigma*
    ev = np.sort(np.linalg.eigvalsh(Sigma))[::-1][:dof]
    ev = np.clip(ev, 0.0, None); tot = ev.sum()
    out["sigma_top1_trace_frac"] = float(ev[0] / tot)
    out["sigma_top5_trace_frac"] = float(ev[:5].sum() / tot)
    out["sigma_rank90_frac"] = float(
        (np.searchsorted(np.cumsum(ev) / tot, 0.90) + 1) / dof)

    # T3b: correlation of internal coordinates under Sigma*, by graph separation
    Ball = np.concatenate([B_bond, B_ang, B_tor])
    Cq = Ball @ Sigma @ Ball.T
    d = np.sqrt(np.clip(np.diag(Cq), 1e-18, None))
    corr = Cq / np.outer(d, d)
    sets = [set(b) for b in bonds] + [set(a) for a in angles] + [set(q) for q in quads]
    near, far = [], []
    for p in range(len(sets)):
        for q in range(p + 1, len(sets)):
            sh = len(sets[p] & sets[q])
            (near if sh >= 2 else (far if sh == 0 else [])).append(abs(corr[p, q]))
    out["corr_shared2plus_median"] = float(np.median(near)) if near else float("nan")
    out["corr_disjoint_median"] = float(np.median(far)) if far else float("nan")
    out["corr_disjoint_p95"] = float(np.percentile(far, 95)) if far else float("nan")

    # sample Sigma* and measure the same channels the paper measures
    w, v = np.linalg.eigh(Sigma)
    w = np.clip(w, 0.0, None)
    L = v * np.sqrt(w)
    star_block = ref[None] + (L @ rng.standard_normal((3 * n, n_draws))).T.reshape(
        n_draws, n, 3)
    out["star"] = measure(star_block, bonds, angles, quads, centres)

    # ---------------- the actual Gauss-Seidel sampler -----------------------
    sweep = []
    for sigma in (0.05, 0.10, 0.20, 0.35, 0.60):
        eps = gauss_seidel_noise(ref, bonds, angles, sigma, n_draws, rng)
        row = measure(ref[None] + eps, bonds, angles, quads, centres)
        row["sigma"] = sigma
        sweep.append(row)
    out["gs_sweep"] = sweep

    # eta: Cartesian dispersion bought per unit of torsional width
    out["eta_star"] = out["star"]["rmsf"] / max(out["star"]["torsion"], 1e-9)
    out["eta_gs"] = float(np.median([s["rmsf"] / max(s["torsion"], 1e-9) for s in sweep]))
    out["eta_phys"] = out["phys"]["rmsf"] / max(out["phys"]["torsion"], 1e-9)
    out["eta_ratio_star_over_gs"] = out["eta_star"] / max(out["eta_gs"], 1e-12)
    out["eta_ratio_phys_over_gs"] = out["eta_phys"] / max(out["eta_gs"], 1e-12)

    # T3d: rank-r additive correction towards Sigma* on top of the GS sampler
    sigma0 = 0.20
    eps0 = gauss_seidel_noise(ref, bonds, angles, sigma0, 400, rng)
    C_gs = np.cov(eps0.reshape(400, -1).T)
    resid = Sigma - C_gs
    wr, vr = np.linalg.eigh(resid)
    order = np.argsort(wr)[::-1]
    base = measure(ref[None] + eps0[:n_draws], bonds, angles, quads, centres)
    rank_rows = []
    for r in (1, 2, 4, 8, 16):
        idx = order[:r]
        Lr = vr[:, idx] * np.sqrt(np.clip(wr[idx], 0.0, None))
        add = (Lr @ rng.standard_normal((r, n_draws))).T.reshape(n_draws, n, 3)
        row = measure(ref[None] + eps0[:n_draws] + add, bonds, angles, quads, centres)
        row["r"] = r
        rank_rows.append(row)
    out["rank_correction"] = {"sigma": sigma0, "base": base, "rows": rank_rows}

    # T4/D2: add the improper to the stiff set of the SAME sampler
    if centres and B_oop is not None:
        tau_oop = np.radians(out["tau_oop_deg"])
        extra = [(B_oop[i], tau_oop) for i in range(len(centres))]
        for sigma in (0.20,):
            eps = gauss_seidel_noise(ref, bonds, angles, sigma, n_draws, rng,
                                     extra=extra)
            out["with_improper"] = measure(ref[None] + eps, bonds, angles, quads,
                                           centres)
            out["with_improper"]["sigma"] = sigma

    # ---------------- T5: graph-free soft gate ------------------------------
    true_pairs = {(int(a), int(b)) for a, b in bonds}
    r0 = np.array([COVALENT_RADIUS.get(int(e), 1.0) for e in z])
    d0 = r0[:, None] + r0[None, :] + 0.30
    gate_rows = []
    for sigma in (0.0, 0.05, 0.15, 0.30, 0.50, 1.00):
        p = ref + sigma * rng.standard_normal(ref.shape)
        dm = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        gate = 0.5 * (1.0 - np.tanh((dm - d0) / 0.15))
        np.fill_diagonal(gate, 0.0)
        pred = {(int(min(i, j)), int(max(i, j)))
                for i, j in zip(*np.where(gate > 0.5)) if i < j}
        tp = len(pred & true_pairs)
        prec = tp / max(len(pred), 1); rec = tp / max(len(true_pairs), 1)
        gate_rows.append({"sigma": sigma, "precision": prec, "recall": rec,
                          "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                          "soft_mass_over_true": float(gate.sum() / (2 * len(true_pairs)))})
    out["soft_gate"] = gate_rows
    return out


def ideal_ring_theorem():
    """P-T1d in its exact form: at perfect planarity the out-of-plane B-vector
    is orthogonal to every bond and every angle B-vector."""
    R = 1.39
    ang = 2 * np.pi * np.arange(6) / 6
    ring = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(6)], 1)
    x = np.vstack([ring, np.array([[R + 1.50, 0.0, 0.0]])])
    n = 7
    bonds = [(i, (i + 1) % 6) for i in range(6)] + [(0, 6)]
    nb = {i: [] for i in range(n)}
    for a, b in bonds:
        nb[a].append(b); nb[b].append(a)
    angles = []
    for j in range(n):
        near = sorted(nb[j])
        for p in range(len(near)):
            for q in range(p + 1, len(near)):
                angles.append((near[p], j, near[q]))
    Bb = np.stack([b_bond(x, i, j, n) for i, j in bonds])
    Ba = np.stack([b_angle(x, i, j, k, n) for i, j, k in angles])
    bo = b_out_of_plane(x, 0, 6, 1, 5, n)
    BA = np.concatenate([Bb, Ba])
    u_, s_, _ = np.linalg.svd(BA.T, full_matrices=False)
    Q = u_[:, s_ > s_.max() * 1e-8]
    e = bo / np.linalg.norm(bo)
    # finite-difference: how far does a 0.1 A and 0.2 A umbrella displacement
    # move the bonds and angles?
    move = {}
    for delta in (0.05, 0.1, 0.2):
        p = x + delta * e.reshape(n, 3)
        move[delta] = (
            float(np.abs(bond_lengths(p[None], bonds)[0]
                         - bond_lengths(x[None], bonds)[0]).max()),
            float(np.abs(bond_angles_degrees(p[None], angles)[0]
                         - bond_angles_degrees(x[None], angles)[0]).max()),
        )
    return {
        "max_abs_dot_with_bonds": float(np.abs(Bb @ bo).max()),
        "max_abs_dot_with_angles": float(np.abs(Ba @ bo).max()),
        "rowspace_projection": float((Q.T @ e) @ (Q.T @ e)),
        "Bnorm2_oop": float(bo @ bo),
        "Bnorm2_bond": float((Bb[0] ** 2).sum()),
        "Bnorm2_angle": float(np.median((Ba ** 2).sum(1))),
        "umbrella_displacement_effect": {
            str(k): {"max_bond_change_A": v[0], "max_angle_change_deg": v[1]}
            for k, v in move.items()},
        "second_order_check_bond_0.2_over_0.1": move[0.2][0] / max(move[0.1][0], 1e-15),
        "second_order_check_angle_0.2_over_0.1": move[0.2][1] / max(move[0.1][1], 1e-15),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path,
                    default=ROOT / "data/MISATO/extracted/frames100_all_v1")
    ap.add_argument("--n-systems", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs/theory_modules_v1/results.json")
    args = ap.parse_args()

    blocked = set()
    with (ROOT / "outputs/mmp_transfer_corpus_v1/pairs.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["split_a"] == "test":
                blocked.add(row["pdb_a"].upper())
            if row["split_b"] == "test":
                blocked.add(row["pdb_b"].upper())

    paths = [p for p in sorted(args.store.glob("*.npz"))
             if p.stem.split("_")[0].upper() not in blocked]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(paths)

    rows = []
    for path in paths:
        if len(rows) >= args.n_systems:
            break
        try:
            row = analyse_system(path, rng)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path.stem}: {exc}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"  {len(rows)} systems", file=sys.stderr, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "n_systems": len(rows), "n_blocked_test_pdbs": len(blocked),
        "seed": args.seed, "sweeps": SWEEPS, "relaxation": RELAX,
        "ideal_ring_theorem": ideal_ring_theorem(), "rows": rows}, indent=1))
    print(f"wrote {args.output} with {len(rows)} systems")


if __name__ == "__main__":
    main()
