"""Wilson B-matrix derivation of the isotropic-noise geometry spectrum: numerical test.

DERIVATION UNDER TEST
---------------------
For internal coordinate q_k, the Wilson B-vector is B_k = dq_k/dx in R^{3N}.
Under isotropic Cartesian noise dx ~ N(0, s^2 I_{3N}), to first order
    Var(q_k) = s^2 ||B_k||^2       (EXACT in the linear regime)
so the synthetic arm's spread is predicted in closed form from GEOMETRY ALONE:
    spread_synth(k) = s * ||B_k||.

Closed forms (to be verified against finite differences):
    bond a-b:            ||B||^2 = 2                      (dimensionless, per Angstrom)
    angle i-j-k:         ||B||^2 = 2/r1^2 + 2/r2^2 - 2 cos(theta)/(r1 r2)   [rad^2/A^2]
    dihedral:            no compact form used; taken by finite difference.

PREDICTIONS, WRITTEN BEFORE RUNNING
-----------------------------------
P0 (analytic vs finite-difference): the two closed forms above reproduce the
   finite-difference ||B|| for bonds and angles to <1e-4 relative.
P1 (bond, sp3-independent): every bond has ||B|| = sqrt(2) = 1.41421 exactly,
   with zero variance across all molecules.
P2 (angle magnitude): median ||B_angle|| is near 1.42 rad/A = 81.4 deg/A
   (value for r1=r2=1.52 A, theta=109.5 deg).
P3 (linear theory reproduces the SYNTHETIC arm): predicted/measured synthetic
   spread within 20% for bond length, within 25% for bond angle.
P4 (dihedral is the nonlinear one): the linear prediction will NOT match the
   measured dihedral circular std, and will UNDER-predict it (ratio
   predicted/measured in 0.5-0.9), because dihedral B-vectors carry 1/sin(bond
   angle) factors that diverge and the finite-amplitude distribution is
   heavy-tailed relative to a wrapped normal.
P5 (ordering): the predicted ratios, computed as s*||B_k|| / measured physical
   spread, reproduce the measured ordering 15.27 > 9.77 > 3.51 and land within
   25% for bond and angle.
P6 (independent force-constant prediction of the PHYSICAL arm, no fitting):
   with kT = 0.5921 kcal/mol at 300 K and GAFF-typical stiffnesses
   f_bond = 2*k_r ~ 700 kcal/mol/A^2 and f_angle = 2*k_theta ~ 126 kcal/mol/rad^2,
   sqrt(kT/f) gives bond 0.029 A (measured 0.0326, expect within 20%) and
   angle 3.93 deg (measured 2.77, expect the prediction to be TOO SOFT by
   ~40% because redundant-coordinate coupling and ring closure stiffen angles).

CONFOUND CHECKS INCLUDED
------------------------
C1. Frame autocorrelation: if the 100 MISATO frames are not decorrelated, the
    physical spread is underestimated and every ratio is inflated. Measured via
    lag-1 autocorrelation of each coordinate and by a half-trajectory split.
C2. Hydrogen: already excluded from internals, but the noise amplitude s is set
    by an RMSF computed over ALL atoms including hydrogens. Quantified by
    recomputing s from a heavy-atom-only RMSF.
C3. Nonlinearity of the bond-length map (|d| is not linear in dx at s ~ 0.39 A):
    isolated by comparing linear prediction, exact Monte Carlo, and measurement.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from isotropic_noise_geometry_cost import (
    bond_angles_degrees, bond_lengths, circular_std_degrees, dihedrals_degrees,
    enumerate_internals, internal_rmsf, noise_arm,
)

KT = 0.5921  # kcal/mol at 300 K


def b_norms_fd(frame: np.ndarray, bonds, angles, quads, h: float = 1e-5):
    """||B_k|| by central finite difference of the exact measurement functions."""
    n = frame.shape[0]
    nb, na, nd = len(bonds), len(angles), len(quads)
    acc = [np.zeros(nb), np.zeros(na), np.zeros(nd)]
    for i in range(n):
        for c in range(3):
            plus = frame.copy(); plus[i, c] += h
            minus = frame.copy(); minus[i, c] -= h
            P = plus[None]; M = minus[None]
            d_b = (bond_lengths(P, bonds)[0] - bond_lengths(M, bonds)[0]) / (2 * h)
            d_a = (bond_angles_degrees(P, angles)[0] - bond_angles_degrees(M, angles)[0]) / (2 * h)
            raw = dihedrals_degrees(P, quads)[0] - dihedrals_degrees(M, quads)[0]
            raw = (raw + 180.0) % 360.0 - 180.0
            d_d = raw / (2 * h)
            acc[0] += d_b ** 2; acc[1] += d_a ** 2; acc[2] += d_d ** 2
    return np.sqrt(acc[0]), np.sqrt(acc[1]), np.sqrt(acc[2])


def b_norm_angle_analytic(frame, angles):
    out = np.zeros(len(angles))
    for m, (i, j, k) in enumerate(angles):
        u = frame[i] - frame[j]; v = frame[k] - frame[j]
        r1 = np.linalg.norm(u); r2 = np.linalg.norm(v)
        c = float(np.dot(u, v) / (r1 * r2))
        out[m] = np.sqrt(2 / r1**2 + 2 / r2**2 - 2 * c / (r1 * r2))
    return out * (180.0 / np.pi)   # deg/A


def analyse(path: Path, rng, do_fd: bool):
    with np.load(path, allow_pickle=True) as st:
        coords = st["ligand_coords"].astype(np.float64)
        bond_index = st["bond_index"]
        elements = st["ligand_atomic_numbers"].astype(int)
    F, N, _ = coords.shape
    if N < 4 or F < 10:
        return None
    bonds, angles, quads = enumerate_internals(bond_index, N)
    heavy = elements != 1
    bonds = [b for b in bonds if heavy[b[0]] and heavy[b[1]]]
    angles = [a for a in angles if all(heavy[i] for i in a)]
    quads = [q for q in quads if all(heavy[i] for i in q)]
    if not bonds or not angles or not quads:
        return None
    rmsf_p = internal_rmsf(coords)
    if not np.isfinite(rmsf_p) or rmsf_p <= 1e-6:
        return None

    # --- replicate the measurement's synthetic arm EXACTLY, and recover s ---
    draw = rng.standard_normal((F, N, 3))
    achieved_unit = internal_rmsf(coords[0][None] + draw)
    s = rmsf_p / achieved_unit                       # the Cartesian noise sigma actually used
    synth = coords[0][None] + draw * s
    # C2 arm: match the noise to the HEAVY-ATOM RMSF instead of the all-atom RMSF.
    rmsf_p_heavy = internal_rmsf(coords[:, heavy, :])
    achieved_unit_heavy = internal_rmsf((coords[0][None] + draw)[:, heavy, :])
    s_heavy = rmsf_p_heavy / achieved_unit_heavy
    synth_heavy = coords[0][None] + draw * s_heavy

    ref = coords[0]
    bn_b_an = np.full(len(bonds), np.sqrt(2.0))
    bn_a_an = b_norm_angle_analytic(ref, angles)
    if do_fd:
        bn_b_fd, bn_a_fd, bn_d_fd = b_norms_fd(ref, bonds, angles, quads)
    else:
        bn_b_fd = bn_a_fd = bn_d_fd = None
    # dihedral B norms always need FD
    if bn_d_fd is None:
        _, _, bn_d_fd = b_norms_fd(ref, bonds, angles, quads)

    row = {"system": path.stem, "n_atoms": N, "n_heavy": int(heavy.sum()),
           "n_bonds": len(bonds), "n_angles": len(angles), "n_dihedrals": len(quads),
           "rmsf_physical": rmsf_p, "achieved_unit_rmsf": achieved_unit, "sigma": s,
           "sqrt3": float(np.sqrt(3.0)),
           "rmsf_heavy_physical": rmsf_p_heavy, "sigma_heavy": s_heavy,
           "b_bond_mean": float(bn_b_an.mean()),
           "b_angle_mean_deg": float(bn_a_an.mean()),
           "b_dihedral_mean_deg": float(bn_d_fd.mean())}
    if do_fd:
        row["fd_bond_relerr"] = float(np.max(np.abs(bn_b_fd - bn_b_an) / bn_b_an))
        row["fd_angle_relerr"] = float(np.max(np.abs(bn_a_fd - bn_a_an) / bn_a_an))

    # --- measured spreads, both arms ---
    for name, ph, sy, bn in (
        ("bond_length", bond_lengths(coords, bonds), bond_lengths(synth, bonds), bn_b_an),
        ("bond_angle", bond_angles_degrees(coords, angles), bond_angles_degrees(synth, angles), bn_a_an),
        ("dihedral", dihedrals_degrees(coords, quads), dihedrals_degrees(synth, quads), bn_d_fd),
    ):
        if name == "dihedral":
            sp_p = float(np.mean(circular_std_degrees(ph)))
            sp_s = float(np.mean(circular_std_degrees(sy)))
            sp_p_half = float(np.mean(circular_std_degrees(ph[: F // 2])))
        else:
            sp_p = float(np.mean(ph.std(axis=0)))
            sp_s = float(np.mean(sy.std(axis=0)))
            sp_p_half = float(np.mean(ph[: F // 2].std(axis=0)))
        pred_s = float(s * bn.mean())
        row[f"{name}_spread_physical"] = sp_p
        row[f"{name}_spread_physical_half"] = sp_p_half
        row[f"{name}_spread_synthetic"] = sp_s
        row[f"{name}_spread_synthetic_predicted"] = pred_s
        row[f"{name}_ratio_measured"] = sp_s / max(sp_p, 1e-9)
        row[f"{name}_ratio_predicted"] = pred_s / max(sp_p, 1e-9)
        row[f"{name}_pred_over_meas_synth"] = pred_s / max(sp_s, 1e-9)
        # C2: heavy-atom-RMSF-matched arm
        if name == "dihedral":
            sp_s_h = float(np.mean(circular_std_degrees(dihedrals_degrees(synth_heavy, quads))))
        elif name == "bond_angle":
            sp_s_h = float(np.mean(bond_angles_degrees(synth_heavy, angles).std(axis=0)))
        else:
            sp_s_h = float(np.mean(bond_lengths(synth_heavy, bonds).std(axis=0)))
        row[f"{name}_ratio_heavymatched"] = sp_s_h / max(sp_p, 1e-9)
        row[f"{name}_ratio_heavymatched_predicted"] = (s_heavy * bn.mean()) / max(sp_p, 1e-9)
        # C1: lag-1 autocorrelation of the physical arm
        x = ph if name != "dihedral" else np.unwrap(np.radians(ph), axis=0)
        xc = x - x.mean(axis=0, keepdims=True)
        num = (xc[:-1] * xc[1:]).sum(axis=0)
        den = (xc ** 2).sum(axis=0)
        row[f"{name}_lag1_autocorr"] = float(np.mean(num / np.clip(den, 1e-12, None)))
        # effective force constant implied by the physical spread (NOT a test, an identity)
        if name == "bond_length":
            row["f_bond_implied"] = KT / max(sp_p, 1e-9) ** 2
        elif name == "bond_angle":
            row["f_angle_implied"] = KT / max(np.radians(sp_p), 1e-9) ** 2
        else:
            row["f_dihedral_implied"] = KT / max(np.radians(sp_p), 1e-9) ** 2
    # lambda_k = f_k * ||B_k||^2 : Cartesian curvature along the coordinate's own direction
    row["lambda_bond"] = row["f_bond_implied"] * 2.0
    row["lambda_angle"] = row["f_angle_implied"] * float(np.mean((bn_a_an * np.pi / 180.0) ** 2))
    row["lambda_dihedral"] = row["f_dihedral_implied"] * float(np.mean((bn_d_fd * np.pi / 180.0) ** 2))
    return row


def main():
    import argparse, csv
    p = argparse.ArgumentParser()
    p.add_argument("--n-systems", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--fd-check", type=int, default=25, help="systems to also finite-difference-check")
    p.add_argument("--out", type=Path, default=Path(str(ROOT/"outputs/wilson_b_theory_v1"/"wilson_b")))
    a = p.parse_args()
    store = ROOT / "data/MISATO/extracted/frames100_all_v1"
    paths = sorted(store.glob("*.npz"))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(paths)
    paths = paths[: a.n_systems]
    rows = []
    for n, path in enumerate(paths):
        try:
            r = analyse(path, rng, do_fd=(n < a.fd_check))
        except Exception as e:
            continue
        if r:
            rows.append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with (a.out.with_suffix(".csv")).open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

    def med(k):
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
        return float(np.median(v)) if v.size else float("nan")
    def iqr(k):
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
        return [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if v.size else [float("nan")]*2

    summary = {"n_systems": len(rows),
               "P0_fd_bond_max_relerr": med("fd_bond_relerr"),
               "P0_fd_angle_max_relerr": med("fd_angle_relerr"),
               "P1_b_bond_mean": med("b_bond_mean"),
               "P2_b_angle_mean_deg_per_A": med("b_angle_mean_deg"),
               "b_dihedral_mean_deg_per_A": med("b_dihedral_mean_deg"),
               "sigma_median": med("sigma"), "sigma_heavy_median": med("sigma_heavy"),
               "achieved_unit_rmsf_median": med("achieved_unit_rmsf"),
               "rmsf_physical_median": med("rmsf_physical"),
               "rmsf_heavy_physical_median": med("rmsf_heavy_physical")}
    for name in ("bond_length", "bond_angle", "dihedral"):
        summary[name] = {
            "physical_spread_median": med(f"{name}_spread_physical"),
            "physical_spread_half_median": med(f"{name}_spread_physical_half"),
            "lag1_autocorr_median": med(f"{name}_lag1_autocorr"),
            "synthetic_spread_measured_median": med(f"{name}_spread_synthetic"),
            "synthetic_spread_predicted_median": med(f"{name}_spread_synthetic_predicted"),
            "P3P4_pred_over_meas_synth_median": med(f"{name}_pred_over_meas_synth"),
            "P3P4_pred_over_meas_synth_iqr": iqr(f"{name}_pred_over_meas_synth"),
            "ratio_measured_median": med(f"{name}_ratio_measured"),
            "ratio_predicted_median": med(f"{name}_ratio_predicted"),
            "ratio_predicted_iqr": iqr(f"{name}_ratio_predicted"),
            "C2_ratio_heavymatched_median": med(f"{name}_ratio_heavymatched"),
            "C2_ratio_heavymatched_iqr": iqr(f"{name}_ratio_heavymatched"),
            "C2_ratio_heavymatched_predicted_median": med(f"{name}_ratio_heavymatched_predicted"),
        }
    summary["implied_force_constants"] = {
        "f_bond_kcal_per_mol_A2": med("f_bond_implied"),
        "f_angle_kcal_per_mol_rad2": med("f_angle_implied"),
        "f_dihedral_kcal_per_mol_rad2": med("f_dihedral_implied"),
        "lambda_bond": med("lambda_bond"),
        "lambda_angle": med("lambda_angle"),
        "lambda_dihedral": med("lambda_dihedral"),
    }
    # condition-number form:  R_k = sqrt(lambda_k * <lambda^-1>_modes),
    # with <lambda^-1>_modes = sigma^2 / kT  (since sigma^2 = kT tr(H^+)/(3N-6))
    lam_inv = med("sigma") ** 2 / KT
    summary["condition_number_form"] = {
        "mean_mode_compliance_A2_per_kcal": lam_inv,
        "R_bond_from_lambda": float(np.sqrt(med("lambda_bond") * lam_inv)),
        "R_angle_from_lambda": float(np.sqrt(med("lambda_angle") * lam_inv)),
        "R_dihedral_from_lambda": float(np.sqrt(med("lambda_dihedral") * lam_inv)),
    }
    (a.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
