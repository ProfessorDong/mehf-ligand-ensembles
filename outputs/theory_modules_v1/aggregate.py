"""Aggregate `results.json` into the numbers RESULT.md reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAU_BOND_A = 0.0326
TAU_ANGLE_RAD = np.radians(2.774)
TAU_TORSION_RAD = np.radians(14.6)


def med(rows, key):
    v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return (float(np.median(v)), len(v)) if v else (float("nan"), 0)


def q(rows, key, lo=25, hi=75):
    v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v else (np.nan, np.nan)


def interp_at_torsion(sweep, target):
    """RMSF / bond / angle / oop of the Gauss-Seidel sampler at the sigma whose
    torsional width equals `target`, by log-linear interpolation in sigma."""
    t = np.array([s["torsion"] for s in sweep])
    sg = np.array([s["sigma"] for s in sweep])
    if target <= t.min() or target >= t.max():
        return None
    i = int(np.searchsorted(t, target))
    f = (target - t[i - 1]) / (t[i] - t[i - 1])
    out = {"sigma": float(sg[i - 1] + f * (sg[i] - sg[i - 1]))}
    for k in ("rmsf", "bond", "angle", "oop"):
        if k in sweep[0]:
            out[k] = float(sweep[i - 1][k] + f * (sweep[i][k] - sweep[i - 1][k]))
    return out


def main():
    data = json.loads((ROOT / "outputs/theory_modules_v1/results.json").read_text())
    rows = data["rows"]
    rep: dict = {"n_systems": len(rows),
                 "n_blocked_test_pdbs": data["n_blocked_test_pdbs"],
                 "ideal_ring_theorem": data["ideal_ring_theorem"]}

    # ---- T1
    t1 = {}
    for k in ("Bnorm2_bond", "Bnorm2_angle", "Bnorm2_torsion", "Bnorm2_oop",
              "oop_bvector_fd_relerr", "oop_rowspace_projection",
              "random_rowspace_projection", "rowspace_dim_BA_over_dof",
              "oop_vs_bond_cos_max", "oop_vs_angle_cos_max", "oop_dof_share",
              "oop_secondorder_bond_ratio", "oop_secondorder_angle_ratio"):
        m, n = med(rows, k)
        t1[k] = {"median": m, "iqr": q(rows, k), "n": n}
    rep["T1"] = t1

    # ---- T2
    t2 = {}
    for k in ("A_var_L2", "A_var_cellB", "A_var_mahalanobis"):
        m, n = med(rows, k)
        t2[k] = {"median": m, "iqr": q(rows, k), "n": n}
    t2["mahalanobis_over_cellB"] = (t2["A_var_mahalanobis"]["median"]
                                    / t2["A_var_cellB"]["median"])
    rep["T2"] = t2

    # ---- T3
    t3 = {}
    for k in ("sigma_top1_trace_frac", "sigma_top5_trace_frac", "sigma_rank90_frac",
              "corr_shared2plus_median", "corr_disjoint_median", "corr_disjoint_p95",
              "eta_star", "eta_gs", "eta_phys",
              "eta_ratio_star_over_gs", "eta_ratio_phys_over_gs"):
        m, n = med(rows, k)
        t3[k] = {"median": m, "iqr": q(rows, k), "n": n}

    # matched-torsion comparison: rescale so every arm has the physical torsion
    gs_r, star_r = [], []
    for r in rows:
        tphys = r["phys"]["torsion"]
        hit = interp_at_torsion(r["gs_sweep"], tphys)
        if hit:
            gs_r.append(hit["rmsf"] / r["phys"]["rmsf"])
        # Sigma* is a covariance: scaling it by c^2 scales every width by c in
        # the small-displacement regime, so match on torsion by a scalar.
        c = tphys / max(r["star"]["torsion"], 1e-9)
        star_r.append(c * r["star"]["rmsf"] / r["phys"]["rmsf"])
    t3["rmsf_over_physical_at_matched_torsion"] = {
        "gauss_seidel": {"median": float(np.median(gs_r)),
                         "iqr": [float(np.percentile(gs_r, 25)),
                                 float(np.percentile(gs_r, 75))], "n": len(gs_r)},
        "sigma_star": {"median": float(np.median(star_r)),
                       "iqr": [float(np.percentile(star_r, 25)),
                               float(np.percentile(star_r, 75))], "n": len(star_r)},
    }

    # rank-r correction, as a fraction of the base and relative to the stiff cost
    ranks = {}
    for r_val in (1, 2, 4, 8, 16):
        rm, bd, an, to = [], [], [], []
        for r in rows:
            base = r["rank_correction"]["base"]
            hit = [x for x in r["rank_correction"]["rows"] if x["r"] == r_val][0]
            rm.append(hit["rmsf"] / base["rmsf"])
            bd.append(hit["bond"] / base["bond"])
            an.append(hit["angle"] / base["angle"])
            to.append(hit["torsion"] / base["torsion"])
        ranks[r_val] = {"rmsf_gain": float(np.median(rm)),
                        "bond_cost": float(np.median(bd)),
                        "angle_cost": float(np.median(an)),
                        "torsion_cost": float(np.median(to))}
    t3["rank_correction"] = ranks
    rep["T3"] = t3

    # ---- T4 / D2
    t4 = {}
    m, n = med(rows, "tau_oop_deg")
    t4["tau_oop_deg"] = {"median": m, "iqr": q(rows, "tau_oop_deg"), "n": n}
    t4["oop_dof_share"] = t1["oop_dof_share"]
    # out-of-plane over-dispersion of the Gauss-Seidel sampler, and the effect
    # of adding the improper to the stiff set, both at sigma = 0.20
    before, after, ang_b, ang_a, rms_b, rms_a, tor_b, tor_a = ([] for _ in range(8))
    for r in rows:
        if "with_improper" not in r or "oop" not in r["phys"]:
            continue
        base = [s for s in r["gs_sweep"] if s["sigma"] == 0.20][0]
        p = r["phys"]
        before.append(base["oop"] / p["oop"])
        after.append(r["with_improper"]["oop"] / p["oop"])
        ang_b.append(base["angle"] / p["angle"])
        ang_a.append(r["with_improper"]["angle"] / p["angle"])
        rms_b.append(base["rmsf"]); rms_a.append(r["with_improper"]["rmsf"])
        tor_b.append(base["torsion"] / p["torsion"])
        tor_a.append(r["with_improper"]["torsion"] / p["torsion"])
    t4["improper_in_stiff_set"] = {
        "n": len(before),
        "oop_ratio_before": float(np.median(before)),
        "oop_ratio_after": float(np.median(after)),
        "angle_ratio_before": float(np.median(ang_b)),
        "angle_ratio_after": float(np.median(ang_a)),
        "torsion_ratio_before": float(np.median(tor_b)),
        "torsion_ratio_after": float(np.median(tor_a)),
        "rmsf_before": float(np.median(rms_b)),
        "rmsf_after": float(np.median(rms_a)),
        "rmsf_relative": float(np.median(np.array(rms_a) / np.array(rms_b))),
    }
    rep["T4"] = t4

    # ---- T5
    gates = {}
    for s in (0.0, 0.05, 0.15, 0.30, 0.50, 1.00):
        f1 = [g["f1"] for r in rows for g in r["soft_gate"] if g["sigma"] == s]
        mass = [g["soft_mass_over_true"] for r in rows for g in r["soft_gate"]
                if g["sigma"] == s]
        gates[s] = {"f1_median": float(np.median(f1)),
                    "f1_q25": float(np.percentile(f1, 25)),
                    "soft_mass_over_true_median": float(np.median(mass))}
    rep["T5"] = gates

    # ---- T6: does one sigma explain the three measured isotropic costs?
    measured = {"bond": 11.55, "angle": 7.88, "torsion": 2.47}
    taus = {"bond": TAU_BOND_A, "angle": TAU_ANGLE_RAD, "torsion": TAU_TORSION_RAD}
    bn = {"bond": np.sqrt(t1["Bnorm2_bond"]["median"]),
          "angle": np.sqrt(t1["Bnorm2_angle"]["median"]),
          "torsion": np.sqrt(t1["Bnorm2_torsion"]["median"])}
    implied = {k: measured[k] * taus[k] / bn[k] for k in measured}
    rep["T6"] = {
        "Bnorm_used": {k: float(v) for k, v in bn.items()},
        "implied_sigma_angstrom": {k: float(v) for k, v in implied.items()},
        "angle_vs_bond_disagreement": float(abs(implied["angle"] / implied["bond"] - 1)),
        "torsion_vs_bond_disagreement": float(abs(implied["torsion"] / implied["bond"] - 1)),
    }

    # ---- weights, recomputed from the corpus widths
    rep["mahalanobis_weights_recomputed"] = {
        "bond": 1.0,
        "angle_per_radian2": float((TAU_BOND_A / TAU_ANGLE_RAD) ** 2),
        "torsion_per_radian2": float((TAU_BOND_A / TAU_TORSION_RAD) ** 2),
        "oop_per_radian2": float(
            (TAU_BOND_A / np.radians(t4["tau_oop_deg"]["median"])) ** 2),
    }

    out = ROOT / "outputs/theory_modules_v1/summary.json"
    out.write_text(json.dumps(rep, indent=1))
    json.dump(rep, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
