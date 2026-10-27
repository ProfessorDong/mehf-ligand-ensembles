"""(c) Is a trained model's residual isotropic?  The seam between the mechanism
and the DiffSBDD refutation, made quantitative.

The mechanism says: an isotropic Cartesian objective exerts no differential
pressure across internal coordinates, so the residual it leaves has no reason to
depart from isotropy, and an isotropic residual damages coordinate k in
proportion to ||B_k||/tau_k.  The DiffSBDD measurement says a trained model's
residual is NOT isotropic.  Both can be true, and this script measures which way
each model sits.

TEST.  If the generated ensemble is the reference ensemble plus an independent
isotropic residual of per-atom SD sigma, then for every channel

    generated_width_k^2 = reference_width_k^2 + (sigma * ||B_k||)^2.

Solve sigma from the BOND channel of each example, predict the ANGLE channel,
and report measured/predicted.  1.0 means the residual is isotropic.

||B_k|| is NOT taken from theory here.  It is taken from
`a2_blaw_amplitude.json`, i.e. from pure isotropic noise measured through the
IDENTICAL estimator (mean over coordinates of the per-coordinate SD, heavy
atoms), at the same amplitude.  That removes the estimator and the
mean-vs-median-over-coordinates difference from the comparison.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AMP = json.loads((HERE / "a2_blaw_amplitude.json").read_text())
SIGMA_GRID = np.array(sorted(float(s) for s in AMP["by_sigma"]))
BEMP = {
    "bond": np.array([AMP["by_sigma"][str(s)]["bond_length"]["implied_Bnorm"]
                      for s in SIGMA_GRID]),
    "angle": np.array([AMP["by_sigma"][str(s)]["bond_angle"]["implied_Bnorm"]
                       for s in SIGMA_GRID]),
    "dih": np.array([AMP["by_sigma"][str(s)]["dihedral"]["implied_Bnorm"]
                     for s in SIGMA_GRID]),
}


def bnorm(channel: str, sigma: float) -> float:
    return float(np.interp(sigma, SIGMA_GRID, BEMP[channel]))


def solve_sigma_from_bond(ref_a: float, gen_a: float) -> float | None:
    """gen^2 = ref^2 + (sigma ||B_bond||(sigma))^2, solved by fixed point."""
    excess2 = gen_a ** 2 - ref_a ** 2
    if excess2 <= 0:
        return None
    excess = float(np.sqrt(excess2))
    sigma = excess / np.sqrt(2.0)
    for _ in range(50):
        sigma = excess / bnorm("bond", sigma)
    return sigma


def predict(channel: str, ref: float, sigma: float) -> float:
    return float(np.sqrt(ref ** 2 + (sigma * bnorm(channel, sigma)) ** 2))


def analyse(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open()))
    out = {"n_rows": len(rows)}
    ratios_angle, ratios_dih, sigmas = [], [], []
    for row in rows:
        try:
            ref_b = float(row["bond_length_reference"])
            gen_b = float(row["bond_length_generated"])
            ref_a = np.radians(float(row["bond_angle_reference"]))
            gen_a = np.radians(float(row["bond_angle_generated"]))
            ref_d = np.radians(float(row["dihedral_reference"]))
            gen_d = np.radians(float(row["dihedral_generated"]))
        except (KeyError, ValueError):
            continue
        sigma = solve_sigma_from_bond(ref_b, gen_b)
        if sigma is None or not np.isfinite(sigma):
            continue
        sigmas.append(sigma)
        ratios_angle.append(gen_a / predict("angle", ref_a, sigma))
        ratios_dih.append(gen_d / predict("dih", ref_d, sigma))
    out["n_used"] = len(sigmas)
    out["sigma_implied_median_A"] = float(np.median(sigmas))
    out["sigma_iqr"] = [float(np.percentile(sigmas, 25)),
                        float(np.percentile(sigmas, 75))]
    for name, values in (("angle", ratios_angle), ("dihedral", ratios_dih)):
        v = np.array(values)
        out[f"{name}_measured_over_isotropic_prediction"] = {
            "median": float(np.median(v)),
            "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
            "frac_above_1": float((v > 1).mean()),
        }
    return out


def main() -> None:
    result = {}
    directory = HERE / "mehf_isotropy"
    for path in sorted(directory.glob("*.per_example.csv")):
        result[path.name.replace(".per_example.csv", "")] = analyse(path)
    (HERE / "c_isotropy_seam.json").write_text(json.dumps(result, indent=2))
    print(f"{'arm':12s} {'n':>4s} {'sigma(A)':>9s} {'angle m/p':>10s} {'IQR':>22s} "
          f"{'dih m/p':>9s}")
    for name, v in result.items():
        a = v["angle_measured_over_isotropic_prediction"]
        d = v["dihedral_measured_over_isotropic_prediction"]
        print(f"{name:12s} {v['n_used']:4d} {v['sigma_implied_median_A']:9.4f} "
              f"{a['median']:10.3f} [{a['iqr'][0]:.3f}, {a['iqr'][1]:.3f}]".ljust(60)
              + f" {d['median']:9.3f}")


if __name__ == "__main__":
    main()
