"""Section 2 numbers: an angle LOSS term versus the inference-time angle PROJECTION.

Reads the 2x2 factorial of `outputs/angle_term_v1/spectrum/` (cells A-D, folds
1-3; A = neither, B = loss only, C = projection only, D = both) and the two
retraining controls (Actrl_f2, Cctrl_f2).

Figure of merit: the dispersion-normalised angle ratio,
    (bond-angle ratio) / (internal-RMSF ratio),
which is invariant to a global amplitude change, so it measures selectivity.

Main effect of the loss   = mean over folds of {B - A, D - C}   (n = 6)
Main effect of projection = mean over folds of {C - A, D - B}   (n = 6)
95% CI: Student t with n - 1 degrees of freedom.
Retraining-noise floor    = control retrain minus its original, same fold.

Published values (outputs/paper_evidence_v1/FINDING.md, section d):
    loss       -0.125 [-0.313, +0.062]  = -1.7% [-4.2%, +0.8%] of base 7.422
    projection -4.213 [-4.505, -3.922]  = -56.8%
    floor      +0.254 (+3.4%), +0.108 (+1.5%)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SPECTRUM = ROOT / "outputs/angle_term_v1/spectrum"


def cell(name: str) -> tuple[float, float]:
    data = json.loads((SPECTRUM / f"{name}.json").read_text())
    return data["bond_angle"]["ratio_median"], data["internal_rmsf"]["ratio_median"]


def normalised(name: str) -> float:
    angle, rmsf = cell(name)
    return angle / rmsf


def t_interval(values: list[float]) -> tuple[float, float, float]:
    x = np.asarray(values)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def main() -> None:
    raw, loss, projection = [], [], []
    for fold in (1, 2, 3):
        a, b, c, d = (f"{k}_f{fold}" for k in "ABCD")
        raw += [cell(b)[0] - cell(a)[0], cell(d)[0] - cell(c)[0]]
        loss += [normalised(b) - normalised(a), normalised(d) - normalised(c)]
        projection += [normalised(c) - normalised(a), normalised(d) - normalised(b)]
    base_raw = float(np.mean([cell(f"A_f{f}")[0] for f in (1, 2, 3)]))
    base = float(np.mean([normalised(f"A_f{f}") for f in (1, 2, 3)]))

    report = {"base_raw_angle_ratio": base_raw, "base_normalised": base}
    for label, values in (("loss_raw", raw), ("loss_normalised", loss),
                          ("projection_normalised", projection)):
        mean, low, high = t_interval(values)
        report[label] = {"per_fold": values, "mean": mean, "ci95": [low, high]}
    for label in ("loss_normalised", "projection_normalised"):
        entry = report[label]
        entry["relative_to_base"] = entry["mean"] / base
        entry["ci95_relative"] = [v / base for v in entry["ci95"]]
    report["retraining_floor"] = {
        f"{ctrl}-{orig}": {"normalised": normalised(ctrl) - normalised(orig),
                           "relative_to_base": (normalised(ctrl) - normalised(orig)) / base}
        for ctrl, orig in (("Actrl_f2", "A_f2"), ("Cctrl_f2", "C_f2"))
    }
    out = ROOT / "outputs/angle_term_v1/effects.json"
    out.write_text(json.dumps(report, indent=2))
    ln, pr = report["loss_normalised"], report["projection_normalised"]
    print(f"loss       {ln['mean']:+.3f} [{ln['ci95'][0]:+.3f}, {ln['ci95'][1]:+.3f}]"
          f"  = {ln['relative_to_base']:+.1%} [{ln['ci95_relative'][0]:+.1%}, {ln['ci95_relative'][1]:+.1%}]")
    print(f"projection {pr['mean']:+.3f} [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}]  = {pr['relative_to_base']:+.1%}")
    for k, v in report["retraining_floor"].items():
        print(f"floor {k}: {v['normalised']:+.3f} ({v['relative_to_base']:+.1%})")


if __name__ == "__main__":
    main()
