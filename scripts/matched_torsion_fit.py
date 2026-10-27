"""Matched-torsion fitting-fold scale for each arm and fold, per the
`outputs/sigma_star_test_v1/PREREGISTRATION.md` rule.

THE RULE IS FIXED HERE, BEFORE ANY GRID POINT WAS READ BY THIS SCRIPT, and
resolves the two ambiguities the pre-registration leaves open (what to do when
the torsion curve crosses 1.0 more than once, and what to do when it never
crosses).  Nothing below is allowed to change once a number has been printed.

INPUT.  For each arm (ISO, ANISO v3, SIGMA*) and each fitting fold (1, 2, 3),
the torsion ratio -- `dihedral.ratio_median` in the spectrum JSON, generated
circular std over reference circular std -- at every scale on the grid
{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}.  All arms share the identical inference
configuration: --angle-projection on, --rigid 0.32, --n-examples 96,
--impropers, seed 20260918, 16 steps.

RULE.
  1. Build the piecewise-linear interpolant of the torsion ratio against scale
     through the seven grid points.
  2. If that interpolant crosses 1.0, the matched-torsion scale s* is the
     crossing.  If it crosses more than once, take the crossing at the SMALLEST
     scale; the tie-break is arbitrary but it is fixed here rather than chosen
     later, and every crossing found is recorded so the choice is visible.
  3. If it never crosses 1.0 anywhere on the grid, the arm/fold is
     UNMATCHABLE.  s* is not extrapolated.  The grid endpoint whose ratio is
     closest to 1.0 is reported as `nearest_grid_scale` for information only and
     is explicitly NOT a matched-torsion scale.
  4. The channel spread at s* is max minus min over the four channel ratios
     {bond_length, bond_angle, dihedral, planar_improper}, each linearly
     interpolated in scale at s* from the same two bracketing grid points.
     Internal RMSF is interpolated the same way and reported, never used.

`test` is not touched.  Only `fit_examples.csv` rows are involved, which is
what the spectrum script reads by default.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
CHANNELS = ["bond_length", "bond_angle", "dihedral", "planar_improper"]
ARMS = {
    "ISO": ("internal_calibration_v1", "proj"),
    "ANISO_v3": ("aniso_prior_v3", "proj"),
    "SIGMA_star": ("sigma_star_v1", "spectrum"),
}


def load(arm: str, fold: int, scale: float) -> dict:
    directory, prefix = ARMS[arm]
    path = ROOT / "outputs" / directory / f"{prefix}_f{fold}_s{scale:.1f}.json"
    return json.loads(path.read_text()), path


def interpolate(x0, x1, y0, y1, x):
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def main() -> None:
    report: dict = {
        "rule": (
            "piecewise-linear interpolation of the torsion (dihedral) ratio on "
            "the grid {1.0,1.5,2.0,2.5,3.0,3.5,4.0}; s* is the crossing of 1.0, "
            "smallest scale if several; no crossing => UNMATCHABLE, not "
            "extrapolated; spread = max-min over the four channel ratios linearly "
            "interpolated at s*"
        ),
        "split": "fitting folds only (fit_examples.csv); test not read",
        "config": {
            "angle_projection": "on", "rigid": 0.32, "n_examples": 96,
            "impropers": True, "seed": 20260918, "steps": 16,
        },
        "arms": {},
    }
    for arm in ARMS:
        report["arms"][arm] = {}
        for fold in (1, 2, 3):
            curve, sources, config_check = [], [], []
            for scale in GRID:
                summary, path = load(arm, fold, scale)
                row = {"scale": scale}
                for channel in CHANNELS + ["internal_rmsf"]:
                    row[channel] = summary[channel]["ratio_median"]
                row["n_examples"] = summary["n_examples_sampled"]
                curve.append(row)
                sources.append(str(path.relative_to(ROOT)))
                config_check.append({
                    "scale": scale,
                    "rigid": summary["rigid_scale"],
                    "internal": summary["internal_scale"],
                    "angle_projection": summary["angle_projection"],
                    "restore": summary["restore_projection_amplitude"],
                    "angle_target_width_radian": summary["angle_target_width_radian"],
                    "projection_iterations": summary["projection_iterations"],
                    "projection_angle_relaxation": summary["projection_angle_relaxation"],
                    "checkpoint": summary["checkpoint"],
                })
            torsion = [r["dihedral"] for r in curve]
            crossings = []
            for i in range(len(GRID) - 1):
                a, b = torsion[i], torsion[i + 1]
                if (a - 1.0) == 0.0:
                    crossings.append(GRID[i])
                elif (a - 1.0) * (b - 1.0) < 0.0:
                    crossings.append(interpolate(a, b, GRID[i], GRID[i + 1], 1.0))
            if abs(torsion[-1] - 1.0) == 0.0:
                crossings.append(GRID[-1])
            entry = {
                "grid": curve,
                "sources": sources,
                "config_check": config_check,
                "torsion_crossings_of_1.0": crossings,
            }
            if crossings:
                star = min(crossings)
                # bracketing grid interval for the chosen crossing
                index = max(i for i in range(len(GRID) - 1) if GRID[i] <= star)
                lo, hi = GRID[index], GRID[index + 1]
                values = {
                    channel: interpolate(
                        lo, hi, curve[index][channel], curve[index + 1][channel], star
                    )
                    for channel in CHANNELS + ["internal_rmsf"]
                }
                entry.update({
                    "matched": True,
                    "matched_torsion_scale": star,
                    "bracket": [lo, hi],
                    "channel_ratios_at_s_star": values,
                    "channel_spread_at_s_star": (
                        max(values[c] for c in CHANNELS)
                        - min(values[c] for c in CHANNELS)
                    ),
                })
            else:
                nearest = min(range(len(GRID)), key=lambda i: abs(torsion[i] - 1.0))
                entry.update({
                    "matched": False,
                    "matched_torsion_scale": None,
                    "unmatchable_reason": (
                        "torsion ratio does not cross 1.0 anywhere on the grid; "
                        "not extrapolated, per the pre-registration"
                    ),
                    "nearest_grid_scale": GRID[nearest],
                    "nearest_grid_torsion_ratio": torsion[nearest],
                    "channel_spread_at_nearest_grid_scale": (
                        max(curve[nearest][c] for c in CHANNELS)
                        - min(curve[nearest][c] for c in CHANNELS)
                    ),
                })
            report["arms"][arm][f"fold{fold}"] = entry

    out = ROOT / "outputs" / "sigma_star_test_v1" / "matched_torsion_fit.json"
    out.write_text(json.dumps(report, indent=2))

    # Human-readable table.
    for arm in ARMS:
        print(f"\n=== {arm}")
        print("fold scale " + " ".join(f"{c:>16s}" for c in CHANNELS) + f"{'int_rmsf':>12s}")
        for fold in (1, 2, 3):
            entry = report["arms"][arm][f"fold{fold}"]
            for row in entry["grid"]:
                print(f"  {fold}  {row['scale']:.1f} "
                      + " ".join(f"{row[c]:16.4f}" for c in CHANNELS)
                      + f"{row['internal_rmsf']:12.4f}")
            if entry["matched"]:
                values = entry["channel_ratios_at_s_star"]
                print(f"  -> fold {fold} s*={entry['matched_torsion_scale']:.4f} "
                      f"spread={entry['channel_spread_at_s_star']:.4f} "
                      f"rmsf={values['internal_rmsf']:.4f} "
                      f"crossings={[round(c, 4) for c in entry['torsion_crossings_of_1.0']]}")
            else:
                print(f"  -> fold {fold} UNMATCHABLE "
                      f"(nearest grid {entry['nearest_grid_scale']}, "
                      f"torsion {entry['nearest_grid_torsion_ratio']:.4f})")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
