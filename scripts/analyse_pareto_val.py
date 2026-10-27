"""Primary analysis for outputs/pareto_v1/PREREGISTRATION.md, written before any
`val` result existed.  It implements that file's "Analysis, fixed now" section
and nothing else.

Per arm, per example and channel: mean over the three fold checkpoints of the
generated/reference spread ratio.  Primary: paired difference in bond-angle
ratio, ANISO minus ISO, median over examples, 95% cluster bootstrap over UniProt
targets (4,000 resamples, seed 20260918).  CONFIRMED iff the median and the
upper CI bound are both below zero AND median RMSF(ANISO) >= median RMSF(ISO)
minus 0.05.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "outputs/pareto_v1/val"
CHANNELS = ("bond_angle", "bond_length", "dihedral", "internal_rmsf", "planar_improper")
RESAMPLES = 4000
SEED = 20260918


def arm_table(arm: str) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    frames = []
    per_fold: dict[str, dict[str, float]] = {}
    for fold in (1, 2, 3):
        path = VAL / f"{arm}_f{fold}.per_example.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}: the val read is incomplete")
        table = pd.read_csv(path).set_index("key")
        frames.append(table[[f"{c}_ratio" for c in CHANNELS]])
        per_fold[f"f{fold}"] = {
            c: float(table[f"{c}_ratio"].median()) for c in CHANNELS
        }
    # Only examples every checkpoint produced, so each value is a 3-way mean.
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    stacked = np.stack([frame.loc[common].to_numpy() for frame in frames])
    mean = pd.DataFrame(
        stacked.mean(axis=0), index=common, columns=[f"{c}_ratio" for c in CHANNELS]
    )
    return mean, per_fold


def cluster_bootstrap_median(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    targets = np.unique(groups)
    members = {t: np.where(groups == t)[0] for t in targets}
    draws = np.empty(RESAMPLES)
    for index in range(RESAMPLES):
        chosen = rng.choice(targets, size=len(targets), replace=True)
        rows = np.concatenate([members[t] for t in chosen])
        draws[index] = np.median(values[rows])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    examples = pd.read_csv(ROOT / "outputs/pareto_v1/val_examples.csv")
    uniprot_of = {
        f"{r.parent}->{r['product']}": str(r.uniprot) for _, r in examples.iterrows()
    }
    aniso, aniso_folds = arm_table("aniso")
    iso, iso_folds = arm_table("iso")
    keys = aniso.index.intersection(iso.index)
    groups = np.array([uniprot_of[k] for k in keys])

    report: dict = {
        "n_examples_paired": int(len(keys)),
        "n_targets": int(len(np.unique(groups))),
        "per_fold_medians": {"ANISO": aniso_folds, "ISO": iso_folds},
        "arm_medians": {
            "ANISO": {c: float(aniso.loc[keys, f"{c}_ratio"].median()) for c in CHANNELS},
            "ISO": {c: float(iso.loc[keys, f"{c}_ratio"].median()) for c in CHANNELS},
        },
        "paired_difference_aniso_minus_iso": {},
    }
    for channel in CHANNELS:
        diff = (
            aniso.loc[keys, f"{channel}_ratio"].to_numpy()
            - iso.loc[keys, f"{channel}_ratio"].to_numpy()
        )
        keep = np.isfinite(diff)
        low, high = cluster_bootstrap_median(diff[keep], groups[keep])
        report["paired_difference_aniso_minus_iso"][channel] = {
            "median": float(np.median(diff[keep])),
            "ci95": [low, high],
            "fraction_aniso_lower": float((diff[keep] < 0).mean()),
            "n": int(keep.sum()),
        }

    primary = report["paired_difference_aniso_minus_iso"]["bond_angle"]
    rmsf_aniso = report["arm_medians"]["ANISO"]["internal_rmsf"]
    rmsf_iso = report["arm_medians"]["ISO"]["internal_rmsf"]
    conditions = {
        "median_D_below_0": primary["median"] < 0.0,
        "upper_CI_below_0": primary["ci95"][1] < 0.0,
        "rmsf_not_traded": rmsf_aniso >= rmsf_iso - 0.05,
    }
    report["decision_conditions"] = conditions
    report["VERDICT"] = "CONFIRMED" if all(conditions.values()) else "NOT CONFIRMED"
    angle_aniso = report["arm_medians"]["ANISO"]["bond_angle"]
    report["secondary_old_bar"] = {
        "aniso_angle_in_0.8_1.35": bool(0.8 <= angle_aniso <= 1.35),
        "aniso_rmsf_at_least_0.60": bool(rmsf_aniso >= 0.60),
    }
    out = ROOT / "outputs/pareto_v1/VAL_RESULT.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
