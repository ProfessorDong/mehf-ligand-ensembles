"""Analysis fixed in outputs/noproj_test_v1/PREREGISTRATION.md, written before any
noproj_test_v1 result existed.

C0 (full projection) = the FROZEN confirmatory files (reference_outputs/sigma_star_test_v1/,
reference_outputs/laplacian_prior_v1/). C1 / C2 = outputs/noproj_test_v1/.
`cluster_bootstrap` and CHANNELS are imported unchanged from
scripts/sigma_star_test_decision.py; `load` repeats the logic of its `load_arm`
(per-example values averaged over the three folds, examples finite in all four
channels on every fold) for a file-name pattern instead of a fixed one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sigma_star_test_decision as D  # noqa: E402

FROZEN = ROOT / "reference_outputs/sigma_star_test_v1"
FROZEN_LAP = ROOT / "reference_outputs/laplacian_prior_v1"
NEW = ROOT / "outputs/noproj_test_v1"
ARMS = ("ISO", "GS", "LAP", "SIG")
C0_NAME = {"ISO": (FROZEN, "test_ISO"), "GS": (FROZEN, "test_ANISO_v3"),
           "SIG": (FROZEN, "test_SIGMA_star"), "LAP": (FROZEN_LAP, "test_LAPLACIAN")}
OOP = "planar_improper_ratio"


def load(directory: Path, stem: str) -> pd.DataFrame:
    per_fold = {}
    for fold in (1, 2, 3):
        frame = pd.read_csv(directory / f"{stem}_f{fold}.per_example.csv").set_index("key")
        frame = frame[frame[list(D.CHANNELS)].notna().all(axis=1)]
        frame = frame[list(D.CHANNELS) + ["internal_rmsf_ratio"]].copy()
        frame["spread"] = frame[list(D.CHANNELS)].max(axis=1) - frame[list(D.CHANNELS)].min(axis=1)
        per_fold[fold] = frame
    shared = per_fold[1].index.intersection(per_fold[2].index).intersection(per_fold[3].index)
    return sum(per_fold[f].loc[shared] for f in (1, 2, 3)) / 3.0


def frames_for(condition: str, point: str) -> dict:
    if condition == "C0":
        return {a: load(*C0_NAME[a]) for a in ARMS}
    return {a: load(NEW, f"{condition}_{point}_{a}") for a in ARMS}


def summarise(frames: dict, uniprot: dict) -> dict:
    common = frames[ARMS[0]].index
    for a in ARMS[1:]:
        common = common.intersection(frames[a].index)
    cluster = np.array([uniprot[k] for k in common])
    out = {"n_examples": int(len(common)), "n_targets": int(len(set(cluster))), "arms": {}, "contrasts": {}}
    for a in ARMS:
        f = frames[a].loc[common]
        out["arms"][a] = {"median_spread": float(f["spread"].median()),
                          "q25_spread": float(f["spread"].quantile(0.25)),
                          "q75_spread": float(f["spread"].quantile(0.75)),
                          "channel_ratio_median": {c: float(f[c].median()) for c in D.CHANNELS},
                          "internal_rmsf_ratio_median": float(f["internal_rmsf_ratio"].median())}
    for metric in (OOP, "spread"):
        for b in ("ISO", "LAP", "GS"):
            diff = (frames["SIG"].loc[common, metric] - frames[b].loc[common, metric]).to_numpy()
            lo, hi = D.cluster_bootstrap(diff, cluster)
            out["contrasts"][f"SIG_minus_{b}:{metric}"] = {
                "mean_difference": float(diff.mean()), "median_difference": float(np.median(diff)),
                "ci95_cluster_bootstrap": [lo, hi], "fraction_SIG_lower": float((diff < 0).mean())}
    return common, out


def main() -> None:
    examples = pd.read_csv(FROZEN / "test_examples.csv")
    uniprot = dict(zip(examples.example_id, examples.uniprot))
    report = {"status": "POST-HOC second read of test, pre-registered in outputs/noproj_test_v1/PREREGISTRATION.md"}
    cells = {}
    for condition, point in (("C0", "Sstar"), ("C1", "Sstar"), ("C2", "Sstar"), ("C1", "S2.5"), ("C2", "S2.5")):
        frames = frames_for(condition, point)
        common, summary = summarise(frames, uniprot)
        cells[(condition, point)] = (frames, common)
        report[f"{condition}_{point}"] = summary

    def ok(key):
        c = report["C2_Sstar"]["contrasts"][key]
        return c["mean_difference"] < 0 and c["ci95_cluster_bootstrap"][1] < 0
    report["P1_SUPPORTED"] = bool(ok(f"SIG_minus_ISO:{OOP}") and ok(f"SIG_minus_LAP:{OOP}"))
    report["P1b_SUPPORTED"] = bool(ok("SIG_minus_ISO:spread") and ok("SIG_minus_LAP:spread"))
    sig_c1_25 = report["C1_S2.5"]["arms"]["SIG"]["channel_ratio_median"][OOP]
    report["P2"] = {"SIG_oop_test_C1_S2.5": sig_c1_25, "fitting_fold_value": 1.03,
                    "exceeds": bool(sig_c1_25 > 1.03)}
    # Projection effect at S*: C0 vs C2 on the examples common to both cells.
    f0, k0 = cells[("C0", "Sstar")]
    f2, k2 = cells[("C2", "Sstar")]
    both = k0.intersection(k2)
    effect = {}
    for a in ARMS:
        m0 = float(f0[a].loc[both, OOP].median())
        m2 = float(f2[a].loc[both, OOP].median())
        effect[a] = {"oop_median_C2": m2, "oop_median_C0": m0, "relative_change_C2_to_C0": (m0 - m2) / m2,
                     "angle_median_C2": float(f2[a].loc[both, "bond_angle_ratio"].median()),
                     "angle_median_C0": float(f0[a].loc[both, "bond_angle_ratio"].median()),
                     "torsion_median_C2": float(f2[a].loc[both, "dihedral_ratio"].median()),
                     "torsion_median_C0": float(f0[a].loc[both, "dihedral_ratio"].median())}
    report["projection_effect_Sstar"] = {"n_examples": int(len(both)), "arms": effect}
    report["P3_SUPPORTED"] = bool(abs(effect["SIG"]["relative_change_C2_to_C0"]) < 0.10)
    report["P4_SUPPORTED"] = bool(abs(effect["ISO"]["relative_change_C2_to_C0"])
                                  > abs(effect["SIG"]["relative_change_C2_to_C0"]))
    (NEW / "RESULT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
