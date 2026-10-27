"""Post-hoc absolute-calibration endpoints on the stored test outputs (see PLAN.md).

Run: python -I posthoc_endpoints.py <MEHF root> <out dir>
Reads only the per-example test CSVs and test_examples.csv. Runs no model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
import sigma_star_test_decision as D  # noqa: E402  (frozen bootstrap function and constants)

REF = ROOT / "reference_outputs"
SRC = {"ISO": (REF / "sigma_star_test_v1", "ISO"), "GS": (REF / "sigma_star_test_v1", "ANISO_v3"),
       "LAP": (REF / "laplacian_prior_v1", "LAPLACIAN"), "SIG": (REF / "sigma_star_test_v1", "SIGMA_star")}
CH4 = ["bond_length", "bond_angle", "dihedral", "planar_improper"]
CH3 = CH4[:3]


def per_fold(arm):
    d, name = SRC[arm]
    out = {}
    for f in (1, 2, 3):
        fr = pd.read_csv(d / f"test_{name}_f{f}.per_example.csv").set_index("key")
        lr = {c: np.log(fr[c + "_ratio"]).abs() for c in CH4 + ["internal_rmsf"]}
        out[f] = pd.DataFrame({
            "S": fr[[c + "_ratio" for c in CH4]].max(axis=1) - fr[[c + "_ratio" for c in CH4]].min(axis=1),
            "E4": sum(lr[c] for c in CH4) / 4.0,
            "E5": sum(lr[c] for c in CH4 + ["internal_rmsf"]) / 5.0,
            "E3": sum(lr[c] for c in CH3) / 3.0,
            **{c + "_gen": fr[c + "_generated"] for c in CH4 + ["internal_rmsf"]},
            **{c + "_ref": fr[c + "_reference"] for c in CH4 + ["internal_rmsf"]},
        })
    return out


examples = pd.read_csv(REF / "sigma_star_test_v1/test_examples.csv")
uniprot = dict(zip(examples["example_id"], examples["uniprot"]))
folds = {a: per_fold(a) for a in SRC}


def fold_mean(arm, col, keys):
    return sum(folds[arm][f].loc[keys, col] for f in (1, 2, 3)) / 3.0


def finite_keys(cols):
    keys = None
    for a in SRC:
        for f in (1, 2, 3):
            k = folds[a][f][cols].dropna().index
            keys = k if keys is None else keys.intersection(k)
    return keys


k4 = finite_keys(["S", "E4", "E5"])
k3 = finite_keys(["E3"])
res = {"n_examples_4ch": int(len(k4)), "n_targets_4ch": int(pd.Series([uniprot[k] for k in k4]).nunique()),
       "n_examples_3ch": int(len(k3)), "n_targets_3ch": int(pd.Series([uniprot[k] for k in k3]).nunique()),
       "summary": {}, "paired": {}, "per_fold_median": {}, "equal_target_median": {}, "abs_width_median": {}}

for metric, keys in (("S", k4), ("E4", k4), ("E5", k4), ("E3", k3)):
    vals = {a: fold_mean(a, metric, keys) for a in SRC}
    res["summary"][metric] = {a: {"median": float(v.median()), "q25": float(v.quantile(.25)),
                                  "q75": float(v.quantile(.75)), "q90": float(v.quantile(.9))}
                              for a, v in vals.items()}
    res["per_fold_median"][metric] = {a: [float(folds[a][f].loc[keys, metric].median()) for f in (1, 2, 3)]
                                      for a in SRC}
    tgt = pd.Series([uniprot[k] for k in keys], index=keys)
    res["equal_target_median"][metric] = {a: float(v.groupby(tgt).median().median()) for a, v in vals.items()}
    cluster = np.array([uniprot[k] for k in keys])
    res["paired"][metric] = {}
    for comp in ("ISO", "GS", "LAP"):
        diff = (vals["SIG"] - vals[comp]).to_numpy()
        lo, hi = D.cluster_bootstrap(diff, cluster)
        res["paired"][metric][f"SIG_minus_{comp}"] = {
            "mean": float(diff.mean()), "median": float(np.median(diff)), "ci95_mean": [lo, hi],
            "frac_sig_lower": float((diff < 0).mean())}

for a in SRC:
    res["abs_width_median"][a] = {c: {"generated": float(fold_mean(a, c + "_gen", k4).median()),
                                      "reference": float(folds[a][1].loc[k4, c + "_ref"].median())}
                                  for c in CH4 + ["internal_rmsf"]}

# consistency check against the frozen record: S medians must reproduce TEST_RESULT.json
frozen = json.load(open(REF / "laplacian_prior_v1/TEST_RESULT.json"))
res["check_S_median_vs_record"] = {a: [res["summary"]["S"][a]["median"], frozen["arms"][SRC[a][1]]["median_spread"]]
                                   for a in SRC}
(OUT / "RESULT.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
