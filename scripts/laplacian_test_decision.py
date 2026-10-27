"""Analysis fixed in outputs/laplacian_prior_v1/PREREGISTRATION.md, written before any
LAPLACIAN number existed.

LAPLACIAN test runs: outputs/laplacian_prior_v1/test_LAPLACIAN_f{1,2,3}.per_example.csv.
ISO, ANISO_v3 and SIGMA_star: the FROZEN confirmatory runs in
reference_outputs/sigma_star_test_v1/ (never re-run for this comparison).
Per-example spread and bootstrap are computed by the functions of
scripts/sigma_star_test_decision.py, imported unchanged.
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
NEW = ROOT / "outputs/laplacian_prior_v1"


def arm(name):
    D.OUT = NEW if name == "LAPLACIAN" else FROZEN
    return D.load_arm(name)


def main():
    examples = pd.read_csv(FROZEN / "test_examples.csv")
    uniprot = dict(zip(examples.example_id, examples.uniprot))
    names = ("ISO", "ANISO_v3", "SIGMA_star", "LAPLACIAN")
    frames, meta = {}, {}
    for n in names:
        frames[n], meta[n] = arm(n)
    common = frames["ISO"].index
    for n in names[1:]:
        common = common.intersection(frames[n].index)
    report = {"status": "ADDED post-hoc comparison; Sigma*/ISO/ANISO numbers are the frozen confirmatory runs",
              "n_examples_scored": int(len(common)),
              "n_targets_scored": int(pd.Series([uniprot[k] for k in common]).nunique()), "arms": {}}
    for n in names:
        f = frames[n].loc[common]
        report["arms"][n] = {"median_spread": float(f["spread"].median()),
                             "q25_spread": float(f["spread"].quantile(0.25)),
                             "q75_spread": float(f["spread"].quantile(0.75)),
                             "channel_ratio_median": {c: float(f[c].median()) for c in D.CHANNELS},
                             "internal_rmsf_ratio_median": float(f["internal_rmsf_ratio"].median()),
                             **meta[n]}
    cluster = np.array([uniprot[k] for k in common])
    for a, b, key in (("SIGMA_star", "LAPLACIAN", "PRIMARY_sigma_star_minus_laplacian"),
                      ("LAPLACIAN", "ISO", "laplacian_minus_iso")):
        diff = (frames[a].loc[common, "spread"] - frames[b].loc[common, "spread"]).to_numpy()
        lo, hi = D.cluster_bootstrap(diff, cluster)
        report[key] = {"mean_difference": float(diff.mean()), "median_difference": float(np.median(diff)),
                       "ci95_cluster_bootstrap": [lo, hi], "fraction_first_lower": float((diff < 0).mean())}
    p = report["PRIMARY_sigma_star_minus_laplacian"]["ci95_cluster_bootstrap"]
    report["READING"] = ("Sigma* more uniform than LAPLACIAN" if p[1] < 0 else
                         "LAPLACIAN more uniform than Sigma*" if p[0] > 0 else "no difference shown")
    (NEW / "TEST_RESULT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
