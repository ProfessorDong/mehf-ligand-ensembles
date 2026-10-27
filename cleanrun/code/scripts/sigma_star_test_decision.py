"""Apply the frozen decision rule of outputs/sigma_star_test_v1/PREREGISTRATION.md.

Reads only the nine test_<ARM>_f<K>.per_example.csv files, the frozen example
list, and nothing else. Written before any of those files existed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/sigma_star_test_v1"
ARMS = ("ISO", "ANISO_v3", "SIGMA_star")
CHANNELS = ("bond_length_ratio", "bond_angle_ratio", "dihedral_ratio",
            "planar_improper_ratio")
N_BOOT, BOOT_SEED = 10000, 20260925


def load_arm(arm: str) -> tuple[pd.DataFrame, dict]:
    """Per-example four-channel spread, averaged over the arm's three folds."""
    per_fold, extras = {}, {}
    for fold in (1, 2, 3):
        frame = pd.read_csv(OUT / f"test_{arm}_f{fold}.per_example.csv")
        frame = frame.set_index("key")
        finite = frame[list(CHANNELS)].notna().all(axis=1)
        frame = frame[finite]
        spread = frame[list(CHANNELS)].max(axis=1) - frame[list(CHANNELS)].min(axis=1)
        per_fold[fold] = spread
        extras[fold] = frame[list(CHANNELS) + ["internal_rmsf_ratio"]]
    shared = per_fold[1].index.intersection(per_fold[2].index).intersection(
        per_fold[3].index)
    mean_spread = sum(per_fold[f].loc[shared] for f in (1, 2, 3)) / 3.0
    channels = sum(extras[f].loc[shared] for f in (1, 2, 3)) / 3.0
    return pd.DataFrame({"spread": mean_spread}).join(channels), {
        "per_fold_median_spread": {f: float(per_fold[f].median()) for f in (1, 2, 3)},
        "n_per_fold": {f: int(len(per_fold[f])) for f in (1, 2, 3)},
    }


def cluster_bootstrap(diff: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """95% percentile CI of the mean paired difference, resampling UniProt targets."""
    rng = np.random.default_rng(BOOT_SEED)
    groups = pd.Series(range(len(diff))).groupby(cluster).apply(
        lambda s: np.asarray(s)).tolist()
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picked = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[p] for p in picked])
        draws[b] = diff[idx].mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    examples = pd.read_csv(OUT / "test_examples.csv")
    uniprot = dict(zip(examples["example_id"], examples["uniprot"]))

    arms, meta = {}, {}
    for arm in ARMS:
        arms[arm], meta[arm] = load_arm(arm)
    common = arms[ARMS[0]].index
    for arm in ARMS[1:]:
        common = common.intersection(arms[arm].index)
    for arm in ARMS:
        arms[arm] = arms[arm].loc[common]

    report = {
        "n_examples_scored": int(len(common)),
        "n_examples_in_split": int(len(examples)),
        "n_targets_scored": int(pd.Series([uniprot[k] for k in common]).nunique()),
        "arms": {},
    }
    for arm in ARMS:
        frame = arms[arm]
        report["arms"][arm] = {
            "median_spread": float(frame["spread"].median()),
            "q25_spread": float(frame["spread"].quantile(0.25)),
            "q75_spread": float(frame["spread"].quantile(0.75)),
            "channel_ratio_median": {c: float(frame[c].median()) for c in CHANNELS},
            "internal_rmsf_ratio_median": float(frame["internal_rmsf_ratio"].median()),
            **meta[arm],
        }

    cluster = np.array([uniprot[k] for k in common])
    for comparator in ("ISO", "ANISO_v3"):
        diff = (arms["SIGMA_star"]["spread"] - arms[comparator]["spread"]).to_numpy()
        low, high = cluster_bootstrap(diff, cluster)
        report[f"paired_vs_{comparator}"] = {
            "mean_difference": float(diff.mean()),
            "median_difference": float(np.median(diff)),
            "ci95_cluster_bootstrap": [low, high],
            "excludes_zero": bool(low < 0 and high < 0) or bool(low > 0 and high > 0),
            "fraction_examples_sigma_star_lower": float((diff < 0).mean()),
        }

    beats_both = (report["arms"]["SIGMA_star"]["median_spread"]
                  < report["arms"]["ISO"]["median_spread"]) and (
                  report["arms"]["SIGMA_star"]["median_spread"]
                  < report["arms"]["ANISO_v3"]["median_spread"])
    primary = report["paired_vs_ISO"]
    report["decision"] = {
        "median_below_both_comparators": bool(beats_both),
        "paired_ci_vs_ISO_excludes_zero": bool(primary["excludes_zero"]),
        "sigma_star_direction_favourable": bool(primary["mean_difference"] < 0),
        "VERDICT": ("CONFIRMED" if (beats_both and primary["excludes_zero"]
                                    and primary["mean_difference"] < 0)
                    else "NOT CONFIRMED"),
    }
    (OUT / "TEST_RESULT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
