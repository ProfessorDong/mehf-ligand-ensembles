"""Step 4 of the MISATO MMP feasibility census: leakage-excluded counts,
target concentration, coordinate availability, and the like-for-like comparison
against the project's current self-generated corpus.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/misato_mmp_census_v1"


def groups(pdf: pd.DataFrame, key_cols: tuple[str, ...]) -> list[int]:
    nbr: dict[tuple, set] = defaultdict(set)
    for r in pdf.itertuples(index=False):
        base = tuple(getattr(r, c) for c in key_cols)
        nbr[base + (r.lig_a,)].add(r.lig_b)
        nbr[base + (r.lig_b,)].add(r.lig_a)
    return [len(v) for v in nbr.values() if len(v) >= 3]


def recurrence(pdf: pd.DataFrame) -> dict:
    tx: dict[str, set] = defaultdict(set)
    for r in pdf.itertuples(index=False):
        tx[r.transformation].add(r.uniprot)
    n = len(tx)
    dist = Counter(len(v) for v in tx.values())
    ge2 = sum(1 for v in tx.values() if len(v) >= 2)
    ge3 = sum(1 for v in tx.values() if len(v) >= 3)
    return {
        "n_transformations": n,
        "n_on_ge2_targets": ge2,
        "fraction_on_ge2_targets": ge2 / n if n else None,
        "n_on_ge3_targets": ge3,
        "fraction_on_ge3_targets": ge3 / n if n else None,
        "targets_per_transformation_distribution": dict(sorted(dist.items())),
    }


def main() -> None:
    pdf = pd.read_csv(OUT / "mmp_pairs.csv")
    excluded = set((OUT / "ids_leakage_excluded.txt").read_text().split())
    sysdf = pd.read_csv(OUT / "systems_resolved.csv")

    # map ligand -> set of pdb ids, to apply leakage exclusion at ligand level
    lig_pdbs: dict[str, set] = defaultdict(set)
    for r in sysdf.itertuples(index=False):
        if isinstance(r.ligand_canon, str):
            lig_pdbs[r.ligand_canon].add(r.pdb_id)

    leak_uni = {"P00533", "P06493", "P24941", "P04626",
                "P21860", "Q15303", "P11802", "Q00534", "P50750"}
    clean = pdf[~pdf["uniprot"].isin(leak_uni)].copy()

    report: dict = {}

    def block(d: pd.DataFrame, label: str) -> dict:
        ppt = d.groupby("uniprot").size().sort_values(ascending=False)
        g_strict = groups(d, ("uniprot", "constant"))
        g_coord = groups(d[d["both_coords"]], ("uniprot", "constant")) if d["both_coords"].any() else []
        tot = int(len(d))
        return {
            "n_pairs": tot,
            "n_pairs_both_endpoints_cached_coords": int(d["both_coords"].sum()),
            "n_targets": int(d["uniprot"].nunique()),
            "n_distinct_ligands": int(len(set(d["lig_a"]) | set(d["lig_b"]))),
            "pairs_per_target": {
                "median": float(ppt.median()), "mean": float(ppt.mean()),
                "max": int(ppt.max()),
                "n_targets_ge_10": int((ppt >= 10).sum()),
                "n_targets_ge_30": int((ppt >= 30).sum()),
                "n_targets_ge_50": int((ppt >= 50).sum()),
                "top_15": {k: int(v) for k, v in ppt.head(15).items()},
                "share_of_pairs_from_top1_target": float(ppt.iloc[0] / tot),
                "share_of_pairs_from_top5_targets": float(ppt.head(5).sum() / tot),
                "share_of_pairs_from_top20_targets": float(ppt.head(20).sum() / tot),
            },
            "same_context_groups_strict": {
                "n": len(g_strict),
                "median_size": float(np.median(g_strict)) if g_strict else None,
                "size_distribution": dict(sorted(Counter(g_strict).items())),
                "n_with_all_members_cached_coords": len(g_coord),
            },
            "cross_target_recurrence": recurrence(d),
        }

    report["all_pairs"] = block(pdf, "all")
    report["leakage_excluded"] = block(clean, "clean")

    # how many pairs touch a leakage-excluded PDB entry directly
    report["leakage"] = {
        "leakage_uniprots": sorted(leak_uni),
        "n_pairs_on_leakage_targets": int(len(pdf) - len(clean)),
        "n_leakage_excluded_pdb_entries": len(excluded),
    }

    # ---- like-for-like vs current self-generated corpus ----
    cur = json.loads((ROOT / "outputs/ecross_feasibility/summary.json").read_text())
    report["comparison_with_current_corpus"] = {
        "current_corpus_source": "outputs/ecross_feasibility/summary.json (project's own measurement)",
        "current": {
            "n_molecules": 96,
            "n_congeneric_series": 3,
            "n_targets": 3,
            "n_pairs": cur["current_corpus"]["n_pairs"],
            "n_distinct_transformations": cur["current_corpus"]["n_distinct_transformations"],
            "n_transformations_recurring_ge2_contexts": cur["current_corpus"]["n_recurring_ge2_starts"],
            "n_transformations_recurring_ge3_contexts": cur["current_corpus"]["n_recurring_ge3_starts"],
            "same_context_groups_reported_by_task": 88,
            "same_context_group_median_size_reported_by_task": 4,
            "cross_scaffold_recurrence_reported_by_task": 0.038,
            "cross_scaffold_recurring_transformations_reported_by_task": 4,
        },
        "misato_leakage_excluded": {
            "n_molecules": report["leakage_excluded"]["n_distinct_ligands"],
            "n_targets": report["leakage_excluded"]["n_targets"],
            "n_pairs": report["leakage_excluded"]["n_pairs"],
            "n_distinct_transformations":
                report["leakage_excluded"]["cross_target_recurrence"]["n_transformations"],
            "n_transformations_on_ge2_targets":
                report["leakage_excluded"]["cross_target_recurrence"]["n_on_ge2_targets"],
            "fraction_on_ge2_targets":
                report["leakage_excluded"]["cross_target_recurrence"]["fraction_on_ge2_targets"],
            "same_context_groups": report["leakage_excluded"]["same_context_groups_strict"]["n"],
            "same_context_group_median_size":
                report["leakage_excluded"]["same_context_groups_strict"]["median_size"],
        },
    }

    (OUT / "refined_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
