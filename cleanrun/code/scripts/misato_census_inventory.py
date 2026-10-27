"""Step 1 of the MISATO MMP feasibility census: inventory what is on this machine.

READ-ONLY on data/. Writes only into outputs/misato_mmp_census_v1/.

Counts, from the data itself (never estimated):
  * MISATO systems covered by the 71-feature trace corpus (train/val/test)
  * MISATO systems covered by the atomic (real-coordinate) window cache
  * their intersection
  * frames / atoms per system in the coordinate cache
  * what rejected_systems.csv rejected and why
  * what the leakage-exclusion file removes and why
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "data/processed/misato_traces_full_13k"
CACHE = ROOT / "data/processed/misato_atomic_window_cache_3k_v1"
LEAK = ROOT / "data/processed/misato_leakage_exclusions_v1.json"
OUT = ROOT / "outputs/misato_mmp_census_v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"step": "inventory"}

    # ---------------- trace corpus ----------------
    trace_ids: dict[str, list[str]] = {}
    n_frames_used = {}
    for split in ("train", "val", "test"):
        d = np.load(TRACES / f"{split}.npz", allow_pickle=True)
        ids = [str(x) for x in d["ids"]]
        trace_ids[split] = ids
        mask = d["frame_mask"]
        n_frames_used[split] = {
            "n_systems": len(ids),
            "trace_shape": list(d["features"].shape),
            "frames_per_system_min": int(mask.sum(axis=1).min()),
            "frames_per_system_max": int(mask.sum(axis=1).max()),
            "frames_per_system_median": float(np.median(mask.sum(axis=1))),
        }
    all_trace = [i for s in ("train", "val", "test") for i in trace_ids[s]]
    trace_set = set(all_trace)
    report["traces"] = {
        "per_split": n_frames_used,
        "n_rows_total": len(all_trace),
        "n_distinct_pdb_ids": len(trace_set),
        "duplicate_ids_across_splits": len(all_trace) - len(trace_set),
        "n_features": 71,
        "id_kind": "4-character RCSB PDB entry ID",
    }

    # ---------------- coordinate window cache ----------------
    idx = pd.read_csv(CACHE / "atomic_windows_cached.csv")
    cache_set = set(idx["molecule_id"].astype(str))
    wc = idx.groupby("molecule_id").size()
    report["coord_cache"] = {
        "n_window_rows": int(len(idx)),
        "n_window_files_on_disk": sum(1 for _ in (CACHE / "windows").glob("*.npz")),
        "n_distinct_pdb_ids": len(cache_set),
        "windows_per_system": {
            "min": int(wc.min()), "max": int(wc.max()),
            "median": float(wc.median()), "mode": int(wc.mode().iloc[0]),
        },
        "context_frames": sorted(idx["context_frames"].unique().tolist()),
        "future_frames": sorted(idx["future_frames"].unique().tolist()),
        "frame_stride": sorted(idx["frame_stride"].unique().tolist()),
        "replicas": sorted(idx["replica"].unique().tolist()),
        "splits_present": idx["split"].value_counts().to_dict(),
        "smiles_column_nonempty": int(idx["smiles"].notna().sum()),
    }

    # ---------------- intersection ----------------
    report["intersection"] = {
        "trace_only": len(trace_set - cache_set),
        "cache_only": len(cache_set - trace_set),
        "both_trace_and_coords": len(trace_set & cache_set),
    }

    # ---------------- atoms per system (sample the npz payloads) ----------------
    rng = np.random.default_rng(0)
    sample = idx.sample(n=min(300, len(idx)), random_state=0)
    lig_atoms, poc_atoms, ctx_shapes, tgt_shapes = [], [], [], []
    keyset = None
    for p in sample["cache_path"]:
        f = ROOT / p
        if not f.exists():
            continue
        z = np.load(f)
        if keyset is None:
            keyset = sorted(z.keys())
        lig_atoms.append(int(z["product_init"].shape[0]))
        poc_atoms.append(int(z["pocket_init"].shape[0]))
        ctx_shapes.append(tuple(int(x) for x in z["context_ligand_coords"].shape))
        tgt_shapes.append(tuple(int(x) for x in z["target_delta"].shape))
    report["coord_cache"]["npz_keys"] = keyset
    report["coord_cache"]["sampled_n_windows"] = len(lig_atoms)
    report["coord_cache"]["ligand_atoms"] = {
        "min": int(np.min(lig_atoms)), "max": int(np.max(lig_atoms)),
        "median": float(np.median(lig_atoms)), "mean": float(np.mean(lig_atoms)),
    }
    report["coord_cache"]["pocket_atoms"] = {
        "min": int(np.min(poc_atoms)), "max": int(np.max(poc_atoms)),
        "median": float(np.median(poc_atoms)), "mean": float(np.mean(poc_atoms)),
    }
    report["coord_cache"]["context_frame_counts"] = dict(
        Counter(s[0] for s in ctx_shapes))
    report["coord_cache"]["target_frame_counts"] = dict(
        Counter(s[0] for s in tgt_shapes))

    # ---------------- rejections ----------------
    rej = pd.read_csv(CACHE / "rejected_systems.csv")

    def bucket(reason: str) -> str:
        r = reason.lower()
        if "above the cap" in r:
            return "ligand atom count above cap"
        if "disconnected" in r:
            return "ligand bond graph disconnected (PBC break / multi-fragment)"
        if "pocket" in r:
            return "no usable pocket"
        return "other"

    report["rejected_systems"] = {
        "n": int(len(rej)),
        "by_reason": dict(Counter(bucket(str(x)) for x in rej["reason"])),
        "pdb_ids": rej["molecule_id"].astype(str).tolist(),
    }

    # ---------------- leakage exclusions ----------------
    lk = json.loads(LEAK.read_text())
    report["leakage_exclusions"] = {
        "n_misato_systems_considered": lk["n_misato_systems"],
        "n_excluded": lk["n_excluded"],
        "source": lk["source"],
        "by_accession": {
            k: {"reason": v.get("reason"), "tier": v.get("tier"),
                "n_misato_excluded": v.get("n_misato_excluded")}
            for k, v in lk["accessions"].items()
        },
        "limitation": lk["limitation"],
    }
    excluded = set(lk["excluded_pdb_ids"])
    report["leakage_exclusions"]["overlap_with_coord_cache"] = len(excluded & cache_set)
    report["leakage_exclusions"]["overlap_with_traces"] = len(excluded & trace_set)

    # ---------------- id lists for downstream steps ----------------
    (OUT / "ids_traces_all.txt").write_text("\n".join(sorted(trace_set)) + "\n")
    (OUT / "ids_coord_cache.txt").write_text("\n".join(sorted(cache_set)) + "\n")
    (OUT / "ids_leakage_excluded.txt").write_text("\n".join(sorted(excluded)) + "\n")

    (OUT / "inventory.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
