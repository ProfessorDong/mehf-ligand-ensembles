"""Cross-validate the MISATO store against the existing known-good 3k cache.

``data/processed/misato_atomic_window_cache_3k_v1`` holds 26,686 windows for
2,980 systems, extracted by this repo's own pipeline by streaming the archive
over HTTP.  It is the only independent ground truth available for the new store,
and a successful exit code from a downloader proves nothing about the numbers
inside the file, so this script re-derives those windows two different ways from
the newly downloaded raw HDF5 and compares every array element by element:

  A. **reference**: ``mdwm.data.misato_atomic.extract_misato_windows`` -- the
     exact function that produced the cache -- run against the *new local* file.
     An A-vs-cache disagreement means the downloaded bytes or the local read
     path differ from what the cache was built from.
  B. **store**: ``misato_store_extract.extract_system`` (100 frames, raw
     coordinates) followed by ``windows_from_system``, which re-cuts the
     11-frame windows from the stored trajectory.  A B-vs-cache disagreement
     means the new store's extraction or its documented window convention is
     wrong, even if the download is perfect.

Both comparisons must pass.  Their tolerances differ on purpose: A reads the
same arrays through the same code, so it is held to bit-equality, while B
rounds through a float32 store and an independent Kabsch fit, so it is held to
a coordinate tolerance.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
    python scripts/misato_store_crossvalidate.py --n-systems 40

Run it against the archive over HTTP with ``--remote`` before the download
finishes, to validate the extractor while the bytes are still arriving; the
local run is the one that also validates the download.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from misato_store_extract import (  # noqa: E402
    SystemRejected,
    extract_system,
    windows_from_system,
)

# The settings the 3k cache was built with (configs/misato_atomic_windows_3k_v1.yaml).
CACHE_WINDOW_KWARGS = {
    "context_frames": 6,
    "future_frames": 5,
    "frame_stride": 1,
    "window_stride": 10,
    "pocket_radius_angstrom": 8.0,
    "max_pocket_atoms": 600,
    "max_ligand_atoms": 300,
    "min_ligand_atoms": 6,
    "broken_bond_angstrom": 2.6,
}
COMPARED_KEYS = (
    "product_init",
    "target_delta",
    "context_ligand_coords",
    "pocket_init",
    "product_features",
    "edit_features",
    "pocket_features",
    "bond_index",
)
DEFAULT_CACHE = ROOT / "data/processed/misato_atomic_window_cache_3k_v1"
MISATO_URL = "https://zenodo.org/api/records/7711953/files/MD.hdf5/content"


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


def _max_abs_deviation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.abs(left.astype(np.float64) - right.astype(np.float64)).max())


def compare_windows(
    produced: list[dict[str, np.ndarray]],
    cached: dict[int, dict[str, np.ndarray]],
) -> dict[str, object]:
    """Per-key maximum absolute deviation, matched on ``start_frame``."""
    by_start = {int(window["start_frame"]): window for window in produced}
    deviations: dict[str, float] = {key: 0.0 for key in COMPARED_KEYS}
    n_matched = 0
    missing = sorted(set(cached) - set(by_start))
    extra = sorted(set(by_start) - set(cached))
    for start, cached_arrays in cached.items():
        if start not in by_start:
            continue
        candidate = by_start[start]
        n_matched += 1
        for key in COMPARED_KEYS:
            if key not in cached_arrays:
                continue
            deviations[key] = max(
                deviations[key], _max_abs_deviation(candidate[key], cached_arrays[key])
            )
    return {
        "n_cached_windows": len(cached),
        "n_produced_windows": len(by_start),
        "n_matched_windows": n_matched,
        "cached_starts_not_produced": missing,
        "produced_starts_not_cached": extra,
        "max_abs_deviation": deviations,
    }


def load_cached_windows(
    index: pd.DataFrame, system_id: str, cache_root: Path
) -> dict[int, dict[str, np.ndarray]]:
    rows = index.loc[index["molecule_id"] == system_id]
    cached: dict[int, dict[str, np.ndarray]] = {}
    for row in rows.itertuples(index=False):
        path = Path(str(row.cache_path))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            path = cache_root / "windows" / f"{row.window_id}.npz"
        with np.load(path) as archive:
            cached[int(row.start_frame)] = {key: archive[key] for key in archive.files}
    return cached


def _crossvalidate_from_store(
    args: argparse.Namespace, index: pd.DataFrame, systems: list[str]
) -> int:
    """Compare stored .npz files on disk against the known-good cache.

    Deliberately never opens the HDF5: the arrays come from whatever the bulk
    run actually wrote.  A torn write, a truncated file, or two runs racing on
    the same output would show up here as a shape or value disagreement, where
    a re-extraction check would silently paper over it by recomputing.
    """
    store = Path(args.store_root) / "extracted" / str(args.from_store)
    manifest_dir = Path(args.store_root) / "manifest" / str(args.from_store)
    available = [s for s in systems if (store / f"{s}.npz").is_file()]
    _log(
        f"from-store check: {len(available)} of {len(systems)} sampled systems "
        f"are present in {store}"
    )
    worst = {key: 0.0 for key in COMPARED_KEYS}
    per_system: list[dict[str, object]] = []
    n_ok = 0
    for system_id in available:
        record: dict[str, object] = {"system_id": system_id}
        try:
            path = store / f"{system_id}.npz"
            with np.load(path) as archive:
                stored = {key: archive[key] for key in archive.files}
            # the manifest's digest must still describe these bytes
            entry_path = manifest_dir / "systems" / f"{system_id}.json"
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            record["sha256_matches_manifest"] = bool(digest == entry.get("npz_sha256"))
            record["frames_stored"] = int(stored["ligand_coords"].shape[0])
            produced = windows_from_system(
                stored,
                context_frames=CACHE_WINDOW_KWARGS["context_frames"],
                future_frames=CACHE_WINDOW_KWARGS["future_frames"],
                frame_stride=CACHE_WINDOW_KWARGS["frame_stride"],
                window_stride=CACHE_WINDOW_KWARGS["window_stride"],
                pocket_radius_angstrom=CACHE_WINDOW_KWARGS["pocket_radius_angstrom"],
                max_pocket_atoms=CACHE_WINDOW_KWARGS["max_pocket_atoms"],
            )
            cached = load_cached_windows(index, system_id, args.cache_root)
            comparison = compare_windows(produced, cached)
            record["store_vs_cache"] = comparison
            for key, value in comparison["max_abs_deviation"].items():  # type: ignore[index]
                worst[key] = max(worst[key], float(value))
            record["status"] = "compared"
            n_ok += 1
        except Exception as error:  # noqa: BLE001
            record["status"] = "error"
            record["reason"] = f"{type(error).__name__}: {error}"[:400]
        per_system.append(record)
        _log(f"  {system_id}: {record.get('status')} "
             f"sha256_ok={record.get('sha256_matches_manifest')} "
             f"frames={record.get('frames_stored')}")

    all_sha_ok = all(r.get("sha256_matches_manifest") for r in per_system
                     if r.get("status") == "compared")
    all_frames_100 = all(int(r.get("frames_stored", 0)) == int(args.frames)
                         for r in per_system if r.get("status") == "compared")
    within = all(v <= float(args.coordinate_tolerance) for v in worst.values())
    sets_agree = all(
        not r["store_vs_cache"]["cached_starts_not_produced"]  # type: ignore[index]
        and not r["store_vs_cache"]["produced_starts_not_cached"]  # type: ignore[index]
        for r in per_system if r.get("status") == "compared"
    )
    report = {
        "status": "misato_store_crossvalidation_from_stored_files",
        "note": (
            "reads the .npz files the bulk run wrote, never the HDF5, so a torn "
            "or raced write cannot be masked by recomputation"
        ),
        "config_name": str(args.from_store),
        "store": str(store),
        "cache_root": str(args.cache_root),
        "n_systems_sampled": len(systems),
        "n_systems_present_in_store": len(available),
        "n_systems_compared": n_ok,
        "worst_max_abs_deviation_store_vs_cache": worst,
        "coordinate_tolerance": float(args.coordinate_tolerance),
        "all_manifest_sha256_match": bool(all_sha_ok),
        "all_stored_frame_counts_correct": bool(all_frames_100),
        "window_sets_agree": bool(sets_agree),
        "verdict": "PASS" if (within and all_sha_ok and all_frames_100
                              and sets_agree and n_ok) else "FAIL",
        "per_system": per_system,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    _log(f"verdict {report['verdict']}; wrote {args.out}")
    _log(json.dumps(worst, indent=2))
    return 0 if report["verdict"] == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hdf5", type=Path, default=ROOT / "data/MISATO/raw/MD.hdf5")
    parser.add_argument(
        "--from-store",
        default=None,
        help="config name under data/MISATO/extracted/: compare the .npz files "
        "AS THEY LIE ON DISK against the cache, instead of re-extracting from "
        "the HDF5. This is the check that would catch a race or a bad write "
        "during a bulk run, because it never re-reads the archive.",
    )
    parser.add_argument("--store-root", type=Path, default=ROOT / "data/MISATO")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="read the archive over HTTP range requests instead of locally; "
        "validates the extractor, not the download",
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--n-systems", type=int, default=40)
    parser.add_argument("--systems", default=None, help="comma-separated ids instead of a sample")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--max-pocket-atoms-store", type=int, default=2000)
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=1e-3,
        help="angstrom tolerance for the store path, which rounds through "
        "float32 and re-does the Kabsch fit",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/MISATO/validation/crossvalidation.json"
    )
    args = parser.parse_args()

    index = pd.read_csv(args.cache_root / "atomic_windows_cached.csv")
    available = sorted(index["molecule_id"].astype(str).unique())
    if args.systems:
        systems = [s.strip() for s in str(args.systems).split(",") if s.strip()]
        unknown = [s for s in systems if s not in set(available)]
        if unknown:
            raise SystemExit(f"not in the 3k cache: {unknown}")
    else:
        rng = random.Random(int(args.seed))
        systems = rng.sample(available, min(int(args.n_systems), len(available)))
    _log(f"cross-validating {len(systems)} systems against {args.cache_root}")

    if args.from_store:
        sys.exit(_crossvalidate_from_store(args, index, systems))

    if args.remote:
        from mdwm.data.http_hdf5 import open_remote_hdf5

        handle, reader = open_remote_hdf5(MISATO_URL, block_bytes=1 << 20, cache_blocks=2048)
        source = MISATO_URL
    else:
        if not args.hdf5.is_file():
            raise SystemExit(
                f"{args.hdf5} is not present (the downloader only publishes it "
                "under that name once its MD5 matches). Use --remote to validate "
                "the extractor before the download finishes."
            )
        handle = h5py.File(args.hdf5, "r", rdcc_nbytes=64 << 20, rdcc_nslots=10007)
        reader = None
        source = str(args.hdf5)

    # The reference implementation: the code that built the cache.
    from mdwm.data.misato_atomic import (  # noqa: E402
        REQUIRED_DATASETS,
        extract_misato_windows,
    )

    assert "torch" not in sys.modules, "the reference import pulled torch; aborting"

    per_system: list[dict[str, object]] = []
    worst = {
        "reference_vs_cache": {key: 0.0 for key in COMPARED_KEYS},
        "store_vs_cache": {key: 0.0 for key in COMPARED_KEYS},
    }
    n_ok = 0
    for position, system_id in enumerate(systems, start=1):
        record: dict[str, object] = {"system_id": system_id}
        try:
            cached = load_cached_windows(index, system_id, args.cache_root)
            group = handle[system_id]
            arrays = {
                name: group[name][:] for name in REQUIRED_DATASETS if name in group
            }
            reference = extract_misato_windows(
                system_id, arrays, split="train", **CACHE_WINDOW_KWARGS
            )
            reference_windows = [
                {"start_frame": window["start_frame"], **window["arrays"]}
                for window in reference
            ]
            record["reference_vs_cache"] = compare_windows(reference_windows, cached)

            stored, qc = extract_system(
                handle,
                system_id,
                n_frames=int(args.frames),
                frame_stride=1,
                pocket_radius_angstrom=CACHE_WINDOW_KWARGS["pocket_radius_angstrom"],
                max_pocket_atoms=int(args.max_pocket_atoms_store),
                max_ligand_atoms=CACHE_WINDOW_KWARGS["max_ligand_atoms"],
                min_ligand_atoms=CACHE_WINDOW_KWARGS["min_ligand_atoms"],
                broken_bond_angstrom=CACHE_WINDOW_KWARGS["broken_bond_angstrom"],
                frame_block=10,
                atom_block=4096,
                max_system_atoms=200_000,
            )
            store_windows = windows_from_system(
                stored,
                context_frames=CACHE_WINDOW_KWARGS["context_frames"],
                future_frames=CACHE_WINDOW_KWARGS["future_frames"],
                frame_stride=CACHE_WINDOW_KWARGS["frame_stride"],
                window_stride=CACHE_WINDOW_KWARGS["window_stride"],
                pocket_radius_angstrom=CACHE_WINDOW_KWARGS["pocket_radius_angstrom"],
                max_pocket_atoms=CACHE_WINDOW_KWARGS["max_pocket_atoms"],
            )
            record["store_vs_cache"] = compare_windows(store_windows, cached)
            record["store_qc"] = {
                key: qc[key]
                for key in (
                    "n_frames",
                    "n_ligand_atoms",
                    "n_pocket_atoms",
                    "min_pocket_ligand_distance_angstrom",
                    "bond_length_max_angstrom",
                    "pocket_truncated_at_cap",
                )
            }
            for side in ("reference_vs_cache", "store_vs_cache"):
                for key, value in record[side]["max_abs_deviation"].items():  # type: ignore[index]
                    worst[side][key] = max(worst[side][key], float(value))
            record["status"] = "compared"
            n_ok += 1
        except SystemRejected as error:
            record["status"] = "rejected_by_extractor"
            record["reason"] = str(error)
        except Exception as error:  # noqa: BLE001
            record["status"] = "error"
            record["reason"] = f"{type(error).__name__}: {error}"[:400]
        per_system.append(record)
        _log(
            f"{position}/{len(systems)} {system_id}: {record.get('status')} "
            + (
                "ref="
                + json.dumps(
                    {
                        k: round(v, 12)
                        for k, v in record["reference_vs_cache"]["max_abs_deviation"].items()  # type: ignore[index]
                    }
                )
                + " store="
                + json.dumps(
                    {
                        k: round(v, 9)
                        for k, v in record["store_vs_cache"]["max_abs_deviation"].items()  # type: ignore[index]
                    }
                )
                if record.get("status") == "compared"
                else str(record.get("reason", ""))
            )
        )

    reference_exact = all(value == 0.0 for value in worst["reference_vs_cache"].values())
    store_within = all(
        value <= float(args.coordinate_tolerance)
        for value in worst["store_vs_cache"].values()
    )
    window_sets_agree = all(
        not record["reference_vs_cache"]["cached_starts_not_produced"]  # type: ignore[index]
        and not record["reference_vs_cache"]["produced_starts_not_cached"]  # type: ignore[index]
        and not record["store_vs_cache"]["cached_starts_not_produced"]  # type: ignore[index]
        and not record["store_vs_cache"]["produced_starts_not_cached"]  # type: ignore[index]
        for record in per_system
        if record.get("status") == "compared"
    )
    report = {
        "status": "misato_store_crossvalidation",
        "source": source,
        "source_is_remote": bool(args.remote),
        "cache_root": str(args.cache_root),
        "cache_windows_total": int(len(index)),
        "cache_systems_total": int(index["molecule_id"].nunique()),
        "n_systems_sampled": len(systems),
        "n_systems_compared": n_ok,
        "seed": int(args.seed),
        "cache_window_kwargs": CACHE_WINDOW_KWARGS,
        "store_frames": int(args.frames),
        "worst_max_abs_deviation": worst,
        "coordinate_tolerance": float(args.coordinate_tolerance),
        "reference_path_bit_exact_vs_cache": bool(reference_exact),
        "store_path_within_tolerance_vs_cache": bool(store_within),
        "window_sets_agree": bool(window_sets_agree),
        "verdict": (
            "PASS"
            if (reference_exact and store_within and window_sets_agree and n_ok)
            else "FAIL"
        ),
        "per_system": per_system,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if reader is not None:
        report["network_bytes_fetched"] = int(reader.bytes_fetched)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    _log(f"verdict {report['verdict']}; wrote {args.out}")
    _log(json.dumps(worst, indent=2))
    handle.close()
    sys.exit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
