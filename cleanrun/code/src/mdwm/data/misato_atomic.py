"""Build MEHF atomic pretraining windows by streaming the MISATO archive.

The development pretraining cache (`data/processed/atomic_window_cache_dev_v1`)
holds 1,350 windows drawn from 30 distinct molecules on a single target.  MISATO
supplies ~17,000 distinct protein-ligand systems at 100 frames each, which is the
scale the backbone needs and which no amount of architecture work substitutes
for.  Nothing here requires new MD: the trajectories already exist, and
:mod:`mdwm.data.http_hdf5` reads them out of a 132.8 GB remote HDF5 file without
downloading it.

Output contract
---------------
Windows are written in *exactly* the format
:func:`mdwm.data.atomic_windows.cache_atomic_windows` produces, so
`mehf_pretrain` consumes them unchanged: one ``.npz`` per window holding
``product_init, target_delta, context_ligand_coords, product_features,
edit_features, pocket_init, pocket_features, bond_index``, plus an index CSV with
the columns ``window_id,target_id,molecule_id,replica,split,start_frame,
context_frames,future_frames,frame_stride,topology_path,trajectory_path,
pose_path,smiles,cache_path``.  ``AtomicWindowDataset`` dereferences only
``cache_path`` when it is populated, so the three path columns carry MISATO URIs
for provenance rather than local files, and ``smiles`` is empty because the
archive does not ship one.

What MISATO does and does not supply
------------------------------------
Verified directly against the archive rather than assumed:

* ``molecules_begin_atom_index`` holds the first atom index of every molecule
  in the system, with the ligand ordered last.  Single-chain entries are
  2-vectors like ``[0, 4419]``, but multi-chain entries are longer -- 4K6W is
  ``[0, 2890, 5813, 8715]`` around a 170-atom ligand -- so the ligand begins at
  the *last* entry.  Splitting on the second entry instead silently folds every
  chain after the first into the "ligand".
* ``atoms_number`` carries true atomic numbers (1, 6, 7, 8, 16, ...).
  ``atoms_element`` is a small *index* (1-7 on 5WIJ) and is deliberately not
  used as an element.
* There is **no bond table, no formal charge, no aromaticity, no hybridization
  and no SMILES**.  Bonds are therefore inferred geometrically from covalent
  radii, ``degree`` follows from those bonds, and the three genuinely unavailable
  channels are written as zero rather than guessed.  This is recorded in the
  summary as a fidelity gap: these windows are suitable for the fixed-topology
  backbone (E2 / frozen-base) role they are built for, and are not a substitute
  for the chemically-resolved development cache in any action-conditioned claim.
* There are **no periodic box vectors**, so the unwrapping and minimum-image
  steps the local pipeline performs cannot be reproduced.  Systems whose ligand
  geometry is broken across the periodic boundary are rejected rather than
  silently repaired, by a bond-graph connectivity gate plus a long-bond sweep.
  Connectivity is the gate that does the work: because the bonds are themselves
  inferred from a possibly-broken frame, an atom displaced by a box length
  simply acquires no bonds, and a bond-length test alone never fires.
* Frames are aligned onto the window's anchor frame by Kabsch superposition of
  the pocket heavy atoms.  The local pipeline aligns on ``protein and backbone``;
  no reliable backbone selection exists here, and the pocket frame is the one
  the ligand dynamics are expressed in.

Every one of these deviations is written into the run summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mdwm.data.http_hdf5 import open_remote_hdf5

MISATO_MD_URL = "https://zenodo.org/record/7711953/files/MD.hdf5"

# The measured sustainable concurrency against Zenodo (see the check in
# build_misato_atomic_windows for the supporting measurements).
MAX_SUSTAINABLE_WORKERS = 2

# Standard atomic weights for the elements MISATO's protein-ligand systems
# contain; the fallback keeps an exotic element from silently becoming zero.
ATOMIC_MASSES: dict[int, float] = {
    1: 1.008, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    11: 22.990, 12: 24.305, 14: 28.085, 15: 30.974, 16: 32.06, 17: 35.45,
    19: 39.098, 20: 40.078, 25: 54.938, 26: 55.845, 27: 58.933, 28: 58.693,
    29: 63.546, 30: 65.38, 33: 74.922, 34: 78.971, 35: 79.904, 44: 101.07,
    45: 102.91, 46: 106.42, 47: 107.87, 48: 112.41, 51: 121.76, 53: 126.90,
    77: 192.22, 78: 195.08, 79: 196.97, 80: 200.59,
}

# Single-bond covalent radii in angstrom, used only to infer the ligand bond
# graph that MISATO does not ship.
COVALENT_RADII: dict[int, float] = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24,
    29: 1.32, 30: 1.22, 33: 1.19, 34: 1.20, 35: 1.20, 44: 1.46,
    45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 51: 1.39, 53: 1.39,
    77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32,
}

DEFAULT_MASS = 12.011
DEFAULT_RADIUS = 0.77

REQUIRED_DATASETS = (
    "trajectory_coordinates",
    "atoms_number",
    "molecules_begin_atom_index",
)

INDEX_COLUMNS = [
    "window_id", "target_id", "molecule_id", "replica", "split",
    "start_frame", "context_frames", "future_frames", "frame_stride",
    "topology_path", "trajectory_path", "pose_path", "smiles", "cache_path",
]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_masses(atomic_numbers: np.ndarray) -> np.ndarray:
    return np.asarray(
        [ATOMIC_MASSES.get(int(value), DEFAULT_MASS) for value in atomic_numbers],
        dtype=np.float32,
    )


def infer_covalent_bonds(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    tolerance_angstrom: float = 0.45,
    max_bond_angstrom: float = 2.2,
) -> np.ndarray:
    """Infer a ligand bond graph from one frame's geometry.

    MISATO ships no bond table.  Two atoms are bonded when their separation is
    below the sum of their covalent radii plus a tolerance, which is the standard
    geometric criterion; the absolute cap keeps a broken or stretched structure
    from producing spurious long bonds.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Bond inference needs coordinates of shape (atoms,3)")
    if len(coordinates) != len(atomic_numbers):
        raise ValueError("Coordinate and atomic-number counts differ")
    radii = np.asarray(
        [COVALENT_RADII.get(int(value), DEFAULT_RADIUS) for value in atomic_numbers],
        dtype=np.float64,
    )
    separation = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    threshold = np.minimum(radii[:, None] + radii[None, :] + tolerance_angstrom, max_bond_angstrom)
    bonded = np.triu(separation <= threshold, k=1)
    rows, columns = np.nonzero(bonded)
    return np.column_stack([rows, columns]).astype(np.int64)


def ligand_product_features(atomic_numbers: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    """The six-column ligand feature block, matching the local extractor's scaling.

    ``formal_charge``, ``aromatic`` and ``hybridization`` are zero because MISATO
    supplies no chemical graph from which to derive them.  They are left at zero
    rather than guessed, and the gap is declared in the summary.
    """
    atomic_numbers = np.asarray(atomic_numbers)
    degree = np.zeros(len(atomic_numbers), dtype=np.float32)
    for left, right in bonds:
        degree[int(left)] += 1.0
        degree[int(right)] += 1.0
    return np.column_stack(
        [
            atomic_numbers.astype(np.float32) / 100.0,
            atomic_masses(atomic_numbers) / 200.0,
            np.zeros(len(atomic_numbers), dtype=np.float32),   # formal_charge: unavailable
            np.zeros(len(atomic_numbers), dtype=np.float32),   # aromatic: unavailable
            degree / 4.0,
            np.zeros(len(atomic_numbers), dtype=np.float32),   # hybridization: unavailable
        ]
    ).astype(np.float32)


def no_edit_features(n_atoms: int) -> np.ndarray:
    """The in-support no-edit block: mapped indicator on, no edit locality.

    Identical to what the local pretraining cache stores and to the neutral code
    ``mdwm.eval.conditioning_variants.neutral_edit_features`` builds, because
    MISATO systems carry no chemical edit at all.
    """
    block = np.zeros((int(n_atoms), 5), dtype=np.float32)
    block[:, 0] = 1.0
    return block


def _kabsch_rotation(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, mobile_center, reference_center


def _max_bond_length(coordinates: np.ndarray, bonds: np.ndarray) -> float:
    if len(bonds) == 0:
        return 0.0
    deltas = coordinates[bonds[:, 0]] - coordinates[bonds[:, 1]]
    return float(np.linalg.norm(deltas, axis=-1).max())


def bond_graph_is_connected(n_atoms: int, bonds: np.ndarray) -> bool:
    """Whether the inferred bond graph joins every ligand atom into one fragment.

    This is the load-bearing check for periodic-boundary damage.  A long-bond
    test alone is vacuous here, because the bonds are themselves inferred from
    the possibly-broken frame: an atom that has drifted a box length away simply
    acquires no bonds and the long-bond test never fires.  A drug-like ligand is
    one connected molecule, so a disconnected graph means either a torn image or
    a multi-fragment entry, and neither should enter the corpus silently.
    """
    if n_atoms <= 0:
        return False
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for left, right in bonds:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    seen = np.zeros(n_atoms, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        current = stack.pop()
        for neighbour in adjacency[current]:
            if not seen[neighbour]:
                seen[neighbour] = True
                stack.append(neighbour)
    return bool(seen.all())


class MisatoSystemRejected(RuntimeError):
    """A system cannot yield trustworthy windows; the reason is recorded."""


def extract_misato_windows(
    system_id: str,
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    context_frames: int = 6,
    future_frames: int = 5,
    frame_stride: int = 1,
    window_stride: int = 10,
    pocket_radius_angstrom: float = 8.0,
    max_pocket_atoms: int = 600,
    max_ligand_atoms: int = 300,
    min_ligand_atoms: int = 6,
    broken_bond_angstrom: float = 2.6,
    target_id: str = "MISATO",
) -> list[dict[str, Any]]:
    """Turn one streamed MISATO system into cache-format window dictionaries."""
    for name in REQUIRED_DATASETS:
        if name not in arrays:
            raise MisatoSystemRejected(f"{system_id}: missing dataset {name}")
    coordinates = np.asarray(arrays["trajectory_coordinates"], dtype=np.float64)
    atomic_numbers = np.asarray(arrays["atoms_number"], dtype=np.int64)
    begin = np.asarray(arrays["molecules_begin_atom_index"], dtype=np.int64)
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise MisatoSystemRejected(f"{system_id}: unexpected coordinate shape {coordinates.shape}")
    n_frames, n_atoms, _ = coordinates.shape
    if n_atoms != len(atomic_numbers):
        raise MisatoSystemRejected(f"{system_id}: atom count disagrees with the coordinate array")
    if begin.size < 2:
        raise MisatoSystemRejected(f"{system_id}: molecules_begin_atom_index has no ligand entry")
    # molecules_begin_atom_index holds the first atom index of EVERY molecule,
    # and MISATO orders the ligand last.  Multi-chain systems therefore have
    # more than two entries -- e.g. 4K6W is [0, 2890, 5813, 8715] with a
    # 170-atom ligand tail -- so the ligand begins at the LAST entry, not the
    # second.  Using begin[1] silently treats every chain after the first as
    # part of the ligand.
    ligand_start = int(begin[-1])
    if not 0 < ligand_start < n_atoms:
        raise MisatoSystemRejected(f"{system_id}: ligand start index {ligand_start} is out of range")

    ligand_slice = slice(ligand_start, n_atoms)
    ligand_numbers = atomic_numbers[ligand_slice]
    n_ligand = int(len(ligand_numbers))
    if n_ligand < min_ligand_atoms:
        raise MisatoSystemRejected(f"{system_id}: ligand has only {n_ligand} atoms")
    if n_ligand > max_ligand_atoms:
        raise MisatoSystemRejected(f"{system_id}: ligand has {n_ligand} atoms, above the cap")

    protein_numbers = atomic_numbers[:ligand_start]
    protein_heavy = np.nonzero(protein_numbers > 1)[0]
    if protein_heavy.size < 3:
        raise MisatoSystemRejected(f"{system_id}: fewer than three protein heavy atoms")

    span = (context_frames - 1 + future_frames) * frame_stride + 1
    if n_frames < span:
        raise MisatoSystemRejected(f"{system_id}: {n_frames} frames is shorter than the window span")

    bonds = infer_covalent_bonds(coordinates[0, ligand_slice], ligand_numbers)
    if len(bonds) == 0:
        raise MisatoSystemRejected(f"{system_id}: no ligand bonds could be inferred")
    # No periodic box is available, so a ligand split across the boundary cannot
    # be made whole.  Reject it rather than emit torn geometry.  Connectivity is
    # checked first because it is the test that actually detects a torn image;
    # the long-bond sweep then catches a ligand that tears later in the run,
    # relative to the graph inferred at the first frame.
    if not bond_graph_is_connected(n_ligand, bonds):
        raise MisatoSystemRejected(
            f"{system_id}: ligand bond graph is disconnected, so the geometry is "
            "broken across the periodic boundary or the entry is multi-fragment"
        )
    for frame in (0, n_frames - 1):
        if _max_bond_length(coordinates[frame, ligand_slice], bonds) > broken_bond_angstrom:
            raise MisatoSystemRejected(
                f"{system_id}: ligand geometry looks broken across the periodic boundary"
            )

    product_features = ligand_product_features(ligand_numbers, bonds)
    edit_features = no_edit_features(n_ligand)

    windows: list[dict[str, Any]] = []
    for start in range(0, n_frames - span + 1, window_stride):
        context_indices = [start + offset * frame_stride for offset in range(context_frames)]
        anchor_index = context_indices[-1]
        future_indices = [
            anchor_index + (offset + 1) * frame_stride for offset in range(future_frames)
        ]
        anchor = coordinates[anchor_index]
        anchor_ligand = anchor[ligand_slice]
        ligand_center = anchor_ligand.mean(axis=0)

        # The pocket is chosen at this window's anchor frame, matching the local
        # extractor, then held fixed across the window's frames.
        distances = np.linalg.norm(
            anchor[:ligand_start][protein_heavy][:, None, :] - anchor_ligand[None, :, :],
            axis=-1,
        ).min(axis=1)
        pocket_indices = protein_heavy[distances <= float(pocket_radius_angstrom)]
        if pocket_indices.size < 3:
            continue
        if pocket_indices.size > max_pocket_atoms:
            order = np.argsort(distances[distances <= float(pocket_radius_angstrom)])
            pocket_indices = pocket_indices[np.sort(order[:max_pocket_atoms])]
        reference_pocket = anchor[pocket_indices]

        ligand_frames = []
        pocket_frames = []
        for frame_index in context_indices + future_indices:
            frame = coordinates[frame_index]
            rotation, mobile_center, reference_center = _kabsch_rotation(
                frame[pocket_indices], reference_pocket
            )
            ligand_frames.append(
                (frame[ligand_slice] - mobile_center) @ rotation.T + reference_center
            )
            pocket_frames.append(
                (frame[pocket_indices] - mobile_center) @ rotation.T + reference_center
            )
        ligand_frames = np.asarray(ligand_frames, dtype=np.float32)
        pocket_frames = np.asarray(pocket_frames, dtype=np.float32)
        context_ligand = ligand_frames[:context_frames]
        anchor_ligand_aligned = context_ligand[-1]

        pocket_numbers = atomic_numbers[pocket_indices]
        pocket_features = np.column_stack(
            [
                (pocket_numbers > 1).astype(np.float32),
                atomic_masses(pocket_numbers) / 200.0,
            ]
        ).astype(np.float32)

        window_id = hashlib.sha256(
            f"{target_id}|{system_id}|1|{start}|{context_frames}|{future_frames}|{frame_stride}".encode(
                "utf-8"
            )
        ).hexdigest()[:20]
        windows.append(
            {
                "window_id": window_id,
                "target_id": target_id,
                "molecule_id": str(system_id),
                "replica": 1,
                "split": str(split),
                "start_frame": int(start),
                "context_frames": int(context_frames),
                "future_frames": int(future_frames),
                "frame_stride": int(frame_stride),
                "arrays": {
                    "product_init": anchor_ligand_aligned,
                    "target_delta": (
                        ligand_frames[context_frames:] - anchor_ligand_aligned[None, :, :]
                    ),
                    "context_ligand_coords": context_ligand,
                    "product_features": product_features,
                    "edit_features": edit_features,
                    "pocket_init": pocket_frames[context_frames - 1],
                    "pocket_features": pocket_features,
                    "bond_index": bonds,
                },
                "n_ligand_atoms": n_ligand,
                "n_pocket_atoms": int(pocket_indices.size),
                "ligand_center": ligand_center,
            }
        )
    if not windows:
        raise MisatoSystemRejected(f"{system_id}: no window had a usable pocket")
    return windows


def load_misato_system_ids(
    root: str | Path, splits: Iterable[str] = ("train", "val", "test")
) -> pd.DataFrame:
    """System ids and their MISATO split assignment, from the on-disk projection.

    Only the ``ids`` key is read; the 71-dimensional descriptor features in these
    files are a different product and are not used here.
    """
    root = Path(root)
    rows: list[dict[str, str]] = []
    for split in splits:
        path = root / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"MISATO split file is missing: {path}")
        with np.load(path, allow_pickle=True) as archive:
            for identifier in archive["ids"]:
                rows.append({"molecule_id": str(identifier), "split": str(split)})
    frame = pd.DataFrame(rows)
    duplicated = frame["molecule_id"].duplicated().sum()
    if duplicated:
        raise ValueError(f"MISATO split files share {duplicated} system ids across splits")
    return frame


# One remote handle per worker process.  The block cache is deliberately NOT
# shared: h5py serialises through a global lock and HTTPRangeReader keeps mutable
# position and cache state, so a shared reader would both serialise the workers
# and corrupt their reads.  Each process pays its own B-tree warm-up in exchange
# for genuinely independent streams.
def _log_progress(
    completed: int,
    total: int,
    started: float,
    per_system: list[dict[str, Any]],
    written_bytes: int,
) -> None:
    elapsed = time.perf_counter() - started
    fetched = sum(int(record["network_bytes"]) for record in per_system)
    print(
        f"[misato] {completed}/{total} systems in {elapsed:.0f}s "
        f"({elapsed / max(completed, 1):.1f} s/system wall), "
        f"{len(per_system)} written, {fetched / 1024**2:.0f} MB fetched, "
        f"{written_bytes / 1024**2:.0f} MB on disk",
        flush=True,
    )


_WORKER_STATE: dict[str, Any] = {}


def _worker_handle(url: str, block_bytes: int, cache_blocks: int):
    state = _WORKER_STATE.get("state")
    if state is None:
        handle, reader = open_remote_hdf5(
            url, block_bytes=block_bytes, cache_blocks=cache_blocks
        )
        state = {"handle": handle, "reader": reader}
        _WORKER_STATE["state"] = state
    return state["handle"], state["reader"]


def _process_system(task: dict[str, Any]) -> dict[str, Any]:
    """Extract and persist one system; safe to call from any worker process.

    Returns a small record rather than arrays, so nothing large crosses the
    process boundary.  The shard CSV is written last and atomically, which is
    what makes the build resumable: its presence means every window for that
    system is already on disk, and an interrupted system leaves no shard and is
    simply redone.
    """
    system_id = str(task["system_id"])
    shard_path = Path(task["shard_dir"]) / f"{system_id}.csv"
    if shard_path.is_file():
        return {"molecule_id": system_id, "status": "skipped"}
    handle, reader = _worker_handle(
        task["url"], int(task["block_bytes"]), int(task["cache_blocks"])
    )
    bytes_before = reader.bytes_fetched
    started = time.perf_counter()
    try:
        if system_id not in handle:
            raise MisatoSystemRejected(f"{system_id}: absent from the archive")
        group = handle[system_id]
        arrays = {name: group[name][:] for name in REQUIRED_DATASETS if name in group}
        windows = extract_misato_windows(
            system_id, arrays, split=str(task["split"]), **task["window_kwargs"]
        )
    except MisatoSystemRejected as error:
        # A property of the data: this system will never be usable.
        return {"molecule_id": system_id, "status": "rejected", "reason": str(error)}
    except Exception as error:  # noqa: BLE001
        # A property of the network (429 exhaustion, 504 gateway timeouts, a
        # dropped connection).  The system is fine; the request was not.  It
        # writes no shard, so simply re-running the build retries it -- and it
        # must NOT be counted alongside data rejections, or a transient outage
        # would masquerade as a corpus-quality statistic.
        return {
            "molecule_id": system_id,
            "status": "transient_failure",
            "reason": f"{type(error).__name__}: {error}"[:300],
        }

    cache_dir = Path(task["cache_dir"])
    url = str(task["url"])
    shard_rows: list[dict[str, Any]] = []
    system_bytes = 0
    for window in windows:
        cache_path = cache_dir / f"{window['window_id']}.npz"
        np.savez_compressed(cache_path, **window["arrays"])
        system_bytes += cache_path.stat().st_size
        shard_rows.append(
            {
                "window_id": window["window_id"],
                "target_id": window["target_id"],
                "molecule_id": window["molecule_id"],
                "replica": window["replica"],
                "split": window["split"],
                "start_frame": window["start_frame"],
                "context_frames": window["context_frames"],
                "future_frames": window["future_frames"],
                "frame_stride": window["frame_stride"],
                "topology_path": f"{url}#/{system_id}",
                "trajectory_path": f"{url}#/{system_id}/trajectory_coordinates",
                "pose_path": f"{url}#/{system_id}",
                "smiles": "",
                "cache_path": str(cache_path),
            }
        )
    # Atomic publish: a reader never sees a half-written shard.
    temporary = shard_path.with_suffix(f".csv.tmp{os.getpid()}")
    pd.DataFrame(shard_rows, columns=INDEX_COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, shard_path)
    return {
        "molecule_id": system_id,
        "status": "written",
        "split": str(task["split"]),
        "n_windows": len(windows),
        "n_ligand_atoms": windows[0]["n_ligand_atoms"],
        "n_pocket_atoms": windows[0]["n_pocket_atoms"],
        "wall_seconds": float(time.perf_counter() - started),
        "network_bytes": int(reader.bytes_fetched - bytes_before),
        "disk_bytes": int(system_bytes),
    }


def build_misato_atomic_windows(cfg: dict[str, Any]) -> dict[str, Any]:
    """Stream MISATO systems and write cache-format atomic pretraining windows."""
    build_cfg = dict(cfg["misato_atomic_windows"])
    output_root = Path(build_cfg["output_root"])
    url = str(build_cfg.get("misato_url", MISATO_MD_URL))
    id_root = Path(build_cfg.get("system_id_root", "data/processed/misato_traces_full_13k"))
    splits = [str(value) for value in build_cfg.get("splits", ["train", "val", "test"])]
    max_systems = build_cfg.get("max_systems")
    max_storage_gb = float(build_cfg.get("max_storage_gb", 50.0))
    exclusion_path = build_cfg.get("leakage_exclusion_path")

    window_kwargs = {
        key: build_cfg[key]
        for key in (
            "context_frames", "future_frames", "frame_stride", "window_stride",
            "pocket_radius_angstrom", "max_pocket_atoms", "max_ligand_atoms",
            "min_ligand_atoms", "broken_bond_angstrom",
        )
        if key in build_cfg
    }

    systems = load_misato_system_ids(id_root, splits)

    excluded_ids: set[str] = set()
    exclusion_record: dict[str, Any] | None = None
    if exclusion_path is not None:
        exclusion_file = Path(exclusion_path)
        if not exclusion_file.is_file():
            raise FileNotFoundError(
                "Leakage exclusion list is required but missing; run "
                "scripts/resolve_misato_leakage_exclusions.py first: "
                f"{exclusion_file}"
            )
        exclusion_record = json.loads(exclusion_file.read_text(encoding="utf-8"))
        excluded_ids = {str(value) for value in exclusion_record["excluded_pdb_ids"]}
    elif not bool(build_cfg.get("leakage_exclusion_waived", False)):
        raise ValueError(
            "Refusing to build a pretraining corpus without a leakage exclusion "
            "list; set leakage_exclusion_path, or leakage_exclusion_waived to "
            "acknowledge the risk explicitly"
        )

    selected = systems.loc[~systems["molecule_id"].isin(excluded_ids)].reset_index(drop=True)
    n_excluded = int(len(systems) - len(selected))
    if max_systems is not None:
        selected = selected.iloc[: int(max_systems)].reset_index(drop=True)

    cache_dir = output_root / "windows"
    shard_dir = output_root / "shards"
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    block_bytes = int(build_cfg.get("block_bytes", 1 << 20))
    cache_blocks = int(build_cfg.get("cache_blocks", 4096))
    n_workers = int(build_cfg.get("n_workers", 1))
    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if n_workers > MAX_SUSTAINABLE_WORKERS:
        # Measured, not assumed: Zenodo throttles by request rate, and a single
        # stream already runs at ~30 requests/minute.  Two streams complete
        # 12/12 systems at 16.3 s/system (1.52x over serial); four streams get
        # 11 of 12 refused with HTTP 429.  Raising block_bytes does not buy
        # headroom either -- 4 MB blocks fetch 2.7x the bytes for no time saved,
        # and 8 MB blocks make Zenodo's proxy return 504.  So the ceiling is
        # politeness and the server's limit, not local CPU.
        raise ValueError(
            f"Refusing more than {MAX_SUSTAINABLE_WORKERS} concurrent streams "
            "against Zenodo: 4 workers were measured to fail with HTTP 429"
        )

    tasks = [
        {
            "system_id": str(row.molecule_id),
            "split": str(row.split),
            "url": url,
            "block_bytes": block_bytes,
            "cache_blocks": cache_blocks,
            "cache_dir": str(cache_dir),
            "shard_dir": str(shard_dir),
            "window_kwargs": window_kwargs,
        }
        for row in selected.itertuples(index=False)
    ]

    rejected: list[dict[str, str]] = []
    per_system: list[dict[str, Any]] = []
    written_bytes = 0
    started = time.perf_counter()

    transient: list[dict[str, str]] = []

    def _absorb(record: dict[str, Any]) -> None:
        nonlocal written_bytes
        if record["status"] == "rejected":
            rejected.append(
                {"molecule_id": record["molecule_id"], "reason": record["reason"]}
            )
        elif record["status"] == "transient_failure":
            transient.append(
                {"molecule_id": record["molecule_id"], "reason": record["reason"]}
            )
        elif record["status"] == "written":
            written_bytes += int(record["disk_bytes"])
            per_system.append(
                {key: value for key, value in record.items() if key != "status"}
            )

    completed = 0
    if n_workers == 1:
        for task in tasks:
            _absorb(_process_system(task))
            completed += 1
            if completed % 10 == 0:
                _log_progress(completed, len(tasks), started, per_system, written_bytes)
    else:
        # Each worker builds its own reader on first use, so the streams are
        # independent; chunksize=1 keeps a slow system from blocking a whole
        # batch behind it.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for record in pool.map(_process_system, tasks, chunksize=1):
                _absorb(record)
                completed += 1
                if completed % 10 == 0:
                    _log_progress(completed, len(tasks), started, per_system, written_bytes)
    if written_bytes / 1024.0**3 > max_storage_gb:
        raise RuntimeError(
            f"MISATO window cache exceeded its storage cap of {max_storage_gb} GB"
        )
    wall_seconds = time.perf_counter() - started

    shards = sorted(shard_dir.glob("*.csv"))
    index = (
        pd.concat([pd.read_csv(path) for path in shards], ignore_index=True)
        if shards
        else pd.DataFrame(columns=INDEX_COLUMNS)
    )
    if index.empty:
        raise ValueError("No MISATO windows were produced")
    leakage = index.groupby("molecule_id")["split"].nunique()
    if int((leakage > 1).sum()):
        raise RuntimeError("System leakage detected across MISATO window splits")
    index_path = output_root / "atomic_windows_cached.csv"
    index.to_csv(index_path, index=False)
    rejected_path = output_root / "rejected_systems.csv"
    pd.DataFrame(rejected, columns=["molecule_id", "reason"]).to_csv(rejected_path, index=False)
    transient_path = output_root / "transient_failures.csv"
    pd.DataFrame(transient, columns=["molecule_id", "reason"]).to_csv(transient_path, index=False)
    per_system_frame = pd.DataFrame(per_system)
    per_system_path = output_root / "per_system_cost.csv"
    per_system_frame.to_csv(per_system_path, index=False)

    total_bytes = sum(path.stat().st_size for path in cache_dir.glob("*.npz"))
    n_done = int(len(per_system_frame))
    mean_wall = float(per_system_frame["wall_seconds"].mean()) if n_done else float("nan")
    mean_network = float(per_system_frame["network_bytes"].mean()) if n_done else float("nan")
    mean_disk = float(per_system_frame["disk_bytes"].mean()) if n_done else float("nan")
    n_full_corpus = int(len(systems) - n_excluded)
    summary = {
        "status": "misato_atomic_pretraining_windows_ready",
        "purpose": (
            "fixed-topology backbone (E2 / frozen-base) pretraining at chemical "
            "scale; not action-conditioned training data"
        ),
        "misato_url": url,
        "misato_archive_streamed_not_downloaded": True,
        "system_id_root": str(id_root),
        "splits_requested": splits,
        "n_systems_listed": int(len(systems)),
        "n_systems_excluded_for_leakage": n_excluded,
        "n_systems_attempted": int(len(selected)),
        "n_systems_written": n_done,
        "n_systems_rejected": int(len(rejected)),
        "n_systems_transient_failure": int(len(transient)),
        "transient_failures": str(transient_path),
        "transient_failure_note": (
            "network faults (HTTP 429 exhaustion, 504 gateway timeouts, dropped "
            "connections), not data problems. These systems wrote no shard, so "
            "re-running the build retries them; they are excluded from the "
            "data-rejection rate on purpose"
        ),
        "data_rejection_rate": float(
            len(rejected) / max(len(rejected) + len(per_system), 1)
        ),
        "n_windows": int(len(index)),
        "n_molecules": int(index["molecule_id"].nunique()),
        "split_window_counts": {
            str(key): int(value) for key, value in index["split"].value_counts().items()
        },
        "split_molecule_counts": {
            str(key): int(value)
            for key, value in index.groupby("split")["molecule_id"].nunique().items()
        },
        "molecule_split_overlap": 0,
        "cached_index": str(index_path),
        "rejected_systems": str(rejected_path),
        "per_system_cost": str(per_system_path),
        "storage_gb": float(total_bytes / 1024.0**3),
        "max_storage_gb": max_storage_gb,
        "wall_seconds": float(wall_seconds),
        "mean_wall_seconds_per_system": mean_wall,
        "mean_network_bytes_per_system": mean_network,
        "mean_disk_bytes_per_system": mean_disk,
        "network_transfer": {
            "total_bytes_fetched": int(
                sum(int(record["network_bytes"]) for record in per_system)
            ),
            "total_gigabytes_fetched": float(
                sum(int(record["network_bytes"]) for record in per_system) / 1024.0**3
            ),
            "block_bytes": block_bytes,
            "cache_blocks_per_worker": cache_blocks,
        },
        "n_workers": n_workers,
        "throughput_systems_per_hour": float(
            3600.0 * len(per_system) / max(wall_seconds, 1.0e-9)
        ),
        "projected_full_corpus": {
            "n_systems": n_full_corpus,
            "network_gb": float(mean_network * n_full_corpus / 1024.0**3) if n_done else None,
            "disk_gb": float(mean_disk * n_full_corpus / 1024.0**3) if n_done else None,
            "wall_hours_single_stream": (
                float(mean_wall * n_full_corpus / 3600.0) if n_done else None
            ),
            "projected_windows": (
                int(round(len(index) / max(n_done, 1) * n_full_corpus)) if n_done else None
            ),
            "basis": (
                "per-system means measured on this run, scaled linearly; the "
                "archive's B-tree cache warms over the first systems, so a long "
                "run is expected to be modestly faster per system than a short one"
            ),
        },
        "leakage_exclusion": exclusion_record,
        "leakage_exclusion_path": (None if exclusion_path is None else str(exclusion_path)),
        "leakage_exclusion_sha256": (
            None if exclusion_path is None else _sha256_file(exclusion_path)
        ),
        "output_contract": (
            "identical to mdwm.data.atomic_windows.cache_atomic_windows: one npz "
            "per window with product_init, target_delta, context_ligand_coords, "
            "product_features, edit_features, pocket_init, pocket_features, "
            "bond_index, plus the standard index CSV"
        ),
        "edit_features": (
            "in-support no-edit block (mapped indicator on, zero edit-locality "
            "atoms); MISATO systems carry no chemical edit"
        ),
        "fidelity_gaps": [
            "bond graph inferred geometrically from covalent radii, not from a chemical graph",
            "formal_charge, aromatic and hybridization feature columns are zero (unavailable in MISATO)",
            "no SMILES is available, so the smiles index column is empty",
            "no periodic box vectors, so no unwrapping or minimum-image correction is applied; "
            "systems are rejected when the inferred ligand bond graph is disconnected or a "
            "bond stretches beyond the cap, rather than being repaired",
            "frames are aligned by Kabsch superposition of pocket heavy atoms rather than "
            "the local pipeline's 'protein and backbone' selection",
        ],
        "split_caveat": (
            "MISATO's own train/val/test assignment is disjoint by PDB entry but "
            "is not guaranteed disjoint by protein; do not treat these splits as "
            "a protein-level generalization test without regrouping by UniProt"
        ),
        "claim_boundary": (
            "fixed-topology pretraining corpus only; no chemical-edit dynamics "
            "label, and no evaluation claim may be made from these windows"
        ),
        "test_outcomes_read": False,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary
