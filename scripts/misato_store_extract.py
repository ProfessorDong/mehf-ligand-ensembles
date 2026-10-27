"""Extract per-system MISATO trajectories from the local raw HDF5 into a reusable store.

This writes the *whole* stored trajectory for each system -- all 100 frames of
the ligand and of its pocket -- rather than pre-cut training windows.  The
existing cache at ``data/processed/misato_atomic_window_cache_3k_v1`` stores
11-frame windows (6 context + 5 target); that was a caching choice, not a
property of MISATO, and a short target horizon is a plausible cause of the
almost-flat MISATO pretraining curve (val MAE 0.5629 -> 0.5581 over 20 epochs).
Storing every frame once makes any future window length a slicing operation
instead of a re-extraction, and costs about 1 MB per system.

Coordinates are stored RAW (unaligned, exactly as MISATO ships them).  Alignment
is a downstream convention -- the existing pipeline Kabsch-fits pocket heavy
atoms onto each window's anchor frame -- and a store that bakes one convention in
cannot serve another.  ``windows_from_system`` in this module reproduces the
existing convention bit-for-bit from the stored arrays, and
``scripts/misato_store_crossvalidate.py`` proves it.

Bounded resources, by construction
----------------------------------
* HDF5 is read in slices, never whole.  The ligand is a contiguous tail slice of
  the atom axis, so it is read directly.  Protein coordinates are read in blocks
  of ``--frame-block`` frames and discarded after the pocket mask is computed;
  the pocket atoms are then re-read by index.  Nothing of system scale is ever
  held for the whole run.
* Pairwise distances are computed in blocks of ``--atom-block`` protein atoms,
  so the distance temporary is bounded independently of system size.
* A watchdog in the parent samples the resident memory of the whole worker
  group, the system's ``MemAvailable`` and the free disk, and pauses then stops
  the pool when any crosses its threshold.  Workers publish atomically, so a
  stop can never leave a half-written output.
* No GPU: ``CUDA_VISIBLE_DEVICES`` is cleared before anything else and the run
  refuses to start if ``torch`` has been imported.

Resumability
------------
Completeness is decided by **content**, never by existence.  A system counts as
done only when its manifest entry says ``written``, records the same extraction
config hash, the same frame count, and a SHA-256 that still matches the bytes of
the ``.npz`` on disk.  A run killed mid-write leaves a ``.npz.tmp<pid>`` that no
check ever looks at, and a run killed after the write but before the manifest
leaves an unclaimed ``.npz`` that is simply redone.  This is deliberately
unlike ``scripts/run_manifest_md.py``, whose "output exists and is non-empty"
test was shown to accept truncated 60 ps runs as complete 1 ns runs.

Usage
-----
    CUDA_VISIBLE_DEVICES="" python scripts/misato_store_extract.py \
        --config-name frames100_pairs_v1 --systems census-pairs --workers 12

    # everything MISATO ships, same settings
    CUDA_VISIBLE_DEVICES="" python scripts/misato_store_extract.py \
        --config-name frames100_all_v1 --systems all --workers 12
"""

from __future__ import annotations

import os

# Before numpy, before h5py, before anything that could look at a device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Iterable  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024.0**3

# ---- chemistry tables, identical to mdwm.data.misato_atomic ---------------- #
ATOMIC_MASSES: dict[int, float] = {
    1: 1.008, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    11: 22.990, 12: 24.305, 14: 28.085, 15: 30.974, 16: 32.06, 17: 35.45,
    19: 39.098, 20: 40.078, 25: 54.938, 26: 55.845, 27: 58.933, 28: 58.693,
    29: 63.546, 30: 65.38, 33: 74.922, 34: 78.971, 35: 79.904, 44: 101.07,
    45: 102.91, 46: 106.42, 47: 107.87, 48: 112.41, 51: 121.76, 53: 126.90,
    77: 192.22, 78: 195.08, 79: 196.97, 80: 200.59,
}
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

MISATO_URL = "https://zenodo.org/api/records/7711953/files/MD.hdf5/content"
DEFAULT_EXCLUSIONS = ROOT / "data/processed/misato_leakage_exclusions_v1.json"
DEFAULT_SPLIT_ROOT = ROOT / "data/processed/misato_traces_full_13k"
DEFAULT_CENSUS = ROOT / "outputs/misato_mmp_census_v1/mmp_pairs.csv"


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# feature construction (kept numerically identical to the reference extractor)
# --------------------------------------------------------------------------- #
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
    coordinates = np.asarray(coordinates, dtype=np.float64)
    radii = np.asarray(
        [COVALENT_RADII.get(int(value), DEFAULT_RADIUS) for value in atomic_numbers],
        dtype=np.float64,
    )
    separation = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )
    threshold = np.minimum(
        radii[:, None] + radii[None, :] + tolerance_angstrom, max_bond_angstrom
    )
    bonded = np.triu(separation <= threshold, k=1)
    rows, columns = np.nonzero(bonded)
    return np.column_stack([rows, columns]).astype(np.int64)


def ligand_product_features(atomic_numbers: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    atomic_numbers = np.asarray(atomic_numbers)
    degree = np.zeros(len(atomic_numbers), dtype=np.float32)
    for left, right in bonds:
        degree[int(left)] += 1.0
        degree[int(right)] += 1.0
    return np.column_stack(
        [
            atomic_numbers.astype(np.float32) / 100.0,
            atomic_masses(atomic_numbers) / 200.0,
            np.zeros(len(atomic_numbers), dtype=np.float32),  # formal_charge: unavailable
            np.zeros(len(atomic_numbers), dtype=np.float32),  # aromatic: unavailable
            degree / 4.0,
            np.zeros(len(atomic_numbers), dtype=np.float32),  # hybridization: unavailable
        ]
    ).astype(np.float32)


def no_edit_features(n_atoms: int) -> np.ndarray:
    block = np.zeros((int(n_atoms), 5), dtype=np.float32)
    block[:, 0] = 1.0
    return block


def pocket_feature_block(pocket_numbers: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            (np.asarray(pocket_numbers) > 1).astype(np.float32),
            atomic_masses(pocket_numbers) / 200.0,
        ]
    ).astype(np.float32)


def bond_graph_is_connected(n_atoms: int, bonds: np.ndarray) -> bool:
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


def _max_bond_length(coordinates: np.ndarray, bonds: np.ndarray) -> float:
    if len(bonds) == 0:
        return 0.0
    deltas = coordinates[bonds[:, 0]] - coordinates[bonds[:, 1]]
    return float(np.linalg.norm(deltas, axis=-1).max())


def kabsch_rotation(
    mobile: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identical to ``mdwm.data.misato_atomic._kabsch_rotation``."""
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, mobile_center, reference_center


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
class SystemRejected(RuntimeError):
    """The system cannot yield trustworthy data; the reason is recorded."""


def _min_distance_to_ligand(
    protein_block: np.ndarray, ligand_frame: np.ndarray, atom_block: int
) -> np.ndarray:
    """Per-protein-atom minimum distance to any ligand atom, in bounded memory.

    The naive ``(n_protein, n_ligand)`` matrix is 36 MB for a 30k-atom protein
    and a 300-atom ligand *per frame*; blocking the protein axis keeps the
    temporary at ``atom_block * n_ligand`` regardless of system size.
    """
    n_protein = protein_block.shape[0]
    out = np.empty(n_protein, dtype=np.float32)
    ligand = np.asarray(ligand_frame, dtype=np.float32)
    for start in range(0, n_protein, atom_block):
        stop = min(start + atom_block, n_protein)
        chunk = protein_block[start:stop]
        deltas = chunk[:, None, :] - ligand[None, :, :]
        out[start:stop] = np.sqrt((deltas * deltas).sum(axis=-1)).min(axis=1)
    return out


def extract_system(
    handle: h5py.File,
    system_id: str,
    *,
    n_frames: int,
    frame_stride: int,
    pocket_radius_angstrom: float,
    max_pocket_atoms: int,
    max_ligand_atoms: int,
    min_ligand_atoms: int,
    broken_bond_angstrom: float,
    frame_block: int,
    atom_block: int,
    max_system_atoms: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One system's full stored trajectory plus its QC record.

    Returns ``(arrays, qc)``.  Raises :class:`SystemRejected` with the reason
    when the system is unusable; that is a property of the data and is recorded,
    not retried.
    """
    if system_id not in handle:
        raise SystemRejected("absent from the archive")
    group = handle[system_id]
    for name in ("trajectory_coordinates", "atoms_number", "molecules_begin_atom_index"):
        if name not in group:
            raise SystemRejected(f"missing dataset {name}")

    coordinates = group["trajectory_coordinates"]
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise SystemRejected(f"unexpected coordinate shape {coordinates.shape}")
    total_frames, n_atoms, _ = coordinates.shape
    if n_atoms > max_system_atoms:
        raise SystemRejected(
            f"{n_atoms} atoms exceeds the per-system cap of {max_system_atoms}"
        )

    frame_indices = np.arange(0, n_frames * frame_stride, frame_stride, dtype=np.int64)
    if frame_indices[-1] >= total_frames:
        raise SystemRejected(
            f"{total_frames} stored frames cannot supply {n_frames} frames at "
            f"stride {frame_stride}"
        )

    atomic_numbers = np.asarray(group["atoms_number"][:], dtype=np.int64)
    begin = np.asarray(group["molecules_begin_atom_index"][:], dtype=np.int64)
    # MISATO also ships per-atom annotations and four per-frame observables.
    # They are a few kilobytes per system and cannot be recomputed from
    # coordinates, so a permanent store keeps them rather than making a future
    # reader go back to the 124 GiB archive for them.  atoms_element is a small
    # internal INDEX, not an element -- it is carried through verbatim and
    # atoms_number remains the only thing treated as chemistry.
    annotations: dict[str, np.ndarray] = {}
    for name in ("atoms_element", "atoms_residue", "atoms_type"):
        if name in group:
            values = np.asarray(group[name][:])
            if values.shape[0] == n_atoms:
                annotations[name] = values
    observables: dict[str, np.ndarray] = {}
    for name in (
        "frames_bSASA",
        "frames_distance",
        "frames_interaction_energy",
        "frames_rmsd_ligand",
    ):
        if name in group:
            values = np.asarray(group[name][:], dtype=np.float64)
            if values.shape[0] >= total_frames:
                observables[name] = values
    if n_atoms != len(atomic_numbers):
        raise SystemRejected("atom count disagrees with the coordinate array")
    if begin.size < 2:
        raise SystemRejected("molecules_begin_atom_index has no ligand entry")
    # MISATO orders the ligand LAST and lists the start of every molecule, so a
    # multi-chain system has more than two entries; begin[1] would fold every
    # chain after the first into the "ligand".
    ligand_start = int(begin[-1])
    if not 0 < ligand_start < n_atoms:
        raise SystemRejected(f"ligand start index {ligand_start} is out of range")

    ligand_numbers = atomic_numbers[ligand_start:]
    n_ligand = int(len(ligand_numbers))
    if n_ligand < min_ligand_atoms:
        raise SystemRejected(f"ligand has only {n_ligand} atoms")
    if n_ligand > max_ligand_atoms:
        raise SystemRejected(f"ligand has {n_ligand} atoms, above the cap")

    protein_numbers = atomic_numbers[:ligand_start]
    protein_heavy = np.nonzero(protein_numbers > 1)[0]
    if protein_heavy.size < 3:
        raise SystemRejected("fewer than three protein heavy atoms")

    # -- ligand: a contiguous tail slice of the atom axis, so one read -------- #
    contiguous = bool(np.array_equal(frame_indices, np.arange(len(frame_indices))))
    if contiguous:
        ligand_coords = np.asarray(
            coordinates[: len(frame_indices), ligand_start:, :], dtype=np.float32
        )
    else:
        ligand_coords = np.stack(
            [
                np.asarray(coordinates[int(index), ligand_start:, :], dtype=np.float32)
                for index in frame_indices
            ]
        )

    if not np.isfinite(ligand_coords).all():
        raise SystemRejected("ligand coordinates contain non-finite values")

    bonds = infer_covalent_bonds(ligand_coords[0], ligand_numbers)
    if len(bonds) == 0:
        raise SystemRejected("no ligand bonds could be inferred")
    if not bond_graph_is_connected(n_ligand, bonds):
        raise SystemRejected(
            "ligand bond graph is disconnected, so the geometry is broken across "
            "the periodic boundary or the entry is multi-fragment"
        )
    # The accept/reject rule is deliberately the reference extractor's -- first
    # and last frame -- so that this store and the existing 3k cache agree on
    # which systems are usable.  The all-frame statistic is recorded as QC.
    bond_lengths = np.linalg.norm(
        ligand_coords[:, bonds[:, 0], :] - ligand_coords[:, bonds[:, 1], :], axis=-1
    )
    for frame in (0, len(frame_indices) - 1):
        if float(bond_lengths[frame].max()) > broken_bond_angstrom:
            raise SystemRejected(
                "ligand geometry looks broken across the periodic boundary"
            )

    # -- pocket: union over stored frames, computed in bounded blocks --------- #
    # Read protein coordinates a few frames at a time, keep only the running
    # minimum distance per protein heavy atom, and drop the coordinates.  Peak
    # memory is frame_block * ligand_start * 12 bytes, not the whole protein
    # trajectory.
    min_distance = np.full(protein_heavy.size, np.inf, dtype=np.float32)
    for block_start in range(0, len(frame_indices), frame_block):
        block = frame_indices[block_start : block_start + frame_block]
        if contiguous:
            protein_block = np.asarray(
                coordinates[int(block[0]) : int(block[-1]) + 1, :ligand_start, :],
                dtype=np.float32,
            )
        else:
            protein_block = np.stack(
                [
                    np.asarray(
                        coordinates[int(index), :ligand_start, :], dtype=np.float32
                    )
                    for index in block
                ]
            )
        for offset in range(protein_block.shape[0]):
            distances = _min_distance_to_ligand(
                protein_block[offset][protein_heavy],
                ligand_coords[block_start + offset],
                atom_block,
            )
            np.minimum(min_distance, distances, out=min_distance)
        del protein_block

    within = min_distance <= float(pocket_radius_angstrom)
    pocket_indices = protein_heavy[within]
    pocket_truncated = False
    if pocket_indices.size < 3:
        raise SystemRejected(
            f"only {pocket_indices.size} protein heavy atoms within "
            f"{pocket_radius_angstrom} A of the ligand at any stored frame"
        )
    if pocket_indices.size > max_pocket_atoms:
        # Keep the closest, in ascending atom index, so the selection is
        # deterministic and still a superset of any per-frame selection that
        # fits under the cap.
        order = np.argsort(min_distance[within], kind="stable")[:max_pocket_atoms]
        pocket_indices = np.sort(pocket_indices[order])
        pocket_truncated = True

    pocket_coords = np.empty(
        (len(frame_indices), pocket_indices.size, 3), dtype=np.float32
    )
    for block_start in range(0, len(frame_indices), frame_block):
        block = frame_indices[block_start : block_start + frame_block]
        if contiguous:
            protein_block = np.asarray(
                coordinates[int(block[0]) : int(block[-1]) + 1, :ligand_start, :],
                dtype=np.float32,
            )
        else:
            protein_block = np.stack(
                [
                    np.asarray(
                        coordinates[int(index), :ligand_start, :], dtype=np.float32
                    )
                    for index in block
                ]
            )
        pocket_coords[block_start : block_start + len(block)] = protein_block[
            :, pocket_indices, :
        ]
        del protein_block

    if not np.isfinite(pocket_coords).all():
        raise SystemRejected("pocket coordinates contain non-finite values")

    pocket_numbers = atomic_numbers[pocket_indices]
    arrays: dict[str, np.ndarray] = {
        "ligand_coords": ligand_coords,
        "pocket_coords": pocket_coords,
        "ligand_atomic_numbers": ligand_numbers.astype(np.int16),
        "pocket_atomic_numbers": pocket_numbers.astype(np.int16),
        "ligand_atom_indices": np.arange(ligand_start, n_atoms, dtype=np.int32),
        "pocket_atom_indices": pocket_indices.astype(np.int32),
        # protein_heavy is ascending and pocket_indices is a sorted subset of it,
        # so searchsorted recovers each pocket atom's slot in min_distance.
        "pocket_min_distance_over_frames": min_distance[
            np.searchsorted(protein_heavy, pocket_indices)
        ].astype(np.float32),
        "bond_index": bonds,
        "product_features": ligand_product_features(ligand_numbers, bonds),
        "edit_features": no_edit_features(n_ligand),
        "pocket_features": pocket_feature_block(pocket_numbers),
        "frame_indices": frame_indices.astype(np.int32),
        "molecules_begin_atom_index": begin,
        "n_system_atoms": np.asarray(n_atoms, dtype=np.int64),
        "n_stored_frames_in_archive": np.asarray(total_frames, dtype=np.int64),
    }
    for name, values in annotations.items():
        arrays[f"ligand_{name}"] = values[ligand_start:]
        arrays[f"pocket_{name}"] = values[pocket_indices]
    for name, values in observables.items():
        arrays[name] = values[frame_indices].astype(np.float32)

    # -- QC ------------------------------------------------------------------ #
    # Cheapest honest clash statistic: the closest approach of a pocket heavy
    # atom to a ligand atom over all stored frames.
    closest = float("inf")
    for frame in range(len(frame_indices)):
        distances = _min_distance_to_ligand(
            pocket_coords[frame], ligand_coords[frame], atom_block
        )
        closest = min(closest, float(distances.min()))
    ligand_displacement = np.linalg.norm(
        ligand_coords - ligand_coords[0][None, :, :], axis=-1
    )
    centroid = ligand_coords.mean(axis=1).astype(np.float64)
    radius_of_gyration = np.sqrt(
        ((ligand_coords - centroid[:, None, :].astype(np.float32)) ** 2)
        .sum(axis=-1)
        .mean(axis=1)
    )
    heavy = ligand_numbers > 1
    # Periodic-image detection.  MISATO ships no box vectors, so a ligand that
    # crosses the periodic boundary cannot be re-imaged -- and a WHOLE-molecule
    # translation by a box vector keeps every bond intact, so the inherited
    # bond-graph connectivity gate cannot see it (that gate only catches a
    # ligand torn apart across the boundary).  Measured on 2VCJ: the centroid
    # moves 47.2 A between two consecutive ~80 ps frames while the radius of
    # gyration stays at 4.68-5.04 A, and MISATO's own ligand RMSD stays at
    # 1.4-2.7 A -- i.e. the ligand never left the pocket and the jump is an
    # imaging artifact.  A rigid jump of tens of angstrom in one frame is not
    # physical, so it is flagged.  It is NOT a rejection: a genuinely unbinding
    # ligand also takes large steps (4K7N moves 7.5 A per frame with MISATO's
    # own RMSD at 42-84 A), and telling the two apart needs the box MISATO does
    # not ship.
    centroid_step = np.linalg.norm(np.diff(centroid, axis=0), axis=-1)
    gyration_spread = (
        float(radius_of_gyration.std() / max(radius_of_gyration.mean(), 1e-9))
        if len(radius_of_gyration)
        else 0.0
    )
    max_step = float(centroid_step.max()) if centroid_step.size else 0.0
    qc: dict[str, Any] = {
        "n_frames": int(len(frame_indices)),
        "frame_stride": int(frame_stride),
        "n_ligand_atoms": n_ligand,
        "n_ligand_heavy_atoms": int(heavy.sum()),
        "n_pocket_atoms": int(pocket_indices.size),
        "n_system_atoms": int(n_atoms),
        "n_bonds": int(len(bonds)),
        "n_molecules_in_system": int(begin.size),
        "ligand_start_atom_index": ligand_start,
        "pocket_truncated_at_cap": bool(pocket_truncated),
        "coords_all_finite": True,
        "bond_length_min_angstrom": float(bond_lengths.min()),
        "bond_length_max_angstrom": float(bond_lengths.max()),
        "bond_length_max_first_last_frame": float(
            max(bond_lengths[0].max(), bond_lengths[-1].max())
        ),
        "bond_length_warning": bool(float(bond_lengths.max()) > broken_bond_angstrom),
        "min_pocket_ligand_distance_angstrom": closest,
        "clash_warning": bool(closest < 1.2),
        "ligand_max_displacement_angstrom": float(ligand_displacement.max()),
        "ligand_rmsd_last_vs_first_angstrom": float(
            np.sqrt((ligand_displacement[-1] ** 2).mean())
        ),
        "ligand_static_trajectory": bool(float(ligand_displacement.max()) < 1e-6),
        "ligand_radius_of_gyration_min": float(radius_of_gyration.min()),
        "ligand_radius_of_gyration_max": float(radius_of_gyration.max()),
        "ligand_radius_of_gyration_rel_spread": gyration_spread,
        "ligand_centroid_step_max_angstrom": max_step,
        "ligand_centroid_step_median_angstrom": (
            float(np.median(centroid_step)) if centroid_step.size else 0.0
        ),
        "n_frames_with_centroid_step_over_10_angstrom": int((centroid_step > 10.0).sum()),
        "periodic_image_jump_suspected": bool(max_step > 10.0 and gyration_spread < 0.2),
        "element_histogram": {
            str(int(z)): int(count)
            for z, count in zip(*np.unique(ligand_numbers, return_counts=True))
        },
        "auxiliary_datasets_present": sorted(annotations) + sorted(observables),
    }
    # What atoms_residue actually is, measured rather than assumed: a residue
    # TYPE code in 0..22, not a residue instance id.  Code 0 is the code for a
    # non-standard residue, which is what a small-molecule ligand is, and it
    # appears on no protein atom.  So for a small-molecule ligand the code
    # independently confirms the boundary that molecules_begin_atom_index[-1]
    # defines -- but for a PEPTIDE or amino-acid ligand the last molecule is
    # built from standard residues and its codes legitimately collide with the
    # protein's (1FTJ's ligand is glutamate, code 9; 1LAF's is an amino acid,
    # code 3; 1D8E's is a hexapeptide spanning codes 6/12/14/15/19/22).  That is
    # a chemistry class, not a boundary error, so the two cases are reported
    # separately and neither is called a warning.
    if "atoms_residue" in annotations:
        residue = annotations["atoms_residue"]
        ligand_codes = np.unique(residue[ligand_start:])
        protein_codes = set(np.unique(residue[:ligand_start]).tolist())
        overlap = sorted(set(ligand_codes.tolist()) & protein_codes)
        qc["ligand_residue_type_codes"] = [int(code) for code in ligand_codes]
        qc["ligand_residue_codes_shared_with_protein"] = [int(c) for c in overlap]
        qc["ligand_is_nonstandard_residue_only"] = bool(
            ligand_codes.tolist() == [0] and 0 not in protein_codes
        )
        # A peptide or amino-acid "ligand": built from the same residue types as
        # the protein.  Relevant because a small-molecule MMP corpus probably
        # does not want these, and because their dynamics are not comparable.
        qc["ligand_looks_peptidic"] = bool(overlap)
        # The boundary is independently confirmed only in the small-molecule
        # case, where an independent dataset agrees; for a peptidic ligand there
        # is no second source, which is stated rather than papered over.
        qc["ligand_boundary_independently_confirmed"] = qc[
            "ligand_is_nonstandard_residue_only"
        ]
    if "frames_rmsd_ligand" in observables:
        # Recorded for reuse, NOT used as a check: MISATO's series is referenced
        # to something other than trajectory frame 0 (its frame-0 value is
        # 0.93 / 1.93 / 5.74 A on 5WIJ / 4K6W / 1KTI, not 0), so it cannot be
        # reproduced from the stored coordinates and correlates with no
        # frame-0-referenced RMSD we can compute (measured |r| < 0.33 for
        # raw-displacement, pocket-superposed and ligand-superposed variants).
        theirs = observables["frames_rmsd_ligand"][frame_indices]
        qc["misato_rmsd_series_min"] = float(theirs.min())
        qc["misato_rmsd_series_max"] = float(theirs.max())
    for name in ("frames_bSASA", "frames_distance", "frames_interaction_energy"):
        if name in observables:
            selected = observables[name][frame_indices]
            qc[f"{name}_min"] = float(selected.min())
            qc[f"{name}_max"] = float(selected.max())
    if qc["ligand_static_trajectory"]:
        raise SystemRejected(
            "ligand does not move across the stored frames, so the trajectory "
            "carries no dynamics"
        )
    return arrays, qc


# --------------------------------------------------------------------------- #
# window reconstruction (the documented reuse path)
# --------------------------------------------------------------------------- #
def windows_from_system(
    arrays: dict[str, np.ndarray],
    *,
    context_frames: int = 6,
    future_frames: int = 5,
    frame_stride: int = 1,
    window_stride: int = 10,
    pocket_radius_angstrom: float = 8.0,
    max_pocket_atoms: int = 600,
) -> list[dict[str, np.ndarray]]:
    """Reproduce the project's atomic-window convention from a stored system.

    This is the function that makes the store reusable: any window length is a
    slice of the stored frames plus a Kabsch fit, with no HDF5 access.  It
    reproduces ``mdwm.data.misato_atomic.extract_misato_windows`` exactly --
    same pocket selection at the window anchor, same superposition, same array
    contract -- which ``misato_store_crossvalidate.py`` verifies numerically
    against the existing 3k cache.
    """
    ligand = np.asarray(arrays["ligand_coords"], dtype=np.float64)
    pocket = np.asarray(arrays["pocket_coords"], dtype=np.float64)
    pocket_numbers = np.asarray(arrays["pocket_atomic_numbers"], dtype=np.int64)
    pocket_atom_indices = np.asarray(arrays["pocket_atom_indices"], dtype=np.int64)
    n_frames = ligand.shape[0]
    span = (context_frames - 1 + future_frames) * frame_stride + 1
    windows: list[dict[str, np.ndarray]] = []
    for start in range(0, n_frames - span + 1, window_stride):
        context_indices = [start + i * frame_stride for i in range(context_frames)]
        anchor_index = context_indices[-1]
        future_indices = [
            anchor_index + (i + 1) * frame_stride for i in range(future_frames)
        ]
        anchor_ligand = ligand[anchor_index]
        # Pocket chosen at this window's anchor frame, from the stored superset.
        distances = np.linalg.norm(
            pocket[anchor_index][:, None, :] - anchor_ligand[None, :, :], axis=-1
        ).min(axis=1)
        selected = np.nonzero(distances <= float(pocket_radius_angstrom))[0]
        if selected.size < 3:
            continue
        if selected.size > max_pocket_atoms:
            order = np.argsort(distances[selected])
            selected = selected[np.sort(order[:max_pocket_atoms])]
        reference_pocket = pocket[anchor_index][selected]
        ligand_frames = []
        pocket_frames = []
        for frame_index in context_indices + future_indices:
            rotation, mobile_center, reference_center = kabsch_rotation(
                pocket[frame_index][selected], reference_pocket
            )
            ligand_frames.append(
                (ligand[frame_index] - mobile_center) @ rotation.T + reference_center
            )
            pocket_frames.append(
                (pocket[frame_index][selected] - mobile_center) @ rotation.T
                + reference_center
            )
        ligand_frames = np.asarray(ligand_frames, dtype=np.float32)
        pocket_frames = np.asarray(pocket_frames, dtype=np.float32)
        context_ligand = ligand_frames[:context_frames]
        anchor_aligned = context_ligand[-1]
        windows.append(
            {
                "start_frame": int(start),
                "product_init": anchor_aligned,
                "target_delta": ligand_frames[context_frames:]
                - anchor_aligned[None, :, :],
                "context_ligand_coords": context_ligand,
                "pocket_init": pocket_frames[context_frames - 1],
                "pocket_features": pocket_feature_block(pocket_numbers[selected]),
                "product_features": np.asarray(arrays["product_features"]),
                "edit_features": np.asarray(arrays["edit_features"]),
                "bond_index": np.asarray(arrays["bond_index"]),
                "pocket_atom_indices": pocket_atom_indices[selected],
            }
        )
    return windows


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, buffer_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(buffer_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def is_complete(
    manifest_path: Path,
    npz_path: Path,
    expected_config_hash: str,
    expected_frames: int,
    *,
    trust_manifest: bool,
) -> tuple[bool, str]:
    """Content-verified completeness.

    Existence proves nothing -- a killed run is exactly what leaves a
    non-empty file behind -- so a system counts as done only when the manifest
    claims it, the extraction config is the same one, the frame count is the one
    asked for, and the bytes on disk still hash to what was recorded.
    """
    if not manifest_path.is_file():
        return False, "no manifest entry"
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "manifest entry unreadable"
    status = record.get("status")
    if status == "excluded_leakage":
        # Leakage is a property of the system, not of the extraction settings.
        return True, str(status)
    if status == "rejected":
        # A rejection CAN be settings-dependent ("100 frames cannot be supplied
        # at stride 2"), so it only stands for the configuration that produced
        # it; under new settings the system is retried.
        if record.get("config_hash") == expected_config_hash:
            return True, "rejected"
        return False, "rejected under a different extraction config"
    if status != "written":
        return False, f"status {status}"
    if record.get("config_hash") != expected_config_hash:
        return False, "extraction config changed"
    if int(record.get("qc", {}).get("n_frames", -1)) != int(expected_frames):
        return False, "recorded frame count differs from the requested one"
    if not npz_path.is_file():
        return False, "output file is missing"
    if int(npz_path.stat().st_size) != int(record.get("npz_bytes", -1)):
        return False, "output size differs from the recorded size"
    if trust_manifest:
        return True, "written (size checked, hash trusted)"
    if sha256_file(npz_path) != record.get("npz_sha256"):
        return False, "output sha256 differs from the recorded digest"
    return True, "written (sha256 verified)"


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _worker_init(hdf5_path: str, rdcc_bytes: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER["handle"] = h5py.File(
        hdf5_path, "r", rdcc_nbytes=int(rdcc_bytes), rdcc_nslots=10007
    )


def _worker_task(task: dict[str, Any]) -> dict[str, Any]:
    system_id = str(task["system_id"])
    npz_path = Path(task["npz_path"])
    manifest_path = Path(task["manifest_path"])
    started = time.perf_counter()
    base = {
        "system_id": system_id,
        "split": task.get("split"),
        "config_hash": task["config_hash"],
        "config_name": task["config_name"],
        "source_hdf5": task["hdf5_path"],
        "source_url": MISATO_URL,
        "host": socket.gethostname(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        arrays, qc = extract_system(
            _WORKER["handle"], system_id, **task["extract_kwargs"]
        )
    except SystemRejected as error:
        record = {**base, "status": "rejected", "reason": str(error)}
        atomic_write_json(manifest_path, record)
        return {"system_id": system_id, "status": "rejected", "reason": str(error)}
    except Exception as error:  # noqa: BLE001
        # A fault of this run, not of the data: leave NO manifest entry so a
        # re-run retries the system, and never let it count as a data rejection.
        return {
            "system_id": system_id,
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}"[:400],
        }

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = npz_path.with_name(npz_path.name + f".tmp{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, npz_path)
    except Exception as error:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        return {
            "system_id": system_id,
            "status": "failed",
            "reason": f"write failed: {type(error).__name__}: {error}"[:400],
        }

    record = {
        **base,
        "status": "written",
        "npz_path": str(npz_path),
        "npz_bytes": int(npz_path.stat().st_size),
        "npz_sha256": sha256_file(npz_path),
        "array_shapes": {
            key: list(np.asarray(value).shape) for key, value in arrays.items()
        },
        "array_dtypes": {
            key: str(np.asarray(value).dtype) for key, value in arrays.items()
        },
        "qc": qc,
        "wall_seconds": float(time.perf_counter() - started),
    }
    # The manifest entry is written LAST and atomically: it is the only thing
    # that marks a system done, so it can never describe a file that is not
    # fully on disk.
    atomic_write_json(manifest_path, record)
    return {
        "system_id": system_id,
        "status": "written",
        "npz_bytes": record["npz_bytes"],
        "wall_seconds": record["wall_seconds"],
        "n_pocket_atoms": qc["n_pocket_atoms"],
        "n_ligand_atoms": qc["n_ligand_atoms"],
    }


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def _process_tree_rss_bytes(root_pid: int) -> int:
    """Resident memory of this process and its children, from /proc only."""
    total = 0
    try:
        import psutil  # local import: the watchdog is the only user

        process = psutil.Process(root_pid)
        for target in [process] + process.children(recursive=True):
            try:
                total += int(target.memory_info().rss)
            except psutil.Error:
                continue
        return total
    except Exception:  # noqa: BLE001
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                statm = (entry / "statm").read_text().split()
                total += int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, IndexError, ValueError):
                continue
        return total


def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


class ResourceWatchdog(threading.Thread):
    """Pause, then stop, if memory or disk crosses its threshold.

    Deliberately conservative: this box is shared, so the trigger is both
    "am I using too much" (worker-group RSS) and "is the machine short"
    (``MemAvailable`` and free disk), and the response is to stop the pool
    rather than to keep going and hope.
    """

    def __init__(
        self,
        *,
        rss_budget_bytes: int,
        min_available_bytes: int,
        disk_path: Path,
        disk_floor_bytes: int,
        stop_event: threading.Event,
        pause_event: threading.Event,
        run_state_path: Path | None = None,
        interval_seconds: float = 5.0,
        grace_seconds: float = 300.0,
    ) -> None:
        super().__init__(name="watchdog", daemon=True)
        self.rss_budget_bytes = rss_budget_bytes
        self.min_available_bytes = min_available_bytes
        self.disk_path = disk_path
        self.disk_floor_bytes = disk_floor_bytes
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.interval_seconds = interval_seconds
        self.grace_seconds = grace_seconds
        self.run_state_path = run_state_path
        self.progress: dict[str, Any] = {}
        self.peak_rss_bytes = 0
        self.reason: str | None = None
        self._violating_since: float | None = None

    def run(self) -> None:
        root = os.getpid()
        while not self.stop_event.is_set():
            rss = _process_tree_rss_bytes(root)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            available = _mem_available_bytes()
            free_disk = shutil.disk_usage(self.disk_path).free
            problems = []
            if rss > self.rss_budget_bytes:
                problems.append(
                    f"worker-group RSS {rss / GIB:.1f} GiB > budget "
                    f"{self.rss_budget_bytes / GIB:.1f} GiB"
                )
            if available < self.min_available_bytes:
                problems.append(
                    f"system MemAvailable {available / GIB:.1f} GiB < floor "
                    f"{self.min_available_bytes / GIB:.1f} GiB"
                )
            if free_disk < self.disk_floor_bytes:
                problems.append(
                    f"free disk {free_disk / GIB:.1f} GiB < floor "
                    f"{self.disk_floor_bytes / GIB:.1f} GiB"
                )
            if problems:
                if self._violating_since is None:
                    self._violating_since = time.monotonic()
                    self.pause_event.set()
                    _log(f"WATCHDOG PAUSE: {'; '.join(problems)}")
                elif time.monotonic() - self._violating_since > self.grace_seconds:
                    self.reason = "; ".join(problems)
                    _log(f"WATCHDOG STOP: {self.reason}")
                    self.stop_event.set()
                    return
            elif self._violating_since is not None:
                self._violating_since = None
                self.pause_event.clear()
                _log("WATCHDOG RESUME: all thresholds satisfied again")
            if self.run_state_path is not None:
                # Liveness is read from HERE, not from a log that keeps its last
                # line forever after the process dies: the pid must still exist
                # and this timestamp must still be moving.
                atomic_write_json(
                    self.run_state_path,
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "heartbeat_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "heartbeat_monotonic": time.monotonic(),
                        "worker_group_rss_bytes": rss,
                        "peak_worker_group_rss_bytes": self.peak_rss_bytes,
                        "system_mem_available_bytes": available,
                        "free_disk_bytes": free_disk,
                        "paused": self.pause_event.is_set(),
                        "stopping": self.stop_event.is_set(),
                        "thresholds": {
                            "rss_budget_bytes": self.rss_budget_bytes,
                            "min_available_bytes": self.min_available_bytes,
                            "disk_floor_bytes": self.disk_floor_bytes,
                        },
                        "progress": dict(self.progress),
                    },
                )
            self.stop_event.wait(self.interval_seconds)


# --------------------------------------------------------------------------- #
# system selection
# --------------------------------------------------------------------------- #
def load_split_map(split_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = split_root / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"MISATO split file is missing: {path}")
        with np.load(path, allow_pickle=True) as archive:
            for identifier in archive["ids"]:
                mapping[str(identifier)] = split
    return mapping


def select_systems(args: argparse.Namespace, split_map: dict[str, str]) -> list[str]:
    source = str(args.systems)
    if source == "all":
        return sorted(split_map)
    if source == "census-pairs":
        import csv

        ids: set[str] = set()
        with Path(args.census_pairs).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ids.add(str(row["pdb_a"]))
                ids.add(str(row["pdb_b"]))
        return sorted(ids)
    path = Path(source)
    if not path.is_file():
        raise SystemExit(
            f"--systems must be 'all', 'census-pairs' or a path to an id list; "
            f"{source!r} is not a file"
        )
    return sorted(
        {line.strip() for line in path.read_text().splitlines() if line.strip()}
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hdf5", type=Path, default=ROOT / "data/MISATO/raw/MD.hdf5")
    parser.add_argument("--store-root", type=Path, default=ROOT / "data/MISATO")
    parser.add_argument("--config-name", default="frames100_pairs_v1")
    parser.add_argument(
        "--systems",
        default="census-pairs",
        help="'all', 'census-pairs', or a path to a newline-separated id list",
    )
    parser.add_argument("--census-pairs", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--limit", type=int, default=None)
    # extraction
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--pocket-radius-angstrom", type=float, default=8.0)
    parser.add_argument("--max-pocket-atoms", type=int, default=2000)
    parser.add_argument("--max-ligand-atoms", type=int, default=300)
    parser.add_argument("--min-ligand-atoms", type=int, default=6)
    parser.add_argument("--broken-bond-angstrom", type=float, default=2.6)
    parser.add_argument("--max-system-atoms", type=int, default=200_000)
    # resources
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--frame-block", type=int, default=10)
    parser.add_argument("--atom-block", type=int, default=4096)
    parser.add_argument("--rdcc-mib", type=int, default=32)
    parser.add_argument("--rss-budget-gib", type=float, default=24.0)
    parser.add_argument("--min-available-gib", type=float, default=100.0)
    parser.add_argument("--disk-floor-gib", type=float, default=25.0)
    parser.add_argument("--watchdog-grace-seconds", type=float, default=300.0)
    parser.add_argument(
        "--trust-manifest",
        action="store_true",
        help="skip re-hashing existing outputs on resume (size and shapes are "
        "still checked); the default re-hashes, which is the honest test",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "torch" in sys.modules:
        raise SystemExit("torch is imported; this extraction must not touch a GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be empty for this extraction")
    if not args.hdf5.is_file():
        raise SystemExit(
            f"{args.hdf5} is not present. Run scripts/misato_store_download.py "
            "first; the raw file only appears under its final name after its MD5 "
            "matches the publisher's."
        )

    store_root = Path(args.store_root)
    out_dir = store_root / "extracted" / args.config_name
    manifest_dir = store_root / "manifest" / args.config_name
    systems_dir = manifest_dir / "systems"
    out_dir.mkdir(parents=True, exist_ok=True)
    systems_dir.mkdir(parents=True, exist_ok=True)

    extract_kwargs = {
        "n_frames": int(args.frames),
        "frame_stride": int(args.frame_stride),
        "pocket_radius_angstrom": float(args.pocket_radius_angstrom),
        "max_pocket_atoms": int(args.max_pocket_atoms),
        "max_ligand_atoms": int(args.max_ligand_atoms),
        "min_ligand_atoms": int(args.min_ligand_atoms),
        "broken_bond_angstrom": float(args.broken_bond_angstrom),
        "frame_block": int(args.frame_block),
        "atom_block": int(args.atom_block),
        "max_system_atoms": int(args.max_system_atoms),
    }
    # The config hash covers only what changes the OUTPUT, so re-tuning workers
    # or block sizes does not invalidate finished systems.
    hashed_config = {
        key: extract_kwargs[key]
        for key in (
            "n_frames", "frame_stride", "pocket_radius_angstrom", "max_pocket_atoms",
            "max_ligand_atoms", "min_ligand_atoms", "broken_bond_angstrom",
        )
    }
    hashed_config["extractor"] = "misato_store_extract.extract_system/v4"
    cfg_hash = config_hash(hashed_config)

    split_map = load_split_map(Path(args.split_root))
    requested = select_systems(args, split_map)
    exclusion_record = json.loads(Path(args.exclusions).read_text(encoding="utf-8"))
    excluded = {str(value) for value in exclusion_record["excluded_pdb_ids"]}

    _log(
        f"config {args.config_name} (hash {cfg_hash}): {len(requested)} systems "
        f"requested, {args.frames} frames at stride {args.frame_stride}"
    )

    # Exclusions are RECORDED, not silently dropped.
    n_excluded = 0
    for system_id in requested:
        if system_id not in excluded:
            continue
        n_excluded += 1
        manifest_path = systems_dir / f"{system_id}.json"
        if not manifest_path.is_file():
            atomic_write_json(
                manifest_path,
                {
                    "system_id": system_id,
                    "split": split_map.get(system_id),
                    "status": "excluded_leakage",
                    "reason": "listed in the MISATO leakage exclusion set",
                    "exclusion_list": str(args.exclusions),
                    "exclusion_list_sha256": hashlib.sha256(
                        Path(args.exclusions).read_bytes()
                    ).hexdigest(),
                    "config_name": args.config_name,
                    "config_hash": cfg_hash,
                    "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )

    candidates = [s for s in requested if s not in excluded]
    unknown = [s for s in candidates if s not in split_map]
    if unknown:
        _log(f"{len(unknown)} requested systems are not in the MISATO split lists")

    # A killed run leaves `<id>.npz.tmp<pid>` behind.  No completeness check
    # ever looks at those, so they are harmless, but they are dead disk: sweep
    # the ones whose writing process is gone (a live pid means a concurrent run,
    # which is left alone).
    swept = 0
    for stale in out_dir.glob("*.npz.tmp*"):
        suffix = stale.name.rsplit(".tmp", 1)[-1]
        if suffix.isdigit() and Path(f"/proc/{suffix}").exists():
            continue
        try:
            stale.unlink()
            swept += 1
        except OSError:
            pass
    if swept:
        _log(f"swept {swept} orphaned .npz.tmp files from a previously killed run")

    pending: list[str] = []
    already: dict[str, int] = {}
    for system_id in candidates:
        complete, why = is_complete(
            systems_dir / f"{system_id}.json",
            out_dir / f"{system_id}.npz",
            cfg_hash,
            int(args.frames),
            trust_manifest=bool(args.trust_manifest),
        )
        if complete:
            already[why] = already.get(why, 0) + 1
        else:
            pending.append(system_id)
    _log(
        f"{n_excluded} excluded (recorded), {len(candidates) - len(pending)} already "
        f"complete {already}, {len(pending)} pending"
    )
    if args.limit is not None:
        pending = pending[: int(args.limit)]
        _log(f"--limit {args.limit}: {len(pending)} systems this run")

    # Memory arithmetic, stated rather than hoped for.
    per_worker_bytes = (
        args.frame_block * args.max_system_atoms * 12  # protein frame block
        + args.atom_block * args.max_ligand_atoms * 8  # distance temporary
        + args.frames * args.max_ligand_atoms * 12  # ligand trajectory
        + args.frames * args.max_pocket_atoms * 12  # pocket trajectory
        + args.rdcc_mib * (1 << 20)  # HDF5 chunk cache
    )
    _log(
        f"per-worker worst-case array footprint "
        f"{per_worker_bytes / GIB:.2f} GiB + ~0.2 GiB interpreter; "
        f"{args.workers} workers => "
        f"{(per_worker_bytes + 0.2 * GIB) * args.workers / GIB:.2f} GiB; "
        f"RSS budget {args.rss_budget_gib} GiB, MemAvailable floor "
        f"{args.min_available_gib} GiB, disk floor {args.disk_floor_gib} GiB"
    )
    if args.dry_run:
        _log("--dry-run: stopping before extraction")
        return

    tasks = [
        {
            "system_id": system_id,
            "split": split_map.get(system_id),
            "npz_path": str(out_dir / f"{system_id}.npz"),
            "manifest_path": str(systems_dir / f"{system_id}.json"),
            "hdf5_path": str(args.hdf5),
            "config_hash": cfg_hash,
            "config_name": args.config_name,
            "extract_kwargs": extract_kwargs,
        }
        for system_id in pending
    ]

    stop_event = threading.Event()
    pause_event = threading.Event()

    def _signal_handler(signum: int, _frame: Any) -> None:
        _log(f"signal {signum}: stopping after the systems in flight")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    watchdog = ResourceWatchdog(
        rss_budget_bytes=int(args.rss_budget_gib * GIB),
        min_available_bytes=int(args.min_available_gib * GIB),
        disk_path=out_dir,
        disk_floor_bytes=int(args.disk_floor_gib * GIB),
        stop_event=stop_event,
        pause_event=pause_event,
        run_state_path=manifest_dir / "run_state.json",
        grace_seconds=float(args.watchdog_grace_seconds),
    )

    counts = {"written": 0, "rejected": 0, "failed": 0}
    failures: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    written_bytes = 0
    started = time.monotonic()
    last_report = started

    if tasks:
        watchdog.start()
        context = mp.get_context("spawn")
        pool = context.Pool(
            processes=max(1, int(args.workers)),
            initializer=_worker_init,
            initargs=(str(args.hdf5), int(args.rdcc_mib) * (1 << 20)),
            maxtasksperchild=200,
        )
        try:
            for record in pool.imap_unordered(_worker_task, tasks, chunksize=1):
                while pause_event.is_set() and not stop_event.is_set():
                    # Consuming no further results backpressures the pool, so
                    # the workers stop advancing while the machine is short.
                    stop_event.wait(2.0)
                status = record["status"]
                counts[status] = counts.get(status, 0) + 1
                if status == "written":
                    written_bytes += int(record["npz_bytes"])
                elif status == "failed":
                    failures.append(record)
                    _log(f"FAILED {record['system_id']}: {record['reason']}")
                else:
                    rejections.append(record)
                done = sum(counts.values())
                watchdog.progress = {
                    "systems_this_run": len(tasks),
                    "systems_done_this_run": done,
                    "counts": dict(counts),
                    "written_bytes_this_run": written_bytes,
                }
                now = time.monotonic()
                if now - last_report > 30.0 or done == len(tasks):
                    rate = done / max(now - started, 1e-9)
                    _log(
                        f"{done}/{len(tasks)} ({counts['written']} written, "
                        f"{counts['rejected']} rejected, {counts['failed']} failed), "
                        f"{rate:.2f} sys/s, {written_bytes / GIB:.2f} GiB written, "
                        f"peak group RSS {watchdog.peak_rss_bytes / GIB:.2f} GiB"
                    )
                    last_report = now
                if stop_event.is_set():
                    _log("stop requested; terminating the pool")
                    break
        finally:
            pool.terminate()
            pool.join()
            stop_event.set()
            watchdog.join(timeout=10.0)

    summary = build_summary(
        store_root=store_root,
        config_name=args.config_name,
        cfg_hash=cfg_hash,
        hashed_config=hashed_config,
        runtime_config={
            "workers": args.workers,
            "frame_block": args.frame_block,
            "atom_block": args.atom_block,
            "rdcc_mib": args.rdcc_mib,
            "rss_budget_gib": args.rss_budget_gib,
            "min_available_gib": args.min_available_gib,
            "disk_floor_gib": args.disk_floor_gib,
            "per_worker_worst_case_bytes": per_worker_bytes,
            "peak_group_rss_bytes": watchdog.peak_rss_bytes,
        },
        systems_requested=requested,
        n_excluded=n_excluded,
        exclusions_path=Path(args.exclusions),
        hdf5_path=Path(args.hdf5),
        run_counts=counts,
        failures=failures,
        watchdog_reason=watchdog.reason,
        wall_seconds=time.monotonic() - started,
    )
    _log(json.dumps({k: v for k, v in summary.items() if k != "qc_aggregate"}, indent=2))
    if failures:
        _log(f"{len(failures)} systems failed transiently; re-run to retry them")
        sys.exit(1)
    if watchdog.reason:
        sys.exit(3)


def build_summary(
    *,
    store_root: Path,
    config_name: str,
    cfg_hash: str,
    hashed_config: dict[str, Any],
    runtime_config: dict[str, Any],
    systems_requested: Iterable[str],
    n_excluded: int,
    exclusions_path: Path,
    hdf5_path: Path,
    run_counts: dict[str, int],
    failures: list[dict[str, Any]],
    watchdog_reason: str | None,
    wall_seconds: float,
) -> dict[str, Any]:
    """Merge the per-system manifest into one JSONL plus a summary.

    The per-system files are the durable state; this is a derived view, rebuilt
    from scratch every run so it can never disagree with them.
    """
    manifest_dir = store_root / "manifest" / config_name
    systems_dir = manifest_dir / "systems"
    records = []
    for path in sorted(systems_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    jsonl = manifest_dir / "manifest.jsonl"
    temporary = jsonl.with_name(jsonl.name + f".tmp{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, jsonl)

    by_status: dict[str, int] = {}
    for record in records:
        by_status[str(record.get("status"))] = by_status.get(str(record.get("status")), 0) + 1
    written = [r for r in records if r.get("status") == "written"]
    qc_keys = (
        "n_pocket_atoms",
        "n_ligand_atoms",
        "min_pocket_ligand_distance_angstrom",
        "bond_length_max_angstrom",
        "ligand_rmsd_last_vs_first_angstrom",
        "ligand_max_displacement_angstrom",
        "ligand_centroid_step_max_angstrom",
    )
    qc_aggregate: dict[str, Any] = {}
    for key in qc_keys:
        values = np.asarray(
            [float(r["qc"][key]) for r in written if key in r.get("qc", {})],
            dtype=np.float64,
        )
        if values.size:
            qc_aggregate[key] = {
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
    flags = {
        "clash_warning": int(sum(1 for r in written if r["qc"].get("clash_warning"))),
        "bond_length_warning": int(
            sum(1 for r in written if r["qc"].get("bond_length_warning"))
        ),
        "pocket_truncated_at_cap": int(
            sum(1 for r in written if r["qc"].get("pocket_truncated_at_cap"))
        ),
        "ligand_boundary_independently_confirmed": int(
            sum(
                1
                for r in written
                if r["qc"].get("ligand_boundary_independently_confirmed")
            )
        ),
        "ligand_looks_peptidic": int(
            sum(1 for r in written if r["qc"].get("ligand_looks_peptidic"))
        ),
        "periodic_image_jump_suspected": int(
            sum(1 for r in written if r["qc"].get("periodic_image_jump_suspected"))
        ),
    }
    total_bytes = int(sum(int(r.get("npz_bytes", 0)) for r in written))
    summary = {
        "status": "misato_store_extraction_summary",
        "config_name": config_name,
        "config_hash": cfg_hash,
        "extraction_config": hashed_config,
        "runtime_config": runtime_config,
        "source_hdf5": str(hdf5_path),
        "source_url": MISATO_URL,
        "leakage_exclusion_path": str(exclusions_path),
        "leakage_exclusion_sha256": hashlib.sha256(
            exclusions_path.read_bytes()
        ).hexdigest(),
        "n_systems_requested": int(len(list(systems_requested))),
        "n_systems_excluded_for_leakage": int(n_excluded),
        "manifest_status_counts": by_status,
        "this_run_counts": run_counts,
        "this_run_failures": failures[:50],
        "watchdog_stop_reason": watchdog_reason,
        "n_written_total": int(len(written)),
        "written_bytes_total": total_bytes,
        "written_gib_total": float(total_bytes / GIB),
        "mean_npz_bytes": float(total_bytes / len(written)) if written else None,
        "qc_aggregate": qc_aggregate,
        "qc_flag_counts": flags,
        "wall_seconds_this_run": float(wall_seconds),
        "manifest_jsonl": str(jsonl),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(manifest_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
