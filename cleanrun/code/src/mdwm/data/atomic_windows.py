"""Leakage-safe atomic trajectory windows for fixed-topology MEHF pretraining."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@lru_cache(maxsize=32)
def _cached_universe(topology_path: str, trajectory_path: str) -> Any:
    """Reuse parsed topologies/readers in the single-worker training process."""
    import MDAnalysis as mda

    return mda.Universe(topology_path, trajectory_path)


def _molecule_split(
    molecule_id: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float | None = None,
) -> str:
    value = int(hashlib.sha256(f"{seed}|{molecule_id}".encode("utf-8")).hexdigest()[:16], 16) / 16**16
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    if test_fraction is None or value < train_fraction + val_fraction + test_fraction:
        return "test"
    return "lockbox"


def build_atomic_window_index(cfg: dict[str, Any]) -> dict[str, Any]:
    """Index windows while keeping every replica/window of a molecule in one split."""
    data_cfg = cfg.get("data", {})
    window_cfg = cfg.get("atomic_windows", {})
    manifest_path = Path(data_cfg["md_run_input_manifest"])
    output_root = Path(window_cfg.get("output_root", "outputs/atomic_window_index"))
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty atomic-window index: {output_root}")
    manifest = pd.read_csv(manifest_path)
    required = {
        "molecule_id", "target_id", "replica", "smiles", "pose_path",
        "topology_output", "trajectory_output",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"MD manifest is missing columns: {missing}")
    context_frames = int(window_cfg.get("context_frames", 6))
    future_frames = int(window_cfg.get("future_frames", 5))
    frame_stride = int(window_cfg.get("frame_stride", 1))
    window_stride = int(window_cfg.get("window_stride", 10))
    if min(context_frames, future_frames, frame_stride, window_stride) < 1:
        raise ValueError("Atomic window sizes and strides must be positive")
    train_fraction = float(window_cfg.get("train_fraction", 0.7))
    val_fraction = float(window_cfg.get("val_fraction", 0.15))
    test_fraction_value = window_cfg.get("test_fraction")
    test_fraction = None if test_fraction_value is None else float(test_fraction_value)
    assigned_fraction = train_fraction + val_fraction + (0.0 if test_fraction is None else test_fraction)
    if (
        not 0 < train_fraction < 1
        or not 0 <= val_fraction < 1
        or (test_fraction is not None and not 0 < test_fraction < 1)
        or assigned_fraction >= 1
    ):
        raise ValueError("Invalid molecule-level split fractions")
    seed = int(window_cfg.get("split_seed", 2026))
    target_ids = [str(value) for value in window_cfg.get("target_ids", [])]
    if target_ids:
        manifest = manifest[manifest["target_id"].astype(str).isin(target_ids)].copy()

    import MDAnalysis as mda

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for run in manifest.itertuples(index=False):
        topology = Path(str(run.topology_output))
        trajectory = Path(str(run.trajectory_output))
        pose = Path(str(run.pose_path))
        missing_paths = [str(path) for path in (topology, trajectory, pose) if not path.is_file()]
        if missing_paths:
            rejected.append({"molecule_id": str(run.molecule_id), "reason": "missing_artifact", "paths": "|".join(missing_paths)})
            continue
        try:
            universe = mda.Universe(str(topology), str(trajectory))
            n_frames = len(universe.trajectory)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"molecule_id": str(run.molecule_id), "reason": "invalid_trajectory", "paths": str(exc)[:300]})
            continue
        needed_span = (context_frames - 1 + future_frames) * frame_stride + 1
        if n_frames < needed_span:
            rejected.append({"molecule_id": str(run.molecule_id), "reason": "too_short", "paths": str(n_frames)})
            continue
        split = _molecule_split(
            str(run.molecule_id), seed, train_fraction, val_fraction, test_fraction
        )
        for start in range(0, n_frames - needed_span + 1, window_stride):
            window_id = hashlib.sha256(
                f"{run.target_id}|{run.molecule_id}|{run.replica}|{start}|{context_frames}|{future_frames}|{frame_stride}".encode("utf-8")
            ).hexdigest()[:20]
            rows.append(
                {
                    "window_id": window_id,
                    "target_id": str(run.target_id),
                    "molecule_id": str(run.molecule_id),
                    "replica": int(run.replica),
                    "split": split,
                    "start_frame": start,
                    "context_frames": context_frames,
                    "future_frames": future_frames,
                    "frame_stride": frame_stride,
                    "topology_path": str(topology),
                    "trajectory_path": str(trajectory),
                    "pose_path": str(pose),
                    "smiles": str(run.smiles),
                }
            )
    index = pd.DataFrame(rows)
    if index.empty:
        raise ValueError("No valid atomic trajectory windows were indexed")
    leakage = index.groupby("molecule_id")["split"].nunique()
    if int((leakage > 1).sum()) != 0:
        raise RuntimeError("Molecule leakage detected across atomic-window splits")
    path_leakage = index.groupby("trajectory_path")["split"].nunique()
    if int((path_leakage > 1).sum()) != 0:
        raise RuntimeError("Trajectory leakage detected across atomic-window splits")

    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "atomic_windows.csv"
    rejected_path = output_root / "rejected_runs.csv"
    index.to_csv(index_path, index=False)
    pd.DataFrame(rejected).to_csv(rejected_path, index=False)
    split_molecules = index.groupby("split")["molecule_id"].nunique().to_dict()
    summary = {
        "status": "atomic_pretraining_index_ready",
        "index_only": True,
        "no_coordinate_artifacts_written": True,
        "md_run_input_manifest": str(manifest_path),
        "atomic_windows": str(index_path),
        "rejected_runs": str(rejected_path),
        "n_windows": int(len(index)),
        "n_trajectories": int(index["trajectory_path"].nunique()),
        "n_molecules": int(index["molecule_id"].nunique()),
        "split_window_counts": {str(k): int(v) for k, v in index["split"].value_counts().items()},
        "split_molecule_counts": {str(k): int(v) for k, v in split_molecules.items()},
        "molecule_split_overlap": 0,
        "trajectory_split_overlap": 0,
        "statistical_unit": "molecule; windows and replicas are correlated observations",
        "lockbox_policy": (
            "not configured"
            if test_fraction is None
            else "molecule-disjoint lockbox; do not inspect until architecture and evaluation protocol are frozen"
        ),
        "claim_boundary": "fixed-topology E2 pretraining only; no chemical-edit dynamics label",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def cache_atomic_windows(cfg: dict[str, Any]) -> dict[str, Any]:
    """Materialize aligned development windows while hard-excluding lockbox data."""
    index_cfg = cfg.get("data", {})
    cache_cfg = cfg.get("atomic_window_cache", {})
    index_path = Path(index_cfg["atomic_window_index"])
    output_root = Path(cache_cfg.get("output_root", "data/processed/atomic_window_cache_v1"))
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite atomic-window cache: {output_root}")
    allowed_splits = tuple(str(value) for value in cache_cfg.get("splits", ["train", "val", "test"]))
    if "lockbox" in allowed_splits or bool(cache_cfg.get("lockbox_access_allowed", False)):
        raise ValueError("Development cache command is forbidden from reading the lockbox split")
    if not allowed_splits or not set(allowed_splits).issubset({"train", "val", "test"}):
        raise ValueError("Cache splits must be a non-empty subset of train/val/test")
    frame = pd.read_csv(index_path)
    selected = frame[frame["split"].astype(str).isin(allowed_splits)].copy()
    if selected.empty:
        raise ValueError("No development windows selected for caching")
    if bool((selected["split"].astype(str) == "lockbox").any()):
        raise RuntimeError("Lockbox row reached development cache despite the hard gate")
    max_storage_gb = float(cache_cfg.get("max_new_storage_gb", 2.0))
    cache_dir = output_root / "windows"
    cache_dir.mkdir(parents=True, exist_ok=False)
    cached_paths: list[str] = []
    cached_bytes = 0
    try:
        for row in selected.itertuples(index=False):
            item = extract_atomic_window(pd.Series(row._asdict()))
            cache_path = cache_dir / f"{item['window_id']}.npz"
            if cache_path.exists():
                raise FileExistsError(f"Duplicate cache window id: {item['window_id']}")
            np.savez_compressed(
                cache_path,
                product_init=item["product_init"],
                target_delta=item["target_delta"],
                context_ligand_coords=item["context_ligand_coords"],
                product_features=item["product_features"],
                edit_features=item["edit_features"],
                pocket_init=item["pocket_init"],
                pocket_features=item["pocket_features"],
                bond_index=item["bond_index"],
            )
            cached_paths.append(str(cache_path))
            cached_bytes += cache_path.stat().st_size
            current_gb = cached_bytes / 1024.0**3
            if current_gb > max_storage_gb:
                raise RuntimeError(f"Atomic cache exceeded storage cap {max_storage_gb} GB")
        selected["cache_path"] = cached_paths
        manifest_path = output_root / "atomic_windows_cached.csv"
        selected.to_csv(manifest_path, index=False)
        total_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
        summary = {
            "status": "development_atomic_window_cache_ready",
            "source_index": str(index_path),
            "cached_index": str(manifest_path),
            "cached_splits": list(allowed_splits),
            "lockbox_accessed": False,
            "n_cached_windows": int(len(selected)),
            "n_cached_molecules": int(selected["molecule_id"].nunique()),
            "split_window_counts": {
                str(key): int(value) for key, value in selected["split"].value_counts().items()
            },
            "storage_gb": float(total_bytes / 1024.0**3),
            "max_new_storage_gb": max_storage_gb,
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary"] = str(summary_path)
        return summary
    except Exception:
        failure_path = output_root / "failed.json"
        failure_path.write_text(
            json.dumps({"status": "failed", "lockbox_accessed": False}, indent=2), encoding="utf-8"
        )
        raise


def build_atomic_exclusion_index(cfg: dict[str, Any]) -> dict[str, Any]:
    """Filter a cached E2 index so no E3 endpoint dynamics reach the base model."""
    exclusion_cfg = cfg.get("atomic_exclusion_index", {})
    source_path = Path(exclusion_cfg["source_atomic_window_index"])
    endpoint_value = exclusion_cfg.get(
        "endpoint_index", exclusion_cfg.get("edit_window_index")
    )
    if endpoint_value is None:
        raise ValueError("atomic_exclusion_index requires endpoint_index or edit_window_index")
    edit_path = Path(endpoint_value)
    output_root = Path(exclusion_cfg["output_root"])
    allowed_parent = Path(exclusion_cfg.get("allowed_output_parent", "data/processed")).resolve()
    resolved_output = output_root.resolve()
    if resolved_output != allowed_parent and allowed_parent not in resolved_output.parents:
        raise ValueError("Atomic exclusion output escapes allowed_output_parent")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite atomic exclusion index: {output_root}")
    for path, label in ((source_path, "source atomic index"), (edit_path, "edit-window index")):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty {label}: {path}")
    source = pd.read_csv(source_path)
    edits = (
        pd.read_json(edit_path, lines=True)
        if edit_path.suffix.lower() in {".jsonl", ".ndjson"}
        else pd.read_csv(edit_path)
    )
    source_required = {"window_id", "molecule_id", "split", "trajectory_path", "cache_path"}
    edit_required = {"start_molecule_id", "end_molecule_id"}
    if not source_required.issubset(source.columns):
        raise ValueError(f"Source atomic index lacks columns: {sorted(source_required - set(source.columns))}")
    if not edit_required.issubset(edits.columns):
        raise ValueError(f"Edit index lacks columns: {sorted(edit_required - set(edits.columns))}")
    exclude_splits = exclusion_cfg.get("exclude_endpoint_splits")
    if exclude_splits is not None:
        if "split" not in edits:
            raise ValueError("exclude_endpoint_splits requires a split column in endpoint_index")
        requested_splits = [str(value) for value in exclude_splits]
        invalid_splits = sorted(set(requested_splits) - {"train", "val", "test"})
        if not requested_splits or invalid_splits:
            raise ValueError(
                f"exclude_endpoint_splits must be non-empty train/val/test values: {invalid_splits}"
            )
        edits = edits[edits["split"].astype(str).isin(requested_splits)].copy()
        if edits.empty:
            raise ValueError("No endpoint rows remain after exclude_endpoint_splits")
    else:
        requested_splits = None
    excluded = set(edits["start_molecule_id"].astype(str)) | set(
        edits["end_molecule_id"].astype(str)
    )
    retained = source[~source["molecule_id"].astype(str).isin(excluded)].copy()
    removed = source[source["molecule_id"].astype(str).isin(excluded)].copy()
    if retained.empty:
        raise ValueError("Endpoint exclusion removed every atomic window")
    if set(retained["split"].astype(str)) != {"train", "val", "test"}:
        raise ValueError("Endpoint exclusion must leave non-empty train/val/test splits")
    if int((retained.groupby("molecule_id")["split"].nunique() > 1).sum()) != 0:
        raise ValueError("Molecule leakage remains after endpoint exclusion")
    if int((retained.groupby("trajectory_path")["split"].nunique() > 1).sum()) != 0:
        raise ValueError("Trajectory leakage remains after endpoint exclusion")
    remaining_overlap = sorted(set(retained["molecule_id"].astype(str)) & excluded)
    if remaining_overlap:
        raise RuntimeError(f"E3 endpoint molecules survived exclusion: {remaining_overlap}")
    output_root.mkdir(parents=True, exist_ok=False)
    index_name = str(
        exclusion_cfg.get(
            "output_index_name", "atomic_windows_excluding_e3_endpoints.csv"
        )
    )
    if Path(index_name).name != index_name or not index_name.endswith(".csv"):
        raise ValueError("output_index_name must be a plain CSV filename")
    index_path = output_root / index_name
    removed_path = output_root / "excluded_atomic_windows.csv"
    retained.to_csv(index_path, index=False)
    removed.to_csv(removed_path, index=False)
    file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "status": "atomic_base_exclusion_index_ready",
        "development_only": True,
        "formal_inductive_claim_allowed": False,
        "source_atomic_window_index": str(source_path),
        "source_atomic_window_index_sha256": file_sha(source_path),
        "edit_window_index": str(edit_path),
        "edit_window_index_sha256": file_sha(edit_path),
        "endpoint_index": str(edit_path),
        "endpoint_index_sha256": file_sha(edit_path),
        "excluded_endpoint_splits": requested_splits,
        "output_index": str(index_path),
        "output_index_sha256": file_sha(index_path),
        "excluded_windows": str(removed_path),
        "excluded_windows_sha256": file_sha(removed_path),
        "n_exclusion_endpoint_ids": len(excluded),
        "n_excluded_molecules_present_in_e2": int(removed["molecule_id"].nunique()),
        "n_excluded_windows": int(len(removed)),
        "n_retained_windows": int(len(retained)),
        "retained_windows_by_split": {
            str(key): int(value) for key, value in retained["split"].value_counts().items()
        },
        "retained_molecules_by_split": {
            str(key): int(value)
            for key, value in retained.groupby("split")["molecule_id"].nunique().items()
        },
        "e3_endpoint_overlap_after_filter": [],
        "split_assignment": "preserved verbatim from the pre-existing E2 development index",
        "statistical_unit": "molecule",
        "claim_boundary": (
            "Constructed after E3 development inspection; useful for leakage-controlled debugging, "
            "not a prospective external test."
        ),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def _kabsch_transform(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3 or len(mobile) < 3:
        raise ValueError("Alignment selections must contain at least three corresponding 3D atoms")
    mobile_center, reference_center = mobile.mean(axis=0), reference.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, mobile_center, reference_center


def _apply_alignment(
    coordinates: np.ndarray, rotation: np.ndarray, mobile_center: np.ndarray, reference_center: np.ndarray
) -> np.ndarray:
    return (coordinates - mobile_center) @ rotation.T + reference_center


def _ligand_features_from_pose(
    pose_path: Path, topology_elements: list[str]
) -> tuple[np.ndarray, list[tuple[int, int]], list[str]]:
    from rdkit import Chem

    mol = next((mol for mol in Chem.SDMolSupplier(str(pose_path), removeHs=False) if mol is not None), None)
    if mol is None:
        raise ValueError(f"Cannot read ligand pose graph: {pose_path}")
    pose_elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if pose_elements != topology_elements:
        raise ValueError("Pose atom order/elements do not match topology ligand order")
    rows = []
    for atom in mol.GetAtoms():
        hybridization = float(int(atom.GetHybridization())) / 8.0
        rows.append(
            [
                atom.GetAtomicNum() / 100.0,
                atom.GetMass() / 200.0,
                atom.GetFormalCharge() / 4.0,
                float(atom.GetIsAromatic()),
                atom.GetDegree() / 4.0,
                hybridization,
            ]
        )
    names = ["atomic_number", "mass", "formal_charge", "aromatic", "degree", "hybridization"]
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return np.asarray(rows, dtype=np.float32), bonds, names


def _minimum_image(delta: np.ndarray, box_vectors: np.ndarray) -> np.ndarray:
    fractional = delta @ np.linalg.inv(box_vectors)
    return delta - np.round(fractional) @ box_vectors


def _unwrap_bonded_molecule(
    coordinates: np.ndarray, bonds: list[tuple[int, int]], box_vectors: np.ndarray
) -> np.ndarray:
    """Make a bonded molecule whole without changing its internal geometry."""
    adjacency: list[list[int]] = [[] for _ in range(len(coordinates))]
    for atom_i, atom_j in bonds:
        adjacency[atom_i].append(atom_j)
        adjacency[atom_j].append(atom_i)
    unwrapped = np.asarray(coordinates, dtype=np.float64).copy()
    visited = np.zeros(len(coordinates), dtype=bool)
    for root in range(len(coordinates)):
        if visited[root]:
            continue
        visited[root] = True
        stack = [root]
        while stack:
            atom_i = stack.pop()
            for atom_j in adjacency[atom_i]:
                if visited[atom_j]:
                    continue
                delta = _minimum_image(coordinates[atom_j] - coordinates[atom_i], box_vectors)
                unwrapped[atom_j] = unwrapped[atom_i] + delta
                visited[atom_j] = True
                stack.append(atom_j)
    return unwrapped


def _nearest_periodic_image(
    coordinates: np.ndarray, reference_center: np.ndarray, box_vectors: np.ndarray
) -> np.ndarray:
    center_delta = coordinates.mean(axis=0) - reference_center
    shift = np.round(center_delta @ np.linalg.inv(box_vectors)) @ box_vectors
    return coordinates - shift


def _points_near_center(points: np.ndarray, center: np.ndarray, box_vectors: np.ndarray) -> np.ndarray:
    return center + _minimum_image(points - center, box_vectors)


def damped_motion_prior(
    context_coordinates: np.ndarray,
    n_future_frames: int,
    *,
    decay_frames: float = 1.0,
    recent_steps: int = 2,
    future_offsets_frames: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Bounded short-memory displacement prior for overdamped molecular motion.

    The most recent finite-difference velocity is exponentially damped instead
    of extrapolated ballistically.  The learned flow can gate this equivariant
    vector path on or off; it is never used as a target.
    """
    context = np.asarray(context_coordinates, dtype=np.float32)
    if context.ndim != 3 or context.shape[-1] != 3 or len(context) < 2:
        raise ValueError("context_coordinates must have shape (time,atoms,3) with time>=2")
    if n_future_frames < 1 or decay_frames <= 0 or recent_steps < 1:
        raise ValueError("future frames, decay, and recent_steps must be positive")
    if future_offsets_frames is None:
        offsets = np.arange(1, n_future_frames + 1, dtype=np.int64)
    else:
        offsets = np.asarray(future_offsets_frames, dtype=np.int64)
        if offsets.shape != (n_future_frames,) or np.any(offsets < 1):
            raise ValueError("future_offsets_frames must contain one positive offset per future frame")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("future_offsets_frames must be strictly increasing")
    differences = np.diff(context, axis=0)
    last_velocity = differences[-min(int(recent_steps), len(differences)) :].mean(axis=0)
    increments = np.exp(-np.arange(int(offsets[-1]), dtype=np.float32) / float(decay_frames))
    cumulative = np.cumsum(increments)[offsets - 1]
    return cumulative[:, None, None] * last_velocity[None, :, :]


def extract_atomic_window(
    row: pd.Series | dict[str, Any],
    *,
    ligand_selection: str = "resname LIG",
    alignment_selection: str = "protein and backbone",
    pocket_radius_angstrom: float = 8.0,
) -> dict[str, Any]:
    """Read one indexed window and align every frame to the anchor protein frame."""
    from MDAnalysis.lib.distances import distance_array

    values = dict(row)
    universe = _cached_universe(str(values["topology_path"]), str(values["trajectory_path"]))
    ligand = universe.select_atoms(ligand_selection)
    alignment = universe.select_atoms(alignment_selection)
    if len(ligand) == 0:
        raise ValueError("Ligand selection is empty")
    if len(alignment) < 3:
        raise ValueError("Protein alignment selection has fewer than three atoms")
    start = int(values["start_frame"])
    context_frames = int(values["context_frames"])
    future_frames = int(values["future_frames"])
    stride = int(values["frame_stride"])
    context_indices = [start + offset * stride for offset in range(context_frames)]
    anchor_index = context_indices[-1]
    future_indices = [anchor_index + (offset + 1) * stride for offset in range(future_frames)]
    if future_indices[-1] >= len(universe.trajectory):
        raise IndexError("Indexed atomic window extends beyond the trajectory")

    universe.trajectory[anchor_index]
    reference_alignment = alignment.positions.copy()
    topology_elements = [str(value).title() for value in ligand.elements]
    ligand_features, ligand_bonds, ligand_feature_names = _ligand_features_from_pose(
        Path(str(values["pose_path"])), topology_elements
    )
    from MDAnalysis.lib.mdamath import triclinic_vectors

    anchor_box = np.asarray(triclinic_vectors(universe.dimensions), dtype=np.float64)
    anchor_ligand = _unwrap_bonded_molecule(ligand.positions.copy(), ligand_bonds, anchor_box)
    protein_heavy = universe.select_atoms("protein and not element H")
    distances = distance_array(protein_heavy.positions, ligand.positions, box=universe.dimensions)
    pocket_indices = protein_heavy.indices[np.min(distances, axis=1) <= float(pocket_radius_angstrom)]
    pocket = universe.atoms[pocket_indices]
    if len(pocket) == 0:
        raise ValueError("Pocket selection is empty")

    ligand_frames, pocket_frames = [], []
    anchor_ligand_center = anchor_ligand.mean(axis=0)
    for frame_index in context_indices + future_indices:
        universe.trajectory[frame_index]
        box_vectors = np.asarray(triclinic_vectors(universe.dimensions), dtype=np.float64)
        rotation, mobile_center, reference_center = _kabsch_transform(
            alignment.positions.copy(), reference_alignment
        )
        whole_ligand = _unwrap_bonded_molecule(ligand.positions.copy(), ligand_bonds, box_vectors)
        aligned_ligand = _apply_alignment(whole_ligand, rotation, mobile_center, reference_center)
        aligned_box = box_vectors @ rotation.T
        aligned_ligand = _nearest_periodic_image(aligned_ligand, anchor_ligand_center, aligned_box)
        pocket_near_ligand = _points_near_center(
            pocket.positions.copy(), whole_ligand.mean(axis=0), box_vectors
        )
        aligned_pocket = _apply_alignment(pocket_near_ligand, rotation, mobile_center, reference_center)
        aligned_pocket = _points_near_center(aligned_pocket, aligned_ligand.mean(axis=0), aligned_box)
        ligand_frames.append(aligned_ligand)
        pocket_frames.append(aligned_pocket)
    ligand_frames = np.asarray(ligand_frames, dtype=np.float32)
    pocket_frames = np.asarray(pocket_frames, dtype=np.float32)
    context_ligand = ligand_frames[:context_frames]
    future_ligand = ligand_frames[context_frames:]
    anchor = context_ligand[-1]

    pocket_features = np.asarray(
        [[atom.element.capitalize() != "H", atom.mass / 200.0] for atom in pocket], dtype=np.float32
    )
    return {
        "window_id": str(values.get("window_id", "")),
        "molecule_id": str(values["molecule_id"]),
        "replica": int(values["replica"]),
        "split": str(values["split"]),
        "product_init": anchor,
        "target_delta": future_ligand - anchor[None, :, :],
        "future_times_ps": (
            (np.asarray(future_indices, dtype=np.float32) - float(anchor_index))
            * float(universe.trajectory.dt)
        ),
        "context_ligand_coords": context_ligand,
        "pocket_init": pocket_frames[context_frames - 1],
        "pocket_future_coords": pocket_frames[context_frames:],
        "product_features": ligand_features,
        "product_feature_names": ligand_feature_names,
        "bond_index": np.asarray(ligand_bonds, dtype=np.int64),
        "pocket_features": pocket_features,
        "pocket_feature_names": ["is_heavy", "mass"],
        "product_to_parent": np.arange(len(ligand), dtype=np.int64),
        "edit_features": np.column_stack(
            [
                np.ones(len(ligand), dtype=np.float32),
                np.zeros((len(ligand), 4), dtype=np.float32),
            ]
        ),
        "alignment_selection": alignment_selection,
        "pocket_atom_indices": pocket_indices.astype(np.int64),
        "fixed_topology_only": True,
    }


class AtomicWindowDataset:
    """Lazy DCD-backed dataset; coordinates are never duplicated on disk."""

    def __init__(self, index: pd.DataFrame) -> None:
        self.index = index.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.index.iloc[int(item)]
        cache_value = row.get("cache_path")
        if cache_value is None or pd.isna(cache_value) or not str(cache_value):
            return extract_atomic_window(row)
        cache_path = Path(str(cache_value))
        if not cache_path.is_file():
            raise FileNotFoundError(f"Cached atomic window is missing: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if "future_times_ps" not in arrays:
            frames = int(arrays["target_delta"].shape[0])
            arrays["future_times_ps"] = (
                np.arange(1, frames + 1, dtype=np.float32) * float(row.get("frame_stride", 1))
            )
        return {
            "window_id": str(row["window_id"]),
            "molecule_id": str(row["molecule_id"]),
            "replica": int(row["replica"]),
            "split": str(row["split"]),
            **arrays,
            "fixed_topology_only": True,
        }


def enumerate_bond_angles(bond_index: np.ndarray, n_atoms: int) -> np.ndarray:
    """Unique ``(i, j, k)`` bond-angle triples with ``j`` at the vertex.

    The graph is the same undirected bond graph the bond-length term uses.
    Each UNORDERED neighbour pair of a vertex contributes exactly one triple
    (``i < k``), so no angle is double counted and no angle appears with its
    two ends swapped; self-bonds, duplicated bonds and out-of-range indices are
    dropped.  The heavy-atom restriction is the caller's: on the MMP transfer
    corpus ``load_system`` has already dropped hydrogens and remapped
    ``bond_index`` into heavy-atom indexing, so the triples are heavy-atom
    only by construction.
    """
    bonds = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    neighbours: list[set[int]] = [set() for _ in range(int(n_atoms))]
    for a, b in bonds:
        a, b = int(a), int(b)
        if a == b or not (0 <= a < n_atoms and 0 <= b < n_atoms):
            continue
        neighbours[a].add(b)
        neighbours[b].add(a)
    triples: list[tuple[int, int, int]] = []
    for j in range(int(n_atoms)):
        near = sorted(neighbours[j])
        for x in range(len(near)):
            for y in range(x + 1, len(near)):
                triples.append((near[x], j, near[y]))
    if not triples:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(triples, dtype=np.int64)


def collate_atomic_windows(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable ligand/pocket sizes and return a MEHFCondition-ready batch."""
    import torch

    if not items:
        raise ValueError("Cannot collate an empty atomic-window batch")
    batch = len(items)
    frames = int(items[0]["target_delta"].shape[0])
    context_frames = int(items[0].get("context_ligand_coords", np.empty((2, 0, 3))).shape[0])
    atom_feature_dim = int(items[0]["product_features"].shape[1])
    edit_feature_dim = int(items[0]["edit_features"].shape[1])
    pocket_feature_dim = int(items[0]["pocket_features"].shape[1])
    max_atoms = max(int(item["product_init"].shape[0]) for item in items)
    max_context_atoms = max(
        int(item.get("context_ligand_coords", np.empty((context_frames, 0, 3))).shape[1])
        for item in items
    )
    max_pocket = max(int(item["pocket_init"].shape[0]) for item in items)
    max_bonds = max(int(item.get("bond_index", np.empty((0, 2))).shape[0]) for item in items)
    product_init = torch.zeros(batch, max_atoms, 3)
    target_delta = torch.zeros(batch, frames, max_atoms, 3)
    # Parent and product atom counts differ for real chemical edits. Context is
    # padded on its own parent-topology axis; the condition remains product-
    # indexed through the explicit parent_motion_prior.
    context_ligand_coords = torch.zeros(batch, context_frames, max_context_atoms, 3)
    product_features = torch.zeros(batch, max_atoms, atom_feature_dim)
    edit_features = torch.zeros(batch, max_atoms, edit_feature_dim)
    pocket_coords = torch.zeros(batch, max_pocket, 3)
    pocket_features = torch.zeros(batch, max_pocket, pocket_feature_dim)
    atom_mask = torch.zeros(batch, max_atoms, dtype=torch.bool)
    pocket_mask = torch.zeros(batch, max_pocket, dtype=torch.bool)
    bond_index = torch.zeros(batch, max_bonds, 2, dtype=torch.long)
    bond_mask = torch.zeros(batch, max_bonds, dtype=torch.bool)
    reference_bond_lengths = torch.zeros(batch, max_bonds)
    # Bond angles travel the same path as bond lengths: enumerated from the
    # very same `bond_index`, padded on their own axis, masked, and referenced
    # against `product_init` -- the identical frame the reference bond lengths
    # are taken from.
    item_angles = [
        enumerate_bond_angles(
            item.get("bond_index", np.empty((0, 2), dtype=np.int64)),
            int(item["product_init"].shape[0]),
        )
        for item in items
    ]
    max_angles = max(int(triples.shape[0]) for triples in item_angles)
    angle_index = torch.zeros(batch, max_angles, 3, dtype=torch.long)
    angle_mask = torch.zeros(batch, max_angles, dtype=torch.bool)
    reference_bond_angles = torch.zeros(batch, max_angles)
    parent_motion_prior = torch.zeros(batch, frames, max_atoms, 3)
    product_to_parent = torch.full((batch, max_atoms), -1, dtype=torch.long)
    future_times_ps = torch.zeros(batch, frames)
    conditioning_flags = [
        all(
            key in item
            for key in (
                "product_force_prior",
                "product_physics_features",
                "action_features",
            )
        )
        for item in items
    ]
    conditioning_partial = [
        any(
            key in item
            for key in (
                "product_force_prior",
                "product_physics_features",
                "action_features",
            )
        )
        for item in items
    ]
    if any(conditioning_partial) and not all(conditioning_flags):
        raise ValueError(
            "A batch may not mix absent or incomplete MEHF conditioning sidecars"
        )
    product_force_prior = None
    product_physics_features = None
    action_features = None
    if all(conditioning_flags):
        physics_dim = int(items[0]["product_physics_features"].shape[1])
        action_dim = int(items[0]["action_features"].shape[0])
        if physics_dim < 1 or action_dim < 1:
            raise ValueError("MEHF conditioning feature dimensions must be positive")
        product_force_prior = torch.zeros(batch, frames, max_atoms, 3)
        product_physics_features = torch.zeros(batch, max_atoms, physics_dim)
        action_features = torch.zeros(batch, action_dim)
    for index, item in enumerate(items):
        if item["target_delta"].shape[0] != frames:
            raise ValueError("All windows in a batch must share the same future horizon")
        n_atoms, n_pocket = item["product_init"].shape[0], item["pocket_init"].shape[0]
        product_init[index, :n_atoms] = torch.as_tensor(item["product_init"])
        target_delta[index, :, :n_atoms] = torch.as_tensor(item["target_delta"])
        item_times = np.asarray(
            item.get("future_times_ps", np.arange(1, frames + 1, dtype=np.float32)),
            dtype=np.float32,
        )
        if item_times.shape != (frames,) or not np.isfinite(item_times).all() or np.any(item_times <= 0):
            raise ValueError("future_times_ps must contain one finite positive value per future frame")
        future_times_ps[index] = torch.as_tensor(item_times)
        if context_frames:
            if item["context_ligand_coords"].shape[0] != context_frames:
                raise ValueError("All windows in a batch must share the same context length")
            n_context_atoms = int(item["context_ligand_coords"].shape[1])
            context_ligand_coords[index, :, :n_context_atoms] = torch.as_tensor(
                item["context_ligand_coords"]
            )
            if "parent_motion_prior" in item:
                prior = np.asarray(item["parent_motion_prior"], dtype=np.float32)
                if prior.shape != (frames, n_atoms, 3):
                    raise ValueError("Explicit parent_motion_prior must be product-indexed")
            else:
                if n_context_atoms != n_atoms:
                    raise ValueError(
                        "Topology-changing windows require an explicit product-indexed parent_motion_prior"
                    )
                prior = damped_motion_prior(item["context_ligand_coords"], frames)
            parent_motion_prior[index, :, :n_atoms] = torch.as_tensor(prior)
        product_features[index, :n_atoms] = torch.as_tensor(item["product_features"])
        edit_features[index, :n_atoms] = torch.as_tensor(item["edit_features"])
        if product_force_prior is not None:
            force = np.asarray(item["product_force_prior"], dtype=np.float32)
            if force.shape == (n_atoms, 3):
                force = np.broadcast_to(force[None], (frames, n_atoms, 3))
            if force.shape != (frames, n_atoms, 3):
                raise ValueError(
                    "product_force_prior must be static (atoms,3) or path-shaped"
                )
            physics = np.asarray(
                item["product_physics_features"], dtype=np.float32
            )
            action = np.asarray(item["action_features"], dtype=np.float32)
            if physics.shape != (n_atoms, product_physics_features.shape[-1]):
                raise ValueError("Inconsistent product physics feature shape")
            if action.shape != (action_features.shape[-1],):
                raise ValueError("Inconsistent global action feature shape")
            if not (
                np.isfinite(force).all()
                and np.isfinite(physics).all()
                and np.isfinite(action).all()
            ):
                raise ValueError("MEHF conditioning contains non-finite values")
            product_force_prior[index, :, :n_atoms] = torch.as_tensor(
                np.array(force, copy=True)
            )
            product_physics_features[index, :n_atoms] = torch.as_tensor(physics)
            action_features[index] = torch.as_tensor(action)
        mapping = np.asarray(
            item.get("product_to_parent", np.arange(n_atoms, dtype=np.int64)),
            dtype=np.int64,
        )
        if mapping.shape != (n_atoms,):
            raise ValueError("product_to_parent must contain one parent index per product atom")
        mapped = mapping[mapping >= 0]
        if len(np.unique(mapped)) != len(mapped):
            raise ValueError("product_to_parent contains duplicate mapped parent atoms")
        product_to_parent[index, :n_atoms] = torch.as_tensor(mapping)
        pocket_coords[index, :n_pocket] = torch.as_tensor(item["pocket_init"])
        pocket_features[index, :n_pocket] = torch.as_tensor(item["pocket_features"])
        atom_mask[index, :n_atoms] = True
        pocket_mask[index, :n_pocket] = True
        bonds = torch.as_tensor(item.get("bond_index", np.empty((0, 2), dtype=np.int64)), dtype=torch.long)
        if len(bonds):
            n_bonds = len(bonds)
            bond_index[index, :n_bonds] = bonds
            bond_mask[index, :n_bonds] = True
            reference_bond_lengths[index, :n_bonds] = (
                product_init[index, bonds[:, 0]] - product_init[index, bonds[:, 1]]
            ).norm(dim=-1)
        triples = torch.as_tensor(item_angles[index], dtype=torch.long)
        if len(triples):
            n_angles = len(triples)
            angle_index[index, :n_angles] = triples
            angle_mask[index, :n_angles] = True
            left = product_init[index, triples[:, 0]] - product_init[index, triples[:, 1]]
            right = product_init[index, triples[:, 2]] - product_init[index, triples[:, 1]]
            # atan2 of |u x v| against u.v: the same angle as arccos of the
            # normalised dot product, but numerically stable for the near-linear
            # angles (nitriles, alkynes) where arccos loses all its precision.
            reference_bond_angles[index, :n_angles] = torch.atan2(
                torch.linalg.cross(left, right).norm(dim=-1).clamp_min(1.0e-12),
                (left * right).sum(dim=-1),
            )
    result = {
        "window_id": [str(item["window_id"]) for item in items],
        "molecule_id": [str(item["molecule_id"]) for item in items],
        "replica": torch.tensor([int(item["replica"]) for item in items]),
        "split": [str(item["split"]) for item in items],
        "product_init": product_init,
        "target_delta": target_delta,
        "future_times_ps": future_times_ps,
        "context_ligand_coords": context_ligand_coords,
        "product_features": product_features,
        "edit_features": edit_features,
        "product_to_parent": product_to_parent,
        "parent_motion_prior": parent_motion_prior,
        "pocket_coords": pocket_coords,
        "pocket_features": pocket_features,
        "atom_mask": atom_mask,
        "pocket_mask": pocket_mask,
        "bond_index": bond_index,
        "bond_mask": bond_mask,
        "reference_bond_lengths": reference_bond_lengths,
        "angle_index": angle_index,
        "angle_mask": angle_mask,
        "reference_bond_angles": reference_bond_angles,
        "fixed_topology_only": all(bool(item.get("fixed_topology_only", True)) for item in items),
    }
    if product_force_prior is not None:
        result.update(
            {
                "product_force_prior": product_force_prior,
                "product_physics_features": product_physics_features,
                "action_features": action_features,
                "conditioning_path": [
                    str(item.get("conditioning_path", "")) for item in items
                ],
            }
        )
    return result
