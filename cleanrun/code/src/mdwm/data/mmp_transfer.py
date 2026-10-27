"""Parent -> product ensemble-transfer dataset for the MISATO matched pairs.

Implements the tensor contract of
``outputs/mmp_transfer_baselines_v1/PHASE_B_SPEC.md`` and the frozen split
protocol of ``PHASE_C_PREREGISTRATION.md`` AMENDMENT 1.

Three facts about the stored corpus drive the implementation and are recorded
here because none of them is obvious from the spec:

1. **The two systems of a pair do not share a coordinate frame.**  Parent and
   product are different PDB entries with independent MD boxes; measured over a
   random dozen usable pairs, the mapped-atom centroids sit 0.9-25.4 A apart
   and the raw mapped-atom RMSD is 1.6-25.7 A, while a Kabsch superposition on
   the mapped atoms brings it to 0.2-1.8 A.  ``parent_motion_prior`` is a
   *vector* basis, so an un-superposed parent contributes displacement vectors
   expressed in a rotated frame -- noise, not signal.  Every parent ensemble is
   therefore rigidly superposed onto the product's frame-0 pose using the
   mapped heavy atoms only, and the same single transform is applied to all of
   its frames so that the parent's internal motion is preserved exactly.
2. **Output nodes are heavy atoms only.**  The store keeps hydrogens; the
   mapping, the frozen torsion metric and the baselines are all heavy-atom
   quantities.  Bond indices and torsion quadruples are therefore relabelled
   into heavy-local numbering, and the original-numbering mapping is kept
   alongside so evaluation can intersect with the baselines.
3. **Pockets are large.**  MISATO pockets run to 606 atoms; the equivariant
   interaction layers cost ``batch x frames x ligand x pocket x hidden``, which
   at 50 frames is several GB per example.  The pocket is cropped to the
   ``max_pocket_atoms`` atoms closest to the ligand, ranked by the store's own
   ``pocket_min_distance_over_frames``.  This is an implementation budget, not
   part of the spec; it is recorded in every example as ``n_pocket_available``.
4. **The stored frames are not superposed.**  Measured over eight random
   training products, the raw ligand displacement from frame 0 has RMS 2.1-5.6
   A while the ligand's internal RMSF is 0.3-1.2 A: roughly 90% of
   ``target_delta`` built from raw coordinates is whole-box tumbling and
   translation.  The frozen primary metric is a torsion distribution and is
   exactly invariant to that motion, so a model trained on raw displacements
   would spend essentially all of its loss on a quantity the metric ignores.
   Each system is therefore rigidly superposed frame-by-frame onto its own
   frame-0 POCKET (not its own ligand: superposing on the ligand would also
   delete the ligand's real motion relative to the binding site).  After that
   the ligand displacement RMS is 0.7-2.9 A against a pocket residual of
   1.0-1.6 A.  The transform is a rigid motion, so every torsion angle, and
   therefore every number the frozen metric and every baseline produce, is
   unchanged; only the learning problem is made well posed.  Set
   ``frame_alignment="none"`` to recover the raw store.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mdwm.data.atomic_windows import collate_atomic_windows

DEFAULT_STORE = Path("data/MISATO/extracted/frames100_all_v1")
DEFAULT_PAIRS = Path("outputs/mmp_transfer_corpus_v1/pairs.csv")
DEFAULT_MAPPINGS = Path("outputs/mmp_transfer_corpus_v1/atom_mappings.json")

SPLIT_SEED = 20260918          # AMENDMENT 1
N_FRAMES = 50                  # frozen by the metric
PRODUCT_INIT_MODES = ("product_frame0", "parent_mean")
FRAME_ALIGNMENTS = ("pocket", "none")


# --------------------------------------------------------------------------
# corpus and splits
# --------------------------------------------------------------------------
def load_usable_pairs(pairs_csv: str | Path = DEFAULT_PAIRS) -> pd.DataFrame:
    """The 4,913 pairs the pre-registration calls usable."""
    frame = pd.read_csv(pairs_csv)
    frame["pdb_a"] = frame["pdb_a"].astype(str).str.upper()
    frame["pdb_b"] = frame["pdb_b"].astype(str).str.upper()
    usable = frame[
        (frame["status"] == "ok")
        & (frame["transformation_agrees"].astype(bool))
        & (frame["split_a"] == frame["split_b"])
    ].copy()
    return usable.reset_index(drop=True)


def target_disjoint_split(
    pairs: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> pd.DataFrame:
    """AMENDMENT 1's PRIMARY split: 70/15/15 by cumulative pair count.

    Targets are enumerated by descending pair count with ties broken by name
    (the deterministic form of ``value_counts().index``), shuffled with
    ``default_rng(20260918)``, then walked, each target joining whichever split
    the cumulative pair count has not yet filled.  This reproduces the recorded
    counts exactly: train 3,489 / 345 targets, val 696 / 106, test 728 / 66.
    """
    counts = pairs.groupby("uniprot").size().to_dict()
    ordered = [key for key, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    targets = np.array(ordered, dtype=object)
    np.random.default_rng(int(seed)).shuffle(targets)
    total = len(pairs)
    train_limit = train_fraction * total
    val_limit = (train_fraction + val_fraction) * total
    assignment: dict[str, str] = {}
    cumulative = 0
    for target in targets:
        if cumulative < train_limit:
            assignment[target] = "train"
        elif cumulative < val_limit:
            assignment[target] = "val"
        else:
            assignment[target] = "test"
        cumulative += counts[target]
    out = pairs.copy()
    out["primary_split"] = out["uniprot"].map(assignment)
    return out


def grouped_folds(pairs: pd.DataFrame, n_folds: int, *, seed: int) -> pd.Series:
    """Fold id per row, grouped by UniProt, balanced by pair count."""
    counts = pairs.groupby("uniprot").size()
    targets = np.array(sorted(counts.index), dtype=object)
    np.random.default_rng(int(seed)).shuffle(targets)
    load = np.zeros(int(n_folds), dtype=np.int64)
    assignment: dict[str, int] = {}
    for target in targets:
        fold = int(np.argmin(load))
        assignment[target] = fold
        load[fold] += int(counts[target])
    return pairs["uniprot"].map(assignment).astype(int)


def directional_examples(pairs: pd.DataFrame) -> pd.DataFrame:
    """Each unordered pair becomes two examples, a->b and b->a."""
    forward = pairs.copy()
    forward["parent"] = forward["pdb_a"]
    forward["product"] = forward["pdb_b"]
    forward["direction"] = "ab"
    reverse = pairs.copy()
    reverse["parent"] = reverse["pdb_b"]
    reverse["product"] = reverse["pdb_a"]
    reverse["direction"] = "ba"
    both = pd.concat([forward, reverse], ignore_index=True)
    both["pair_key"] = both["pdb_a"] + "|" + both["pdb_b"]
    both["example_id"] = both["parent"] + "->" + both["product"]
    return both


# --------------------------------------------------------------------------
# store access
# --------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _load_system_cached(path: str, frame_alignment: str) -> dict[str, Any]:
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if frame_alignment not in FRAME_ALIGNMENTS:
        raise ValueError(f"frame_alignment must be one of {FRAME_ALIGNMENTS}")
    if frame_alignment == "pocket":
        ligand = arrays["ligand_coords"].astype(np.float64)
        pocket = arrays["pocket_coords"].astype(np.float64)
        aligned_ligand = np.empty_like(ligand)
        aligned_pocket = np.empty_like(pocket)
        aligned_ligand[0], aligned_pocket[0] = ligand[0], pocket[0]
        for frame in range(1, ligand.shape[0]):
            rotation, translation = kabsch_transform(pocket[frame], pocket[0])
            aligned_ligand[frame] = ligand[frame] @ rotation + translation
            aligned_pocket[frame] = pocket[frame] @ rotation + translation
        arrays["ligand_coords"] = aligned_ligand.astype(np.float32)
        arrays["pocket_coords"] = aligned_pocket.astype(np.float32)
    numbers = arrays["ligand_atomic_numbers"]
    heavy = np.flatnonzero(numbers > 1).astype(np.int64)
    position = -np.ones(len(numbers), dtype=np.int64)
    position[heavy] = np.arange(len(heavy), dtype=np.int64)
    bonds = arrays["bond_index"].astype(np.int64)
    keep = (position[bonds[:, 0]] >= 0) & (position[bonds[:, 1]] >= 0)
    local_bonds = position[bonds[keep]]
    return {
        "heavy_indices": heavy,                     # original numbering
        "heavy_position": position,                 # original -> heavy-local, -1 for H
        "coords": arrays["ligand_coords"][:, heavy].astype(np.float32),
        "atomic_numbers": numbers[heavy],
        "bond_index": local_bonds,
        "product_features": arrays["product_features"][heavy].astype(np.float32),
        "pocket_coords": arrays["pocket_coords"].astype(np.float32),
        "pocket_features": arrays["pocket_features"].astype(np.float32),
        "pocket_min_distance": arrays["pocket_min_distance_over_frames"].astype(np.float32),
        "n_frames": int(arrays["ligand_coords"].shape[0]),
    }


def load_system(
    pdb: str,
    store: str | Path = DEFAULT_STORE,
    frame_alignment: str = "pocket",
) -> dict[str, Any]:
    return _load_system_cached(
        str(Path(store) / f"{pdb.upper()}.npz"), str(frame_alignment)
    )


@lru_cache(maxsize=1)
def _load_mappings_cached(path: str) -> dict[str, dict[str, list[int]]]:
    with open(path) as handle:
        return json.load(handle)


def load_mappings(path: str | Path = DEFAULT_MAPPINGS) -> dict[str, dict[str, list[int]]]:
    return _load_mappings_cached(str(path))


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation carrying `mobile` onto `target` (both (n,3))."""
    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (mobile - mobile_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    flip = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation = u @ flip @ vt
    return rotation, target_centre - mobile_centre @ rotation


def _graph_neighbours(n_atoms: int, bonds: np.ndarray) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for a, b in bonds:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    return adjacency


def _graph_distances(n_atoms: int, adjacency: list[list[int]], source: int) -> np.ndarray:
    distance = np.full(n_atoms, np.inf)
    distance[source] = 0.0
    frontier = [source]
    while frontier:
        nxt: list[int] = []
        for node in frontier:
            for neighbour in adjacency[node]:
                if distance[neighbour] == np.inf:
                    distance[neighbour] = distance[node] + 1.0
                    nxt.append(neighbour)
        frontier = nxt
    return distance


def _parent_mean_init(
    product_frame0: np.ndarray,
    parent_mean: np.ndarray,
    product_to_parent: np.ndarray,
    bonds: np.ndarray,
) -> np.ndarray:
    """Regime 1 initial pose: no product coordinates for mapped atoms.

    Mapped atoms take the parent's mean structure.  Each added atom is rebuilt
    from the local frame of its three nearest mapped atoms -- nearest by bond
    graph distance in the product topology, ties broken by product frame-0
    distance -- so its product frame-0 offset is expressed in an internal frame
    and re-realised on the parent mean structure.
    """
    n_atoms = product_frame0.shape[0]
    init = np.zeros_like(product_frame0)
    mapped = product_to_parent >= 0
    init[mapped] = parent_mean[product_to_parent[mapped]]
    added = np.flatnonzero(~mapped)
    if added.size == 0:
        return init
    if int(mapped.sum()) < 3:
        # Degenerate topology: fall back to a rigid placement of the whole
        # product on whatever mapped atoms exist.
        if int(mapped.sum()) == 0:
            return product_frame0.copy()
        offset = init[mapped].mean(axis=0) - product_frame0[mapped].mean(axis=0)
        init[~mapped] = product_frame0[~mapped] + offset
        return init
    adjacency = _graph_neighbours(n_atoms, bonds)
    mapped_indices = np.flatnonzero(mapped)
    for atom in added:
        hops = _graph_distances(n_atoms, adjacency, int(atom))[mapped_indices]
        euclid = np.linalg.norm(
            product_frame0[mapped_indices] - product_frame0[atom], axis=-1
        )
        order = np.lexsort((euclid, hops))[:3]
        anchors = mapped_indices[order]
        source = product_frame0[anchors]
        destination = init[anchors]
        rotation, translation = kabsch_transform(source, destination)
        init[atom] = product_frame0[atom] @ rotation + translation
    return init


# --------------------------------------------------------------------------
# one example
# --------------------------------------------------------------------------
def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def mapped_torsion_indices(
    product: dict[str, Any],
    product_to_parent: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Heavy-local rotatable torsions whose four atoms are all mapped."""
    from mdwm.eval.ensemble_metrics import rotatable_torsions

    quadruples = rotatable_torsions(product["atomic_numbers"], product["bond_index"])
    return [q for q in quadruples if all(product_to_parent[i] >= 0 for i in q)]


def pair_mapping(
    parent_id: str,
    product_id: str,
    pair_key: str,
    direction: str,
    mappings: dict[str, dict[str, list[int]]],
    parent: dict[str, Any],
    product: dict[str, Any],
) -> np.ndarray:
    """product heavy-local index -> parent heavy-local index, -1 when added."""
    record = mappings[pair_key]
    a_indices = np.asarray(record["a_atom_indices"], dtype=np.int64)
    b_indices = np.asarray(record["b_atom_indices"], dtype=np.int64)
    if direction == "ab":
        parent_original, product_original = a_indices, b_indices
    else:
        parent_original, product_original = b_indices, a_indices
    mapping = -np.ones(product["coords"].shape[1], dtype=np.int64)
    parent_local = parent["heavy_position"][parent_original]
    product_local = product["heavy_position"][product_original]
    keep = (parent_local >= 0) & (product_local >= 0)
    mapping[product_local[keep]] = parent_local[keep]
    return mapping


def _edit_masks(
    product_to_parent: np.ndarray,
    product_bonds: np.ndarray,
    parent_bonds: np.ndarray,
    n_parent_atoms: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_atoms = len(product_to_parent)
    mapped = product_to_parent >= 0
    adjacency = _graph_neighbours(n_atoms, product_bonds)
    attachment = np.zeros(n_atoms, dtype=bool)
    for atom in range(n_atoms):
        if mapped[atom] and any(not mapped[n] for n in adjacency[atom]):
            attachment[atom] = True
    parent_adjacency = _graph_neighbours(n_parent_atoms, parent_bonds)
    bond_changed = np.zeros(n_atoms, dtype=bool)
    for atom in range(n_atoms):
        if not mapped[atom]:
            continue
        product_side = {
            int(product_to_parent[n]) for n in adjacency[atom] if mapped[n]
        }
        parent_side = set(parent_adjacency[int(product_to_parent[atom])])
        if product_side != parent_side:
            bond_changed[atom] = True
    deleted = int(n_parent_atoms - int(mapped.sum()))
    return attachment, bond_changed, deleted


POCKET_SELECTIONS = ("min_over_frames", "frame0")


def pocket_ranking_distance(product: dict[str, Any], pocket_selection: str) -> np.ndarray:
    """Per-pocket-atom distance used to choose the cropped pocket atoms.

    ``min_over_frames`` (legacy, default): the store's minimum over all 100
    trajectory frames -- it reads the frames the model is asked to generate.
    ``frame0``: min over ligand heavy atoms of the frame-0 distance only, a pure
    function of frame 0 of ``coords`` and ``pocket_coords``.
    """
    if pocket_selection == "min_over_frames":
        return product["pocket_min_distance"]
    if pocket_selection == "frame0":
        pocket0 = product["pocket_coords"][0].astype(np.float64)
        ligand0 = product["coords"][0].astype(np.float64)
        delta = pocket0[:, None, :] - ligand0[None, :, :]
        return np.sqrt((delta ** 2).sum(-1)).min(axis=1)
    raise ValueError(f"pocket_selection must be one of {POCKET_SELECTIONS}, got {pocket_selection!r}")


def build_example(
    *,
    parent_id: str,
    product_id: str,
    pair_key: str,
    direction: str,
    uniprot: str,
    split_name: str,
    store: str | Path = DEFAULT_STORE,
    mappings: dict[str, dict[str, list[int]]] | None = None,
    product_init_mode: str = "product_frame0",
    n_frames: int = N_FRAMES,
    max_pocket_atoms: int | None = 96,
    pocket_selection: str = "min_over_frames",
    frame_alignment: str = "pocket",
    target_frames: Sequence[int] | None = None,
    parent_frames: Sequence[int] | None = None,
    frame_seed: int = 0,
    include_target: bool = True,
    disable_mapping: bool = False,
) -> dict[str, Any]:
    """One directional example in the padded-batch item format.

    ``disable_mapping`` is the pre-registered NO-MAPPING ablation, exactly as
    PHASE_C_PREREGISTRATION.md words it: ``product_to_parent`` is emitted as -1
    everywhere, so ``parent_motion_prior`` is identically zero and the parent's
    MD ensemble is unreachable by the model.  The real mapping is still used to
    admit the pair and to superpose the parent block, neither of which reaches
    any emitted tensor once the prior is zero.  ``edit_features`` is left as
    computed, because it carries edit TOPOLOGY, not the parent's ensemble; the
    edit channel is a separate Phase D ablation.  This flag is a DATA-path
    switch and the model is retrained under it -- nothing is ever zeroed on a
    trained model (guard 2)."""
    if product_init_mode not in PRODUCT_INIT_MODES:
        raise ValueError(f"product_init_mode must be one of {PRODUCT_INIT_MODES}")
    mappings = load_mappings() if mappings is None else mappings
    parent = load_system(parent_id, store, frame_alignment)
    product = load_system(product_id, store, frame_alignment)
    mapping = pair_mapping(
        parent_id, product_id, pair_key, direction, mappings, parent, product
    )
    if int((mapping >= 0).sum()) < 3:
        raise ValueError(f"{parent_id}->{product_id}: fewer than three mapped atoms")

    if parent_frames is None:
        rng = np.random.default_rng(_stable_seed(pair_key, direction, "parent"))
        parent_frames = np.sort(
            rng.choice(parent["n_frames"], n_frames, replace=False)
        )
    parent_frames = np.asarray(parent_frames, dtype=np.int64)
    if target_frames is None:
        rng = np.random.default_rng(
            _stable_seed(pair_key, direction, "target", frame_seed)
        )
        target_frames = np.sort(
            rng.choice(product["n_frames"], n_frames, replace=False)
        )
    target_frames = np.asarray(target_frames, dtype=np.int64)
    if len(parent_frames) != n_frames or len(target_frames) != n_frames:
        raise ValueError("parent and target frame blocks must both hold n_frames")

    product_frame0 = product["coords"][0].astype(np.float64)
    parent_block = parent["coords"][parent_frames].astype(np.float64)

    # Superpose the whole parent block onto the product's frame-0 pose using the
    # mapped heavy atoms.  One transform for the block, so parent-internal
    # motion is untouched; see the module docstring.
    mapped_product = np.flatnonzero(mapping >= 0)
    mapped_parent = mapping[mapped_product]
    parent_mean_raw = parent_block.mean(axis=0)
    rotation, translation = kabsch_transform(
        parent_mean_raw[mapped_parent], product_frame0[mapped_product]
    )
    parent_block = parent_block @ rotation + translation
    parent_mean = parent_block.mean(axis=0)

    if product_init_mode == "product_frame0":
        product_init = product_frame0
    else:
        product_init = _parent_mean_init(
            product_frame0, parent_mean, mapping, product["bond_index"]
        )

    # parent_motion_prior: displacements relative to the block's first frame,
    # gathered through the map, exactly zero on added atoms.
    displacement = parent_block - parent_block[:1]
    prior = np.zeros((n_frames, len(mapping), 3), dtype=np.float32)
    prior[:, mapped_product] = displacement[:, mapped_parent].astype(np.float32)

    attachment, bond_changed, deleted = _edit_masks(
        mapping, product["bond_index"], parent["bond_index"], parent["coords"].shape[1]
    )
    edit_features = np.column_stack(
        [
            (mapping >= 0).astype(np.float32),
            (mapping < 0).astype(np.float32),
            attachment.astype(np.float32),
            bond_changed.astype(np.float32),
            np.full(len(mapping), np.log1p(float(deleted)), dtype=np.float32),
        ]
    )

    pocket_coords = product["pocket_coords"][0].astype(np.float32)
    pocket_features = product["pocket_features"].astype(np.float32)
    n_pocket_available = int(pocket_coords.shape[0])
    if max_pocket_atoms is not None and n_pocket_available > int(max_pocket_atoms):
        keep = np.argsort(
            pocket_ranking_distance(product, pocket_selection), kind="stable"
        )[: int(max_pocket_atoms)]
        keep = np.sort(keep)
        pocket_coords = pocket_coords[keep]
        pocket_features = pocket_features[keep]

    if disable_mapping:
        mapping = np.full(len(mapping), -1, dtype=mapping.dtype)
        prior = np.zeros_like(prior)

    item: dict[str, Any] = {
        "window_id": f"{parent_id}->{product_id}",
        "molecule_id": product_id,
        "replica": 0,
        "split": split_name,
        "product_init": product_init.astype(np.float32),
        "product_features": product["product_features"],
        "edit_features": edit_features,
        "bond_index": product["bond_index"],
        "product_to_parent": mapping,
        "parent_motion_prior": prior,
        "context_ligand_coords": parent_block.astype(np.float32),
        "pocket_init": pocket_coords,
        "pocket_features": pocket_features,
        # A nominal uniform time axis.  The drawn frame indices are deliberately
        # NOT exposed here: at evaluation they index the reference block, and a
        # model must not be able to read them.
        "future_times_ps": np.arange(1, n_frames + 1, dtype=np.float32),
        "fixed_topology_only": False,
        "parent": parent_id,
        "product": product_id,
        "uniprot": str(uniprot),
        "direction": direction,
        "pair_key": pair_key,
        "parent_frames": parent_frames,
        "target_frames": target_frames,
        "n_pocket_available": n_pocket_available,
        "frame_alignment": str(frame_alignment),
        "pair_work": int(len(mapping) * pocket_coords.shape[0]),
    }
    target_block = product["coords"][target_frames].astype(np.float64)
    if include_target:
        item["target_delta"] = (target_block - product_init[None]).astype(np.float32)
    else:
        item["target_delta"] = np.zeros(
            (n_frames, len(mapping), 3), dtype=np.float32
        )
    return item


class MMPTransferDataset:
    """Directional parent->product examples over a fixed pair table."""

    def __init__(
        self,
        examples: pd.DataFrame,
        *,
        store: str | Path = DEFAULT_STORE,
        mappings_path: str | Path = DEFAULT_MAPPINGS,
        product_init_mode: str = "product_frame0",
        n_frames: int = N_FRAMES,
        max_pocket_atoms: int | None = 96,
        pocket_selection: str = "min_over_frames",
        frame_alignment: str = "pocket",
        split_name: str = "train",
        epoch: int = 0,
        disable_mapping: bool = False,
    ) -> None:
        self.examples = examples.reset_index(drop=True)
        self.store = Path(store)
        self.mappings_path = Path(mappings_path)
        self.product_init_mode = product_init_mode
        self.n_frames = int(n_frames)
        self.max_pocket_atoms = max_pocket_atoms
        self.pocket_selection = str(pocket_selection)
        self.frame_alignment = str(frame_alignment)
        self.split_name = split_name
        self.epoch = int(epoch)
        self.disable_mapping = bool(disable_mapping)

    def set_epoch(self, epoch: int) -> None:
        """Redraw target frame blocks; the parent block stays per-pair fixed."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.examples.iloc[int(item)]
        return build_example(
            parent_id=str(row["parent"]),
            product_id=str(row["product"]),
            pair_key=str(row["pair_key"]),
            direction=str(row["direction"]),
            uniprot=str(row["uniprot"]),
            split_name=self.split_name,
            store=self.store,
            mappings=load_mappings(self.mappings_path),
            product_init_mode=self.product_init_mode,
            n_frames=self.n_frames,
            max_pocket_atoms=self.max_pocket_atoms,
            pocket_selection=self.pocket_selection,
            frame_alignment=self.frame_alignment,
            frame_seed=self.epoch,
            disable_mapping=self.disable_mapping,
        )


def collate_mmp_transfer(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad exactly as `collate_atomic_windows` does, then carry pair metadata."""
    batch = collate_atomic_windows(items)
    batch["parent"] = [str(item["parent"]) for item in items]
    batch["product"] = [str(item["product"]) for item in items]
    batch["uniprot"] = [str(item["uniprot"]) for item in items]
    batch["direction"] = [str(item["direction"]) for item in items]
    batch["pair_key"] = [str(item["pair_key"]) for item in items]
    batch["n_pocket_available"] = [int(item["n_pocket_available"]) for item in items]
    return batch


def length_bucketed_batches(
    examples: pd.DataFrame,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    max_pair_work: int | None = None,
) -> list[list[int]]:
    """Same padded-cost budgeting as `mehf_pretrain._length_bucketed_batches`."""
    cost = examples["pair_work"].to_numpy(dtype=np.int64)
    order = np.argsort(cost, kind="stable")
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        candidate_max = max(current_max, int(cost[index]))
        over_budget = (
            max_pair_work is not None
            and current
            and (len(current) + 1) * candidate_max > int(max_pair_work)
        )
        if current and (len(current) >= batch_size or over_budget):
            batches.append(current)
            current, current_max = [], 0
            candidate_max = int(cost[index])
        current.append(int(index))
        current_max = candidate_max
    if current:
        batches.append(current)
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(batches)
    return batches


def estimate_pair_work(
    examples: pd.DataFrame,
    *,
    store: str | Path = DEFAULT_STORE,
    max_pocket_atoms: int | None = 96,
) -> pd.DataFrame:
    """Add `pair_work` = product heavy atoms x cropped pocket atoms."""
    out = examples.copy()
    work = []
    for product_id in out["product"]:
        system = load_system(str(product_id), store, "none")
        n_atoms = int(system["coords"].shape[1])
        n_pocket = int(system["pocket_coords"].shape[1])
        if max_pocket_atoms is not None:
            n_pocket = min(n_pocket, int(max_pocket_atoms))
        work.append(n_atoms * n_pocket)
    out["pair_work"] = work
    return out


def load_excluded_systems(path: str | Path) -> frozenset[str]:
    """PDB ids listed in an exclusions JSON (key ``excluded``)."""
    import json
    systems = json.loads(Path(path).read_text())["excluded"]
    if not systems:
        raise ValueError(f"{path} lists no excluded systems")
    return frozenset(str(item).upper() for item in systems)


def drop_excluded(examples: pd.DataFrame, excluded: Iterable[str]) -> pd.DataFrame:
    """Drop every pair (both directions) in which either system is excluded.

    Applied AFTER the target-disjoint split and the fold assignment so that no target
    moves between splits or folds because of the exclusion.
    """
    excluded = {str(item).upper() for item in excluded}
    hit = examples["pdb_a"].isin(excluded) | examples["pdb_b"].isin(excluded)
    return examples[~hit].reset_index(drop=True)


def usable_examples(
    *,
    pairs_csv: str | Path = DEFAULT_PAIRS,
    mappings_path: str | Path = DEFAULT_MAPPINGS,
    store: str | Path = DEFAULT_STORE,
    split: str | Iterable[str] = "train",
) -> pd.DataFrame:
    """Usable pairs, primary split applied, expanded into both directions."""
    pairs = target_disjoint_split(load_usable_pairs(pairs_csv))
    wanted = {split} if isinstance(split, str) else set(split)
    if not wanted <= {"train", "val", "test"}:
        raise ValueError("split must be drawn from train/val/test")
    selected = pairs[pairs["primary_split"].isin(wanted)]
    examples = directional_examples(selected)
    mappings = load_mappings(mappings_path)
    keep = examples["pair_key"].isin(mappings.keys())
    return examples[keep].reset_index(drop=True)
