"""Frozen metric suite for ligand conformational-ensemble comparison.

Two sample-size hazards were measured on this corpus before this module was
written, and both are enforced here rather than left to the caller:

1. The torsion Jensen-Shannon distance between two disjoint halves of ONE
   trajectory moves from 0.64 to 0.28 as the frames per side go from 5 to 50.
   A JS distance is therefore meaningless unless the two ensembles being
   compared carry the same number of samples.
2. The energy score's self-distance term, if computed as a V-statistic, carries
   a (K-1)/K factor, so a model evaluated with K=2 samples is scored on a
   different scale from one evaluated with K=100.

Every function here takes ensembles that have already been subsampled to a
common size by `ensemble_report`, which records that size in its output.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

TORSION_BINS = 36                  # 10 degrees per bin
HISTOGRAM_EPS = 1e-3               # identical smoothing everywhere


# --------------------------------------------------------------------------
# torsion geometry
# --------------------------------------------------------------------------
def _connected_without(n_atoms: int, bonds: list[tuple[int, int]], skip: int) -> bool:
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for index, (a, b) in enumerate(bonds):
        if index == skip:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen, stack = {0}, [0]
    while stack:
        for neighbour in adjacency[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == n_atoms


def rotatable_torsions(atomic_numbers: np.ndarray,
                       bond_index: np.ndarray) -> list[tuple[int, int, int, int]]:
    """a-b-c-d quadruples about each non-ring heavy-atom bond.

    Indices refer to the ORIGINAL atom order, so a caller holding an atom
    mapping between two ligands can intersect these directly.
    """
    heavy = np.flatnonzero(atomic_numbers > 1)
    if len(heavy) < 4:
        return []
    local = {int(a): i for i, a in enumerate(heavy)}
    bonds = sorted({tuple(sorted((local[int(a)], local[int(b)])))
                    for a, b in bond_index
                    if int(a) in local and int(b) in local})
    if not bonds:
        return []
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(heavy))}
    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    weights = atomic_numbers[heavy]
    quadruples = []
    for index, (b, c) in enumerate(bonds):
        if _connected_without(len(heavy), bonds, index):
            continue                                   # ring bond, not rotatable
        left = [n for n in adjacency[b] if n != c]
        right = [n for n in adjacency[c] if n != b]
        if not left or not right:
            continue                                   # terminal bond
        a = max(left, key=lambda i: (weights[i], -i))
        d = max(right, key=lambda i: (weights[i], -i))
        quadruples.append((int(heavy[a]), int(heavy[b]),
                           int(heavy[c]), int(heavy[d])))
    return quadruples


def dihedral_degrees(coordinates: np.ndarray,
                     quadruple: tuple[int, int, int, int]) -> np.ndarray:
    """Signed dihedral in degrees for every frame of a (frames, atoms, 3) array."""
    p0, p1, p2, p3 = (coordinates[:, i, :] for i in quadruple)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    axis = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * axis).sum(-1, keepdims=True) * axis
    w = b2 - (b2 * axis).sum(-1, keepdims=True) * axis
    return np.degrees(np.arctan2((np.cross(axis, v) * w).sum(-1), (v * w).sum(-1)))


def angle_histogram(angles: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(angles, bins=TORSION_BINS, range=(-180.0, 180.0))
    density = counts.astype(np.float64) + HISTOGRAM_EPS
    return density / density.sum()


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon DISTANCE: sqrt of the base-2 divergence, in [0, 1]."""
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))


# --------------------------------------------------------------------------
# proper scoring rule
# --------------------------------------------------------------------------
def energy_score(samples: np.ndarray, observations: np.ndarray) -> float:
    """Energy score with an UNBIASED (U-statistic) self-distance term.

    ES = E||X - y||  -  0.5 * E||X - X'||, X, X' independent.

    The self-distance must exclude i == j. Including it, as a V-statistic does,
    multiplies the second term by (K-1)/K, which makes scores computed with
    different sample counts incomparable. Lower is better.
    """
    generated = np.asarray(samples, dtype=float).reshape(len(samples), -1)
    observed = np.asarray(observations, dtype=float).reshape(len(observations), -1)
    if len(generated) < 2:
        raise ValueError("energy_score needs at least 2 generated samples for an "
                         "unbiased self-distance term")
    if len(observed) < 1:
        raise ValueError("energy_score needs at least 1 observation")
    cross = np.linalg.norm(generated[:, None, :] - observed[None, :, :],
                           axis=-1).mean()
    pairwise = np.linalg.norm(generated[:, None, :] - generated[None, :, :], axis=-1)
    k = len(generated)
    self_distance = pairwise.sum() / (k * (k - 1))       # diagonal is zero, excluded
    return float(cross - 0.5 * self_distance)


# --------------------------------------------------------------------------
# ensemble-shape metrics
# --------------------------------------------------------------------------
def _superpose(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    target = reference - reference.mean(axis=0)
    out = np.empty_like(frames)
    for index, frame in enumerate(frames):
        centred = frame - frame.mean(axis=0)
        u, _, vt = np.linalg.svd(centred.T @ target)
        flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
        out[index] = centred @ (u @ flip @ vt)
    return out


def rmsf_profile(frames: np.ndarray) -> np.ndarray:
    """Per-atom RMSF after removing global translation and rotation."""
    aligned = _superpose(frames, frames.mean(axis=0))
    return np.sqrt(((aligned - aligned.mean(axis=0)) ** 2).sum(-1).mean(axis=0))


def rmsf_recovery(generated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    a, b = rmsf_profile(generated), rmsf_profile(reference)
    correlation = spearmanr(a, b)[0] if len(a) > 2 else float("nan")
    return {"rmsf_spearman": float(correlation),
            "rmsf_mae_angstrom": float(np.abs(a - b).mean()),
            "rmsf_mean_ratio": float(a.mean() / b.mean()) if b.mean() > 0 else float("nan")}


def interaction_wasserstein(generated: np.ndarray, reference: np.ndarray,
                            pocket: np.ndarray) -> float:
    """1-D Wasserstein between the per-frame minimum ligand-pocket distance."""
    def closest(frames: np.ndarray) -> np.ndarray:
        return np.array([
            np.linalg.norm(frame[:, None, :] - pocket[None, :, :], axis=-1).min()
            for frame in frames])
    return float(wasserstein_distance(closest(generated), closest(reference)))


# --------------------------------------------------------------------------
# the frozen report
# --------------------------------------------------------------------------
def ensemble_report(generated: np.ndarray, reference: np.ndarray, *,
                    atomic_numbers: np.ndarray, bond_index: np.ndarray,
                    pocket: np.ndarray | None = None,
                    torsions: list[tuple[int, int, int, int]] | None = None,
                    n_samples: int | None = None,
                    seed: int = 20260918) -> dict[str, float]:
    """Compare two ensembles of the SAME ligand, both (frames, atoms, 3).

    Both sides are subsampled without replacement to a common size, which is
    reported as `n_samples`, because the torsion JS distance and the energy
    score both depend on it.
    """
    generated = np.asarray(generated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if generated.shape[1:] != reference.shape[1:]:
        raise ValueError("generated and reference must describe the same ligand")
    size = min(len(generated), len(reference))
    if n_samples is not None:
        size = min(size, int(n_samples))
    if size < 2:
        raise ValueError("need at least 2 samples per side")
    rng = np.random.default_rng(seed)
    generated = generated[rng.choice(len(generated), size, replace=False)]
    reference = reference[rng.choice(len(reference), size, replace=False)]

    if torsions is None:
        torsions = rotatable_torsions(atomic_numbers, bond_index)
    report: dict[str, float] = {"n_samples": float(size),
                                "n_torsions": float(len(torsions))}
    if torsions:
        distances = [js_distance(angle_histogram(dihedral_degrees(generated, q)),
                                 angle_histogram(dihedral_degrees(reference, q)))
                     for q in torsions]
        report["torsion_js_distance"] = float(np.mean(distances))
        report["torsion_js_distance_max"] = float(np.max(distances))
    else:
        report["torsion_js_distance"] = float("nan")
        report["torsion_js_distance_max"] = float("nan")

    report["energy_score"] = energy_score(generated, reference)
    report.update(rmsf_recovery(generated, reference))
    if pocket is not None:
        report["interaction_wasserstein"] = interaction_wasserstein(
            generated, reference, np.asarray(pocket, dtype=float))
    return report
