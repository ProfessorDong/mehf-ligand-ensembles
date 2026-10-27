"""Exact harmonic prior Sigma* = (B^T F B)^+ as a sampling distribution.

Measured motivation, `outputs/sigma_star_v1/FINDING.md`: drawing each stiff
internal coordinate independently at its physical width, which is what the
Gauss-Seidel sampler does, leaves the SOFT subspace white, and physical soft
motion is collective. The resulting noise is uniform to within a channel spread
of 3.295 across bond, angle, torsion and out-of-plane; sampling from the exact
harmonic covariance instead gives 0.582, a factor of 5.7.

The O(N^3) objection to Sigma* holds for proteins and not for ligands: a
30-heavy-atom ligand is a 90x90 eigenproblem, and the reference geometry is
fixed per example so the factor is built once and cached.

The internal coordinate set must be COMPLETE. With only bonds and angles the
torsional directions lie in the null space of H = B^T F B and the pseudo-inverse
hands them unbounded amplitude. Bonds, angles, proper torsions and out-of-plane
coordinates are all included, each at its measured physical width.

Known limitation, measured and not fixable by retuning: the torsional channel
comes out at 0.564 of the physical width, because real torsional motion is
multi-modal between rotamer wells while a harmonic model oscillates about one
mean. The deficit is in the shape of the distribution, not its scale.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

TAU_BOND_ANGSTROM = 0.032637
TAU_ANGLE_RADIAN = math.radians(2.774)
TAU_TORSION_RADIAN = math.radians(14.6)
TAU_OOP_RADIAN = math.radians(6.48)

# Clean-protocol widths: median over TRAINING-split systems only (n = 73 / 247 / 247 / 15
# systems for bond / angle / torsion / oop).  Provenance and exact criterion:
# outputs/clean_protocol_v1/tau_train_only.json (scripts/tau_train_only.py).  The constants
# above are untouched because released checkpoints depend on them.
TAU_BOND_ANGSTROM_TRAIN_ONLY = 0.0324516934526607
TAU_ANGLE_RADIAN_TRAIN_ONLY = math.radians(2.7689192924707644)
TAU_TORSION_RADIAN_TRAIN_ONLY = math.radians(14.162180993580918)
TAU_OOP_RADIAN_TRAIN_ONLY = math.radians(6.415573887321512)

# Clean-protocol widths, LARGE sample: median over ALL 2,330 store systems of train-split UniProt
# targets (1,998 for oop) with the 28 clean-protocol exclusions removed -- the full population,
# not a draw.  Estimator identical to the legacy one (reproduces the stored per-system values
# bit-exactly).  Provenance: outputs/clean_protocol_v1/tau_train_only_large.json
# (scripts/tau_train_only_large.py).  This supersedes the small-sample *_TRAIN_ONLY set above.
TAU_BOND_ANGSTROM_TRAIN_ONLY_LARGE = 0.03275677838383294
TAU_ANGLE_RADIAN_TRAIN_ONLY_LARGE = math.radians(2.7707469663417585)
TAU_TORSION_RADIAN_TRAIN_ONLY_LARGE = math.radians(14.12894962939264)
TAU_OOP_RADIAN_TRAIN_ONLY_LARGE = math.radians(6.465122828119934)

# (bond, angle, torsion, oop) widths per named set.  "legacy" is what every released
# checkpoint and the paper used; "train_only" is the clean-protocol re-estimate.
TAU_SETS = {
    "legacy": (TAU_BOND_ANGSTROM, TAU_ANGLE_RADIAN, TAU_TORSION_RADIAN, TAU_OOP_RADIAN),
    "train_only": (TAU_BOND_ANGSTROM_TRAIN_ONLY, TAU_ANGLE_RADIAN_TRAIN_ONLY,
                   TAU_TORSION_RADIAN_TRAIN_ONLY, TAU_OOP_RADIAN_TRAIN_ONLY),
    "train_only_large": (TAU_BOND_ANGSTROM_TRAIN_ONLY_LARGE, TAU_ANGLE_RADIAN_TRAIN_ONLY_LARGE,
                         TAU_TORSION_RADIAN_TRAIN_ONLY_LARGE, TAU_OOP_RADIAN_TRAIN_ONLY_LARGE),
}

_TOPOLOGY_CACHE: dict[bytes, tuple] = {}


def _enumerate(bond_pairs: list[tuple[int, int]], n_atoms: int):
    """Angles, proper torsions and out-of-plane centres from a bond list."""
    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in bond_pairs:
        neighbours[a].append(b)
        neighbours[b].append(a)
    angles = []
    for j in range(n_atoms):
        near = sorted(neighbours[j])
        for x in range(len(near)):
            for y in range(x + 1, len(near)):
                angles.append((near[x], j, near[y]))
    torsions = []
    for j, k in bond_pairs:
        for i in neighbours[j]:
            if i == k:
                continue
            for l in neighbours[k]:
                if l in (j, i):
                    continue
                torsions.append((i, j, k, l))
    centres = [(j, *sorted(v)[:3]) for j, v in neighbours.items() if len(v) == 3]
    return angles, torsions, centres


def topology(bond_index: torch.Tensor, n_atoms: int):
    """Cached internal-coordinate lists for one molecule's bond graph."""
    key = bond_index.detach().cpu().numpy().tobytes() + bytes([n_atoms & 0xFF])
    hit = _TOPOLOGY_CACHE.get(key)
    if hit is not None:
        return hit
    pairs = []
    seen = set()
    for a, b in bond_index.detach().cpu().tolist():
        a, b = int(a), int(b)
        if a < 0 or b < 0 or a == b or a >= n_atoms or b >= n_atoms:
            continue
        edge = (min(a, b), max(a, b))
        if edge in seen:
            continue
        seen.add(edge)
        pairs.append(edge)
    built = (pairs, *_enumerate(pairs, n_atoms))
    _TOPOLOGY_CACHE[key] = built
    return built


def _b_rows(coords: torch.Tensor, pairs, angles, torsions, centres, tau_set: str = "legacy"):
    """Wilson B rows by autograd, one row per internal coordinate.

    Autograd rather than a hand-written analytic form for every coordinate type:
    the values below are the coordinate definitions themselves, so the rows
    cannot disagree with the quantities the rest of the project measures.
    """
    if tau_set not in TAU_SETS:
        raise ValueError(f"tau_set must be one of {sorted(TAU_SETS)}")
    tau_bond, tau_angle, tau_torsion, tau_oop = TAU_SETS[tau_set]
    x = coords.detach().clone().requires_grad_(True)
    values, taus = [], []
    for a, b in pairs:
        values.append(torch.linalg.norm(x[a] - x[b]))
        taus.append(tau_bond)
    for i, j, k in angles:
        u, v = x[i] - x[j], x[k] - x[j]
        cosine = (u * v).sum() / (u.norm() * v.norm()).clamp_min(1e-9)
        values.append(torch.arccos(cosine.clamp(-1 + 1e-7, 1 - 1e-7)))
        taus.append(tau_angle)
    for i, j, k, l in torsions:
        b0, b1, b2 = x[i] - x[j], x[k] - x[j], x[l] - x[k]
        b1n = b1 / b1.norm().clamp_min(1e-9)
        w1 = b0 - (b0 * b1n).sum() * b1n
        w2 = b2 - (b2 * b1n).sum() * b1n
        values.append(torch.atan2((torch.linalg.cross(b1n, w1) * w2).sum(),
                                  (w1 * w2).sum()))
        taus.append(tau_torsion)
    for j, a, b, c in centres:
        va, vb, vc = x[a] - x[j], x[b] - x[j], x[c] - x[j]
        normal = torch.linalg.cross(vb, vc)
        normal = normal / normal.norm().clamp_min(1e-9)
        ua = va / va.norm().clamp_min(1e-9)
        values.append(torch.arcsin((ua * normal).sum().clamp(-1 + 1e-7, 1 - 1e-7)))
        taus.append(tau_oop)
    if not values:
        return None, None
    stacked = torch.stack(values)
    rows = torch.autograd.grad(
        stacked, x, grad_outputs=torch.eye(len(values), device=x.device, dtype=x.dtype),
        is_grads_batched=True,
    )[0].reshape(len(values), -1)
    weights = torch.tensor(taus, device=x.device, dtype=x.dtype).square().reciprocal()
    return rows.detach(), weights


def sigma_star_factor(
    coords: torch.Tensor, bond_index: torch.Tensor, *, cutoff: float = 1.0e-6,
    tau_set: str = "legacy",
):
    """Factor L with L L^T = Sigma*, shape (3N, K) over the non-rigid subspace.

    Returns (L, n_rigid_modes), or (None, None) if the internal set is
    degenerate for this molecule.
    """
    n_atoms = int(coords.shape[0])
    pairs, angles, torsions, centres = topology(bond_index, n_atoms)
    if not pairs or not angles:
        return None, None
    rows, weights = _b_rows(coords, pairs, angles, torsions, centres, tau_set)
    if rows is None:
        return None, None
    hessian = rows.T @ (weights[:, None] * rows)
    values, vectors = torch.linalg.eigh(hessian.double())
    keep = values > values.max().clamp_min(1e-30) * cutoff
    if int(keep.sum()) == 0:
        return None, None
    amplitude = values[keep].reciprocal().sqrt()
    factor = (vectors[:, keep] * amplitude[None, :]).to(coords.dtype)
    return factor, int((~keep).sum())


def sample_harmonic_noise(
    coords: torch.Tensor,
    bond_index: torch.Tensor,
    n_frames: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Draw `n_frames` displacements from N(0, Sigma*) for one molecule."""
    factor, _ = sigma_star_factor(coords, bond_index)
    if factor is None:
        return None
    draw = torch.randn(
        (n_frames, factor.shape[1]), generator=generator,
        device=coords.device, dtype=coords.dtype,
    )
    return (draw @ factor.T).reshape(n_frames, coords.shape[0], 3)
