"""Prior-only comparison, no network: how does each local prior distribute noise
across the four internal-coordinate channels at MATCHED overall dispersion?

For each of the systems used by the Sigma* control (`exact_sigma_star_noise.py`:
same store, seed 20260924, 8-40 heavy atoms), draw 100 frames of local noise from
  ISO        N(0, I)
  LAPLACIAN  N(0, L^+ (x) I_3), L the heavy-atom bond-graph Laplacian
  SIGMA*     N(0, (B^T F B)^+) over bonds, angles, torsions, out-of-plane
around the ensemble-mean pose, scale each draw so that its internal RMSF equals the
physical ensemble's, and report generated/physical width per channel and the channel
spread (max - min). Scaling removes each prior's own amplitude, so only the SHAPE is
compared -- exactly as MEHF uses the priors. Part of outputs/laplacian_prior_v1 (P1).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from exact_sigma_star_noise import improper_centres, sigma_star_sample  # noqa: E402
from isotropic_noise_geometry_cost import (  # noqa: E402
    bond_angles_degrees, bond_lengths, circular_std_degrees, dihedrals_degrees,
    enumerate_internals, internal_rmsf,
)

CHANNELS = ("bond", "angle", "torsion", "oop")


def laplacian_factor(pairs, n):
    lap = np.zeros((n, n))
    for a, b in pairs:
        if lap[a, b] == 0:
            lap[a, b] = lap[b, a] = -1.0
            lap[a, a] += 1.0; lap[b, b] += 1.0
    values, vectors = np.linalg.eigh(lap)
    keep = values > values.max() * 1e-8
    return vectors[:, keep] / np.sqrt(values[keep])[None, :]


def widths(block, bonds, angles, torsions, centres):
    out = {"bond": float(np.mean(bond_lengths(block, bonds).std(axis=0))),
           "angle": float(np.mean(bond_angles_degrees(block, angles).std(axis=0))),
           "torsion": float(np.mean(circular_std_degrees(dihedrals_degrees(block, torsions)))),
           "rmsf": internal_rmsf(block)}
    if centres:
        quads = [(c[1], c[0], c[2], c[3]) for c in centres]
        out["oop"] = float(np.mean(circular_std_degrees(dihedrals_degrees(block, quads))))
    return out


def matched(reference, noise, target_rmsf):
    scale = 1.0
    for _ in range(4):
        got = internal_rmsf(reference[None] + scale * noise)
        scale *= target_rmsf / max(got, 1e-12)
    return reference[None] + scale * noise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=ROOT / "data/MISATO/extracted/frames100_all_v1")
    ap.add_argument("--n-systems", type=int, default=150)
    ap.add_argument("--max-atoms", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/laplacian_prior_v1/prior_noise_profiles.json")
    args = ap.parse_args()

    paths = sorted(args.store.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(paths)
    ratios = {p: defaultdict(list) for p in ("ISO", "LAPLACIAN", "SIGMA_star")}
    spreads = {p: [] for p in ratios}
    used = 0
    for path in paths:
        if used >= args.n_systems:
            break
        with np.load(path, allow_pickle=True) as z:
            coords = z["ligand_coords"].astype(np.float64)
            bond_index = z["bond_index"]
            elements = z["ligand_atomic_numbers"].astype(int)
        heavy = np.where(elements != 1)[0]
        if not (8 <= len(heavy) <= args.max_atoms):
            continue
        remap = {int(a): i for i, a in enumerate(heavy)}
        pairs = [(remap[int(a)], remap[int(b)]) for a, b in np.asarray(bond_index).reshape(-1, 2)
                 if int(a) in remap and int(b) in remap and int(a) != int(b)]
        if not pairs:
            continue
        block = coords[:, heavy, :]
        n = len(heavy)
        bonds, angles, torsions = enumerate_internals(np.asarray(pairs), n)
        centres = improper_centres(np.asarray(pairs), n)
        if not bonds or not angles or not torsions or not centres:
            continue
        reference = block.mean(axis=0)
        frames = block.shape[0]
        try:
            star, _ = sigma_star_sample(reference, bonds, angles, torsions, centres, frames, rng)
        except np.linalg.LinAlgError:
            continue
        lap = laplacian_factor(pairs, n)
        noise = {"ISO": rng.standard_normal((frames, n, 3)),
                 "LAPLACIAN": np.einsum("ak,fkc->fac", lap, rng.standard_normal((frames, lap.shape[1], 3))),
                 "SIGMA_star": star}
        physical = widths(block, bonds, angles, torsions, centres)
        if any(physical[c] <= 1e-9 for c in CHANNELS):
            continue
        for prior, eps in noise.items():
            gen = widths(matched(reference, eps, physical["rmsf"]), bonds, angles, torsions, centres)
            r = {c: gen[c] / physical[c] for c in CHANNELS}
            for c in CHANNELS:
                ratios[prior][c].append(r[c])
            spreads[prior].append(max(r.values()) - min(r.values()))
        used += 1

    out = {"n_systems": used, "seed": args.seed, "matching": "internal RMSF equal to the physical ensemble's",
           "priors": {p: {"ratio_median": {c: float(np.median(ratios[p][c])) for c in CHANNELS},
                          "spread_median": float(np.median(spreads[p])),
                          "spread_iqr": [float(np.percentile(spreads[p], 25)), float(np.percentile(spreads[p], 75))]}
                      for p in ratios}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
