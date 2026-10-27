"""STAGE 1, EXPLORATORY.  Sweep the (rigid_scale, internal_scale) plane of a
RETRAINED decomposed-prior checkpoint at SAMPLING time.

    export CUDA_VISIBLE_DEVICES=0
    PYTHONPATH=src python scripts/run_subspace_plane.py \
        --checkpoint outputs/subspace_prior_v1/v1_f1/checkpoint.pt \
        --examples fit --output outputs/subspace_prior_stage1/plane_v1_f1_fit.json

`outputs/subspace_prior_v1/PREREGISTRATION.md` section "OUTCOME AND TWO-STAGE
CONTINUATION" is binding.  The pre-registered gate returned FAIL and that
verdict stands; everything this script produces is EXPLORATORY.

Why a sweep is legitimate on a fixed checkpoint.  `rigid_noise_scale` and
`internal_noise_scale` multiply the two halves of `_structured_noise`, which is
drawn at sampling time; for a given trained velocity field they are sampling
knobs, so one retrained field lets the whole plane be read without retraining.
What retraining changes is the field the knobs act on -- that is exactly the
difference between V1 (field trained under the ORIGINAL prior, scales set
afterwards) and this script (field trained under the DECOMPOSED prior, scales
moment-matched afterwards).

Only moments are computed: `mdwm.eval.subspace_dispersion.block_moments`,
imported unmodified, which is the pre-registration's internal RMSF and centroid
spread.  No Jensen-Shannon distance is computed anywhere in this path, so it
may be run on the fitting folds (`--examples fit`), which is where the
calibration and the selection happen.

The batches are built ONCE and reused for every grid point, and the sampling
generator is seeded exactly as `sample_product_ensembles` seeds it, so grid
points are common-random-number paired: any difference between two points is
the scales and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdwm.data.mmp_transfer import (  # noqa: E402
    DEFAULT_MAPPINGS,
    DEFAULT_STORE,
    _stable_seed,
    build_example,
    collate_mmp_transfer,
    load_mappings,
    load_system,
)
from mdwm.eval.ensemble_metrics import dihedral_degrees  # noqa: E402
from mdwm.eval.mmp_transfer_eval import circular_spread_degrees, draw_frames  # noqa: E402
from mdwm.eval.subspace_dispersion import block_moments, reference_moments  # noqa: E402
from mdwm.training.mehf_transfer import build_condition, build_model  # noqa: E402


DEFAULT_RIGID = [0.15, 0.30, 0.45, 0.60, 0.90]
DEFAULT_INTERNAL = [1.5, 2.5, 4.0, 6.5, 10.0]


def build_batches(
    usable: pd.DataFrame,
    *,
    store: str,
    mappings: dict[str, Any],
    frame_alignment: str,
    product_init_mode: str,
    max_pocket_atoms: int | None,
    seed: int,
    batch_size: int,
    disable_mapping: bool,
) -> list[dict[str, Any]]:
    """Collated conditioning batches, built once.  Mirrors the batching loop of
    `mmp_transfer_eval.sample_product_ensembles` exactly, including its
    `include_target=False` contract and its generator seed."""
    rows = list(usable.itertuples())
    batches: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start: start + batch_size]
        items, keys = [], []
        for row in chunk:
            frames = draw_frames(
                str(row.pair_key), str(row.direction),
                load_system(str(row.product), store, frame_alignment)["n_frames"], seed,
            )
            items.append(
                build_example(
                    parent_id=str(row.parent), product_id=str(row.product),
                    pair_key=str(row.pair_key), direction=str(row.direction),
                    uniprot=str(row.uniprot), split_name="eval",
                    store=store, mappings=mappings,
                    product_init_mode=product_init_mode,
                    max_pocket_atoms=max_pocket_atoms,
                    frame_alignment=frame_alignment,
                    parent_frames=frames["parent"],
                    target_frames=frames["reference"],
                    include_target=False,
                    disable_mapping=disable_mapping,
                )
            )
            keys.append(f"{row.parent}->{row.product}")
        batches.append({
            "batch": collate_mmp_transfer(items),
            "keys": keys,
            "n_atoms": [int(item["product_init"].shape[0]) for item in items],
            "generator_seed": _stable_seed("sample", keys[0], seed),
            "uniprot": [str(row.uniprot) for row in chunk],
        })
    return batches


def sample_grid_point(
    model,
    batches: list[dict[str, Any]],
    *,
    device: torch.device,
    steps: int,
) -> dict[str, np.ndarray]:
    ensembles: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for record in batches:
            condition, _ = build_condition(record["batch"], device)
            generator = torch.Generator(device=device).manual_seed(
                record["generator_seed"]
            )
            path = model.sample(condition, n_samples=1, steps=steps, generator=generator)
            coords = (condition.product_init[:, None] + path[0]).cpu().numpy()
            for index, key in enumerate(record["keys"]):
                ensembles[key] = coords[
                    index, :, : record["n_atoms"][index]
                ].astype(np.float64)
    return ensembles


def select_examples(
    frame: pd.DataFrame, n_examples: int, mode: str, seed: int
) -> pd.DataFrame:
    """Draw the calibration sample, either as the original head or balanced.

    The example CSVs are written in per-target blocks, so a head slice is
    dominated by whichever targets happen to come first.  `balanced` walks the
    targets round robin -- one row from each in turn -- so the sample's target
    composition is as flat as the corpus allows.
    """
    if mode == "head":
        return frame.head(n_examples).reset_index(drop=True)
    groups = {
        target: rows.index.tolist()
        for target, rows in frame.groupby("uniprot", sort=True)
    }
    order = sorted(groups)
    np.random.default_rng(seed).shuffle(order)
    picked: list[int] = []
    depth = 0
    while len(picked) < n_examples:
        added = False
        for target in order:
            rows = groups[target]
            if depth < len(rows):
                picked.append(rows[depth])
                added = True
                if len(picked) >= n_examples:
                    break
        if not added:  # every target exhausted
            break
        depth += 1
    return frame.loc[picked].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--examples", choices=["fit", "dev"], default="fit")
    parser.add_argument("--n-examples", type=int, default=160)
    parser.add_argument(
        "--example-sampling", choices=["head", "balanced"], default="head",
        help=(
            "`head` is the original behaviour and is kept so earlier runs stay "
            "reproducible.  It is BIASED: the example CSVs are ordered in "
            "per-target blocks, so head(160) covers 20-45 targets and lets one "
            "target supply 38-68 percent of the rows.  `balanced` draws "
            "round-robin over targets (targets shuffled with --seed, rows "
            "within a target kept in file order), which gives every target the "
            "same weight until it runs out."
        ),
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--rigid", type=float, nargs="+", default=DEFAULT_RIGID)
    parser.add_argument("--internal", type=float, nargs="+", default=DEFAULT_INTERNAL)
    parser.add_argument(
        "--extra-points", type=str, default=None,
        help='JSON list of [rigid, internal] pairs appended to the product grid, '
             'e.g. the floor probe [[1e-06, 1e-06]].',
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "0", "1"):
        raise SystemExit(f"Use GPU 0 or GPU 1, one job per card; got {visible!r}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_cfg = dict(checkpoint.get("data_cfg", {}))
    model_cfg = dict(checkpoint.get("model_cfg", {}))
    store = data_cfg.get("store", str(DEFAULT_STORE))
    mappings = load_mappings(data_cfg.get("mappings_path", str(DEFAULT_MAPPINGS)))
    frame_alignment = str(data_cfg.get("frame_alignment", "pocket"))
    product_init_mode = str(data_cfg.get("product_init_mode", "product_frame0"))
    max_pocket_atoms = data_cfg.get("max_pocket_atoms", 96)
    disable_mapping = bool(data_cfg.get("disable_mapping", False))

    directory = args.checkpoint.parent
    fit = pd.read_csv(directory / "fit_examples.csv")
    dev = pd.read_csv(directory / "dev_examples.csv")
    if set(fit["uniprot"]) & set(dev["uniprot"]):
        raise ValueError("Fitting and dev folds share a UniProt target")
    chosen = (fit if args.examples == "fit" else dev)
    chosen = select_examples(
        chosen, int(args.n_examples), args.example_sampling, int(args.seed)
    )

    reference, geometries, usable = reference_moments(
        chosen, store=store, mappings=mappings,
        frame_alignment=frame_alignment, seed=int(args.seed),
    )
    # TORSION circular spread of each reference ensemble.  The Cartesian
    # internal RMSF and the torsion spread are NOT the same moment: the prior's
    # `local` term is isotropic per-atom noise, and a dihedral read off four
    # atoms picks up far more angular variance from uncorrelated displacement
    # than physical internal motion of the same RMSF does.  The reported metric
    # lives in torsion space, so the torsion spread is measured alongside.
    reference_spread: dict[str, float] = {}
    torsions_by_key: dict[str, Any] = {}
    for row in usable.itertuples():
        key = f"{row.parent}->{row.product}"
        geometry = geometries.get(key)
        if geometry is None:
            continue
        quads = geometry.get("torsions") or []
        torsions_by_key[key] = quads
        if not quads:
            continue
        coords = geometry["product"]["coords"].astype(np.float64)
        frames = draw_frames(
            str(row.pair_key), str(row.direction),
            geometry["product"]["n_frames"], int(args.seed),
        )
        reference_spread[key] = float(np.mean([
            circular_spread_degrees(dihedral_degrees(coords, q)[frames["reference"]])
            for q in quads
        ]))
    batches = build_batches(
        usable, store=store, mappings=mappings, frame_alignment=frame_alignment,
        product_init_mode=product_init_mode, max_pocket_atoms=max_pocket_atoms,
        seed=int(args.seed), batch_size=int(args.batch_size),
        disable_mapping=disable_mapping,
    )
    uniprot_by_key = {
        f"{r.parent}->{r['product']}": str(r.uniprot) for _, r in usable.iterrows()
    }

    dummy = {
        "product_features": torch.zeros(1, 1, int(checkpoint["atom_feature_dim"])),
        "edit_features": torch.zeros(1, 1, int(checkpoint["edit_feature_dim"])),
        "pocket_features": torch.zeros(1, 1, int(checkpoint["pocket_feature_dim"])),
    }
    model = build_model(model_cfg, dummy)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if model.internal_scale_head is not None:
        raise SystemExit(
            "This sweep is defined for two SCALARS; a conditional head (V3) makes "
            "the internal scale per molecule and is not a point in this plane."
        )

    points = [(float(r), float(i)) for r in args.rigid for i in args.internal]
    if args.extra_points:
        points += [(float(r), float(i)) for r, i in json.loads(args.extra_points)]

    plane: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    n_frames_generated = 0
    for rigid, internal in points:
        started = time.time()
        model.rigid_noise_scale = rigid
        model.internal_noise_scale = internal
        ensembles = sample_grid_point(model, batches, device=device, steps=args.steps)
        ratios = {"internal_rmsf": [], "centroid_dispersion": [], "orientation_dispersion": []}
        spread_ratios: list[float] = []
        for key, block in ensembles.items():
            n_frames_generated = int(len(block))
            generated = block_moments(block)
            for name, value in generated.items():
                ratios[name].append(value / max(reference[key][name], 1.0e-9))
            if key in reference_spread and torsions_by_key.get(key):
                generated_spread = float(np.mean([
                    circular_spread_degrees(dihedral_degrees(block, q))
                    for q in torsions_by_key[key]
                ]))
                spread_ratios.append(
                    generated_spread / max(reference_spread[key], 1.0e-9)
                )
            per_example.append({
                "rigid_scale": rigid, "internal_scale": internal, "key": key,
                "uniprot": uniprot_by_key[key],
                **{f"ratio_{n}": generated[n] / max(reference[key][n], 1.0e-9)
                   for n in ratios},
            })
        record: dict[str, Any] = {"rigid_scale": rigid, "internal_scale": internal}
        if spread_ratios:
            record["torsion_spread_ratio_median"] = float(np.median(spread_ratios))
            record["torsion_spread_ratio_q25"] = float(np.percentile(spread_ratios, 25))
            record["torsion_spread_ratio_q75"] = float(np.percentile(spread_ratios, 75))
        for name, values in ratios.items():
            record[f"{name}_ratio_median"] = float(np.median(values))
            record[f"{name}_ratio_q25"] = float(np.percentile(values, 25))
            record[f"{name}_ratio_q75"] = float(np.percentile(values, 75))
        record["max_deviation"] = max(
            abs(record["internal_rmsf_ratio_median"] - 1.0),
            abs(record["centroid_dispersion_ratio_median"] - 1.0),
        )
        record["both_in_window"] = bool(
            0.85 <= record["internal_rmsf_ratio_median"] <= 1.15
            and 0.85 <= record["centroid_dispersion_ratio_median"] <= 1.15
        )
        record["seconds"] = round(time.time() - started, 1)
        plane.append(record)
        print(json.dumps(record), flush=True)

    summary = {
        "stage": "STAGE 1 -- EXPLORATORY.  The pre-registered gate returned FAIL "
                 "and that verdict stands; nothing here is a confirmed result.",
        "checkpoint": str(args.checkpoint),
        "examples": args.examples,
        "n_examples": int(len(usable)),
        "n_targets": int(usable["uniprot"].nunique()),
        "n_samples_per_side": int(n_frames_generated),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "training_rigid_noise_scale": float(model_cfg.get("rigid_noise_scale", 1.0)),
        "training_internal_noise_scale": float(model_cfg.get("internal_noise_scale", 1.0)),
        "window": [0.85, 1.15],
        "plane": plane,
        "cuda_visible_devices": visible,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(per_example).to_csv(args.output.with_suffix(".per_example.csv"), index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "plane"}, indent=2))


if __name__ == "__main__":
    main()
