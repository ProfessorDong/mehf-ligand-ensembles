"""No-retraining baselines for the clean-protocol comparison table.

Scored with EXACTLY the metric code of `mehf_internal_geometry_spectrum.py` (imported:
`spreads`, `system_internals`, `out_of_plane_degrees`, `CHANNELS`; the reference-ensemble /
planar-centre logic of its main() is reproduced line for line) and written in its per-example
schema (key, <channel>_{reference,generated,ratio} for bond_length, bond_angle, dihedral,
internal_rmsf, planar_improper), so the CSVs drop into the same comparison.

  harm_source   draw the model's own SOURCE distribution with no learned flow: the model's
                `_structured_noise` (rigid translation/rotation + Sigma* local noise rescaled
                to isotropic amplitude) around product_init, 50 frames, multiplier 1.0 (the
                coarse-stage source).  The network is built from the arm's config but its
                weights are never loaded and never used.  No angle projection, no flow.
  parent_copy   product_init + parent_motion_prior: the mapped analogue's displacement
                trajectory applied to the target ligand, unmapped atoms zero displacement.

Examples: the fit_examples.csv (balanced sample of --n-examples, as the spectrum script) and
the dev_examples.csv (all rows) of a reference fold directory.  Never val, never test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mdwm.data.mmp_transfer import DEFAULT_MAPPINGS, DEFAULT_STORE, load_mappings, load_system  # noqa: E402
from mdwm.eval.mmp_transfer_eval import draw_frames  # noqa: E402
from mdwm.training.mehf_transfer import build_condition, build_model  # noqa: E402
from mehf_internal_geometry_spectrum import (  # noqa: E402
    CHANNELS, out_of_plane_degrees, spreads, system_internals,
)
from mehf_internal_geometry_spectrum import pocket_clash_columns  # noqa: E402
from run_subspace_plane import build_batches, select_examples  # noqa: E402


def reference_and_internals(chosen, store, frame_alignment, seed):
    """Verbatim logic of mehf_internal_geometry_spectrum.main(), --impropers on."""
    reference, internals, keep = {}, {}, []
    rejected = []
    for position, row in enumerate(chosen.itertuples()):
        system = load_system(str(row.product), store, frame_alignment)
        geometry = system_internals(system)
        if geometry is None:
            rejected.append((f"{row.parent}->{row.product}", "no bonds/angles/dihedrals"))
            continue
        coords = np.asarray(system["coords"], dtype=np.float64)
        frames = draw_frames(str(row.pair_key), str(row.direction), coords.shape[0], seed)
        key = f"{row.parent}->{row.product}"
        block = coords[frames["reference"]]
        if geometry["impropers"]:
            deviation = np.abs(out_of_plane_degrees(block, geometry["impropers"])).mean(axis=0)
            geometry["impropers"] = [c for c, v in zip(geometry["impropers"], deviation) if v < 10.0]
        reference[key] = spreads(block, geometry, impropers=True)
        internals[key] = geometry
        keep.append(position)
    return reference, internals, chosen.iloc[keep].reset_index(drop=True), rejected


def score(ensembles, reference, internals, clash=None):
    rows = []
    for key, block in ensembles.items():
        generated = spreads(np.asarray(block, dtype=np.float64), internals[key], impropers=True)
        row = {"key": key}
        for name in CHANNELS + ("planar_improper",):
            row[f"{name}_reference"] = reference[key][name]
            row[f"{name}_generated"] = generated[name]
            row[f"{name}_ratio"] = generated[name] / max(reference[key][name], 1.0e-9)
        if clash is not None:
            row.update(clash(key, block))
        rows.append(row)
    return rows


def summarise(rows, extra):
    out = dict(extra, n_examples_sampled=len(rows), heavy_atoms_only=True)
    for name in CHANNELS + ("planar_improper",):
        v = np.array([r[f"{name}_ratio"] for r in rows])
        v = v[np.isfinite(v)]
        if v.size:
            out[name] = {"ratio_median": float(np.median(v)), "ratio_q25": float(np.percentile(v, 25)),
                         "ratio_q75": float(np.percentile(v, 75)), "n_examples": int(v.size)}
    if rows and "full_clash_frame_rate" in rows[0]:
        for tag in ("full", "crop"):
            fr = np.array([r[f"{tag}_clash_frame_rate"] for r in rows])
            pf = np.array([r[f"{tag}_clash_pairs_per_frame"] for r in rows])
            out[f"{tag}_pocket_clash"] = {"frame_rate_mean": float(fr.mean()), "pairs_per_frame_mean": float(pf.mean()),
                                          "pair_fraction_mean": float(np.mean([r[f"{tag}_clash_pair_fraction"] for r in rows])),
                                          "examples_with_any_clash": float((fr > 0).mean())}
    return out


def write(path: Path, rows, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(path.with_suffix(".per_example.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="clean-protocol HARM yaml (model + data cfg)")
    ap.add_argument("--dims-checkpoint", type=Path, required=True,
                    help="any checkpoint, read ONLY for the corpus feature dimensions")
    ap.add_argument("--fold-dir", type=Path, required=True,
                    help="dir holding fit_examples.csv and dev_examples.csv of this fold")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-examples", type=int, default=96)
    ap.add_argument("--internal-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ap.add_argument("--rigid", type=float, default=0.32)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--n-dev-examples", type=int, default=160,
                    help="balanced dev sample, the SAME selection the trained arms' dev grid uses")
    ap.add_argument("--tau-set", default="train_only_large", help="harmonic tau set of the SOURCE control (diagnostic: legacy)")
    ap.add_argument("--projection", choices=("off", "on"), required=True,
                    help="on = the arms' inference projection (8 sweeps, bond+angle relaxation 0.5, hard angle "
                         "target, clash step relaxation 1.0 against the 96-atom crop, restore off)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/clean_protocol_v1/baselines_v2")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, model_cfg = dict(cfg["data"]), dict(cfg["model"])
    model_cfg["harmonic_tau_set"] = args.tau_set
    if not model_cfg.get("harmonic_local_noise"):
        raise SystemExit("config must be a HARM (harmonic_local_noise) config")
    if data_cfg.get("pocket_selection") != "frame0":
        raise SystemExit("config must set pocket_selection: frame0")
    if data_cfg["split"] != "train":
        raise SystemExit("baselines are scored on train-fold examples only")
    store = data_cfg.get("store", str(DEFAULT_STORE))
    mappings = load_mappings(data_cfg.get("mappings_path", str(DEFAULT_MAPPINGS)))
    frame_alignment = str(data_cfg.get("frame_alignment", "pocket"))
    dims = torch.load(args.dims_checkpoint, map_location="cpu", weights_only=False)
    dummy = {"product_features": torch.zeros(1, 1, int(dims["atom_feature_dim"])),
             "edit_features": torch.zeros(1, 1, int(dims["edit_feature_dim"])),
             "pocket_features": torch.zeros(1, 1, int(dims["pocket_feature_dim"]))}
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg, dummy).to(device).eval()      # weights NOT loaded
    assert model.harmonic_local_noise and model.harmonic_tau_set == args.tau_set
    if args.projection == "on":
        model.angle_projection = True
        model.restore_projection_amplitude = False
        model.angle_target_width_radian = 0.0
        model.projection_iterations = 8
        model.projection_angle_relaxation = 0.5
    else:
        model.projection_iterations = 0
    suffix = ("_proj" if args.projection == "on" else "") + ("" if args.tau_set == "train_only_large" else f"_tau-{args.tau_set}")

    sets = {"fit": select_examples(pd.read_csv(args.fold_dir / "fit_examples.csv"),
                                   args.n_examples, "balanced", args.seed),
            "dev": select_examples(pd.read_csv(args.fold_dir / "dev_examples.csv"),
                                   args.n_dev_examples, "balanced", args.seed)}
    for split, chosen in sets.items():
        reference, internals, usable, rejected = reference_and_internals(chosen, store, frame_alignment, args.seed)
        print(f"fold {args.fold} {split}: {len(chosen)} requested, {len(usable)} usable, "
              f"{len(rejected)} rejected {rejected}", flush=True)
        batches = build_batches(
            usable, store=store, mappings=mappings, frame_alignment=frame_alignment,
            product_init_mode=str(data_cfg.get("product_init_mode", "product_frame0")),
            max_pocket_atoms=data_cfg.get("max_pocket_atoms", 96),
            pocket_selection="frame0", seed=args.seed, batch_size=4,
            disable_mapping=bool(data_cfg.get("disable_mapping", False)),
        )
        meta = {"projection": args.projection, "fold": args.fold, "split": split, "n_requested": len(chosen), "n_rejected": len(rejected),
                "rejected": rejected, "seed": args.seed, "config": str(args.config),
                "examples_from": str(args.fold_dir)}
        clash = lambda key, block: pocket_clash_columns(  # noqa: E731
            np.asarray(block, dtype=np.float64), key.split("->")[1], store, frame_alignment,
            data_cfg.get("max_pocket_atoms", 96), "frame0")
        # (b) parent motion transferred directly
        ens_b: dict[str, np.ndarray] = {}
        for rec in batches:
            cond, _ = build_condition(rec["batch"], device)
            path = cond.parent_motion_prior
            if args.projection == "on":
                with torch.no_grad():
                    path = model._project_geometry(path, cond, generator=None)
            coords = (cond.product_init[:, None] + path).cpu().numpy()
            for i, key in enumerate(rec["keys"]):
                ens_b[key] = coords[i, :, : rec["n_atoms"][i]].astype(np.float64)
        rows = score(ens_b, reference, internals, clash)
        write(args.out / f"parent_copy_f{args.fold}_{split}{suffix}.json", rows,
              summarise(rows, dict(meta, baseline="parent_motion_transferred")))
        # (a) Sigma* source, no flow, per scale
        for scale in args.internal_scales:
            model.rigid_noise_scale, model.internal_noise_scale = args.rigid, float(scale)
            ens_a: dict[str, np.ndarray] = {}
            with torch.no_grad():
                for rec in batches:
                    cond, _ = build_condition(rec["batch"], device)
                    gen = torch.Generator(device=device).manual_seed(rec["generator_seed"])
                    noise = model._structured_noise(cond, 50, multiplier=1.0, generator=gen)
                    if args.projection == "on":
                        noise = model._project_geometry(noise, cond, generator=gen)
                    coords = (cond.product_init[:, None] + noise).cpu().numpy()
                    for i, key in enumerate(rec["keys"]):
                        ens_a[key] = coords[i, :, : rec["n_atoms"][i]].astype(np.float64)
            rows = score(ens_a, reference, internals, clash)
            write(args.out / f"harm_source_f{args.fold}_{split}_s{scale}{suffix}.json", rows,
                  summarise(rows, dict(meta, baseline="sigma_star_source_no_flow",
                                       rigid_scale=args.rigid, internal_scale=float(scale),
                                       tau_set=args.tau_set, projection=args.projection)))
            print(f"  harm_source s={scale}: n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
