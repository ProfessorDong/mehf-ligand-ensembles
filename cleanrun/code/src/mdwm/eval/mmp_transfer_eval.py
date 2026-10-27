"""Phase C evaluation: MEHF against the frozen Phase A baseline suite.

Everything is paired by construction.  For one directional example the frame
draws are made ONCE -- a 100-frame permutation gives the 50 reference frames and
the 50 disjoint floor frames, and a separate draw gives the 50 parent frames --
and those same indices are handed to the model, to B1, to B4 and to the floor.
The torsion set is the same list of heavy-local quadruples for every predictor,
and every predictor carries exactly 50 samples per side.

The model never sees a product frame other than frame 0: examples are built with
``include_target=False``, so ``target_delta`` is structurally zero and the
reference block cannot reach the conditioning tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from mdwm.data.mmp_transfer import (
    DEFAULT_MAPPINGS,
    DEFAULT_STORE,
    _stable_seed,
    build_example,
    collate_mmp_transfer,
    load_mappings,
    load_system,
    mapped_torsion_indices,
    pair_mapping,
)
from mdwm.eval.ensemble_metrics import (
    angle_histogram,
    dihedral_degrees,
    ensemble_report,
    js_distance,
    rotatable_torsions,
)
from mdwm.models.mehf import MEHFCondition, MappedEditHierarchicalFlow
from mdwm.training.mehf_transfer import build_condition, build_model

N_SIDE = 50
BASELINE_KEYS = ("floor", "B1_copy_parent", "B4_parent_widened", "B3_prior", "B2_static")


def circular_spread_degrees(angles: np.ndarray) -> float:
    """Circular standard deviation, for reading how wide an ensemble is."""
    resultant = float(np.abs(np.mean(np.exp(1j * np.radians(np.asarray(angles))))))
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(resultant, 1.0e-12)))))


# --------------------------------------------------------------------------
# B3 and B4, reusing the Phase A reference logic
# --------------------------------------------------------------------------
def smooth(histogram: np.ndarray, width_bins: int) -> np.ndarray:
    """Circular box smoothing, the B4 widening operator (Phase A)."""
    if width_bins <= 0:
        return histogram
    kernel = np.ones(2 * width_bins + 1)
    kernel /= kernel.sum()
    padded = np.concatenate(
        [histogram[-width_bins:], histogram, histogram[:width_bins]]
    )
    widened = np.convolve(padded, kernel, mode="same")[width_bins:-width_bins]
    return widened / widened.sum()


def pooled_torsion_prior(
    products: Sequence[str],
    *,
    store: str | Path = DEFAULT_STORE,
    frame_alignment: str = "pocket",
) -> np.ndarray:
    """B3: one knowledge-based torsion histogram pooled over training products."""
    angles: list[np.ndarray] = []
    for pdb in products:
        try:
            system = load_system(str(pdb), store, frame_alignment)
        except Exception:                                        # noqa: BLE001
            continue
        coords = system["coords"].astype(np.float64)
        for quadruple in rotatable_torsions(
            system["atomic_numbers"], system["bond_index"]
        ):
            angles.append(dihedral_degrees(coords, quadruple))
    if not angles:
        return np.full(36, 1.0 / 36)
    return angle_histogram(np.concatenate(angles))


# --------------------------------------------------------------------------
# one example
# --------------------------------------------------------------------------
def draw_frames(
    pair_key: str, direction: str, n_frames: int, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_stable_seed(pair_key, direction, "eval", seed))
    order = rng.permutation(n_frames)
    return {
        "reference": np.sort(order[:N_SIDE]),
        "floor": np.sort(order[N_SIDE: 2 * N_SIDE]),
        "parent": np.sort(rng.choice(n_frames, N_SIDE, replace=False)),
    }


def example_geometry(
    parent_id: str,
    product_id: str,
    pair_key: str,
    direction: str,
    *,
    store: str | Path = DEFAULT_STORE,
    mappings: dict[str, Any] | None = None,
    frame_alignment: str = "pocket",
) -> dict[str, Any] | None:
    mappings = load_mappings() if mappings is None else mappings
    if pair_key not in mappings:
        return None
    parent = load_system(parent_id, store, frame_alignment)
    product = load_system(product_id, store, frame_alignment)
    mapping = pair_mapping(
        parent_id, product_id, pair_key, direction, mappings, parent, product
    )
    if int((mapping >= 0).sum()) < 3:
        return None
    torsions = mapped_torsion_indices(product, mapping)
    if not torsions:
        return None
    return {
        "parent": parent, "product": product, "mapping": mapping, "torsions": torsions,
    }


def baseline_scores(
    geometry: dict[str, Any],
    frames: dict[str, np.ndarray],
    prior: np.ndarray,
    b4_width: int,
) -> dict[str, float]:
    product_coords = geometry["product"]["coords"].astype(np.float64)
    parent_coords = geometry["parent"]["coords"].astype(np.float64)
    mapping = geometry["mapping"]
    scores: dict[str, list[float]] = {key: [] for key in BASELINE_KEYS}
    for quadruple in geometry["torsions"]:
        product_angles = dihedral_degrees(product_coords, quadruple)
        reference = angle_histogram(product_angles[frames["reference"]])
        scores["floor"].append(
            js_distance(reference, angle_histogram(product_angles[frames["floor"]]))
        )
        scores["B2_static"].append(
            js_distance(reference, angle_histogram(product_angles[:1]))
        )
        scores["B3_prior"].append(js_distance(reference, prior))
        mapped = tuple(int(mapping[atom]) for atom in quadruple)
        parent_angles = dihedral_degrees(parent_coords, mapped)
        parent_histogram = angle_histogram(parent_angles[frames["parent"]])
        scores["B1_copy_parent"].append(js_distance(reference, parent_histogram))
        scores["B4_parent_widened"].append(
            js_distance(reference, smooth(parent_histogram, b4_width))
        )
    return {key: float(np.mean(value)) for key, value in scores.items()}


def calibrate_b4_width(
    calibration: pd.DataFrame,
    prior: np.ndarray,
    *,
    store: str | Path,
    mappings: dict[str, Any],
    frame_alignment: str,
    seed: int,
    widths: Sequence[int] = (0, 1, 2, 3, 5, 8),
) -> tuple[int, dict[int, float]]:
    """B4's single widening width, chosen on a slice disjoint from evaluation."""
    curve: dict[int, float] = {}
    cached: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for row in calibration.itertuples():
        geometry = example_geometry(
            str(row.parent), str(row.product), str(row.pair_key), str(row.direction),
            store=store, mappings=mappings, frame_alignment=frame_alignment,
        )
        if geometry is None:
            continue
        cached.append(
            (geometry,
             draw_frames(str(row.pair_key), str(row.direction),
                         geometry["product"]["n_frames"], seed))
        )
    for width in widths:
        values = [
            baseline_scores(geometry, frames, prior, width)["B4_parent_widened"]
            for geometry, frames in cached
        ]
        if values:
            curve[int(width)] = float(np.mean(values))
    best = min(curve, key=curve.get) if curve else 0
    return int(best), curve


# --------------------------------------------------------------------------
# the model's ensemble
# --------------------------------------------------------------------------
@torch.no_grad()
def sample_product_ensembles(
    model: MappedEditHierarchicalFlow,
    examples: pd.DataFrame,
    *,
    device: torch.device,
    store: str | Path,
    mappings: dict[str, Any],
    frame_alignment: str,
    product_init_mode: str,
    max_pocket_atoms: int | None,
    seed: int,
    steps: int,
    batch_size: int,
    disable_mapping: bool = False,
    pocket_selection: str = "min_over_frames",
    temperature_by_key: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Per example, a (50, atoms, 3) product ensemble in heavy-local order.

    Also returns the DEGENERACY PROBE ensemble ``product_init +
    parent_motion_prior``: the exact function the model's fourth vector basis
    realises when its gate saturates at one.  If MEHF's torsion histograms sit
    on top of that probe, the model has learned nothing beyond re-playing the
    parent's motion on the product pose, and that must be reported.
    """
    out: dict[str, np.ndarray] = {}
    probe: dict[str, np.ndarray] = {}
    rows = list(examples.itertuples())
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
                    pocket_selection=pocket_selection,
                    frame_alignment=frame_alignment,
                    parent_frames=frames["parent"],
                    target_frames=frames["reference"],
                    include_target=False,          # the model sees no product frame
                    disable_mapping=disable_mapping,
                )
            )
            keys.append(f"{row.parent}->{row.product}")
        batch = collate_mmp_transfer(items)
        condition, _ = build_condition(batch, device)
        generator = torch.Generator(device=device).manual_seed(
            _stable_seed("sample", keys[0], seed)
        )
        if temperature_by_key is None:
            path = model.sample(
                condition, n_samples=1, steps=steps, generator=generator
            )
        else:
            # PARENT-ANCHORED PER-EXAMPLE CALIBRATION.  `sampling_temperature`
            # multiplies only the residual `(interpolated + fine)` inside
            # `sample`, so a (batch,1,1,1) tensor broadcasts exactly as the
            # scalar does and gives each example in the batch its own width.
            # Saved and restored so the model object is never left mutated.
            saved = model.sampling_temperature
            # An example whose geometry is unusable carries no calibrated
            # temperature; it falls back to the uncalibrated scalar and is
            # dropped from the report by the same geometry check downstream.
            fallback = float(
                saved.flatten()[0] if torch.is_tensor(saved) else saved
            )
            model.sampling_temperature = torch.tensor(
                [float(temperature_by_key.get(k, fallback)) for k in keys],
                device=device, dtype=condition.product_init.dtype,
            ).view(-1, 1, 1, 1)
            try:
                path = model.sample(
                    condition, n_samples=1, steps=steps, generator=generator
                )
            finally:
                model.sampling_temperature = saved
        coords = (condition.product_init[:, None] + path[0]).cpu().numpy()
        for index, (item, key) in enumerate(zip(items, keys)):
            n_atoms = int(item["product_init"].shape[0])
            out[key] = coords[index, :, :n_atoms].astype(np.float64)
            probe[key] = (
                item["product_init"][None] + item["parent_motion_prior"]
            ).astype(np.float64)
    return out, probe


def parent_anchor_rmsf(
    geometry: dict[str, Any], frames: dict[str, np.ndarray]
) -> float:
    """Mean internal RMSF of the PARENT ensemble over the mapped atoms.

    Conditioning-side only: the parent's ensemble is an input the model already
    receives, and the product's reference block never enters.  This is the
    anchor the per-example width calibration matches, so the calibration is
    computable at test time on unseen pairs and reads no target quantity.
    """
    from mdwm.eval.ensemble_metrics import rmsf_profile

    mapped = np.flatnonzero(geometry["mapping"] >= 0)
    parent_atoms = geometry["mapping"][mapped]
    block = geometry["parent"]["coords"][frames["parent"]][:, parent_atoms]
    return float(rmsf_profile(block.astype(np.float64)).mean())


def internal_rmsf(block: np.ndarray) -> float:
    from mdwm.eval.ensemble_metrics import rmsf_profile

    return float(rmsf_profile(np.asarray(block, dtype=np.float64)).mean())


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic,
    *,
    n_resamples: int = 2000,
    seed: int = 20260918,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    clusters = frame["uniprot"].to_numpy()
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    draws = []
    for _ in range(int(n_resamples)):
        chosen = rng.choice(unique, len(unique), replace=True)
        rows = np.concatenate([index_by_cluster[c] for c in chosen])
        value = statistic(frame.iloc[rows])
        if np.isfinite(value):
            draws.append(float(value))
    draws = np.asarray(draws, dtype=float)
    if draws.size == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_resamples": 0}
    return {
        "point": float(statistic(frame)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "n_resamples": int(draws.size),
    }


def _recovery(frame: pd.DataFrame, reducer) -> float:
    b1 = reducer(frame["B1_copy_parent"])
    mehf = reducer(frame["MEHF"])
    floor = reducer(frame["floor"])
    headroom = b1 - floor
    if abs(headroom) < 1.0e-12:
        return float("nan")
    return float((b1 - mehf) / headroom)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run_mmp_transfer_eval(cfg: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = dict(cfg.get("evaluation", {}))
    checkpoint_path = Path(eval_cfg["checkpoint"])
    output_root = Path(eval_cfg.get("output_root", "outputs/mehf_transfer_eval"))
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_cfg = dict(checkpoint.get("data_cfg", {}))
    data_cfg.update(eval_cfg.get("data_overrides", {}))
    store = data_cfg.get("store", str(DEFAULT_STORE))
    mappings_path = data_cfg.get("mappings_path", str(DEFAULT_MAPPINGS))
    frame_alignment = str(data_cfg.get("frame_alignment", "pocket"))
    product_init_mode = str(data_cfg.get("product_init_mode", "product_frame0"))
    max_pocket_atoms = data_cfg.get("max_pocket_atoms", 96)
    pocket_selection = str(data_cfg.get("pocket_selection", "min_over_frames"))
    mappings = load_mappings(mappings_path)
    seed = int(eval_cfg.get("seed", 20260918))

    checkpoint_dir = checkpoint_path.parent
    evaluation = pd.read_csv(eval_cfg.get(
        "evaluation_examples", checkpoint_dir / "dev_examples.csv"))
    calibration_pool = pd.read_csv(eval_cfg.get(
        "calibration_examples", checkpoint_dir / "fit_examples.csv"))
    if set(evaluation["uniprot"]) & set(calibration_pool["uniprot"]):
        raise ValueError("Calibration and evaluation sets share a UniProt target")
    max_eval = eval_cfg.get("max_evaluation_examples")
    if max_eval is not None:
        evaluation = evaluation.head(int(max_eval)).reset_index(drop=True)

    # B3 over training products only; B4's width on a disjoint calibration slice.
    prior_products = sorted(set(calibration_pool["product"].astype(str)))
    rng = np.random.default_rng(seed)
    if len(prior_products) > int(eval_cfg.get("prior_systems", 400)):
        prior_products = list(
            rng.choice(prior_products, int(eval_cfg.get("prior_systems", 400)),
                       replace=False)
        )
    prior = pooled_torsion_prior(
        prior_products, store=store, frame_alignment=frame_alignment
    )
    calibration = calibration_pool.head(
        int(eval_cfg.get("calibration_examples_used", 100))
    )
    b4_width, b4_curve = calibrate_b4_width(
        calibration, prior, store=store, mappings=mappings,
        frame_alignment=frame_alignment, seed=seed,
    )

    device = torch.device(str(eval_cfg.get("device", "cuda:0")))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dummy = {
        "product_features": torch.zeros(1, 1, int(checkpoint["atom_feature_dim"])),
        "edit_features": torch.zeros(1, 1, int(checkpoint["edit_feature_dim"])),
        "pocket_features": torch.zeros(1, 1, int(checkpoint["pocket_feature_dim"])),
    }
    model = build_model(dict(checkpoint.get("model_cfg", {})), dummy)
    model.load_state_dict(checkpoint["state_dict"])
    # SAMPLING-TIME DISPERSION CALIBRATION.  One scalar, moment-matched on the
    # FITTING folds only (generated circular spread vs reference circular
    # spread), then frozen.  It is set AFTER construction deliberately: the
    # constructor's (0,2] bound is a training-time sanity range, and the
    # calibrated value is allowed to exceed it.  It touches no weight and no
    # training objective, and it is never chosen by looking at torsion JS
    # distance (pre-registration guard 3).
    temperature = eval_cfg.get("sampling_temperature")
    if temperature is not None:
        temperature = float(temperature)
        if not 0.0 < temperature <= 8.0:
            raise ValueError("sampling_temperature calibration must lie in (0,8]")
        model.sampling_temperature = temperature
    # STAGE 1 (EXPLORATORY) SUBSPACE SCALES, PREREGISTRATION.md section
    # "OUTCOME AND TWO-STAGE CONTINUATION".  Identical contract to the
    # temperature above: these multiply the two halves of the sampling-time
    # structured noise, they touch no weight, they are moment-matched on the
    # FITTING folds, and they are frozen before any torsion JS distance is
    # computed (guard 3).  Absent, the checkpoint's own trained scales stand.
    for _scale_name in ("rigid_noise_scale", "internal_noise_scale"):
        _scale = eval_cfg.get(_scale_name)
        if _scale is not None:
            if not 0.0 < float(_scale) <= 50.0:
                raise ValueError(f"{_scale_name} must lie in (0, 50]")
            setattr(model, _scale_name, float(_scale))
    model = model.to(device).eval()

    sampling_kwargs = dict(
        device=device, store=store, mappings=mappings,
        frame_alignment=frame_alignment, product_init_mode=product_init_mode,
        max_pocket_atoms=max_pocket_atoms, pocket_selection=pocket_selection,
        seed=seed,
        steps=int(eval_cfg.get("sampling_steps", 16)),
        batch_size=int(eval_cfg.get("batch_size", 4)),
        disable_mapping=bool(data_cfg.get("disable_mapping", False)),
    )

    # ---------------------------------------------------------------
    # PARENT-ANCHORED PER-EXAMPLE WIDTH CALIBRATION
    #
    # The global scalar fails STRICTENING condition 3: the dispersion deficit
    # is target-specific (per-example required temperature spans 0.58 to 5.25),
    # so one number lands only 1 of 5 grouped folds inside [0.85, 1.15].  This
    # rule instead matches each example's generated internal RMSF to
    # `c * parent internal RMSF`, with the SINGLE scalar c fitted on the
    # fitting folds only.  Every input is conditioning-side -- the parent's
    # ensemble and the model's own output -- so nothing about the product's
    # reference block is read, and guard 3 holds: no torsion JS distance enters.
    #
    # CAVEAT, and it is part of the finding rather than a footnote: anchoring
    # on the parent imports the parent's width into the model, so the generated
    # width stops being the model's own prediction.
    #
    # `internal RMSF(T)` is affine in T (verified: linear to 2% over T in
    # [1,3]), so two probe passes determine each example's own line exactly and
    # a third pass samples at the solved temperature.
    # ---------------------------------------------------------------
    calibration = str(eval_cfg.get("calibration", "none"))
    calibration_record: dict[str, Any] = {"calibration": calibration}
    temperature_by_key: dict[str, float] | None = None
    if calibration == "parent_anchored":
        c_scalar = float(eval_cfg["calibration_c"])
        probe_low = float(eval_cfg.get("calibration_probe_low", 1.0))
        probe_high = float(eval_cfg.get("calibration_probe_high", 2.5))
        low, _ = sample_product_ensembles(model, evaluation, **sampling_kwargs)
        saved_temperature = model.sampling_temperature
        model.sampling_temperature = probe_high / probe_low * saved_temperature
        try:
            high, _ = sample_product_ensembles(model, evaluation, **sampling_kwargs)
        finally:
            model.sampling_temperature = saved_temperature
        temperature_by_key, anchors = {}, []
        for row in evaluation.itertuples():
            geometry = example_geometry(
                str(row.parent), str(row.product), str(row.pair_key),
                str(row.direction), store=store, mappings=mappings,
                frame_alignment=frame_alignment,
            )
            key = f"{row.parent}->{row.product}"
            if geometry is None or key not in low:
                continue
            drawn = draw_frames(
                str(row.pair_key), str(row.direction),
                geometry["product"]["n_frames"], seed,
            )
            # The anchor is the parent's internal RMSF over the MAPPED atoms,
            # so the generated side is measured on the same atom set; added
            # atoms have no parent counterpart and cannot be anchored.
            mapped = np.flatnonzero(geometry["mapping"] >= 0)
            target = c_scalar * parent_anchor_rmsf(geometry, drawn)
            g_low = internal_rmsf(low[key][:, mapped])
            g_high = internal_rmsf(high[key][:, mapped])
            slope = (g_high - g_low) / (probe_high - probe_low)
            intercept = g_low - slope * probe_low
            value = (
                (target - intercept) / slope if abs(slope) > 1.0e-9 else probe_low
            )
            value = float(np.clip(value, 0.05, 8.0))
            temperature_by_key[key] = value
            anchors.append(value)
        calibration_record.update({
            "calibration_c": c_scalar,
            "calibration_probe_temperatures": [probe_low, probe_high],
            "per_example_temperature_median": float(np.median(anchors)),
            "per_example_temperature_q10": float(np.quantile(anchors, 0.10)),
            "per_example_temperature_q90": float(np.quantile(anchors, 0.90)),
            "n_calibrated_examples": len(anchors),
            "note": (
                "c fitted on the FITTING folds only; the anchor is the parent's "
                "internal RMSF, a conditioning-side quantity, so the product's "
                "reference block never enters the calibration"
            ),
        })

    ensembles, probes = sample_product_ensembles(
        model, evaluation, temperature_by_key=temperature_by_key, **sampling_kwargs
    )

    rows: list[dict[str, Any]] = []
    skipped = 0
    for row in evaluation.itertuples():
        geometry = example_geometry(
            str(row.parent), str(row.product), str(row.pair_key), str(row.direction),
            store=store, mappings=mappings, frame_alignment=frame_alignment,
        )
        if geometry is None:
            skipped += 1
            continue
        frames = draw_frames(
            str(row.pair_key), str(row.direction), geometry["product"]["n_frames"], seed
        )
        record: dict[str, Any] = {
            "parent": str(row.parent), "product": str(row.product),
            "uniprot": str(row.uniprot), "direction": str(row.direction),
            "n_mapped_torsions": len(geometry["torsions"]),
            "n_mapped_atoms": int((geometry["mapping"] >= 0).sum()),
            "n_atoms": int(len(geometry["mapping"])),
        }
        record.update(baseline_scores(geometry, frames, prior, b4_width))

        key = f"{row.parent}->{row.product}"
        generated = ensembles[key]
        reference = geometry["product"]["coords"][frames["reference"]].astype(np.float64)
        report = ensemble_report(
            generated, reference,
            atomic_numbers=geometry["product"]["atomic_numbers"],
            bond_index=geometry["product"]["bond_index"],
            torsions=geometry["torsions"],
            n_samples=N_SIDE,
        )
        record["MEHF"] = float(report["torsion_js_distance"])
        record["MEHF_n_samples"] = float(report["n_samples"])
        for extra in ("energy_score", "rmsf_spearman", "rmsf_mae_angstrom",
                      "rmsf_mean_ratio", "torsion_js_distance_max"):
            record[f"MEHF_{extra}"] = float(report.get(extra, float("nan")))
        # Independent recomputation of the primary number outside ensemble_report,
        # on exactly the same frames, as a wiring check.
        direct = [
            js_distance(
                angle_histogram(dihedral_degrees(generated, q)),
                angle_histogram(
                    dihedral_degrees(
                        geometry["product"]["coords"].astype(np.float64), q
                    )[frames["reference"]]
                ),
            )
            for q in geometry["torsions"]
        ]
        record["MEHF_direct"] = float(np.mean(direct))

        # Degeneracy probe and ensemble-width readout, same torsions, same frames.
        probe = probes[key]
        product_coords = geometry["product"]["coords"].astype(np.float64)
        probe_js, agreement, spreads = [], [], {"MEHF": [], "reference": [],
                                                "probe": [], "parent": []}
        for quadruple in geometry["torsions"]:
            reference_angles = dihedral_degrees(product_coords, quadruple)[
                frames["reference"]
            ]
            reference_histogram = angle_histogram(reference_angles)
            model_angles = dihedral_degrees(generated, quadruple)
            probe_angles = dihedral_degrees(probe, quadruple)
            mapped = tuple(int(geometry["mapping"][atom]) for atom in quadruple)
            parent_angles = dihedral_degrees(
                geometry["parent"]["coords"].astype(np.float64), mapped
            )[frames["parent"]]
            probe_js.append(
                js_distance(reference_histogram, angle_histogram(probe_angles))
            )
            agreement.append(
                js_distance(angle_histogram(model_angles),
                            angle_histogram(probe_angles))
            )
            spreads["MEHF"].append(circular_spread_degrees(model_angles))
            spreads["reference"].append(circular_spread_degrees(reference_angles))
            spreads["probe"].append(circular_spread_degrees(probe_angles))
            spreads["parent"].append(circular_spread_degrees(parent_angles))
        record["probe_transferred_parent"] = float(np.mean(probe_js))
        record["MEHF_vs_probe_js"] = float(np.mean(agreement))
        for name, values in spreads.items():
            record[f"spread_{name}_degrees"] = float(np.mean(values))
        rows.append(record)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "per_pair.csv", index=False)

    keys = ["MEHF", *BASELINE_KEYS, "probe_transferred_parent"]
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "n_pairs_evaluated": int(len(frame)),
        "n_targets": int(frame["uniprot"].nunique()) if len(frame) else 0,
        "n_skipped": skipped,
        "n_samples_per_side": N_SIDE,
        "b4_widening_bins": b4_width,
        "b4_calibration_curve": b4_curve,
        "b3_prior_systems": len(prior_products),
        "product_init_mode": product_init_mode,
        "frame_alignment": frame_alignment,
        "max_pocket_atoms": max_pocket_atoms,
        "pocket_selection": pocket_selection,
        "disable_mapping": bool(data_cfg.get("disable_mapping", False)),
        "sampling_steps": int(eval_cfg.get("sampling_steps", 16)),
        "sampling_temperature": float(
            model.sampling_temperature.flatten()[0]
            if torch.is_tensor(model.sampling_temperature)
            else model.sampling_temperature
        ),
        **calibration_record,
        "device": str(device),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_gpu_memory_gb": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda" else None
        ),
        "median_mapped_torsions": (
            float(frame["n_mapped_torsions"].median()) if len(frame) else float("nan")
        ),
        "metric_wiring_check_max_abs_difference": (
            float((frame["MEHF"] - frame["MEHF_direct"]).abs().max())
            if len(frame) else float("nan")
        ),
    }
    for key in keys:
        summary[key] = {
            "median": float(frame[key].median()),
            "mean": float(frame[key].mean()),
            "q25": float(frame[key].quantile(0.25)),
            "q75": float(frame[key].quantile(0.75)),
        }

    n_resamples = int(eval_cfg.get("bootstrap_resamples", 2000))
    summary["recovery_fraction_median"] = cluster_bootstrap(
        frame, lambda f: _recovery(f, np.median),
        n_resamples=n_resamples, seed=seed,
    )
    summary["recovery_fraction_mean"] = cluster_bootstrap(
        frame, lambda f: _recovery(f, np.mean),
        n_resamples=n_resamples, seed=seed,
    )
    for other in ("B1_copy_parent", "B4_parent_widened", "B3_prior", "B2_static", "floor"):
        summary[f"median_difference_MEHF_minus_{other}"] = cluster_bootstrap(
            frame,
            lambda f, o=other: float(np.median(f["MEHF"]) - np.median(f[o])),
            n_resamples=n_resamples, seed=seed,
        )
    for column in ("MEHF_vs_probe_js", "spread_MEHF_degrees",
                   "spread_reference_degrees", "spread_probe_degrees",
                   "spread_parent_degrees", "MEHF_energy_score",
                   "MEHF_rmsf_spearman", "MEHF_rmsf_mae_angstrom",
                   "MEHF_rmsf_mean_ratio"):
        if column in frame:
            summary[column] = {"median": float(frame[column].median()),
                               "mean": float(frame[column].mean())}
    summary["note"] = (
        "torsion JS distance on fully-mapped product torsions, lower is better; "
        "every predictor uses the same pairs, torsions, frame draws and 50 "
        "samples per side; R = (B1 - MEHF) / (B1 - floor)"
    )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
