"""Phase C training: MEHF on the MISATO matched-molecular-pair transfer task.

The objective is the model's existing :meth:`MappedEditHierarchicalFlow.flow_loss`
with no term added and none removed, as PHASE_B_SPEC.md requires.  Everything
new here is plumbing: the directional pair examples, the length-bucketed
batching copied from ``mehf_pretrain``, the per-epoch redraw of target frames,
and a hard guard that this phase may read only the primary split's ``train``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mdwm.data.mmp_transfer import (
    MMPTransferDataset,
    collate_mmp_transfer,
    estimate_pair_work,
    grouped_folds,
    length_bucketed_batches,
    usable_examples,
)
from mdwm.models.mehf import MEHFCondition, MappedEditHierarchicalFlow
from mdwm.training.subspace_prior import OnlineScaleCalibrator, internal_rmsf_torch


def build_condition(
    batch: dict[str, Any], device: torch.device
) -> tuple[MEHFCondition, torch.Tensor]:
    """The Phase B conditioning contract, and nothing else.

    `product_force_prior`, `product_physics_features` and `action_features` are
    deliberately absent: this corpus ships none of them, and a zero-filled
    stand-in would be exactly the out-of-support input that destroyed three
    earlier results in this project.
    """
    keys = (
        "product_init", "target_delta", "product_features", "edit_features",
        "parent_motion_prior", "pocket_coords", "pocket_features",
        "atom_mask", "pocket_mask", "bond_index", "bond_mask",
        "reference_bond_lengths", "future_times_ps", "product_to_parent",
    )
    tensors = {key: batch[key].to(device) for key in keys}
    # The bond-angle tensors are optional so that batches built by older
    # collates (and by hand, in the tests) stay loadable; when they are absent
    # the angle term is simply inactive.
    angle = {
        key: (None if batch.get(key) is None else batch[key].to(device))
        for key in ("angle_index", "angle_mask", "reference_bond_angles")
    }
    condition = MEHFCondition(
        product_init=tensors["product_init"],
        product_features=tensors["product_features"],
        edit_features=tensors["edit_features"],
        parent_motion_prior=tensors["parent_motion_prior"],
        pocket_coords=tensors["pocket_coords"],
        pocket_features=tensors["pocket_features"],
        future_times_ps=tensors["future_times_ps"],
        product_to_parent=tensors["product_to_parent"],
        bond_index=tensors["bond_index"],
        bond_mask=tensors["bond_mask"],
        reference_bond_lengths=tensors["reference_bond_lengths"],
        angle_index=angle["angle_index"],
        angle_mask=angle["angle_mask"],
        reference_bond_angles=angle["reference_bond_angles"],
        atom_mask=tensors["atom_mask"],
        pocket_mask=tensors["pocket_mask"],
    )
    return condition, tensors["target_delta"]


def build_model(model_cfg: dict[str, Any], first_batch: dict[str, Any]) -> MappedEditHierarchicalFlow:
    return MappedEditHierarchicalFlow(
        atom_feature_dim=int(first_batch["product_features"].shape[-1]),
        edit_feature_dim=int(first_batch["edit_features"].shape[-1]),
        pocket_feature_dim=int(first_batch["pocket_features"].shape[-1]),
        coarse_stride=int(model_cfg.get("coarse_stride", 5)),
        hidden_dim=int(model_cfg.get("hidden_dim", 96)),
        n_spatial_layers=int(model_cfg.get("n_spatial_layers", 3)),
        n_temporal_layers=int(model_cfg.get("n_temporal_layers", 2)),
        nhead=int(model_cfg.get("nhead", 8)),
        bond_loss_weight=float(model_cfg.get("bond_loss_weight", 1.0)),
        angle_loss_weight=float(model_cfg.get("angle_loss_weight", 0.0)),
        clash_loss_weight=float(model_cfg.get("clash_loss_weight", 10.0)),
        clash_distance_angstrom=float(model_cfg.get("clash_distance_angstrom", 1.2)),
        translation_noise_angstrom=float(model_cfg.get("translation_noise_angstrom", 0.5)),
        rotation_noise_radian=float(model_cfg.get("rotation_noise_radian", 0.08)),
        local_noise_angstrom=float(model_cfg.get("local_noise_angstrom", 0.06)),
        rigid_noise_scale=float(model_cfg.get("rigid_noise_scale", 1.0)),
        internal_noise_scale=float(model_cfg.get("internal_noise_scale", 1.0)),
        internal_scale_head=bool(model_cfg.get("internal_scale_head", False)),
        internal_scale_head_hidden=int(model_cfg.get("internal_scale_head_hidden", 32)),
        internal_scale_head_limit=float(model_cfg.get("internal_scale_head_limit", 1.5)),
        fine_noise_fraction=float(model_cfg.get("fine_noise_fraction", 0.25)),
        temporal_noise_mode=str(model_cfg.get("temporal_noise_mode", "independent")),
        projection_iterations=int(model_cfg.get("projection_iterations", 8)),
        projection_relaxation=float(model_cfg.get("projection_relaxation", 0.5)),
        projection_angle_relaxation=float(
            model_cfg.get("projection_angle_relaxation", 0.5)
        ),
        angle_projection=bool(model_cfg.get("angle_projection", False)),
        anisotropic_local_noise=bool(
            model_cfg.get("anisotropic_local_noise", False)),
        tau_bond_angstrom=float(model_cfg.get("tau_bond_angstrom", 0.032637)),
        tau_angle_radian=float(model_cfg.get("tau_angle_radian", 0.048766)),
        reference_internal_rmsf_angstrom=float(
            model_cfg.get("reference_internal_rmsf_angstrom", 0.4640)),
        anisotropic_tau_absolute=bool(
            model_cfg.get("anisotropic_tau_absolute", False)),
        anisotropic_structure_bonds=bool(
            model_cfg.get("anisotropic_structure_bonds", True)),
        anisotropic_sweeps=int(model_cfg.get("anisotropic_sweeps", 0)),
        harmonic_local_noise=bool(model_cfg.get("harmonic_local_noise", False)),
        laplacian_local_noise=bool(model_cfg.get("laplacian_local_noise", False)),
        angle_target_width_radian=float(
            model_cfg.get("angle_target_width_radian", 0.0)
        ),
        restore_projection_amplitude=bool(
            model_cfg.get("restore_projection_amplitude", False)
        ),
        restore_iterations=int(model_cfg.get("restore_iterations", 3)),
        projection_clash_relaxation=float(model_cfg.get("projection_clash_relaxation", 1.0)),
        use_mean_backbone=bool(model_cfg.get("use_mean_backbone", True)),
        mean_loss_weight=float(model_cfg.get("mean_loss_weight", 2.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        physical_time_conditioning=bool(model_cfg.get("physical_time_conditioning", False)),
        fine_conditioning_mode=str(
            model_cfg.get("fine_conditioning_mode", "teacher_forced_coarse")
        ),
        sampling_temperature=float(model_cfg.get("sampling_temperature", 1.0)),
        sample_moment_matching=bool(model_cfg.get("sample_moment_matching", False)),
    )


def build_folded_examples(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Directional train examples, split into a fit fold and a dev fold.

    Phase C development reads the PRIMARY split's ``train`` only.  `val` and
    `test` are unreadable here by construction: requesting them raises.
    """
    data_cfg = cfg.get("data", {})
    split = str(data_cfg.get("split", "train"))
    if split != "train":
        raise ValueError(
            "Phase C development may read only the primary split's train; "
            f"refused split={split!r}"
        )
    examples = usable_examples(
        pairs_csv=data_cfg.get("pairs_csv", "outputs/mmp_transfer_corpus_v1/pairs.csv"),
        mappings_path=data_cfg.get(
            "mappings_path", "outputs/mmp_transfer_corpus_v1/atom_mappings.json"
        ),
        store=data_cfg.get("store", "data/MISATO/extracted/frames100_all_v1"),
        split="train",
    )
    max_pairs = data_cfg.get("max_pairs")
    if max_pairs is not None:
        # Subset by PAIR, never by directional example, so both directions of a
        # kept pair stay together and no target is half present.
        keys = examples["pair_key"].drop_duplicates()
        keep = set(
            keys.sample(
                min(int(max_pairs), len(keys)),
                random_state=int(data_cfg.get("subset_seed", 20260918)),
            )
        )
        examples = examples[examples["pair_key"].isin(keep)].reset_index(drop=True)
    examples = estimate_pair_work(
        examples,
        store=data_cfg.get("store", "data/MISATO/extracted/frames100_all_v1"),
        max_pocket_atoms=data_cfg.get("max_pocket_atoms", 96),
    )
    n_folds = int(data_cfg.get("n_folds", 5))
    held_out = int(data_cfg.get("held_out_fold", 0))
    examples["fold"] = grouped_folds(
        examples, n_folds, seed=int(data_cfg.get("fold_seed", 20260918))
    )
    fit = examples[examples["fold"] != held_out].reset_index(drop=True)
    dev = examples[examples["fold"] == held_out].reset_index(drop=True)
    if fit.empty or dev.empty:
        raise ValueError("Grouped CV produced an empty fold")
    if set(fit["uniprot"]) & set(dev["uniprot"]):
        raise ValueError("Grouped CV folds share a UniProt target")
    meta = {
        "n_train_pairs_available": int(examples["pair_key"].nunique()),
        "n_directional_examples": int(len(examples)),
        "n_folds": n_folds,
        "held_out_fold": held_out,
        "n_fit_examples": int(len(fit)),
        "n_dev_examples": int(len(dev)),
        "n_fit_targets": int(fit["uniprot"].nunique()),
        "n_dev_targets": int(dev["uniprot"].nunique()),
    }
    return fit, dev, meta


def _loader(
    examples: pd.DataFrame,
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    *,
    split_name: str,
    shuffle: bool,
    epoch: int,
) -> DataLoader:
    dataset = MMPTransferDataset(
        examples,
        store=data_cfg.get("store", "data/MISATO/extracted/frames100_all_v1"),
        mappings_path=data_cfg.get(
            "mappings_path", "outputs/mmp_transfer_corpus_v1/atom_mappings.json"
        ),
        product_init_mode=str(data_cfg.get("product_init_mode", "product_frame0")),
        n_frames=int(data_cfg.get("n_frames", 50)),
        max_pocket_atoms=data_cfg.get("max_pocket_atoms", 96),
        frame_alignment=str(data_cfg.get("frame_alignment", "pocket")),
        split_name=split_name,
        epoch=epoch,
        disable_mapping=bool(data_cfg.get("disable_mapping", False)),
    )
    return DataLoader(
        dataset,
        batch_sampler=length_bucketed_batches(
            examples,
            int(train_cfg.get("batch_size", 4)),
            shuffle=shuffle,
            seed=int(train_cfg.get("seed", 2026)) + epoch,
            max_pair_work=train_cfg.get("max_pair_work_per_batch"),
        ),
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_mmp_transfer,
    )


def run_train_mehf_transfer(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(cfg.get("data", {}))
    model_cfg = dict(cfg.get("model", {}))
    train_cfg = dict(cfg.get("train", {}))
    output_cfg = dict(cfg.get("outputs", {}))

    output_root = Path(output_cfg.get("root", "outputs/mehf_transfer"))
    if output_root.exists() and any(output_root.iterdir()) and not bool(
        output_cfg.get("overwrite", False)
    ):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(str(train_cfg.get("device", "cuda:0")))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    fit, dev, meta = build_folded_examples(cfg)
    fit_loader = _loader(fit, data_cfg, train_cfg, split_name="fit", shuffle=True, epoch=0)
    dev_loader = _loader(dev, data_cfg, train_cfg, split_name="dev", shuffle=False, epoch=0)
    first = next(iter(fit_loader))
    model = build_model(model_cfg, first).to(device)
    n_parameters = int(sum(p.numel() for p in model.parameters()))

    # ---- subspace-prior plumbing (V2 online calibration, V3 scale head) ----
    subspace_cfg = dict(train_cfg.get("subspace", {}))
    calibrator = None
    if bool(subspace_cfg.get("online_calibration", False)):
        from mdwm.data.mmp_transfer import DEFAULT_MAPPINGS, load_mappings

        calibrator = OnlineScaleCalibrator(
            fit,
            store=data_cfg.get("store", "data/MISATO/extracted/frames100_all_v1"),
            mappings=load_mappings(
                data_cfg.get("mappings_path", str(DEFAULT_MAPPINGS))
            ),
            frame_alignment=str(data_cfg.get("frame_alignment", "pocket")),
            product_init_mode=str(data_cfg.get("product_init_mode", "product_frame0")),
            max_pocket_atoms=data_cfg.get("max_pocket_atoms", 96),
            disable_mapping=bool(data_cfg.get("disable_mapping", False)),
            seed=int(subspace_cfg.get("calibration_seed", 20260918)),
            steps=int(subspace_cfg.get("calibration_steps", 16)),
            batch_size=int(train_cfg.get("batch_size", 4)),
            n_examples=int(subspace_cfg.get("calibration_examples", 48)),
            damping=float(subspace_cfg.get("calibration_damping", 0.8)),
            last_update_epoch=int(subspace_cfg.get("calibration_last_epoch", 4)),
        )
    head_weight = float(subspace_cfg.get("internal_head_loss_weight", 0.0))
    if (model.internal_scale_head is not None) != (head_weight > 0.0):
        raise ValueError(
            "internal_scale_head and internal_head_loss_weight must be enabled together"
        )
    running_log_reference = torch.zeros((), device=device)
    running_initialised = False
    head_momentum = float(subspace_cfg.get("internal_head_momentum", 0.99))

    def head_regression_loss(
        condition: MEHFCondition, target: torch.Tensor, train: bool
    ) -> torch.Tensor:
        """V3's auxiliary: regress the per-molecule internal log multiplier.

        The label is the example's own reference internal RMSF, centred by a
        running mean of its logarithm, so the head learns only the RELATIVE
        per-molecule modulation; the global level stays with
        `internal_noise_scale`.  The label is a training-fold target quantity
        and a moment, never a torsion Jensen-Shannon distance.  The head shares
        no parameter with the velocity fields, so this term cannot perturb the
        flow-matching objective.
        """
        nonlocal running_log_reference, running_initialised
        with torch.no_grad():
            reference = internal_rmsf_torch(
                condition.product_init[:, None] + target, condition.atom_mask
            )
            log_reference = torch.log(reference.clamp_min(1.0e-4))
            if train:
                batch_mean = log_reference.mean()
                if not running_initialised:
                    running_log_reference = batch_mean.detach()
                    running_initialised = True
                else:
                    running_log_reference = (
                        head_momentum * running_log_reference
                        + (1.0 - head_momentum) * batch_mean.detach()
                    )
            label = torch.clamp(
                log_reference - running_log_reference,
                -model.internal_scale_head_limit,
                model.internal_scale_head_limit,
            )
        return (model.internal_log_multiplier(condition) - label).square().mean()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    epochs = int(train_cfg.get("epochs", 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1),
        eta_min=float(train_cfg.get("minimum_learning_rate", 1.0e-5)),
    )
    max_steps = train_cfg.get("max_steps")
    history: list[dict[str, float]] = []
    best_value, best_state = float("inf"), None
    steps_done = 0
    stop = False

    for epoch in range(epochs):
        if epoch:
            fit_loader = _loader(
                fit, data_cfg, train_cfg, split_name="fit", shuffle=True, epoch=epoch
            )
        model.train()
        components: dict[str, list[float]] = {}
        for batch in fit_loader:
            condition, target = build_condition(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.flow_loss(target, condition)
            total = losses["loss"]
            if head_weight > 0.0:
                head_loss = head_regression_loss(condition, target, True)
                losses = {**losses, "internal_head_loss": head_loss.detach()}
                total = total + head_weight * head_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.get("gradient_clip", 1.0))
            )
            optimizer.step()
            for key, value in losses.items():
                components.setdefault(key, []).append(float(value.detach().cpu()))
            steps_done += 1
            if max_steps is not None and steps_done >= int(max_steps):
                stop = True
                break
        model.eval()
        dev_components: dict[str, list[float]] = {}
        torch.manual_seed(seed + 10_000 + epoch)
        with torch.no_grad():
            for batch in dev_loader:
                condition, target = build_condition(batch, device)
                dev_losses = dict(model.flow_loss(target, condition))
                if head_weight > 0.0:
                    # Reported only.  `dev_loss` -- the selection metric -- stays
                    # exactly the flow objective, so the auxiliary never enters
                    # checkpoint selection.
                    dev_losses["internal_head_loss"] = head_regression_loss(
                        condition, target, False
                    )
                for key, value in dev_losses.items():
                    dev_components.setdefault(key, []).append(float(value.cpu()))
        record = {
            "epoch": float(epoch + 1),
            "steps": float(steps_done),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"fit_{k}": float(np.mean(v)) for k, v in components.items()},
            **{f"dev_{k}": float(np.mean(v)) for k, v in dev_components.items()},
            "rigid_noise_scale": float(model.rigid_noise_scale),
            "internal_noise_scale": float(model.internal_noise_scale),
        }
        selectable = True
        if calibrator is not None:
            # V2 learns the two scales here, between epochs, by matching the
            # FITTING folds' generated dispersion.  While they are still
            # moving, `dev_loss` is measured under a different prior each
            # epoch and is not comparable, so those epochs are not eligible
            # for checkpoint selection; once the scales freeze, selection is
            # ordinary `dev_loss` again (pre-registration guard 1).
            calibration = calibrator.update(model, device, epoch + 1)
            record.update({
                "calibration_internal_ratio": calibration["internal_rmsf_ratio_median"],
                "calibration_centroid_ratio": calibration[
                    "centroid_dispersion_ratio_median"
                ],
                "rigid_noise_scale": calibration["rigid_noise_scale_after"],
                "internal_noise_scale": calibration["internal_noise_scale_after"],
            })
            selectable = bool(calibration["updated"] == 0.0)
        record["selectable"] = float(selectable)
        history.append(record)
        # Selection is on the held-out fold's TRAINING objective, never on any
        # quantity containing the reported torsion metric (pre-registration
        # guard 3).  `dev_mean_path_loss` is a poor criterion here -- the
        # conditional mean of a ligand's displacement is near zero, so it is
        # flat to three decimals across a run -- so the default is the total.
        selection_name = str(train_cfg.get("selection_metric", "dev_loss"))
        selection = record[selection_name]
        if selectable and selection < best_value:
            best_value = selection
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()
        print(json.dumps(record), flush=True)
        if stop:
            break

    if best_state is None and history:
        # Only reachable if every epoch ran while V2's scales were still
        # moving; the last epoch is then the only one measured under the final
        # prior, so it is the honest fallback.
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_value = history[-1][str(train_cfg.get("selection_metric", "dev_loss"))]
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    # The two subspace scales are plain floats, not parameters, so they travel
    # with the checkpoint through `model_cfg`; a rebuilt model then samples
    # under exactly the prior it was trained under.
    model_cfg = {
        **model_cfg,
        "rigid_noise_scale": float(model.rigid_noise_scale),
        "internal_noise_scale": float(model.internal_noise_scale),
    }
    checkpoint_path = output_root / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model_cfg": model_cfg,
            "data_cfg": data_cfg,
            "atom_feature_dim": int(first["product_features"].shape[-1]),
            "edit_feature_dim": int(first["edit_features"].shape[-1]),
            "pocket_feature_dim": int(first["pocket_features"].shape[-1]),
        },
        checkpoint_path,
    )
    fit[["parent", "product", "uniprot", "direction", "pair_key", "fold"]].to_csv(
        output_root / "fit_examples.csv", index=False
    )
    dev[["parent", "product", "uniprot", "direction", "pair_key", "fold"]].to_csv(
        output_root / "dev_examples.csv", index=False
    )
    pd.DataFrame(history).to_csv(output_root / "history.csv", index=False)
    summary = {
        **meta,
        "n_parameters": n_parameters,
        "rigid_noise_scale": float(model.rigid_noise_scale),
        "internal_noise_scale": float(model.internal_noise_scale),
        "internal_scale_head": bool(model.internal_scale_head is not None),
        "internal_scale_head_parameters": int(
            0 if model.internal_scale_head is None
            else sum(p.numel() for p in model.internal_scale_head.parameters())
        ),
        "scale_calibration_history": (
            None if calibrator is None else calibrator.history
        ),
        "device": str(device),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_gpu_memory_gb": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda" else None
        ),
        "epochs_run": len(history),
        "steps_done": steps_done,
        "best_selection_value": best_value,
        "selection_metric": str(train_cfg.get("selection_metric", "dev_loss")),
        "checkpoint": str(checkpoint_path),
        "product_init_mode": str(data_cfg.get("product_init_mode", "product_frame0")),
        "frame_alignment": str(data_cfg.get("frame_alignment", "pocket")),
        "max_pocket_atoms": data_cfg.get("max_pocket_atoms", 96),
        "disable_mapping": bool(data_cfg.get("disable_mapping", False)),
        "history": history,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
