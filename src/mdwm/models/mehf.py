"""Mapped Edit Hierarchical Flow (MEHF) for post-edit atomic trajectories.

The module implements the architectural core described in the theory report:

* a fixed *product* topology produced by the edited-state constructor;
* parent motion transferred only through an explicit product-to-parent map;
* an SE(3)-equivariant velocity field over the complete future path;
* coarse anchor generation followed by fine residual interpolation.

It deliberately contains no claim that existing 39-dimensional trace data can
train this all-atom model.  Training requires fixed-topology pretraining followed
by same-parent-microstate edit interventions.
"""

from __future__ import annotations

import contextlib

from dataclasses import dataclass
import math

import torch
from torch import nn

from mdwm.models.trace_flow import timestep_embedding


@dataclass
class MEHFCondition:
    """Product-indexed conditions for one padded batch.

    `parent_motion_prior` is a vector path already transferred through the atom
    map and resampled to the requested output times.  It is zero for added atoms.
    `edit_features` should at least identify mapped, added, and attachment atoms.
    """

    product_init: torch.Tensor
    product_features: torch.Tensor
    edit_features: torch.Tensor
    parent_motion_prior: torch.Tensor
    pocket_coords: torch.Tensor
    pocket_features: torch.Tensor
    future_times_ps: torch.Tensor | None = None
    product_force_prior: torch.Tensor | None = None
    product_physics_features: torch.Tensor | None = None
    action_features: torch.Tensor | None = None
    product_to_parent: torch.Tensor | None = None
    bond_index: torch.Tensor | None = None
    bond_mask: torch.Tensor | None = None
    reference_bond_lengths: torch.Tensor | None = None
    angle_index: torch.Tensor | None = None
    angle_mask: torch.Tensor | None = None
    reference_bond_angles: torch.Tensor | None = None
    atom_mask: torch.Tensor | None = None
    pocket_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class MEHFParentNoisePlan:
    """Reusable latent innovations coupled through parent atom identities.

    Sibling products may have different atom orders and added atoms.  This plan
    shares rigid translation, angular velocity, and mapped-parent local noise;
    added atoms receive no independent local innovation.  It therefore isolates
    chemical-conditioning effects instead of mixing them with Monte Carlo noise.
    """

    coarse_translation: torch.Tensor
    coarse_rotation: torch.Tensor
    coarse_parent_local: torch.Tensor
    fine_translation: torch.Tensor
    fine_rotation: torch.Tensor
    fine_parent_local: torch.Tensor


def transfer_parent_motion_to_product(
    parent_coords: torch.Tensor,
    product_to_parent: torch.Tensor,
    n_output_frames: int,
) -> torch.Tensor:
    """Map parent displacements to product atoms and linearly resample in time.

    `product_to_parent[b, q]` is the parent atom index for product atom `q`, or
    -1 for a newly added product atom.  Absolute translations cancel before the
    gather, so the result is an SE(3)-equivariant vector prior with exact zeros on
    added atoms.
    """
    if parent_coords.ndim != 4 or parent_coords.shape[-1] != 3:
        raise ValueError("parent_coords must have shape (batch,time,parent_atoms,3)")
    if product_to_parent.ndim != 2 or product_to_parent.shape[0] != parent_coords.shape[0]:
        raise ValueError("product_to_parent must have shape (batch,product_atoms)")
    if n_output_frames < 2:
        raise ValueError("n_output_frames must be at least two")
    batch, parent_frames, parent_atoms, _ = parent_coords.shape
    if parent_frames < 2:
        raise ValueError("parent_coords must contain at least two frames")
    mapped = product_to_parent >= 0
    if torch.any(product_to_parent[mapped] >= parent_atoms):
        raise ValueError("product_to_parent contains an out-of-range parent atom index")
    safe_indices = product_to_parent.clamp_min(0)
    displacement = parent_coords - parent_coords[:, :1]
    gathered = torch.gather(
        displacement,
        2,
        safe_indices[:, None, :, None].expand(batch, parent_frames, -1, 3),
    )
    gathered = gathered * mapped[:, None, :, None]
    if parent_frames == n_output_frames:
        return gathered
    # interpolate expects (N,C,L), so flatten batch/product/vector channels.
    values = gathered.permute(0, 2, 3, 1).reshape(batch, -1, parent_frames)
    values = nn.functional.interpolate(values, size=n_output_frames, mode="linear", align_corners=True)
    return values.view(batch, product_to_parent.shape[1], 3, n_output_frames).permute(0, 3, 1, 2)


def mapped_edit_node_features(
    product_to_parent: torch.Tensor,
    *,
    attachment_mask: torch.Tensor | None = None,
    bond_change_mask: torch.Tensor | None = None,
    deleted_atom_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create the minimum auditable per-product-node edit representation.

    Channels are mapped, added, attachment, bond-changed, and global deleted-atom
    count.  The last channel is repeated on product nodes so deletion information
    remains visible even though deleted atoms are not output nodes.
    """
    if product_to_parent.ndim != 2:
        raise ValueError("product_to_parent must have shape (batch,product_atoms)")
    mapped = product_to_parent >= 0
    added = ~mapped
    attachment = torch.zeros_like(mapped) if attachment_mask is None else attachment_mask.bool()
    bond_changed = torch.zeros_like(mapped) if bond_change_mask is None else bond_change_mask.bool()
    if attachment.shape != mapped.shape or bond_changed.shape != mapped.shape:
        raise ValueError("attachment_mask and bond_change_mask must match product_to_parent")
    if deleted_atom_count is None:
        deleted = torch.zeros(product_to_parent.shape[0], device=product_to_parent.device)
    else:
        deleted = deleted_atom_count.to(device=product_to_parent.device, dtype=torch.float32).reshape(-1)
        if deleted.shape[0] != product_to_parent.shape[0]:
            raise ValueError("deleted_atom_count must contain one value per batch item")
    deleted = torch.log1p(deleted).view(-1, 1).expand_as(product_to_parent).to(torch.float32)
    return torch.stack(
        [mapped.to(torch.float32), added.to(torch.float32), attachment.to(torch.float32),
         bond_changed.to(torch.float32), deleted],
        dim=-1,
    )


class _EquivariantInteractionLayer(nn.Module):
    """Scalar messages plus relative-vector aggregation; no Cartesian linear maps."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.ligand_message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.ligand_vector_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.pocket_message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.pocket_vector_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.node_update = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        pocket_h: torch.Tensor,
        pocket_coords: torch.Tensor,
        atom_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # h/coords: (B,T,N,H)/(B,T,N,3); pocket: (B,M,H)/(B,M,3)
        batch, frames, atoms, _ = h.shape
        relative = coords.unsqueeze(3) - coords.unsqueeze(2)  # i - j
        distance_sq = relative.square().sum(dim=-1, keepdim=True)
        radial = torch.log1p(distance_sq)
        unit_relative = relative / distance_sq.clamp_min(1.0e-8).sqrt()
        h_i = h.unsqueeze(3).expand(-1, -1, -1, atoms, -1)
        h_j = h.unsqueeze(2).expand(-1, -1, atoms, -1, -1)
        ligand_message = self.ligand_message(torch.cat([h_i, h_j, radial], dim=-1))
        pair_mask = atom_mask[:, None, :, None] & atom_mask[:, None, None, :]
        eye = torch.eye(atoms, device=h.device, dtype=torch.bool).view(1, 1, atoms, atoms)
        pair_mask = pair_mask & ~eye
        ligand_message = ligand_message * pair_mask.unsqueeze(-1)
        ligand_count = pair_mask.sum(dim=3, keepdim=True).clamp_min(1)
        ligand_scalar = ligand_message.sum(dim=3) / ligand_count
        ligand_vector = (
            torch.tanh(self.ligand_vector_weight(ligand_message)) * unit_relative * pair_mask.unsqueeze(-1)
        ).sum(dim=3) / ligand_count

        if pocket_coords.shape[1] == 0:
            pocket_scalar = torch.zeros_like(ligand_scalar)
            pocket_vector = torch.zeros_like(ligand_vector)
        else:
            pocket_relative = coords.unsqueeze(3) - pocket_coords[:, None, None, :, :]
            pocket_distance_sq = pocket_relative.square().sum(dim=-1, keepdim=True)
            pocket_radial = torch.log1p(pocket_distance_sq)
            pocket_unit_relative = pocket_relative / pocket_distance_sq.clamp_min(1.0e-8).sqrt()
            hp = pocket_h[:, None, None, :, :].expand(batch, frames, atoms, -1, -1)
            hi = h.unsqueeze(3).expand(-1, -1, -1, pocket_coords.shape[1], -1)
            pocket_message = self.pocket_message(torch.cat([hi, hp, pocket_radial], dim=-1))
            cross_mask = atom_mask[:, None, :, None] & pocket_mask[:, None, None, :]
            pocket_message = pocket_message * cross_mask.unsqueeze(-1)
            pocket_count = cross_mask.sum(dim=3, keepdim=True).clamp_min(1)
            pocket_scalar = pocket_message.sum(dim=3) / pocket_count
            pocket_vector = (
                torch.tanh(self.pocket_vector_weight(pocket_message))
                * pocket_unit_relative
                * cross_mask.unsqueeze(-1)
            ).sum(dim=3) / pocket_count

        h = h + self.node_update(torch.cat([h, ligand_scalar, pocket_scalar], dim=-1))
        h = h * atom_mask[:, None, :, None]
        return h, ligand_vector, pocket_vector


class MappedEditEquivariantVelocity(nn.Module):
    """SE(3)-equivariant conditional velocity on a product displacement path."""

    def __init__(
        self,
        atom_feature_dim: int,
        edit_feature_dim: int,
        pocket_feature_dim: int,
        hidden_dim: int = 128,
        n_spatial_layers: int = 3,
        n_temporal_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
        edit_adapter_dim: int = 0,
        edit_adapter_product_conditioning: bool = False,
        edit_adapter_locality_gating: bool = False,
        product_physics_feature_dim: int = 0,
        action_feature_dim: int = 0,
        use_product_force_prior: bool = True,
        use_product_physics_conditioning: bool = True,
        use_action_conditioning: bool = True,
        physical_time_conditioning: bool = False,
        physical_time_scale_ps: float = 10.0,
    ) -> None:
        super().__init__()
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.hidden_dim = hidden_dim
        self.physical_time_conditioning = bool(physical_time_conditioning)
        self.physical_time_scale_ps = float(physical_time_scale_ps)
        if self.physical_time_scale_ps <= 0:
            raise ValueError("physical_time_scale_ps must be positive")
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_feature_dim + edit_feature_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edit_adapter: nn.ModuleDict | None = None
        self.edit_adapter_product_conditioning = bool(edit_adapter_product_conditioning)
        self.edit_adapter_locality_gating = bool(edit_adapter_locality_gating)
        self.product_physics_feature_dim = int(product_physics_feature_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.use_product_force_prior = bool(use_product_force_prior)
        self.use_product_physics_conditioning = bool(
            use_product_physics_conditioning
        )
        self.use_action_conditioning = bool(use_action_conditioning)
        if self.product_physics_feature_dim < 0 or self.action_feature_dim < 0:
            raise ValueError("Conditioning feature dimensions must be non-negative")
        if int(edit_adapter_dim) > 0:
            # Parameter-efficient intervention adaptation. One residual updates
            # hidden scalars; the second directly modulates the five equivariant
            # vector bases. Both outputs are zero initialized, so insertion is
            # exactly function preserving for an E2 checkpoint.
            adapter_input_dim = edit_feature_dim + (
                atom_feature_dim if self.edit_adapter_product_conditioning else 0
            )
            self.edit_adapter = nn.ModuleDict(
                {
                    "hidden": nn.Sequential(
                        nn.Linear(adapter_input_dim, int(edit_adapter_dim), bias=False),
                        nn.SiLU(),
                        nn.Linear(int(edit_adapter_dim), hidden_dim, bias=False),
                    ),
                    "gate": nn.Sequential(
                        nn.Linear(adapter_input_dim, int(edit_adapter_dim), bias=False),
                        nn.SiLU(),
                        nn.Linear(int(edit_adapter_dim), 5, bias=False),
                    ),
                }
            )
            if self.product_physics_feature_dim:
                self.edit_adapter["physics_encoder"] = nn.Sequential(
                    nn.Linear(
                        self.product_physics_feature_dim,
                        int(edit_adapter_dim),
                        bias=False,
                    ),
                    nn.SiLU(),
                )
                self.edit_adapter["physics_hidden"] = nn.Linear(
                    int(edit_adapter_dim), hidden_dim, bias=False
                )
                self.edit_adapter["physics_gate"] = nn.Linear(
                    int(edit_adapter_dim), 5, bias=False
                )
            if self.action_feature_dim:
                self.edit_adapter["action_encoder"] = nn.Sequential(
                    nn.Linear(
                        self.action_feature_dim,
                        int(edit_adapter_dim),
                        bias=False,
                    ),
                    nn.SiLU(),
                )
                self.edit_adapter["action_hidden"] = nn.Linear(
                    int(edit_adapter_dim), hidden_dim, bias=False
                )
                self.edit_adapter["action_gate"] = nn.Linear(
                    int(edit_adapter_dim), 5, bias=False
                )
            nn.init.zeros_(self.edit_adapter["hidden"][-1].weight)
            nn.init.zeros_(self.edit_adapter["gate"][-1].weight)
            for key in ("physics_hidden", "physics_gate", "action_hidden", "action_gate"):
                if key in self.edit_adapter:
                    nn.init.zeros_(self.edit_adapter[key].weight)
        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.physical_time_encoder: nn.Module | None = None
        if self.physical_time_conditioning:
            self.physical_time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
            )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first is intentional; disable the incompatible nested-tensor
        # fast path explicitly instead of emitting a warning on every load.
        self.temporal = nn.TransformerEncoder(
            temporal_layer, num_layers=n_temporal_layers, enable_nested_tensor=False
        )
        self.spatial = nn.ModuleList(
            [_EquivariantInteractionLayer(hidden_dim) for _ in range(n_spatial_layers)]
        )
        # Five equivariant vector bases: ligand message, pocket message,
        # displacement from product anchor, mapped parent motion, and a known
        # product-force prior.  The network learns scalar residual/gating only.
        self.vector_gates = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5)
        )
        nn.init.zeros_(self.vector_gates[-1].weight)
        nn.init.zeros_(self.vector_gates[-1].bias)

    @staticmethod
    def _validate(condition: MEHFCondition, noisy_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy_delta.ndim != 4 or noisy_delta.shape[-1] != 3:
            raise ValueError("noisy_delta must have shape (batch,time,product_atoms,3)")
        batch, frames, atoms, _ = noisy_delta.shape
        if condition.product_init.shape != (batch, atoms, 3):
            raise ValueError("product_init shape is inconsistent with noisy_delta")
        if condition.parent_motion_prior.shape != noisy_delta.shape:
            raise ValueError("parent_motion_prior must be product-indexed and match noisy_delta")
        if condition.product_force_prior is not None and condition.product_force_prior.shape != noisy_delta.shape:
            raise ValueError("product_force_prior must match noisy_delta when provided")
        if (
            condition.product_physics_features is not None
            and condition.product_physics_features.shape[:2] != (batch, atoms)
        ):
            raise ValueError("product_physics_features must have shape (batch,atoms,features)")
        if condition.action_features is not None and (
            condition.action_features.ndim != 2
            or condition.action_features.shape[0] != batch
        ):
            raise ValueError("action_features must have shape (batch,features)")
        if (
            condition.product_to_parent is not None
            and condition.product_to_parent.shape != (batch, atoms)
        ):
            raise ValueError("product_to_parent must have shape (batch,atoms)")
        atom_mask = condition.atom_mask
        if atom_mask is None:
            atom_mask = torch.ones(batch, atoms, dtype=torch.bool, device=noisy_delta.device)
        pocket_mask = condition.pocket_mask
        if pocket_mask is None:
            pocket_mask = torch.ones(
                batch, condition.pocket_coords.shape[1], dtype=torch.bool, device=noisy_delta.device
            )
        return atom_mask.bool(), pocket_mask.bool()

    def forward(
        self,
        noisy_delta: torch.Tensor,
        flow_time: torch.Tensor,
        condition: MEHFCondition,
        *,
        include_edit_adapter: bool = True,
    ) -> torch.Tensor:
        atom_mask, pocket_mask = self._validate(condition, noisy_delta)
        batch, frames, atoms, _ = noisy_delta.shape
        if flow_time.ndim == 1:
            flow_time = flow_time[:, None].expand(-1, frames)
        if flow_time.shape != (batch, frames):
            raise ValueError("flow_time must have shape (batch,) or (batch,time)")

        parent_magnitude = condition.parent_motion_prior.norm(dim=-1).mean(dim=1, keepdim=False).unsqueeze(-1)
        base_edit_features = condition.edit_features
        adapter_edit_features = condition.edit_features
        if self.edit_adapter is not None:
            # E2 only observes the neutral fixed-topology code: every valid
            # product atom is mapped and no local edit flag is active. Feeding
            # unseen action channels through the frozen E2 atom encoder would
            # expose random, never-trained weights. Keep the base exactly on its
            # training support and give only the action residual to adapters.
            neutral = torch.zeros_like(condition.edit_features)
            neutral[..., 0] = atom_mask.to(neutral.dtype)
            base_edit_features = neutral
            adapter_edit_features = condition.edit_features - neutral
        adapter_features = adapter_edit_features
        if self.edit_adapter is not None and self.edit_adapter_product_conditioning:
            # Product chemistry is part of the intervention, not merely the
            # location of an edit. The zero-initialized adapter may therefore
            # learn element/environment-specific corrections while the frozen
            # E2 path remains exactly on its original support.
            adapter_features = torch.cat(
                [condition.product_features, adapter_edit_features], dim=-1
            )
        adapter_locality = torch.ones(
            batch, atoms, 1, device=noisy_delta.device, dtype=noisy_delta.dtype
        )
        if self.edit_adapter_locality_gating:
            if adapter_edit_features.shape[-1] < 2:
                raise ValueError("Locality-gated edit adapters require local edit channels")
            # Channels 1:4 are added, attachment, and bond-change indicators.
            # Product chemistry is allowed to modulate the correction only at
            # intervention-local nodes; equivariant spatial messages then carry
            # supported effects to the rest of the molecule.
            local_channels = adapter_edit_features[..., 1:4]
            adapter_locality = local_channels.abs().amax(dim=-1, keepdim=True)
        atom_input = torch.cat(
            [condition.product_features, base_edit_features, parent_magnitude], dim=-1
        )
        base_h = self.atom_encoder(atom_input)
        physics_latent: torch.Tensor | None = None
        action_latent: torch.Tensor | None = None
        if self.edit_adapter is not None and include_edit_adapter:
            base_h = base_h + adapter_locality * self.edit_adapter["hidden"](
                adapter_features
            )
            if (
                self.product_physics_feature_dim
                and self.use_product_physics_conditioning
            ):
                physics = condition.product_physics_features
                if physics is None or physics.shape != (
                    batch,
                    atoms,
                    self.product_physics_feature_dim,
                ):
                    raise ValueError(
                        "Configured product physics adapter requires matching features"
                    )
                physics_latent = self.edit_adapter["physics_encoder"](physics)
                base_h = base_h + self.edit_adapter["physics_hidden"](
                    physics_latent
                )
            if self.action_feature_dim and self.use_action_conditioning:
                action = condition.action_features
                if action is None or action.shape != (batch, self.action_feature_dim):
                    raise ValueError(
                        "Configured global action adapter requires matching features"
                    )
                # The intervention token is global: every product atom receives
                # it, while equivariant geometry determines the resulting vector.
                action_latent = self.edit_adapter["action_encoder"](action)
                base_h = base_h + self.edit_adapter["action_hidden"](
                    action_latent
                )[:, None]
        time_h = self.time_encoder(timestep_embedding(flow_time.reshape(-1), self.hidden_dim)).view(
            batch, frames, self.hidden_dim
        )
        if self.physical_time_encoder is not None:
            if condition.future_times_ps is None:
                raise ValueError("future_times_ps is required when physical time conditioning is enabled")
            physical_time = condition.future_times_ps.to(
                device=noisy_delta.device, dtype=noisy_delta.dtype
            )
            if physical_time.shape != (batch, frames):
                raise ValueError("future_times_ps must have shape (batch,time)")
            if torch.any(~torch.isfinite(physical_time)) or torch.any(physical_time <= 0):
                raise ValueError("future_times_ps must be finite and positive")
            scaled_time = torch.log1p(physical_time / self.physical_time_scale_ps)
            physical_h = self.physical_time_encoder(
                timestep_embedding(scaled_time.reshape(-1), self.hidden_dim)
            ).view(batch, frames, self.hidden_dim)
            time_h = time_h + physical_h
        h = base_h[:, None, :, :] + time_h[:, :, None, :]
        h = self.temporal(h.transpose(1, 2).reshape(batch * atoms, frames, self.hidden_dim))
        h = h.view(batch, atoms, frames, self.hidden_dim).transpose(1, 2)

        coords = condition.product_init[:, None, :, :] + noisy_delta
        pocket_h = self.pocket_encoder(condition.pocket_features)
        ligand_vector = torch.zeros_like(noisy_delta)
        pocket_vector = torch.zeros_like(noisy_delta)
        for layer in self.spatial:
            h, ligand_update, pocket_update = layer(
                h, coords, pocket_h, condition.pocket_coords, atom_mask, pocket_mask
            )
            ligand_vector = ligand_vector + ligand_update
            pocket_vector = pocket_vector + pocket_update

        gate_logits = self.vector_gates(h)
        if self.edit_adapter is not None and include_edit_adapter:
            gate_logits = gate_logits + (
                adapter_locality * self.edit_adapter["gate"](adapter_features)
            )[:, None]
            if (
                self.product_physics_feature_dim
                and self.use_product_physics_conditioning
            ):
                assert physics_latent is not None
                gate_logits = gate_logits + self.edit_adapter["physics_gate"](
                    physics_latent
                )[:, None]
            if self.action_feature_dim and self.use_action_conditioning:
                assert action_latent is not None
                gate_logits = gate_logits + self.edit_adapter["action_gate"](
                    action_latent
                )[:, None, None]
        gates = torch.tanh(gate_logits)
        anchor_vector = noisy_delta
        force_prior = (
            torch.zeros_like(noisy_delta)
            if condition.product_force_prior is None or not self.use_product_force_prior
            else condition.product_force_prior
        )
        bases = torch.stack(
            [ligand_vector, pocket_vector, anchor_vector, condition.parent_motion_prior, force_prior], dim=-2
        )
        velocity = (gates.unsqueeze(-1) * bases).sum(dim=-2)
        return velocity * atom_mask[:, None, :, None]


def _coarse_indices(n_frames: int, stride: int, device: torch.device) -> torch.Tensor:
    if n_frames < 2:
        raise ValueError("MEHF requires at least two future frames")
    if stride < 1:
        raise ValueError("coarse_stride must be positive")
    indices = list(range(0, n_frames, stride))
    if indices[-1] != n_frames - 1:
        indices.append(n_frames - 1)
    return torch.tensor(indices, dtype=torch.long, device=device)


def interpolate_coarse_path(coarse: torch.Tensor, indices: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Piecewise-linear SE(3)-equivariant interpolation of coarse displacements."""
    out = torch.empty(
        coarse.shape[0], n_frames, coarse.shape[2], 3, device=coarse.device, dtype=coarse.dtype
    )
    for segment in range(indices.numel() - 1):
        left, right = int(indices[segment]), int(indices[segment + 1])
        alpha = torch.linspace(0.0, 1.0, right - left + 1, device=coarse.device, dtype=coarse.dtype)
        out[:, left : right + 1] = (
            (1.0 - alpha)[None, :, None, None] * coarse[:, segment : segment + 1]
            + alpha[None, :, None, None] * coarse[:, segment + 1 : segment + 2]
        )
    return out


class MappedEditHierarchicalFlow(nn.Module):
    """Coarse whole-path flow followed by fine residual flow."""

    def __init__(
        self,
        atom_feature_dim: int,
        edit_feature_dim: int,
        pocket_feature_dim: int,
        coarse_stride: int = 5,
        hidden_dim: int = 128,
        n_spatial_layers: int = 3,
        n_temporal_layers: int = 2,
        nhead: int = 4,
        bond_loss_weight: float = 0.0,
        angle_loss_weight: float = 0.0,
        clash_loss_weight: float = 0.0,
        clash_distance_angstrom: float = 1.2,
        translation_noise_angstrom: float = 0.5,
        rotation_noise_radian: float = 0.08,
        local_noise_angstrom: float = 0.08,
        rigid_noise_scale: float = 1.0,
        internal_noise_scale: float = 1.0,
        internal_scale_head: bool = False,
        internal_scale_head_hidden: int = 32,
        internal_scale_head_limit: float = 1.5,
        fine_noise_fraction: float = 0.25,
        temporal_noise_mode: str = "independent",
        noise_time_scale_ps: float = 10.0,
        projection_iterations: int = 0,
        projection_relaxation: float = 0.5,
        projection_angle_relaxation: float = 0.5,
        angle_projection: bool = False,
        angle_target_width_radian: float = 0.0,
        anisotropic_local_noise: bool = False,
        tau_bond_angstrom: float = 0.032637,
        tau_angle_radian: float = 0.048766,
        reference_internal_rmsf_angstrom: float = 0.4640,
        anisotropic_tau_absolute: bool = False,
        anisotropic_structure_bonds: bool = True,
        anisotropic_sweeps: int = 0,
        harmonic_local_noise: bool = False,
        laplacian_local_noise: bool = False,
        restore_projection_amplitude: bool = False,
        restore_iterations: int = 3,
        projection_clash_relaxation: float = 1.0,
        use_mean_backbone: bool = False,
        mean_loss_weight: float = 1.0,
        dropout: float = 0.0,
        edit_adapter_dim: int = 0,
        edit_adapter_product_conditioning: bool = False,
        edit_adapter_locality_gating: bool = False,
        product_physics_feature_dim: int = 0,
        action_feature_dim: int = 0,
        use_product_force_prior: bool = True,
        use_product_physics_conditioning: bool = True,
        use_action_conditioning: bool = True,
        center_mean_edit_residual_translation: bool = False,
        physical_time_conditioning: bool = False,
        physical_time_scale_ps: float = 10.0,
        fine_conditioning_mode: str = "teacher_forced_coarse",
        sampling_temperature: float = 1.0,
        share_conditioning_encoders: bool = False,
        sample_moment_matching: bool = False,
        sample_moment_matching_iterations: int = 1,
    ) -> None:
        super().__init__()
        self.coarse_stride = coarse_stride
        self.bond_loss_weight = float(bond_loss_weight)
        # BOND ANGLE TERM.  Constructed symmetrically to the bond length term:
        # a penalty on the deviation of the (rotation- and translation-
        # invariant) angle from its reference value, plus an optional
        # inference projection sharing `projection_iterations`.  Both default
        # OFF -- weight 0.0 and `angle_projection` False -- so a model built
        # without them is bit-identical to the original and every historical
        # checkpoint and config reproduces exactly.
        self.angle_loss_weight = float(angle_loss_weight)
        self.angle_projection = bool(angle_projection)
        # 0.0 keeps the hard constraint, so every existing checkpoint and config
        # reproduces exactly; the corpus-measured physical angle width is
        # 2.794 degrees = 0.048766 rad.
        self.angle_target_width_radian = float(angle_target_width_radian)
        # OFF by default, so every existing checkpoint and config reproduces
        # exactly.  The widths are the corpus measurements reported in
        # outputs/isotropic_noise_cost_v1/FINDING.md (heavy atoms: bond
        # 0.0326 A, angle 2.794 deg, internal RMSF 0.4640 A), not fitted values.
        self.anisotropic_local_noise = bool(anisotropic_local_noise)
        self.tau_bond_angstrom = float(tau_bond_angstrom)
        self.tau_angle_radian = float(tau_angle_radian)
        self.reference_internal_rmsf_angstrom = float(reference_internal_rmsf_angstrom)
        # Both default to the aniso_prior_v2 behaviour, so its checkpoints
        # reproduce exactly.
        self.anisotropic_tau_absolute = bool(anisotropic_tau_absolute)
        self.anisotropic_structure_bonds = bool(anisotropic_structure_bonds)
        # 0 = borrow projection_iterations, the aniso_prior_v2 behaviour.
        self.anisotropic_sweeps = int(anisotropic_sweeps)
        # Exact Sigma* prior.  OFF by default so every existing checkpoint and
        # config reproduces bit-identically.
        self.harmonic_local_noise = bool(harmonic_local_noise)
        self._harmonic_cache: dict = {}
        # Bond-graph Laplacian prior (EigenFold / HarmonicFlow), the comparison arm
        # of outputs/laplacian_prior_v1.  OFF by default; mutually exclusive with
        # the Sigma* prior.
        self.laplacian_local_noise = bool(laplacian_local_noise)
        self._laplacian_cache: dict = {}
        if self.laplacian_local_noise and self.harmonic_local_noise:
            raise ValueError("laplacian_local_noise and harmonic_local_noise are exclusive")
        # AMPLITUDE RESTORE (step 4 of the anisotropic-sampler recipe).  A
        # projection removes amplitude from the stiff directions and therefore
        # leaves the sample globally narrower than the model drew it.  This puts
        # that amplitude back by rescaling the projected displacement with ONE
        # positive scalar per sample, chosen so the sample's own internal RMSF
        # returns to its pre-projection value.  It is inference-only -- it lives
        # inside `_project_geometry`, which is reached only from `sample()` --
        # and defaults OFF, so nothing already measured moves.
        self.restore_projection_amplitude = bool(restore_projection_amplitude)
        self.restore_iterations = int(restore_iterations)
        if self.restore_iterations < 1:
            raise ValueError("restore_iterations must be positive")
        self.clash_loss_weight = float(clash_loss_weight)
        self.clash_distance_angstrom = float(clash_distance_angstrom)
        self.translation_noise_angstrom = float(translation_noise_angstrom)
        self.rotation_noise_radian = float(rotation_noise_radian)
        self.local_noise_angstrom = float(local_noise_angstrom)
        # SUBSPACE PRIOR (outputs/subspace_prior_v1/PREREGISTRATION.md).  The
        # three hand-set scales above allocate the prior's budget as rigid
        # translation, rigid rotation and internal displacement; only the last
        # changes a torsion angle.  These two multipliers let the rigid and the
        # internal subspace be scaled independently.  Both default to 1.0, so a
        # model built without them is bit-identical to the original prior and
        # every historical checkpoint keeps loading.
        self.rigid_noise_scale = float(rigid_noise_scale)
        self.internal_noise_scale = float(internal_noise_scale)
        if self.rigid_noise_scale <= 0.0 or self.internal_noise_scale <= 0.0:
            raise ValueError("subspace noise scales must be positive")
        self.internal_scale_head_limit = float(internal_scale_head_limit)
        self.fine_noise_fraction = float(fine_noise_fraction)
        self.temporal_noise_mode = str(temporal_noise_mode)
        self.noise_time_scale_ps = float(noise_time_scale_ps)
        if self.temporal_noise_mode not in {"independent", "brownian"}:
            raise ValueError("temporal_noise_mode must be independent or brownian")
        if self.noise_time_scale_ps <= 0:
            raise ValueError("noise_time_scale_ps must be positive")
        self.projection_iterations = int(projection_iterations)
        self.projection_relaxation = float(projection_relaxation)
        self.projection_angle_relaxation = float(projection_angle_relaxation)
        self.projection_clash_relaxation = float(projection_clash_relaxation)
        self.use_mean_backbone = bool(use_mean_backbone)
        self.mean_loss_weight = float(mean_loss_weight)
        self.center_mean_edit_residual_translation = bool(
            center_mean_edit_residual_translation
        )
        self.sampling_temperature = float(sampling_temperature)
        self.share_conditioning_encoders = bool(share_conditioning_encoders)
        self.sample_moment_matching = bool(sample_moment_matching)
        self.sample_moment_matching_iterations = int(
            sample_moment_matching_iterations
        )
        if not 0.0 < self.sampling_temperature <= 2.0:
            raise ValueError("sampling_temperature must be in (0,2]")
        if not 1 <= self.sample_moment_matching_iterations <= 8:
            raise ValueError(
                "sample_moment_matching_iterations must lie in [1,8]"
            )
        self.fine_conditioning_mode = str(fine_conditioning_mode)
        if self.fine_conditioning_mode not in {
            "teacher_forced_coarse",
            "denoised_coarse",
        }:
            raise ValueError(
                "fine_conditioning_mode must be teacher_forced_coarse or "
                "denoised_coarse"
            )
        if (
            self.projection_iterations < 0
            or not 0.0 < self.projection_relaxation <= 1.0
            or not 0.0 < self.projection_angle_relaxation <= 1.0
            or not 0.0 < self.projection_clash_relaxation <= 1.0
        ):
            raise ValueError("projection iterations must be non-negative and relaxations in (0,1]")
        kwargs = dict(
            atom_feature_dim=atom_feature_dim,
            edit_feature_dim=edit_feature_dim,
            pocket_feature_dim=pocket_feature_dim,
            hidden_dim=hidden_dim,
            n_spatial_layers=n_spatial_layers,
            n_temporal_layers=n_temporal_layers,
            nhead=nhead,
            dropout=float(dropout),
            edit_adapter_dim=int(edit_adapter_dim),
            edit_adapter_product_conditioning=bool(edit_adapter_product_conditioning),
            edit_adapter_locality_gating=bool(edit_adapter_locality_gating),
            product_physics_feature_dim=int(product_physics_feature_dim),
            action_feature_dim=int(action_feature_dim),
            use_product_force_prior=bool(use_product_force_prior),
            use_product_physics_conditioning=bool(
                use_product_physics_conditioning
            ),
            use_action_conditioning=bool(use_action_conditioning),
            physical_time_conditioning=bool(physical_time_conditioning),
            physical_time_scale_ps=float(physical_time_scale_ps),
        )
        self.coarse_flow = MappedEditEquivariantVelocity(**kwargs)
        self.fine_flow = MappedEditEquivariantVelocity(**kwargs)
        # V3: a per-molecule multiplier on the INTERNAL scale only.  It is a
        # standalone MLP over mask-pooled conditioning features and shares no
        # parameter with either velocity field, so the auxiliary objective that
        # trains it cannot reach into the flow-matching objective.
        self.internal_scale_head = (
            nn.Sequential(
                nn.Linear(
                    atom_feature_dim + edit_feature_dim + pocket_feature_dim + 2,
                    int(internal_scale_head_hidden),
                ),
                nn.SiLU(),
                nn.Linear(int(internal_scale_head_hidden), 1),
            )
            if internal_scale_head
            else None
        )
        if self.internal_scale_head is not None:
            nn.init.zeros_(self.internal_scale_head[-1].weight)
            nn.init.zeros_(self.internal_scale_head[-1].bias)
        self.mean_field = MappedEditEquivariantVelocity(**kwargs) if self.use_mean_backbone else None
        if self.share_conditioning_encoders and self.coarse_flow.edit_adapter is not None:
            fields = [self.fine_flow]
            if self.mean_field is not None:
                fields.append(self.mean_field)
            for key in ("physics_encoder", "action_encoder"):
                if key not in self.coarse_flow.edit_adapter:
                    continue
                for field in fields:
                    assert field.edit_adapter is not None
                    field.edit_adapter[key] = self.coarse_flow.edit_adapter[key]

    def predict_mean(
        self,
        condition: MEHFCondition,
        *,
        include_edit_adapter: bool = True,
    ) -> torch.Tensor:
        """Deterministic equivariant conditional mean path.

        Persistence is the exact initialization because all vector gates are
        zero-initialized. Training then learns only mean motion supported by
        product geometry, pocket context, parent history, and mapped edit.
        """
        if self.mean_field is None:
            return torch.zeros_like(condition.parent_motion_prior)
        batch, frames, _, _ = condition.parent_motion_prior.shape
        physical_time = torch.linspace(
            0.0, 1.0, frames, device=condition.product_init.device, dtype=condition.product_init.dtype
        ).view(1, frames).expand(batch, -1)
        full_prediction = self.mean_field(
            torch.zeros_like(condition.parent_motion_prior),
            physical_time,
            condition,
            include_edit_adapter=include_edit_adapter,
        )
        if (
            not include_edit_adapter
            or not self.center_mean_edit_residual_translation
            or self.mean_field.edit_adapter is None
        ):
            return full_prediction
        base_prediction = self.mean_field(
            torch.zeros_like(condition.parent_motion_prior),
            physical_time,
            condition,
            include_edit_adapter=False,
        )
        residual = full_prediction - base_prediction
        atom_mask = condition.atom_mask
        if atom_mask is None:
            atom_mask = torch.ones(
                residual.shape[0], residual.shape[2],
                dtype=torch.bool, device=residual.device,
            )
        weights = atom_mask[:, None, :, None].to(residual.dtype)
        translation = (residual * weights).sum(dim=2, keepdim=True) / weights.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        return base_prediction + (residual - translation) * weights

    @staticmethod
    def _subset_condition(condition: MEHFCondition, indices: torch.Tensor) -> MEHFCondition:
        return MEHFCondition(
            product_init=condition.product_init,
            product_features=condition.product_features,
            edit_features=condition.edit_features,
            parent_motion_prior=condition.parent_motion_prior.index_select(1, indices),
            pocket_coords=condition.pocket_coords,
            pocket_features=condition.pocket_features,
            future_times_ps=(
                None
                if condition.future_times_ps is None
                else condition.future_times_ps.index_select(1, indices)
            ),
            product_force_prior=(
                None if condition.product_force_prior is None else condition.product_force_prior.index_select(1, indices)
            ),
            product_physics_features=condition.product_physics_features,
            action_features=condition.action_features,
            product_to_parent=condition.product_to_parent,
            bond_index=condition.bond_index,
            bond_mask=condition.bond_mask,
            reference_bond_lengths=condition.reference_bond_lengths,
            angle_index=condition.angle_index,
            angle_mask=condition.angle_mask,
            reference_bond_angles=condition.reference_bond_angles,
            atom_mask=condition.atom_mask,
            pocket_mask=condition.pocket_mask,
        )

    def _pooled_condition_features(self, condition: MEHFCondition) -> torch.Tensor:
        """Mask-pooled conditioning summary for the per-molecule scale head.

        Every term is conditioning-side: the product's atom features, the edit
        features, the pocket features, the ligand's size, and the INTERNAL
        spread of the parent's own motion prior.  The product's reference block
        is not among the model's inputs and cannot reach here.
        """
        mask = (
            torch.ones(
                condition.product_init.shape[:2],
                device=condition.product_init.device,
                dtype=condition.product_init.dtype,
            )
            if condition.atom_mask is None
            else condition.atom_mask.to(condition.product_init.dtype)
        )
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        def pool(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            w = weights.unsqueeze(-1)
            return (value * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

        pocket_mask = (
            torch.ones(
                condition.pocket_features.shape[:2],
                device=condition.product_init.device,
                dtype=condition.product_init.dtype,
            )
            if condition.pocket_mask is None
            else condition.pocket_mask.to(condition.product_init.dtype)
        )
        motion = condition.parent_motion_prior
        frame_centre = (motion * mask[:, None, :, None]).sum(dim=2, keepdim=True) / (
            counts[:, None, :, None]
        )
        internal = motion - frame_centre
        internal = internal - internal.mean(dim=1, keepdim=True)
        spread = torch.sqrt(
            (internal.square().sum(dim=-1) * mask[:, None, :]).sum(dim=(1, 2))
            / (counts.squeeze(-1) * motion.shape[1]).clamp_min(1.0)
            + 1.0e-12
        )
        return torch.cat(
            [
                pool(condition.product_features, mask),
                pool(condition.edit_features, mask),
                pool(condition.pocket_features, pocket_mask),
                torch.log(counts),
                torch.log(spread.clamp_min(1.0e-4)).unsqueeze(-1),
            ],
            dim=-1,
        )

    def internal_log_multiplier(self, condition: MEHFCondition) -> torch.Tensor:
        """V3's per-molecule log multiplier on the internal scale, shape (B,)."""
        if self.internal_scale_head is None:
            return torch.zeros(
                condition.product_init.shape[0],
                device=condition.product_init.device,
                dtype=condition.product_init.dtype,
            )
        raw = self.internal_scale_head(
            self._pooled_condition_features(condition)
        ).squeeze(-1)
        return self.internal_scale_head_limit * torch.tanh(
            raw / max(self.internal_scale_head_limit, 1.0e-6)
        )

    def _subspace_scales(
        self, condition: MEHFCondition
    ) -> tuple[float, torch.Tensor | float]:
        """(rigid, internal) multipliers; internal is per-example under V3.

        V3's multiplier is resolved ONCE per call to `flow_loss` or `sample`,
        from the caller's original condition, and reused for the coarse and the
        fine draw alike.  Resolving it separately inside each draw would feed
        the head a frame-subset in one place and a mean-augmented motion prior
        in the other, so training and sampling would apply different scales to
        the same molecule.
        """
        if self.internal_scale_head is None:
            return self.rigid_noise_scale, self.internal_noise_scale
        override = getattr(self, "_internal_multiplier_override", None)
        if override is None:
            multiplier = torch.exp(
                self.internal_log_multiplier(condition).detach()
            ).view(-1, 1, 1, 1)
        else:
            multiplier = override
            if multiplier.shape[0] != condition.product_init.shape[0]:
                raise ValueError("Cached internal multiplier does not match the batch")
        return self.rigid_noise_scale, self.internal_noise_scale * multiplier

    @contextlib.contextmanager
    def _resolved_internal_multiplier(
        self, condition: MEHFCondition, repeats: int = 1
    ):
        if self.internal_scale_head is None:
            yield
            return
        with torch.no_grad():
            multiplier = torch.exp(
                self.internal_log_multiplier(condition)
            ).repeat_interleave(repeats, dim=0).view(-1, 1, 1, 1)
        self._internal_multiplier_override = multiplier
        try:
            yield
        finally:
            self._internal_multiplier_override = None

    def _structured_noise(
        self,
        condition: MEHFCondition,
        n_frames: int,
        *,
        multiplier: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Short-time molecular prior: rigid translation/rotation plus local noise."""
        batch, atoms, _ = condition.product_init.shape
        device, dtype = condition.product_init.device, condition.product_init.dtype
        mask = (
            torch.ones(batch, atoms, dtype=torch.bool, device=device)
            if condition.atom_mask is None else condition.atom_mask
        )
        weights = mask.to(dtype).unsqueeze(-1)
        center = (condition.product_init * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        centered = condition.product_init - center
        translation = self._standard_path_noise(
            (batch, n_frames, 1, 3), condition, generator=generator
        ) * (self.translation_noise_angstrom * multiplier)
        omega = self._standard_path_noise(
            (batch, n_frames, 1, 3), condition, generator=generator
        ) * (self.rotation_noise_radian * multiplier)
        rotation = torch.linalg.cross(
            omega.expand(-1, -1, atoms, -1),
            centered[:, None, :, :].expand(-1, n_frames, -1, -1),
        )
        local = self._standard_path_noise(
            (batch, n_frames, atoms, 3), condition, generator=generator
        ) * (self.local_noise_angstrom * multiplier)
        rigid_scale, internal_scale = self._subspace_scales(condition)
        local = local * internal_scale
        if self.harmonic_local_noise:
            local = self._harmonic_local_noise(local, condition, generator=generator)
        elif self.laplacian_local_noise:
            local = self._laplacian_local_noise(local, condition, generator=generator)
        elif self.anisotropic_local_noise:
            local = self._structure_local_noise(local, condition, generator=generator)
        return (
            (translation + rotation) * rigid_scale + local
        ) * mask[:, None, :, None]

    def _laplacian_local_noise(
        self,
        noise: torch.Tensor,
        condition: MEHFCondition,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Local noise with covariance L^+ (x) I_3, L the heavy-atom bond-graph Laplacian.

        The harmonic prior of EigenFold and HarmonicFlow: bonded atoms co-move, so
        bond VECTORS are damped, while angles, torsions and out-of-plane motion are
        governed only by graph distance.  Treated exactly like Sigma* in
        `_harmonic_local_noise`: one factor per molecule, cached, and the draw
        rescaled to the RMS amplitude of the isotropic draw so that only the SHAPE
        of the noise differs (the Laplacian's global scale constant drops out).
        The null space (one translation per connected component) is removed.
        """
        if condition.bond_index is None:
            return noise
        batch, frames, atoms, _ = noise.shape
        out = noise.clone()
        for index in range(batch):
            live = atoms
            if condition.atom_mask is not None:
                live = int(condition.atom_mask[index].sum())
            if live < 2:
                continue
            bonds = condition.bond_index[index]
            key = (bonds.detach().cpu().numpy().tobytes(), live)
            factor = self._laplacian_cache.get(key)
            if factor is None:
                laplacian = torch.zeros(live, live, dtype=torch.float64)
                for a, b in bonds.detach().cpu().tolist():
                    a, b = int(a), int(b)
                    if a < 0 or b < 0 or a == b or a >= live or b >= live:
                        continue
                    if laplacian[a, b] != 0:
                        continue
                    laplacian[a, b] = laplacian[b, a] = -1.0
                    laplacian[a, a] += 1.0
                    laplacian[b, b] += 1.0
                values, vectors = torch.linalg.eigh(laplacian)
                keep = values > values.max().clamp_min(1e-30) * 1.0e-8
                if int(keep.sum()) == 0:
                    continue
                factor = (vectors[:, keep] * values[keep].rsqrt()[None, :]).to(noise.dtype)
                if len(self._laplacian_cache) > 8192:
                    self._laplacian_cache.clear()
                self._laplacian_cache[key] = factor
            factor = factor.to(noise.device)
            draw = torch.randn(
                (frames, factor.shape[1], 3), generator=generator,
                device=noise.device, dtype=noise.dtype,
            )
            shaped = torch.einsum("ak,fkc->fac", factor, draw)
            target = noise[index, :, :live].square().sum(-1).mean().clamp_min(1e-12).sqrt()
            achieved = shaped.square().sum(-1).mean().clamp_min(1e-12).sqrt()
            out[index, :, :live] = shaped * (target / achieved)
        return out

    def _harmonic_local_noise(
        self,
        noise: torch.Tensor,
        condition: MEHFCondition,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Local noise drawn from the exact harmonic covariance Sigma*.

        The Gauss-Seidel sampler gives each stiff coordinate its physical width
        but leaves the SOFT subspace white, and physical soft motion is
        collective; measured channel spread 3.295 against Sigma*'s 0.582
        (`outputs/sigma_star_v1/FINDING.md`). For ligand-sized molecules the
        exact factor is a 90x90 eigenproblem and the reference geometry is fixed
        per example, so it is built once and cached.

        The draw is rescaled to the amplitude the isotropic path would have had,
        so `internal_noise_scale` keeps its meaning and only the SHAPE changes.
        """
        from mdwm.models.harmonic_prior import sigma_star_factor

        if condition.bond_index is None:
            return noise
        batch, frames, atoms, _ = noise.shape
        out = noise.clone()
        for index in range(batch):
            live = atoms
            if condition.atom_mask is not None:
                live = int(condition.atom_mask[index].sum())
            if live < 4:
                continue
            reference = condition.product_init[index, :live]
            bonds = condition.bond_index[index]
            key = (
                bonds.detach().cpu().numpy().tobytes(),
                reference.detach().to(torch.float32).cpu().numpy().round(3).tobytes(),
            )
            factor = self._harmonic_cache.get(key)
            if factor is None:
                with torch.enable_grad():
                    factor, _ = sigma_star_factor(reference.detach().double(), bonds)
                if factor is None:
                    continue
                factor = factor.to(noise.dtype)
                if len(self._harmonic_cache) > 8192:
                    self._harmonic_cache.clear()
                self._harmonic_cache[key] = factor
            factor = factor.to(noise.device)
            draw = torch.randn(
                (frames, factor.shape[1]), generator=generator,
                device=noise.device, dtype=noise.dtype,
            )
            shaped = (draw @ factor.T).reshape(frames, live, 3)
            target = noise[index, :, :live].square().sum(-1).mean().clamp_min(1e-12).sqrt()
            achieved = shaped.square().sum(-1).mean().clamp_min(1e-12).sqrt()
            out[index, :, :live] = shaped * (target / achieved)
        return out

    @torch.no_grad()
    def _structure_local_noise(
        self,
        noise: torch.Tensor,
        condition: MEHFCondition,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Give the local noise the internal-coordinate structure of real motion.

        Isotropic per-atom noise spends the same amplitude in every Cartesian
        direction, while physical thermal motion is near rigid along bonds and
        angles and soft along torsions.  At matched dispersion that mismatch
        costs an order of magnitude in the stiff channels, and no inference-time
        correction repairs it (`outputs/angle_term_v1/FINDING.md`): every
        operator that puts the angles right has to take the dispersion budget
        away, because the budget being in the wrong place IS the defect.

        So the budget is placed correctly here, in the process that creates it.
        For each stiff internal coordinate k this drives the noise's component
        along the Wilson B-vector to a draw at that coordinate's PHYSICAL
        thermal width, leaving every other direction untouched:

            w_k ~ N(0, tau_k^2);   repeat:  eps += (w_k - B_k . eps) B_k/||B_k||^2

        which is Gauss-Seidel on `B_S eps = w`, O(N) per sweep.  The B-vectors
        are evaluated once at the reference geometry, so the map is linear and
        the sweeps converge to a fixed target.  Widths scale with the noise
        amplitude relative to the corpus's physical internal RMSF, so the SHAPE
        of the anisotropy is held fixed as the path amplitude varies.

        Every B-vector is built from unit bond vectors and invariant scalars and
        sums to zero over its atoms, so this is SE(3)-equivariant and does not
        move the centroid, exactly as the inference projection is.
        """
        reference = condition.product_init
        batch, atoms, _ = reference.shape
        frames = noise.shape[1]
        # Widths either SCALE with the noise amplitude (the original form, which
        # ties the stiff directions to the overall amplitude, so raising the
        # amplitude to recover dispersion re-inflates the angles), or are held
        # ABSOLUTE at the physical values.  Physically the stiff widths do not
        # depend on how much collective motion there is: a bond angle's thermal
        # width is 2.8 degrees whatever the ligand is doing.  Absolute widths
        # let the overall amplitude feed only the soft, collective directions.
        if self.anisotropic_tau_absolute:
            amplitude = noise.new_tensor(1.0)
        else:
            amplitude = (
                noise.detach().square().sum(dim=-1).mean().clamp_min(1.0e-12).sqrt()
                / self.reference_internal_rmsf_angstrom
            )
        eps = noise

        bonds = condition.bond_index
        # Bonds already carry a bond-length loss and an 8-iteration inference
        # projection.  Structuring them in the prior as well constrains them a
        # third time; aniso_prior_v2 measured the result, bond length ratio
        # 0.393, i.e. 2.5x UNDER-dispersed.  This switch leaves bonds to the
        # existing machinery and structures angles only.
        bond_ok = (
            self.anisotropic_structure_bonds
            and bonds is not None
            and condition.bond_mask is not None
            and bonds.shape[1] > 0
        )
        if bond_ok:
            safe = bonds.clamp_min(0)
            head = torch.gather(reference, 1, safe[:, :, 0, None].expand(-1, -1, 3))
            tail = torch.gather(reference, 1, safe[:, :, 1, None].expand(-1, -1, 3))
            unit = head - tail
            unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            bond_valid = condition.bond_mask[:, None, :].to(noise.dtype)
            bond_target = self._draw_like(
                (batch, frames, safe.shape[1]), noise, condition, generator
            ) * (self.tau_bond_angstrom * amplitude)

        angles = condition.angle_index
        angle_ok = (
            angles is not None
            and condition.angle_mask is not None
            and angles.shape[1] > 0
        )
        if angle_ok:
            triple = angles.clamp_min(0)
            left, right = self._gathered_angle_vertices(reference[:, None], triple)
            left, right = left[:, 0], right[:, 0]
            left_norm = left.norm(dim=-1).clamp_min(1.0e-8)
            right_norm = right.norm(dim=-1).clamp_min(1.0e-8)
            left_unit = left / left_norm[..., None]
            right_unit = right / right_norm[..., None]
            cosine = (left_unit * right_unit).sum(dim=-1).clamp(-1.0, 1.0)
            sine = (1.0 - cosine.square()).clamp_min(1.0e-12).sqrt().clamp_min(1.0e-6)
            grad_i = -(right_unit - cosine[..., None] * left_unit) / (
                left_norm * sine
            )[..., None]
            grad_k = -(left_unit - cosine[..., None] * right_unit) / (
                right_norm * sine
            )[..., None]
            grad_j = -(grad_i + grad_k)
            angle_norm_squared = (
                grad_i.square().sum(-1) + grad_j.square().sum(-1)
                + grad_k.square().sum(-1)
            ).clamp_min(1.0e-12)
            angle_valid = condition.angle_mask[:, None, :].to(noise.dtype)
            angle_target = self._draw_like(
                (batch, frames, triple.shape[1]), noise, condition, generator
            ) * (self.tau_angle_radian * amplitude)

        # The sweep count is the prior's own.  Borrowing the inference
        # projection's 8 leaves a residual proportional to the starting
        # amplitude, so the achieved stiff width tracks the amplitude instead
        # of the physical target -- measured: angle 3.78 deg at internal scale
        # 3.5, 9.57 deg at 7.0, with tau held absolute.
        for _ in range(self.anisotropic_sweeps or self.projection_iterations or 8):
            if bond_ok:
                gathered = [
                    torch.gather(
                        eps, 2, safe[:, None, :, position, None].expand(-1, frames, -1, 3)
                    )
                    for position in range(2)
                ]
                component = ((gathered[0] - gathered[1]) * unit[:, None]).sum(-1)
                # Under-relaxed: bonds and angles are a REDUNDANT system, and
                # un-relaxed Gauss-Seidel on redundant constraints oscillates and
                # diverges.  The existing bond projection uses the same factor.
                delta = (
                    (bond_target - component) * bond_valid
                    * (0.5 * self.projection_relaxation)
                )[..., None] * unit[:, None]
                eps = eps.scatter_add(
                    2, safe[:, None, :, 0, None].expand(-1, frames, -1, 3), delta
                ).scatter_add(
                    2, safe[:, None, :, 1, None].expand(-1, frames, -1, 3), -delta
                )
            if angle_ok:
                gathered = [
                    torch.gather(
                        eps, 2, triple[:, None, :, position, None].expand(-1, frames, -1, 3)
                    )
                    for position in range(3)
                ]
                component = (
                    (gathered[0] * grad_i[:, None]).sum(-1)
                    + (gathered[1] * grad_j[:, None]).sum(-1)
                    + (gathered[2] * grad_k[:, None]).sum(-1)
                )
                step = (
                    self.projection_angle_relaxation
                    * (angle_target - component)
                    / angle_norm_squared[:, None] * angle_valid
                )[..., None]
                for position, gradient in ((0, grad_i), (1, grad_j), (2, grad_k)):
                    eps = eps.scatter_add(
                        2,
                        triple[:, None, :, position, None].expand(-1, frames, -1, 3),
                        step * gradient[:, None],
                    )
        return eps

    def _draw_like(
        self,
        shape: tuple[int, ...],
        like: torch.Tensor,
        condition: MEHFCondition,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Standard normal on the same device/dtype, honouring the run generator."""
        if generator is not None:
            return torch.randn(
                shape, generator=generator, device=like.device, dtype=like.dtype
            )
        return torch.randn(shape, device=like.device, dtype=like.dtype)

    def make_parent_noise_plan(
        self,
        reference_condition: MEHFCondition,
        *,
        n_samples: int,
        parent_atom_count: int,
        generator: torch.Generator | None = None,
    ) -> MEHFParentNoisePlan:
        """Draw one cross-action CRN plan from a single parent condition."""
        if reference_condition.product_init.shape[0] != 1:
            raise ValueError("A parent CRN reference condition must have batch size one")
        if n_samples < 1 or parent_atom_count < 1:
            raise ValueError("Parent CRN samples and atom count must be positive")
        frames = int(reference_condition.parent_motion_prior.shape[1])
        indices = _coarse_indices(
            frames, self.coarse_stride, reference_condition.product_init.device
        )

        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value.repeat_interleave(n_samples, dim=0)

        repeated = MEHFCondition(
            product_init=repeat(reference_condition.product_init),
            product_features=repeat(reference_condition.product_features),
            edit_features=repeat(reference_condition.edit_features),
            parent_motion_prior=repeat(reference_condition.parent_motion_prior),
            pocket_coords=repeat(reference_condition.pocket_coords),
            pocket_features=repeat(reference_condition.pocket_features),
            future_times_ps=(
                None
                if reference_condition.future_times_ps is None
                else repeat(reference_condition.future_times_ps)
            ),
            atom_mask=(
                None
                if reference_condition.atom_mask is None
                else repeat(reference_condition.atom_mask)
            ),
        )
        coarse = self._subset_condition(repeated, indices)

        def standard(condition: MEHFCondition, n_frames: int, atoms: int) -> torch.Tensor:
            return self._standard_path_noise(
                (n_samples, n_frames, atoms, 3),
                condition,
                generator=generator,
            )

        return MEHFParentNoisePlan(
            coarse_translation=standard(coarse, int(indices.numel()), 1),
            coarse_rotation=standard(coarse, int(indices.numel()), 1),
            coarse_parent_local=standard(
                coarse, int(indices.numel()), parent_atom_count
            ),
            fine_translation=standard(repeated, frames, 1),
            fine_rotation=standard(repeated, frames, 1),
            fine_parent_local=standard(repeated, frames, parent_atom_count),
        )

    def _structured_noise_from_parent_plan(
        self,
        condition: MEHFCondition,
        *,
        translation_standard: torch.Tensor,
        rotation_standard: torch.Tensor,
        parent_local_standard: torch.Tensor,
        multiplier: float,
    ) -> torch.Tensor:
        mapping = condition.product_to_parent
        if mapping is None:
            raise ValueError("Parent-coupled noise requires product_to_parent")
        batch, atoms, _ = condition.product_init.shape
        frames = int(condition.parent_motion_prior.shape[1])
        if translation_standard.shape != (batch, frames, 1, 3):
            raise ValueError("Parent CRN translation shape does not match condition")
        if rotation_standard.shape != (batch, frames, 1, 3):
            raise ValueError("Parent CRN rotation shape does not match condition")
        if parent_local_standard.shape[:2] != (batch, frames):
            raise ValueError("Parent CRN local path shape does not match condition")
        if parent_local_standard.shape[-1] != 3:
            raise ValueError("Parent CRN local vectors must be three-dimensional")
        mapped = mapping >= 0
        if torch.any(mapping[mapped] >= parent_local_standard.shape[2]):
            raise ValueError("Parent CRN plan does not cover every mapped parent atom")
        safe = mapping.clamp_min(0)
        local = torch.gather(
            parent_local_standard,
            2,
            safe[:, None, :, None].expand(batch, frames, atoms, 3),
        ) * mapped[:, None, :, None]
        mask = (
            torch.ones(batch, atoms, dtype=torch.bool, device=condition.product_init.device)
            if condition.atom_mask is None
            else condition.atom_mask.bool()
        )
        weights = mask.to(condition.product_init.dtype).unsqueeze(-1)
        center = (condition.product_init * weights).sum(dim=1, keepdim=True) / (
            weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        centered = condition.product_init - center
        rotation = torch.linalg.cross(
            rotation_standard.expand(-1, -1, atoms, -1),
            centered[:, None].expand(-1, frames, -1, -1),
        )
        rigid_scale, internal_scale = self._subspace_scales(condition)
        noise = (
            (
                translation_standard * self.translation_noise_angstrom
                + rotation * self.rotation_noise_radian
            )
            * rigid_scale
            + local * self.local_noise_angstrom * internal_scale
        ) * float(multiplier)
        return noise * mask[:, None, :, None]

    def _standard_path_noise(
        self,
        shape: tuple[int, int, int, int],
        condition: MEHFCondition,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample either white or physical-time Brownian path noise."""
        batch, frames, _, _ = shape
        device, dtype = condition.product_init.device, condition.product_init.dtype
        innovations = torch.randn(
            *shape, device=device, dtype=dtype, generator=generator
        )
        if self.temporal_noise_mode == "independent":
            return innovations
        if condition.future_times_ps is None:
            times = torch.arange(
                1, frames + 1, device=device, dtype=dtype
            ).view(1, frames).expand(batch, -1)
        else:
            times = condition.future_times_ps.to(device=device, dtype=dtype)
        if times.shape != (batch, frames):
            raise ValueError("Brownian noise requires future_times_ps to match path shape")
        increments = torch.cat([times[:, :1], times[:, 1:] - times[:, :-1]], dim=1)
        if torch.any(~torch.isfinite(increments)) or torch.any(increments <= 0):
            raise ValueError("Brownian noise requires finite strictly increasing times")
        scale = torch.sqrt(increments / self.noise_time_scale_ps).view(
            batch, frames, 1, 1
        )
        return torch.cumsum(innovations * scale, dim=1)

    def _cfm_loss(
        self,
        field: MappedEditEquivariantVelocity,
        target: torch.Tensor,
        condition: MEHFCondition,
        atom_mask: torch.Tensor | None,
        noise_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = self._structured_noise(
            condition, target.shape[1], multiplier=noise_multiplier
        )
        # Independent noise times per physical frame implement BioMD-style masking;
        # physical time remains the tensor axis and is never conflated with flow time.
        flow_time = torch.rand(target.shape[:2], device=target.device, dtype=target.dtype)
        noisy = flow_time[:, :, None, None] * target + (1.0 - flow_time[:, :, None, None]) * noise
        prediction = field(noisy, flow_time, condition)
        squared = (prediction - (target - noise)).square().sum(dim=-1)
        endpoint_estimate = noisy + (1.0 - flow_time[:, :, None, None]) * prediction
        if atom_mask is None:
            return squared.mean(), endpoint_estimate
        weights = atom_mask[:, None, :].to(squared.dtype)
        loss = (squared * weights).sum() / (weights.sum() * target.shape[1]).clamp_min(1.0)
        return loss, endpoint_estimate

    def _coarse_path_for_fine(
        self,
        coarse_target: torch.Tensor,
        coarse_endpoint_estimate: torch.Tensor,
    ) -> torch.Tensor:
        """Choose the frozen coarse context seen by the fine residual flow.

        The prospective mode uses the detached one-step denoised estimate from
        the coarse CFM objective. This exposes the fine model to coarse-model
        errors during training without allowing its loss to game the coarse
        estimator. Historical checkpoints retain exact teacher forcing.
        """
        if coarse_target.shape != coarse_endpoint_estimate.shape:
            raise ValueError("Coarse target and endpoint estimate must match")
        if self.fine_conditioning_mode == "teacher_forced_coarse":
            return coarse_target
        return coarse_endpoint_estimate.detach()

    @staticmethod
    def _gathered_angle_vertices(
        coords: torch.Tensor, angle_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two bond vectors of every angle, `j` at the vertex."""
        safe = angle_index.clamp_min(0)
        frames = coords.shape[1]
        corner = [
            torch.gather(
                coords, 2, safe[:, None, :, position, None].expand(-1, frames, -1, 3)
            )
            for position in range(3)
        ]
        return corner[0] - corner[1], corner[2] - corner[1]

    @staticmethod
    def _angles_from_vectors(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Bond angles in radians, stable at the linear limit.

        `atan2(|u x v|, u.v)` is the same angle as `arccos` of the normalised
        dot product, and is the form used to build the reference angles in the
        collate, but it does not lose its precision (nor, in the loss, its
        gradient) as the angle approaches 0 or pi.  The quantity is invariant
        under a global rotation and translation, so a penalty built from it
        leaves the model SE(3)-equivariant.
        """
        cross = torch.linalg.cross(left, right, dim=-1).norm(dim=-1)
        return torch.atan2(cross.clamp_min(1.0e-12), (left * right).sum(dim=-1))

    def _geometry_losses(
        self, estimated_delta: torch.Tensor, condition: MEHFCondition
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = condition.product_init[:, None, :, :] + estimated_delta
        zero = coords.sum() * 0.0
        bond_loss = zero
        if (
            condition.bond_index is not None
            and condition.bond_mask is not None
            and condition.reference_bond_lengths is not None
            and condition.bond_index.shape[1] > 0
        ):
            safe = condition.bond_index.clamp_min(0)
            atom_i = torch.gather(
                coords, 2, safe[:, None, :, 0, None].expand(-1, coords.shape[1], -1, 3)
            )
            atom_j = torch.gather(
                coords, 2, safe[:, None, :, 1, None].expand(-1, coords.shape[1], -1, 3)
            )
            lengths = (atom_i - atom_j).norm(dim=-1)
            weights = condition.bond_mask[:, None, :].to(lengths.dtype)
            bond_loss = (
                (lengths - condition.reference_bond_lengths[:, None, :]).square() * weights
            ).sum() / (weights.sum() * coords.shape[1]).clamp_min(1.0)
        angle_loss = zero
        if (
            condition.angle_index is not None
            and condition.angle_mask is not None
            and condition.reference_bond_angles is not None
            and condition.angle_index.shape[1] > 0
        ):
            left, right = self._gathered_angle_vertices(coords, condition.angle_index)
            angles = self._angles_from_vectors(left, right)
            weights = condition.angle_mask[:, None, :].to(angles.dtype)
            angle_loss = (
                (angles - condition.reference_bond_angles[:, None, :]).square() * weights
            ).sum() / (weights.sum() * coords.shape[1]).clamp_min(1.0)
        clash_loss = zero
        if condition.pocket_coords.shape[1] > 0:
            distances = torch.cdist(coords.flatten(0, 1), condition.pocket_coords.repeat_interleave(coords.shape[1], 0))
            distances = distances.view(coords.shape[0], coords.shape[1], coords.shape[2], -1)
            atom_mask = (
                torch.ones(coords.shape[0], coords.shape[2], dtype=torch.bool, device=coords.device)
                if condition.atom_mask is None else condition.atom_mask
            )
            pocket_mask = (
                torch.ones(coords.shape[0], condition.pocket_coords.shape[1], dtype=torch.bool, device=coords.device)
                if condition.pocket_mask is None else condition.pocket_mask
            )
            # product feature channel 0 stores Z/100 in the provided atomic interface.
            ligand_heavy = condition.product_features[..., 0] > 0.015
            valid = atom_mask[:, None, :, None] & ligand_heavy[:, None, :, None] & pocket_mask[:, None, None, :]
            penalty = torch.relu(self.clash_distance_angstrom - distances).square()
            clash_loss = (penalty * valid).sum() / valid.sum().clamp_min(1)
        return bond_loss, angle_loss, clash_loss

    def flow_loss(self, target_delta: torch.Tensor, condition: MEHFCondition) -> dict[str, torch.Tensor]:
        with self._resolved_internal_multiplier(condition):
            return self._flow_loss(target_delta, condition)

    def _flow_loss(self, target_delta: torch.Tensor, condition: MEHFCondition) -> dict[str, torch.Tensor]:
        mean_prediction = self.predict_mean(condition)
        mean_error = nn.functional.smooth_l1_loss(
            mean_prediction, target_delta, reduction="none", beta=0.5
        ).sum(dim=-1)
        if condition.atom_mask is None:
            mean_loss = mean_error.mean()
        else:
            mean_weights = condition.atom_mask[:, None, :].to(mean_error.dtype)
            mean_loss = (mean_error * mean_weights).sum() / (
                mean_weights.sum() * target_delta.shape[1]
            ).clamp_min(1.0)
        residual_target = target_delta - mean_prediction.detach()
        indices = _coarse_indices(residual_target.shape[1], self.coarse_stride, residual_target.device)
        coarse_target = residual_target.index_select(1, indices)
        coarse_condition = self._subset_condition(condition, indices)
        coarse_loss, coarse_endpoint_estimate = self._cfm_loss(
            self.coarse_flow, coarse_target, coarse_condition, condition.atom_mask
        )
        coarse_for_fine = self._coarse_path_for_fine(
            coarse_target, coarse_endpoint_estimate
        )
        interpolated = interpolate_coarse_path(
            coarse_for_fine, indices, residual_target.shape[1]
        )
        fine_target = residual_target - interpolated
        fine_condition = MEHFCondition(
            product_init=condition.product_init,
            product_features=condition.product_features,
            edit_features=condition.edit_features,
            parent_motion_prior=(
                condition.parent_motion_prior + mean_prediction.detach() + interpolated.detach()
            ),
            pocket_coords=condition.pocket_coords,
            pocket_features=condition.pocket_features,
            future_times_ps=condition.future_times_ps,
            product_force_prior=condition.product_force_prior,
            product_physics_features=condition.product_physics_features,
            action_features=condition.action_features,
            product_to_parent=condition.product_to_parent,
            bond_index=condition.bond_index,
            bond_mask=condition.bond_mask,
            reference_bond_lengths=condition.reference_bond_lengths,
            angle_index=condition.angle_index,
            angle_mask=condition.angle_mask,
            reference_bond_angles=condition.reference_bond_angles,
            atom_mask=condition.atom_mask,
            pocket_mask=condition.pocket_mask,
        )
        fine_loss, fine_estimate = self._cfm_loss(
            self.fine_flow,
            fine_target,
            fine_condition,
            condition.atom_mask,
            noise_multiplier=self.fine_noise_fraction,
        )
        bond_loss, angle_loss, clash_loss = self._geometry_losses(
            mean_prediction + interpolated + fine_estimate, condition
        )
        total = (
            coarse_loss
            + fine_loss
            + self.mean_loss_weight * mean_loss
            + self.bond_loss_weight * bond_loss
            + self.angle_loss_weight * angle_loss
            + self.clash_loss_weight * clash_loss
        )
        return {
            "loss": total,
            "coarse_flow_loss": coarse_loss,
            "fine_flow_loss": fine_loss,
            "mean_path_loss": mean_loss,
            "coarse_context_mae_angstrom": (
                coarse_for_fine - coarse_target
            ).abs().mean(),
            "bond_geometry_loss": bond_loss,
            "angle_geometry_loss": angle_loss,
            "pocket_clash_loss": clash_loss,
        }

    @staticmethod
    def _integrate(
        field: MappedEditEquivariantVelocity,
        initial: torch.Tensor,
        condition: MEHFCondition,
        steps: int,
        *,
        include_edit_adapter: bool = True,
    ) -> torch.Tensor:
        state = initial
        dt = 1.0 / max(int(steps), 1)
        for step in range(max(int(steps), 1)):
            time = torch.full(state.shape[:2], step * dt, device=state.device, dtype=state.dtype)
            first = field(
                state, time, condition, include_edit_adapter=include_edit_adapter
            )
            proposal = state + dt * first
            next_time = torch.full(
                state.shape[:2], min((step + 1) * dt, 1.0), device=state.device, dtype=state.dtype
            )
            second = field(
                proposal,
                next_time,
                condition,
                include_edit_adapter=include_edit_adapter,
            )
            state = state + 0.5 * dt * (first + second)
        return state

    @staticmethod
    def _internal_dispersion(
        coords: torch.Tensor, condition: MEHFCondition
    ) -> torch.Tensor:
        """(B,) internal RMSF, translation and rotation removed.

        Deliberately the project's single existing definition rather than a new
        one: `internal_rmsf_torch` mirrors `eval.ensemble_metrics.rmsf_profile`,
        which is the function the geometry-spectrum measurement reports as the
        `internal_rmsf` channel.  Imported lazily so `models` keeps no import
        edge into `training`.  The quantity is invariant under a global rotation
        and translation, so scaling by it preserves SE(3)-equivariance.
        """
        from mdwm.training.subspace_prior import internal_rmsf_torch

        return internal_rmsf_torch(coords, condition.atom_mask)

    @torch.no_grad()
    def _project_geometry(
        self,
        path: torch.Tensor,
        condition: MEHFCondition,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """SE(3)-equivariant inference projection for bonds, angles and hard clashes."""
        if self.projection_iterations == 0:
            return path
        target_dispersion = (
            self._internal_dispersion(condition.product_init[:, None] + path, condition)
            if self.restore_projection_amplitude else None
        )
        coords = condition.product_init[:, None] + path
        batch, frames, atoms, _ = coords.shape
        atom_mask = (
            torch.ones(batch, atoms, dtype=torch.bool, device=coords.device)
            if condition.atom_mask is None else condition.atom_mask.bool()
        )
        # SOFT angle target.  Driving every angle to its reference value is a
        # HARD constraint: it gives the coordinate zero width, which is why it
        # removes dispersion budget rather than reshaping it.  With a width set,
        # each angle is instead driven to `reference + w`, w ~ N(0, width^2),
        # i.e. the coordinate is given its PHYSICAL thermal width.  The draw is
        # made once per call, not per sweep, so the sweeps converge to a fixed
        # target.  Width 0.0 reproduces the hard constraint exactly.
        angle_offset = None
        if (
            self.angle_target_width_radian > 0.0
            and self.angle_projection
            and condition.reference_bond_angles is not None
        ):
            # Drawn from the SAMPLER's generator, so a seeded run is
            # reproducible; an unseeded draw here would make every rerun of the
            # same command a different measurement.
            angle_offset = torch.randn(
                (batch, frames, condition.reference_bond_angles.shape[1]),
                generator=generator,
                device=coords.device, dtype=coords.dtype,
            ) * self.angle_target_width_radian
        for _ in range(self.projection_iterations):
            if (
                condition.bond_index is not None
                and condition.bond_mask is not None
                and condition.reference_bond_lengths is not None
                and condition.bond_index.shape[1] > 0
            ):
                bonds = condition.bond_index.clamp_min(0)
                atom_i = torch.gather(
                    coords, 2, bonds[:, None, :, 0, None].expand(-1, frames, -1, 3)
                )
                atom_j = torch.gather(
                    coords, 2, bonds[:, None, :, 1, None].expand(-1, frames, -1, 3)
                )
                delta = atom_i - atom_j
                lengths = delta.norm(dim=-1).clamp_min(1.0e-8)
                valid = condition.bond_mask[:, None, :].to(coords.dtype)
                magnitude = (
                    0.5
                    * self.projection_relaxation
                    * (lengths - condition.reference_bond_lengths[:, None, :])
                    * valid
                )
                bond_correction = magnitude[..., None] * delta / lengths[..., None]
                correction = torch.zeros_like(coords)
                counts = torch.zeros(batch, frames, atoms, 1, device=coords.device, dtype=coords.dtype)
                index_i = bonds[:, None, :, 0, None].expand(-1, frames, -1, 3)
                index_j = bonds[:, None, :, 1, None].expand(-1, frames, -1, 3)
                correction.scatter_add_(2, index_i, -bond_correction)
                correction.scatter_add_(2, index_j, bond_correction)
                count_values = valid[..., None].expand(-1, frames, -1, -1)
                counts.scatter_add_(
                    2, bonds[:, None, :, 0, None].expand(-1, frames, -1, 1), count_values
                )
                counts.scatter_add_(
                    2, bonds[:, None, :, 1, None].expand(-1, frames, -1, 1), count_values
                )
                coords = coords + correction / counts.clamp_min(1.0)

            if (
                self.angle_projection
                and condition.angle_index is not None
                and condition.angle_mask is not None
                and condition.reference_bond_angles is not None
                and condition.angle_index.shape[1] > 0
            ):
                # Gauss-Newton step on the angle deviation, moving the three
                # atoms along the EQUIVARIANT gradient of the angle itself.
                # The gradient is built only from the two unit bond vectors and
                # invariant scalars, so it rotates with the molecule, and it
                # sums to zero over the triple, so the centroid does not move:
                # the step is SE(3)-equivariant exactly as the bond step is.
                triples = condition.angle_index.clamp_min(0)
                left, right = self._gathered_angle_vertices(coords, triples)
                left_norm = left.norm(dim=-1).clamp_min(1.0e-8)
                right_norm = right.norm(dim=-1).clamp_min(1.0e-8)
                left_unit = left / left_norm[..., None]
                right_unit = right / right_norm[..., None]
                cosine = (left_unit * right_unit).sum(dim=-1).clamp(-1.0, 1.0)
                angles = self._angles_from_vectors(left, right)
                # |g_i| = 1/|u| exactly, so the 1/sin here is cancelled by the
                # 1/||g||^2 below: the step vanishes smoothly at the linear
                # limit instead of diverging.
                sine = (1.0 - cosine.square()).clamp_min(1.0e-12).sqrt().clamp_min(1.0e-6)
                grad_i = -(right_unit - cosine[..., None] * left_unit) / (
                    left_norm * sine
                )[..., None]
                grad_k = -(left_unit - cosine[..., None] * right_unit) / (
                    right_norm * sine
                )[..., None]
                grad_j = -(grad_i + grad_k)
                valid = condition.angle_mask[:, None, :].to(coords.dtype)
                norm_squared = (
                    grad_i.square().sum(dim=-1)
                    + grad_j.square().sum(dim=-1)
                    + grad_k.square().sum(dim=-1)
                ).clamp_min(1.0e-12)
                target = condition.reference_bond_angles[:, None, :]
                if angle_offset is not None:
                    target = target + angle_offset
                step = (
                    self.projection_angle_relaxation
                    * (angles - target)
                    / norm_squared
                    * valid
                )[..., None]
                correction = torch.zeros_like(coords)
                counts = torch.zeros(
                    batch, frames, atoms, 1, device=coords.device, dtype=coords.dtype
                )
                count_values = valid[..., None].expand(-1, frames, -1, -1)
                for position, gradient in ((0, grad_i), (1, grad_j), (2, grad_k)):
                    correction.scatter_add_(
                        2,
                        triples[:, None, :, position, None].expand(-1, frames, -1, 3),
                        -step * gradient,
                    )
                    counts.scatter_add_(
                        2,
                        triples[:, None, :, position, None].expand(-1, frames, -1, 1),
                        count_values,
                    )
                coords = coords + correction / counts.clamp_min(1.0)

            if condition.pocket_coords.shape[1] > 0:
                relative = coords[:, :, :, None] - condition.pocket_coords[:, None, None]
                distances = relative.norm(dim=-1).clamp_min(1.0e-8)
                pocket_mask = (
                    torch.ones(
                        batch, condition.pocket_coords.shape[1], dtype=torch.bool, device=coords.device
                    )
                    if condition.pocket_mask is None else condition.pocket_mask.bool()
                )
                ligand_heavy = condition.product_features[..., 0] > 0.015
                eligible = (
                    atom_mask[:, None, :, None]
                    & ligand_heavy[:, None, :, None]
                    & pocket_mask[:, None, None, :]
                )
                masked_distances = torch.where(
                    eligible, distances, torch.full_like(distances, float("inf"))
                )
                nearest_distance, nearest_index = masked_distances.min(dim=-1)
                nearest_relative = torch.gather(
                    relative,
                    3,
                    nearest_index[..., None, None].expand(-1, -1, -1, 1, 3),
                ).squeeze(3)
                has_clash = torch.isfinite(nearest_distance) & (
                    nearest_distance < self.clash_distance_angstrom
                )
                penetration = torch.relu(
                    self.clash_distance_angstrom + 1.0e-4 - nearest_distance
                )
                push = (
                    penetration[..., None]
                    * nearest_relative
                    / nearest_distance.clamp_min(1.0e-8)[..., None]
                    * has_clash[..., None]
                )
                coords = coords + self.projection_clash_relaxation * push
            coords = torch.where(atom_mask[:, None, :, None], coords, condition.product_init[:, None])
        projected = coords - condition.product_init[:, None]
        if target_dispersion is None:
            return projected
        # Per-sample scalar solve.  The Kabsch-superposed RMSF is only
        # APPROXIMATELY linear in this scalar -- the ensemble mean structure the
        # frames are superposed on is itself `product_init + s * mean_t(path)`
        # and so moves with s -- hence a fixed, pre-declared number of
        # fixed-point steps rather than one closed-form division.
        scale = torch.ones_like(target_dispersion)
        for _ in range(self.restore_iterations):
            achieved = self._internal_dispersion(
                condition.product_init[:, None]
                + projected * scale[:, None, None, None],
                condition,
            )
            scale = scale * target_dispersion / achieved.clamp_min(1.0e-8)
        return projected * scale[:, None, None, None]

    @torch.no_grad()
    def sample(
        self,
        condition: MEHFCondition,
        n_samples: int = 1,
        steps: int = 32,
        generator: torch.Generator | None = None,
        *,
        include_edit_adapter: bool = True,
        parent_noise_plan: MEHFParentNoisePlan | None = None,
    ) -> torch.Tensor:
        with self._resolved_internal_multiplier(condition, repeats=int(n_samples)):
            return self._sample(
                condition, n_samples=n_samples, steps=steps, generator=generator,
                include_edit_adapter=include_edit_adapter,
                parent_noise_plan=parent_noise_plan,
            )

    def _sample(
        self,
        condition: MEHFCondition,
        n_samples: int = 1,
        steps: int = 32,
        generator: torch.Generator | None = None,
        *,
        include_edit_adapter: bool = True,
        parent_noise_plan: MEHFParentNoisePlan | None = None,
    ) -> torch.Tensor:
        batch, frames, atoms, _ = condition.parent_motion_prior.shape
        mean_path = self.predict_mean(
            condition, include_edit_adapter=include_edit_adapter
        )
        indices = _coarse_indices(frames, self.coarse_stride, condition.product_init.device)
        coarse_condition = self._subset_condition(condition, indices)
        def repeat(value: torch.Tensor) -> torch.Tensor:
            return value.repeat_interleave(n_samples, dim=0)

        repeated_coarse = MEHFCondition(
            product_init=repeat(coarse_condition.product_init),
            product_features=repeat(coarse_condition.product_features),
            edit_features=repeat(coarse_condition.edit_features),
            parent_motion_prior=repeat(coarse_condition.parent_motion_prior),
            pocket_coords=repeat(coarse_condition.pocket_coords),
            pocket_features=repeat(coarse_condition.pocket_features),
            future_times_ps=(
                None
                if coarse_condition.future_times_ps is None
                else repeat(coarse_condition.future_times_ps)
            ),
            product_force_prior=(
                None if coarse_condition.product_force_prior is None else repeat(coarse_condition.product_force_prior)
            ),
            product_physics_features=(
                None
                if coarse_condition.product_physics_features is None
                else repeat(coarse_condition.product_physics_features)
            ),
            action_features=(
                None
                if coarse_condition.action_features is None
                else repeat(coarse_condition.action_features)
            ),
            product_to_parent=(
                None
                if coarse_condition.product_to_parent is None
                else repeat(coarse_condition.product_to_parent)
            ),
            bond_index=None if coarse_condition.bond_index is None else repeat(coarse_condition.bond_index),
            bond_mask=None if coarse_condition.bond_mask is None else repeat(coarse_condition.bond_mask),
            reference_bond_lengths=(
                None if coarse_condition.reference_bond_lengths is None
                else repeat(coarse_condition.reference_bond_lengths)
            ),
            angle_index=(
                None if coarse_condition.angle_index is None
                else repeat(coarse_condition.angle_index)
            ),
            angle_mask=(
                None if coarse_condition.angle_mask is None
                else repeat(coarse_condition.angle_mask)
            ),
            reference_bond_angles=(
                None if coarse_condition.reference_bond_angles is None
                else repeat(coarse_condition.reference_bond_angles)
            ),
            atom_mask=None if coarse_condition.atom_mask is None else repeat(coarse_condition.atom_mask),
            pocket_mask=None if coarse_condition.pocket_mask is None else repeat(coarse_condition.pocket_mask),
        )
        if parent_noise_plan is None:
            coarse_noise = self._structured_noise(
                repeated_coarse, indices.numel(), generator=generator
            )
        else:
            if parent_noise_plan.coarse_translation.shape[0] != n_samples:
                raise ValueError("Parent CRN plan sample count does not match request")
            coupled = lambda value: value.repeat(batch, 1, 1, 1)
            coarse_noise = self._structured_noise_from_parent_plan(
                repeated_coarse,
                translation_standard=coupled(parent_noise_plan.coarse_translation),
                rotation_standard=coupled(parent_noise_plan.coarse_rotation),
                parent_local_standard=coupled(parent_noise_plan.coarse_parent_local),
                multiplier=1.0,
            )
        coarse = self._integrate(
            self.coarse_flow,
            coarse_noise,
            repeated_coarse,
            steps,
            include_edit_adapter=include_edit_adapter,
        )
        interpolated = interpolate_coarse_path(coarse, indices, frames)
        fine_condition = MEHFCondition(
            product_init=repeat(condition.product_init),
            product_features=repeat(condition.product_features),
            edit_features=repeat(condition.edit_features),
            parent_motion_prior=(
                repeat(condition.parent_motion_prior) + repeat(mean_path) + interpolated
            ),
            pocket_coords=repeat(condition.pocket_coords),
            pocket_features=repeat(condition.pocket_features),
            future_times_ps=(
                None if condition.future_times_ps is None else repeat(condition.future_times_ps)
            ),
            product_force_prior=(
                None if condition.product_force_prior is None else repeat(condition.product_force_prior)
            ),
            product_physics_features=(
                None
                if condition.product_physics_features is None
                else repeat(condition.product_physics_features)
            ),
            action_features=(
                None
                if condition.action_features is None
                else repeat(condition.action_features)
            ),
            product_to_parent=(
                None
                if condition.product_to_parent is None
                else repeat(condition.product_to_parent)
            ),
            bond_index=None if condition.bond_index is None else repeat(condition.bond_index),
            bond_mask=None if condition.bond_mask is None else repeat(condition.bond_mask),
            reference_bond_lengths=(
                None if condition.reference_bond_lengths is None else repeat(condition.reference_bond_lengths)
            ),
            angle_index=(
                None if condition.angle_index is None else repeat(condition.angle_index)
            ),
            angle_mask=(
                None if condition.angle_mask is None else repeat(condition.angle_mask)
            ),
            reference_bond_angles=(
                None if condition.reference_bond_angles is None
                else repeat(condition.reference_bond_angles)
            ),
            atom_mask=None if condition.atom_mask is None else repeat(condition.atom_mask),
            pocket_mask=None if condition.pocket_mask is None else repeat(condition.pocket_mask),
        )
        if parent_noise_plan is None:
            fine_noise = self._structured_noise(
                fine_condition,
                frames,
                multiplier=self.fine_noise_fraction,
                generator=generator,
            )
        else:
            coupled = lambda value: value.repeat(batch, 1, 1, 1)
            fine_noise = self._structured_noise_from_parent_plan(
                fine_condition,
                translation_standard=coupled(parent_noise_plan.fine_translation),
                rotation_standard=coupled(parent_noise_plan.fine_rotation),
                parent_local_standard=coupled(parent_noise_plan.fine_parent_local),
                multiplier=self.fine_noise_fraction,
            )
        fine = self._integrate(
            self.fine_flow,
            fine_noise,
            fine_condition,
            steps,
            include_edit_adapter=include_edit_adapter,
        )
        path = self._project_geometry(
            repeat(mean_path)
            + self.sampling_temperature * (interpolated + fine),
            fine_condition,
            generator=generator,
        )
        if self.sample_moment_matching and n_samples >= 2:
            # The residual flow and nonlinear geometry projection need not have
            # zero conditional mean, even though `mean_path` is trained as the
            # deterministic first moment.  Moment projection removes this
            # avoidable Monte Carlo bias without reading the target outcome.
            # Reapplying geometry after every centering step preserves the same
            # bond/clash contract as ordinary samples.  All operations are
            # permutation- and SE(3)-equivariant.
            for _ in range(self.sample_moment_matching_iterations):
                grouped = path.view(batch, n_samples, frames, atoms, 3)
                grouped = (
                    grouped
                    - grouped.mean(dim=1, keepdim=True)
                    + mean_path[:, None]
                )
                path = self._project_geometry(
                    grouped.reshape(batch * n_samples, frames, atoms, 3),
                    fine_condition,
                    generator=generator,
                )
        return path.view(batch, n_samples, frames, atoms, 3).transpose(0, 1)
