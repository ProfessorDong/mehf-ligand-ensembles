"""World model with an explicit latent transition that is closed under composition.

`ActionConditionedTracePredictor` maps (context, action) straight to a future
delta path. That is enough to answer "what does this one edit do", but it cannot
*imagine*: there is no object representing the product's dynamic state that could
be fed back in to apply a second edit. Predicting two edits deep would require
re-encoding a 10-frame prediction with a 60-frame encoder.

This model separates the three roles a world model needs:

    h  = Enc(X)                  state latent -- reference-free, one vector space
    h' = h + Trans(h, g(a))      transition  -- same space in and out, so it composes
    du = Dec(h', x_ref)          decoder     -- deltas against an explicit reference frame

Because `Trans` is residual and its output layer is zero-initialised, an untrained
model satisfies h' = h exactly: the prior is latent persistence, which is also the
strongest trivial baseline on this task.

Keeping the reference frame *out* of the latent is what makes composition
well-posed. A product's latent means "this molecule's dynamic state" regardless of
which starting molecule it was reached from, so the latent reached by an edit and
the latent the encoder assigns to that product's own trajectory are directly
comparable -- which is exactly the target the composition objective needs.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentTransitionWorldModel(nn.Module):
    def __init__(
        self,
        n_trace_features: int,
        n_action_features: int,
        context_len: int,
        horizon: int,
        d_model: int = 128,
        action_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        transition_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_trace_features = n_trace_features
        self.d_model = d_model

        self.input_proj = nn.Linear(n_trace_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        # Attention pooling rather than reading only the final position: the last
        # context frame is a single noisy MD snapshot, and the previous model threw
        # away the other 59 frames of evidence about the trajectory.
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.state_norm = nn.LayerNorm(d_model)

        self.action_encoder = nn.Sequential(
            nn.Linear(n_action_features, action_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(action_dim, action_dim),
            nn.GELU(),
        )

        hidden = transition_hidden or d_model * 2
        self.transition = nn.Sequential(
            nn.LayerNorm(d_model + action_dim),
            nn.Linear(d_model + action_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model + n_trace_features),
            nn.Linear(d_model + n_trace_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * n_trace_features),
        )
        # Aleatoric uncertainty is essential for MD: independently simulated
        # products are samples from a conditional path distribution, not paired
        # deterministic targets.  Keep a separate heteroscedastic scale head so
        # Gaussian NLL/CRPS can be used before escalating to conditional flows.
        self.scale_decoder = nn.Sequential(
            nn.LayerNorm(d_model + n_trace_features),
            nn.Linear(d_model + n_trace_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * n_trace_features),
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        # Zero-init the transition output so the model starts from h' = h.
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)

    def encode_state(self, x_context: torch.Tensor) -> torch.Tensor:
        """Context window (B, C, F) -> state latent (B, D)."""
        x = self.input_proj(x_context) + self.pos_embedding[:, : x_context.shape[1], :]
        encoded = self.encoder(x)
        query = self.pool_query.expand(encoded.shape[0], -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded, need_weights=False)
        return self.state_norm(pooled.squeeze(1))

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(action)

    def transition_step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Apply one chemical edit in latent space. Closed: (B, D) -> (B, D)."""
        z_action = self.encode_action(action)
        return state + self.transition(torch.cat([state, z_action], dim=-1))

    def context_only(self, x_context: torch.Tensor) -> torch.Tensor:
        """Context-continuation baseline sharing the exact decoder and reference.

        This is intentionally an explicit method rather than an all-zero action:
        action encoders may learn a bias, whereas this path contains no action
        computation at all.  It makes the additive value of the residual edit
        transition directly measurable on the same held-out rows.
        """
        state = self.encode_state(x_context)
        return self.decode(state, x_context[:, -1, :])

    def action_effect(self, x_context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Additive edit contribution relative to the independently decodable base."""
        state = self.encode_state(x_context)
        base = self.decode(state, x_context[:, -1, :])
        edited = self.decode(self.transition_step(state, action), x_context[:, -1, :])
        return edited - base

    def decode(self, state: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """State latent + reference frame (B, F) -> delta path (B, H, F)."""
        out = self.decoder(torch.cat([state, x_ref], dim=-1))
        return out.view(state.shape[0], self.horizon, self.n_trace_features)

    def decode_log_scale(self, state: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """Log standard deviation of the normalized future-delta path."""
        out = self.scale_decoder(torch.cat([state, x_ref], dim=-1))
        return out.view(state.shape[0], self.horizon, self.n_trace_features).clamp(-5.0, 3.0)

    def context_distribution(self, x_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encode_state(x_context)
        x_ref = x_context[:, -1, :]
        return self.decode(state, x_ref), self.decode_log_scale(state, x_ref)

    def distribution(self, x_context: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Action-conditioned diagonal Gaussian path distribution."""
        state = self.encode_state(x_context)
        edited_state = self.transition_step(state, action)
        x_ref = x_context[:, -1, :]
        return self.decode(edited_state, x_ref), self.decode_log_scale(edited_state, x_ref)

    def forward(self, x_context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Single-edit prediction, matching ActionConditionedTracePredictor's signature."""
        # Use one encoder draw so dropout cannot inject a spurious difference
        # between the base and edit paths during training.
        state = self.encode_state(x_context)
        base = self.decode(state, x_context[:, -1, :])
        edited = self.decode(self.transition_step(state, action), x_context[:, -1, :])
        return base + (edited - base)

    def rollout(
        self,
        x_context: torch.Tensor,
        actions: list[torch.Tensor],
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply a sequence of edits in imagination without re-simulating intermediates.

        Deltas are decoded against the *starting* molecule's final context frame,
        so a k-step rollout is directly comparable to the observed trace of the
        product reached by the composed edit.
        """
        state = self.encode_state(x_context)
        x_ref = x_context[:, -1, :]
        states: list[torch.Tensor] = []
        for action in actions:
            state = self.transition_step(state, action)
            states.append(state)
        delta = self.decode(state, x_ref)
        if return_states:
            return delta, states
        return delta
