"""Generative head over the post-edit interaction trace.

The world model has always regressed a single future delta path. Under an L2
objective the optimum is the *conditional mean* of the post-edit trace, which is
why a linear ridge matches the Transformer and why persistence stays competitive:
averaging away the stochasticity of MD is most of what the task rewards.

This replaces the point prediction with a distribution,

    p(tau(m') | tau(m), a)

parameterised as a conditional flow from Gaussian noise to the observed delta
path, in the manner of the stochastic-interpolant models used for trajectory
generation. The array being generated here is small -- horizon x features, e.g.
10 x 39 -- so a compact DiT-style stack over the time axis is sufficient.

Conditioning enters through the latent transition that already exists: the flow is
conditioned on [h', x_ref], where h' is the state latent reached by applying the
edit. Nothing about the transition changes, so composition and the imagination
horizon carry over -- traces can be sampled at any rollout depth.

The deterministic decoder is retained alongside the flow head. It supplies the
point prediction used by the group-centered objective and by the existing
endpoint metrics, so adding the generative head costs no previously reported
capability.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mdwm.models.latent_transition import LatentTransitionWorldModel


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of the flow time t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    The modulation projection is zero-initialised, so at initialisation each block
    is the identity and the network predicts a zero velocity field. Training then
    departs from "no transport" rather than from an arbitrary flow.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.modulation = nn.Linear(d_model, 6 * d_model)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        return x + gate2.unsqueeze(1) * self.mlp(h)


class TraceFlowHead(nn.Module):
    """Velocity field v(u_t, t | cond) over the (horizon, features) delta path."""

    def __init__(
        self,
        n_features: int,
        horizon: int,
        cond_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, horizon, d_model))
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([_DiTBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_modulation = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, n_features)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.zeros_(self.out_modulation.weight)
        nn.init.zeros_(self.out_modulation.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, u_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.time_mlp(timestep_embedding(t, self.d_model)) + self.cond_mlp(cond)
        x = self.input_proj(u_t) + self.pos_embedding
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.out_modulation(c).chunk(2, dim=-1)
        x = self.out_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out_proj(x)


class GenerativeTraceWorldModel(nn.Module):
    """Action-conditioned world model that samples post-edit traces.

    Combines the existing latent transition with a flow head. `predict` gives the
    deterministic point estimate as before; `sample` draws plausible post-edit
    delta paths.
    """

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
        flow_layers: int = 3,
        flow_d_model: int = 128,
    ) -> None:
        super().__init__()
        self.dynamics = LatentTransitionWorldModel(
            n_trace_features=n_trace_features,
            n_action_features=n_action_features,
            context_len=context_len,
            horizon=horizon,
            d_model=d_model,
            action_dim=action_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.horizon = horizon
        self.n_trace_features = n_trace_features
        self.flow_head = TraceFlowHead(
            n_features=n_trace_features,
            horizon=horizon,
            cond_dim=d_model + n_trace_features,
            d_model=flow_d_model,
            nhead=nhead,
            num_layers=flow_layers,
            dropout=0.0,
        )

    def condition(self, x_context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Conditioning vector [h'; x_ref] for the flow head."""
        state = self.dynamics.transition_step(self.dynamics.encode_state(x_context), action)
        return torch.cat([state, x_context[:, -1, :]], dim=-1)

    def condition_rollout(self, x_context: torch.Tensor, actions: list[torch.Tensor]) -> torch.Tensor:
        """Conditioning after applying a sequence of edits in imagination.

        The reference frame stays that of the *starting* complex, so a depth-k
        sample is a statement about the trace of the molecule reached by the whole
        route, measured from where the route began -- directly comparable to the
        observed trace of the composed transformation.
        """
        state = self.dynamics.encode_state(x_context)
        for action in actions:
            state = self.dynamics.transition_step(state, action)
        return torch.cat([state, x_context[:, -1, :]], dim=-1)

    def predict(self, x_context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Deterministic point prediction, unchanged from the non-generative model."""
        return self.dynamics(x_context, action)

    def flow_loss(
        self,
        x_context: torch.Tensor,
        action: torch.Tensor,
        target_delta: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Conditional flow matching on the straight path from noise to data.

        u_t = t * u_1 + (1 - t) * u_0 with u_0 ~ N(0, I) and u_1 the observed delta
        path; the regression target is the constant velocity u_1 - u_0.
        """
        cond = self.condition(x_context, action) if cond is None else cond
        u_1 = target_delta
        u_0 = torch.randn_like(u_1)
        t = torch.rand(u_1.shape[0], device=u_1.device)
        u_t = t.view(-1, 1, 1) * u_1 + (1.0 - t.view(-1, 1, 1)) * u_0
        return nn.functional.mse_loss(self.flow_head(u_t, t, cond), u_1 - u_0)

    @torch.no_grad()
    def sample(
        self,
        x_context: torch.Tensor,
        action: torch.Tensor,
        n_samples: int = 1,
        steps: int = 50,
        generator: torch.Generator | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw post-edit delta paths. Returns (n_samples, B, horizon, features).

        Integrates the learned velocity field with midpoint (Heun) steps, which is
        markedly more accurate than Euler at the same step count for the nearly
        straight paths that conditional flow matching produces.

        Pass `cond` from `condition_rollout` to sample at an arbitrary edit depth.
        """
        if cond is None:
            cond = self.condition(x_context, action)
        batch = x_context.shape[0]
        cond_rep = cond.repeat_interleave(n_samples, dim=0)
        u = torch.randn(
            batch * n_samples, self.horizon, self.n_trace_features,
            device=x_context.device, dtype=x_context.dtype, generator=generator,
        )
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((u.shape[0],), step * dt, device=u.device)
            v1 = self.flow_head(u, t, cond_rep)
            u_mid = u + dt * v1
            t_next = torch.full((u.shape[0],), min((step + 1) * dt, 1.0), device=u.device)
            v2 = self.flow_head(u_mid, t_next, cond_rep)
            u = u + dt * 0.5 * (v1 + v2)
        return u.view(batch, n_samples, self.horizon, self.n_trace_features).transpose(0, 1)


class GaussianTraceHead(nn.Module):
    """Heteroscedastic Gaussian head: the cheap uncertainty baseline.

    Predicts an independent Gaussian per (frame, feature). It cannot represent
    correlated or multimodal paths the way the flow head can, but it is stable,
    trains in one pass, and yields calibrated per-channel intervals -- which makes
    it the right control for judging whether the flow head earns its complexity.
    """

    def __init__(self, cond_dim: int, horizon: int, n_features: int, d_model: int = 128) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim), nn.Linear(cond_dim, d_model), nn.GELU(),
            nn.Linear(d_model, 2 * horizon * n_features),
        )

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(cond).view(-1, 2, self.horizon, self.n_features)
        mean, log_var = out[:, 0], out[:, 1].clamp(-10.0, 10.0)
        return mean, log_var

    def nll(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean, log_var = self(cond)
        return 0.5 * (log_var + (target - mean) ** 2 / log_var.exp() + math.log(2 * math.pi)).mean()
