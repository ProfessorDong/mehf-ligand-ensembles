from __future__ import annotations

import torch
from torch import nn


class ActionConditionedTracePredictor(nn.Module):
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
        activity_fp_dim: int = 0,
        action_injection: str = "concat",
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_trace_features = n_trace_features
        if action_injection not in ("concat", "token"):
            raise ValueError(f"action_injection must be 'concat' or 'token', got {action_injection!r}")
        self.action_injection = action_injection
        # action_features = [start_fp, end_fp, delta_fp], each n_action_features//3 bits
        fp_bits = n_action_features // 3 if (n_action_features % 3 == 0 and activity_fp_dim > 0) else 0
        self._fp_bits = fp_bits
        self._activity_fp_dim = activity_fp_dim if fp_bits > 0 else 0
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
        self.action_encoder = nn.Sequential(
            nn.Linear(n_action_features, action_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(action_dim, action_dim),
            nn.GELU(),
        )
        # "token" injection: the encoded edit is prepended as an extra sequence position so the
        # self-attention layers can condition the whole trajectory read on the edit, instead of
        # only meeting the pooled context at the head. The fused vector keeps the same width
        # (d_model + action_dim) so the heads are unchanged; this requires action_dim == d_model.
        if action_injection == "token":
            if action_dim != d_model:
                raise ValueError(f"token injection requires action_dim == d_model, got {action_dim} != {d_model}")
            self.action_to_token = nn.Linear(action_dim, d_model)
            self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.action_to_token = None
            self.action_type_embedding = None
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + action_dim),
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * n_trace_features),
        )
        if self._activity_fp_dim > 0:
            self.activity_fp_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(fp_bits, self._activity_fp_dim),
                nn.GELU(),
                nn.Linear(self._activity_fp_dim, self._activity_fp_dim),
                nn.GELU(),
            )
        else:
            self.activity_fp_encoder = None
        activity_head_in = d_model + action_dim + self._activity_fp_dim
        self.activity_head = nn.Sequential(
            nn.LayerNorm(activity_head_in),
            nn.Linear(activity_head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.action_type_embedding is not None:
            nn.init.trunc_normal_(self.action_type_embedding, std=0.02)

    def encode(self, x_context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_context) + self.pos_embedding[:, : x_context.shape[1], :]
        action_encoded = self.action_encoder(action)
        if self.action_injection == "token":
            action_token = self.action_to_token(action_encoded).unsqueeze(1) + self.action_type_embedding
            seq = torch.cat([action_token, x], dim=1)          # (B, 1 + T, d_model)
            encoded_seq = self.encoder(seq)
            # [last context frame (has attended to the edit); contextualised edit token]
            return torch.cat([encoded_seq[:, -1, :], encoded_seq[:, 0, :]], dim=-1)
        encoded = self.encoder(x)[:, -1, :]
        return torch.cat([encoded, action_encoded], dim=-1)

    def forward(self, x_context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        out = self.head(self.encode(x_context, action))
        return out.view(x_context.shape[0], self.horizon, self.n_trace_features)

    def predict_activity(self, x_context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        fused = self.encode(x_context, action)
        if self.activity_fp_encoder is not None:
            fp_diff = action[:, self._fp_bits * 2 :]  # delta_fp slice
            fused = torch.cat([fused, self.activity_fp_encoder(fp_diff)], dim=-1)
        return self.activity_head(fused).squeeze(-1)
