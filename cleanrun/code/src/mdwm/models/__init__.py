"""Model definitions and the backwards-compatible Exp2 model factory."""

from __future__ import annotations

from typing import Any

from mdwm.models.action_transformer import ActionConditionedTracePredictor
from mdwm.models.latent_transition import LatentTransitionWorldModel


def build_exp2_world_model(
    *,
    n_trace_features: int,
    n_action_features: int,
    context_len: int,
    horizon: int,
    model_cfg: dict[str, Any],
):
    """Build an Exp2 predictor from config without breaking old checkpoints.

    ``action_transformer`` remains the default for historical configurations.
    ``latent_transition`` is a context-only latent backbone plus a residual,
    zero-initialised action-conditioned transition; it is the model appropriate
    for testing whether edits add predictive value beyond trajectory continuation.
    """
    cfg = dict(model_cfg)
    model_type = str(cfg.pop("type", "action_transformer")).lower()
    common = {
        "n_trace_features": n_trace_features,
        "n_action_features": n_action_features,
        "context_len": context_len,
        "horizon": horizon,
    }
    if model_type in {"action_transformer", "action-conditioned-transformer", "legacy"}:
        return ActionConditionedTracePredictor(**common, **cfg)
    if model_type in {"latent_transition", "latent-transition", "additive_residual"}:
        return LatentTransitionWorldModel(**common, **cfg)
    raise ValueError(
        f"Unknown model.type {model_type!r}; expected 'action_transformer' or 'latent_transition'."
    )


__all__ = ["ActionConditionedTracePredictor", "LatentTransitionWorldModel", "build_exp2_world_model"]
