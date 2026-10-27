#!/bin/bash
# OPTIONAL. Retrains the 14 released checkpoints from their configs, writing to
# outputs_retrain/ so the released checkpoints in outputs/ are never overwritten.
# About 70 min per checkpoint on one RTX A6000. GPU training is not bit-deterministic
# (float atomics), so retrained weights will differ slightly from the released ones.
source "$(dirname "$0")/common.sh"
GPU="${GPU:-0}"
for cfg in configs/*.yaml; do
  tmp=$(mktemp --suffix .yaml)
  sed -E 's#^(  root: )outputs/#\1outputs_retrain/#' "$cfg" > "$tmp"
  log "train $(basename $cfg) -> $(grep '  root:' $tmp | awk '{print $2}')"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mehf_transfer.py --config "$tmp"
  rm -f "$tmp"
done
