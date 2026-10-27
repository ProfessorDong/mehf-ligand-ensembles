#!/bin/bash
# Retrains all nine arms (iso/harm/lap x folds 1-3). ~15 GPU-hours. GPU required.
# SKIP THIS if you only want to check the analysis: checkpoints/ already holds the nine.
# Retraining will NOT reproduce them bit-identically (non-deterministic GPU reductions).
source "$(dirname "$0")/_common.sh"; preflight
cd "$CLEANRUN/code" || die "cannot cd to code/"
say "peak memory is 11.8 GB per run; at most THREE concurrently on a 48 GB card, four will OOM"
for f in 1 2 3; do
  for arm in iso harm lap; do
    tgt="$OUT/${arm}_f${f}"
    guard "$tgt/checkpoint.pt" || continue
    mkdir -p "$tgt"
    say "train $arm fold $f -> $tgt"
    CUDA_VISIBLE_DEVICES=$GPU $PY "$CLEANRUN/code/scripts/run_mehf_transfer.py" \
      --config "$CLEANRUN/configs/clean_f${f}_${arm}.yaml" > "$OUT/train_${arm}_f${f}.log" 2>&1 \
      || { say "FAILED: see $OUT/train_${arm}_f${f}.log"; exit 1; }
  done
done
say "done. Observed wall times: iso ~4,700 s, harm ~7,100 s, lap ~5,100 s per fold."
