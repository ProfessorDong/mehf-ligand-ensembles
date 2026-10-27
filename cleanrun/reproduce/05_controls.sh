#!/bin/bash
# The two model-free controls, in BOTH projection variants. ~1 h, light GPU.
#   A: the Sigma* source sampled directly, no learned flow
#   B: the analogue's mapped motion transferred to the target, no model at all
source "$(dirname "$0")/_common.sh"; preflight
cd "$CLEANRUN/code" || die "cannot cd to code/"
guard "$OUT/controls" || exit 0
mkdir -p "$OUT/controls"
CUDA_VISIBLE_DEVICES=$GPU $PY "$CLEANRUN/code/scripts/clean_protocol_test_controls.py" \
  --examples "$CLEANRUN/inputs/test_examples_clean.csv" --out "$OUT/controls" 2>&1 | tail -20
say "expect: control A unmatched at every scale (torsion never reaches 1); control B matched at 0.877 with spread 0.836"
