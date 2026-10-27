#!/bin/bash
# Re-estimates the four widths on training-split systems only. Seconds, no GPU.
source "$(dirname "$0")/_common.sh"; preflight
cd "$CLEANRUN/code" || die "cannot cd to code/"
guard "$OUT/tau_train_only_large.json" || exit 0
$PY "$CLEANRUN/code/scripts/tau_train_only_large.py" --out "$OUT/tau_train_only_large.json"
say "recorded values for comparison:"
$PY -c "import json;d=json.load(open('$CLEANRUN/inputs/tau_train_only_large.json'));print(json.dumps(d,indent=1)[:600])"
