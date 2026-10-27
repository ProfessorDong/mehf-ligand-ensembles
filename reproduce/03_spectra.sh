#!/bin/bash
# Depends on: 00. Regenerates all 200 spectrum results (fitting-fold grids, val, test, ablations) from
# the released checkpoints. LANES parallel jobs spread over GPUS. ~3 h on 2 A6000s.
source "$(dirname "$0")/common.sh"
GPUS="${GPUS:-0}"; LANES="${LANES:-3}"
read -ra G <<< "$(echo $GPUS | tr ',' ' ')"
mapfile -t CMDS < <(grep '^python ' reproduce/spectra_commands.txt)
run_lane() {
  local lane=$1 gpu=$2 i=0
  for cmd in "${CMDS[@]}"; do
    if (( i++ % LANES != lane )); then continue; fi
    out=$(echo "$cmd" | sed -E 's/.*--output ([^ ]+).*/\1/')
    [ -f "$out" ] && continue
    mkdir -p "$(dirname "$out")"
    CUDA_VISIBLE_DEVICES=$gpu eval "${cmd/python/$PY}" > "$out.log" 2>&1 && log "ok  $out" || log "FAIL $out (see $out.log)"
  done
}
for ((l=0; l<LANES; l++)); do run_lane $l "${G[$((l % ${#G[@]}))]}" & done
wait
