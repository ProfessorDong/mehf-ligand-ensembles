#!/bin/bash
# Checks the inputs are present and intact. Safe to run any time. No GPU.
source "$(dirname "$0")/_common.sh"; preflight
say "store: $(ls "$CLEANRUN"/inputs/data/MISATO/extracted/frames100_all_v1/*.npz 2>/dev/null | wc -l) npz (expect 16536)"
say "empty files in store: $(find "$CLEANRUN"/inputs/data/MISATO/extracted/frames100_all_v1 -type f -size 0 2>/dev/null | wc -l) (expect 0)"
say "checkpoints: $(ls -d "$CLEANRUN"/checkpoints/*_f? 2>/dev/null | wc -l) (expect 9)"
say "configs: $(ls "$CLEANRUN"/configs/clean_f*.yaml 2>/dev/null | wc -l) (expect 9)"
echo
say "pre-registration hashes"
sha256sum "$CLEANRUN/frozen/PREREGISTRATION.md"
echo "  expect 3863992a30606022fa11736e86104faa4bcf6c6d36d76273a9f678e4ae9f4daf"
head -638 "$CLEANRUN/frozen/PREREGISTRATION.md" | head -c -1 | sha256sum
echo "  expect a51668b7af5d8ba9ec4243597fbcf1046ba9e72bebd7ee6c661cad5590ff322e (draft 1, predates all training)"
sha256sum "$CLEANRUN/frozen/LAP_DECISION.txt" "$CLEANRUN/frozen/s_star_frozen_v2.json" "$CLEANRUN/frozen/s_star_frozen_lap.json" 2>/dev/null
echo
say "GPU $GPU free memory:"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null
say "free disk: $(df -h "$CLEANRUN/code" | tail -1 | awk '{print $4}')"
