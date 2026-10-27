#!/bin/bash
# Gated MISATO store pipeline: wait for the verified raw file, cross-validate
# against the existing known-good cache, and only then extract.
#
# The gate is the point.  A successful download tells you nothing about the
# numbers inside the file, so bulk extraction must not start until the extractor
# has been shown to reproduce data that was built independently.  If the
# cross-validation fails, this script stops and says so; it does not extract.
#
#   setsid bash scripts/misato_store_pipeline.sh \
#     > runlogs/misato_store_pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#
# Environment overrides: WORKERS, XVAL_SYSTEMS, ALL_BUDGET_HOURS, MAX_WAIT_HOURS.

set -u -o pipefail

export CUDA_VISIBLE_DEVICES=""
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
RAW="$ROOT/data/MISATO/raw/MD.hdf5"
WORKERS="${WORKERS:-8}"
XVAL_SYSTEMS="${XVAL_SYSTEMS:-40}"
ALL_BUDGET_HOURS="${ALL_BUDGET_HOURS:-10}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-24}"

cd "$ROOT" || exit 1
log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] pipeline: $*"; }

log "waiting for $RAW (the downloader publishes it under that name only after its MD5 matches)"
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while [ ! -f "$RAW" ]; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "STOP: $RAW did not appear within $MAX_WAIT_HOURS h"
    exit 1
  fi
  sleep 60
done
log "raw file present: $(stat -c %s "$RAW") bytes"
log "md5 sidecar:"; cat "$ROOT/data/MISATO/raw/MD.hdf5.md5" || true

log "=== GATE: cross-validating against data/processed/misato_atomic_window_cache_3k_v1 ==="
PYTHONPATH="$ROOT/src" "$PY" scripts/misato_store_crossvalidate.py \
  --n-systems "$XVAL_SYSTEMS" \
  --out "$ROOT/data/MISATO/validation/crossvalidation_local.json"
status=$?
if [ "$status" -ne 0 ]; then
  log "STOP: cross-validation did not pass (exit $status). NOT extracting."
  log "see data/MISATO/validation/crossvalidation_local.json"
  exit 2
fi
log "cross-validation PASSED; proceeding to extraction"

# The root filesystem reports itself rotational, so a dozen concurrent readers
# of a 124 GiB file could spend their time seeking.  Measure a small batch
# before committing to a worker count instead of guessing: extraction is
# resumable, so a warm-up costs nothing but the systems it actually finishes.
log "=== warm-up: ${WARMUP_SYSTEMS:-80} systems, to measure real throughput ==="
"$PY" scripts/misato_store_extract.py \
  --config-name frames100_pairs_v1 --systems census-pairs \
  --workers "$WORKERS" --limit "${WARMUP_SYSTEMS:-80}"
log "warm-up exited $?"
log "warm-up rate: $("$PY" - <<PY2
import json
s = json.load(open("$ROOT/data/MISATO/manifest/frames100_pairs_v1/summary.json"))
done = max(sum(s["this_run_counts"].values()), 1)
per = s["wall_seconds_this_run"] / done
print(json.dumps({"systems": done, "seconds_per_system": round(per, 3),
                  "projected_hours_4622_pairs": round(per * 4622 / 3600, 2),
                  "projected_hours_16971_all": round(per * 16971 / 3600, 2),
                  "mean_npz_bytes": s["mean_npz_bytes"],
                  "peak_group_rss_gib": round(s["runtime_config"]["peak_group_rss_bytes"] / 1024**3, 2)}))
PY2
)"

log "=== extraction: MMP-pair systems at 100 frames ==="
"$PY" scripts/misato_store_extract.py \
  --config-name frames100_pairs_v1 --systems census-pairs --workers "$WORKERS"
log "pair extraction exited $?"

# Extend to every system only if the measured cost of the pair run projects
# under budget.  Measured, not assumed.
decision=$("$PY" - <<PY
import json
s = json.load(open("$ROOT/data/MISATO/manifest/frames100_pairs_v1/summary.json"))
done = max(sum(s["this_run_counts"].values()), 1)
per = s["wall_seconds_this_run"] / done
hours = per * 16971 / 3600.0
gb = (s["mean_npz_bytes"] or 0) * 16971 / 1024**3
print(json.dumps({"per_system_seconds": round(per, 3), "projected_hours_all": round(hours, 2),
                  "projected_gib_all": round(gb, 1),
                  "go": hours <= float("$ALL_BUDGET_HOURS")}))
PY
)
log "all-systems projection: $decision"
if echo "$decision" | grep -q '"go": true'; then
  log "=== extraction: every MISATO system at 100 frames ==="
  "$PY" scripts/misato_store_extract.py \
    --config-name frames100_all_v1 --systems all --workers "$WORKERS"
  log "all-systems extraction exited $?"
else
  log "skipping the all-systems pass: projection exceeds ALL_BUDGET_HOURS=$ALL_BUDGET_HOURS h."
  log "run it deliberately with: $PY scripts/misato_store_extract.py --config-name frames100_all_v1 --systems all --workers $WORKERS"
fi

log "done"
