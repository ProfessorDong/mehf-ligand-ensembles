# Sourced by every stage. Run stages from anywhere; they cd to the MEHF root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-python}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
