# Shared settings for every reproduce step. Portable: no absolute paths baked in.
#
# Overridable from the environment:
#   PY         path to the python in the MDWM conda env   (default: `which python`)
#   MEHF_DATA  path to the MISATO store's parent `data/`  (default: cleanrun/inputs/data)
#   GPU        CUDA device index                          (default: 1)
#   OUT        where regenerated output goes              (default: cleanrun/rerun)
set -u

CLEANRUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(command -v python)}"
GPU="${GPU:-1}"
OUT="${OUT:-$CLEANRUN/code/outputs/clean_protocol_v1}"
MEHF_DATA="${MEHF_DATA:-$CLEANRUN/inputs/data}"

# cleanrun's OWN code must come first on the path. This is not cosmetic: a stock
# MEHF tree's src/ has no `pocket_selection`, so picking it up by accident runs
# the LEAKY over-all-frames pocket loader and produces a wrong answer silently.
export PYTHONPATH="$CLEANRUN/code/src:$CLEANRUN/code/scripts"
export MEHF_DATA

mkdir -p "$OUT"
say()  { echo "[$(date '+%F %H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
guard() { [ -e "$1" ] && { echo "exists, refusing to clobber: $1"; return 1; }; return 0; }

preflight() {
  [ -x "$PY" ] || die "python not found. Set PY=/path/to/mdwm-env/bin/python"

  "$PY" - <<'CHECK' || die "environment check failed (see above)"
import importlib.util, sys, inspect
missing = [m for m in ("torch", "numpy", "yaml", "rdkit") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing modules: {', '.join(missing)}. Activate the MDWM conda env.")

# The decisive guard: the loader on the path must be cleanrun's, not a stock MEHF's.
try:
    from mdwm.data import mmp_transfer as mt
except Exception as e:
    sys.exit(f"cannot import mdwm.data.mmp_transfer: {e}")

src = inspect.getsourcefile(mt)
if not hasattr(mt, "POCKET_SELECTIONS") or not hasattr(mt, "pocket_ranking_distance"):
    sys.exit(
        "WRONG CODE ON THE PATH.\n"
        f"  loaded: {src}\n"
        "  This module has no `pocket_selection` support, so it is a pre-supplementary-task\n"
        "  MEHF source. Running with it would silently use the LEAKY over-all-frames pocket\n"
        "  crop and reproduce the wrong result. Use cleanrun/code/src (PYTHONPATH is set by\n"
        "  _common.sh; something else must be overriding it)."
    )
print(f"  code OK: {src}")
CHECK

  local store="$MEHF_DATA/MISATO/extracted/frames100_all_v1"
  [ -d "$store" ] || die "MISATO store not found at $store. Point MEHF_DATA at the data/ folder of a stock MEHF checkout, e.g. MEHF_DATA=/path/to/MEHF/data"
  local n; n=$(find "$store" -name '*.npz' | wc -l)
  [ "$n" -eq 16536 ] || die "store has $n npz, expected 16536 — wrong or incomplete data/"
  say "preflight OK (store $n npz, GPU $GPU, out $OUT)"
}
