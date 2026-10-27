#!/bin/bash
# Re-derives the 28 excluded systems and the post-exclusion split counts. ~3 min, no GPU.
source "$(dirname "$0")/_common.sh"; preflight
guard "$OUT/excluded_systems.json" || exit 0
say "re-deriving exclusions (96th-nearest frame-0 distance to ligand HEAVY atoms vs the pool bound)"
cd "$CLEANRUN/code" || die "cannot cd to code/"
"$PY" scripts/clean_protocol_exclusions.py --out "$OUT/excluded_systems.json" --counts-out "$OUT/split_counts_clean.json" || die "exclusion script failed"
say "comparing against the recorded list"
"$PY" - "$OUT/excluded_systems.json" "$CLEANRUN/inputs/excluded_systems.json" <<'P'
import json,sys
a=json.load(open(sys.argv[1]))["excluded"]; b=json.load(open(sys.argv[2]))["excluded"]
print(f"  regenerated {len(a)}, recorded {len(b)}")
print("  MATCH" if sorted(a)==sorted(b) else f"  DIFFER: only-new {sorted(set(a)-set(b))} only-old {sorted(set(b)-set(a))}")
P
