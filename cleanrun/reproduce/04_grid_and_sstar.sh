#!/bin/bash
# Noise-scale grid, then the torsion-matching rule, then freeze s*. ~3 h. GPU required.
# The grid is {1.0 .. 5.0}: the original {1.0 .. 4.0} did not bracket a torsion ratio of 1
# for the harmonic arm, which is recorded as Amendment 7 of the pre-registration.
source "$(dirname "$0")/_common.sh"; preflight
cd "$CLEANRUN/code" || die "cannot cd to code/"
CKPT=${CKPT:-$CLEANRUN/checkpoints}     # set CKPT=$OUT to use your own retrain
mkdir -p "$OUT/grid"
for f in 1 2 3; do for arm in iso harm lap; do
  C="--checkpoint $CKPT/${arm}_f$f/checkpoint.pt --rigid 0.32 --impropers --restore off \
     --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5 \
     --angle-projection on --full-pocket-clash"
  for which in fit dev; do
    [ $which = fit ] && X="--n-examples 96" || X="--n-examples 160 --use-dev"
    for s in 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
      o="$OUT/grid/${arm}_${which}_f${f}_s$s.json"; guard "$o" || continue
      CUDA_VISIBLE_DEVICES=$GPU $PY "$CLEANRUN/code/scripts/mehf_internal_geometry_spectrum.py" \
        $C $X --internal $s --output "$o" > "$OUT/grid_${arm}_${which}_f${f}_s$s.log" 2>&1
    done
  done
done; done
say "applying the torsion-matching rule (interpolation between MEASURED bracketing points, never extrapolation)"
$PY "$CLEANRUN/code/scripts/clean_protocol_sstar_v2.py" --grid "$OUT/grid" --out "$OUT/s_star_frozen_v2.json"
sha256sum "$OUT/s_star_frozen_v2.json"
echo "  recorded: dabcd5f38db88b6b770a91289c0d24a32d33987ca477da6eec8b5b06bc3e1072"
