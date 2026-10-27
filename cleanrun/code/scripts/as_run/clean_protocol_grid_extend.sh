#!/bin/bash
# extend the clean-protocol grid to further scales, ALL arms, fit and dev (uniform grid). usage: fold scale...
cd /home/ieb24002/MEHF
f=$1; shift
PY=/home/ieb24002/anaconda3/envs/MDWM/bin/python
LOG=outputs/clean_protocol_v1/logs
for arm in iso harm; do
  C="--checkpoint outputs/clean_protocol_v1/${arm}_f$f/checkpoint.pt --rigid 0.32 --impropers --restore off --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5 --angle-projection on --full-pocket-clash"
  for which in fit dev; do
    if [ $which = fit ]; then X="--n-examples 96"; else X="--n-examples 160 --use-dev"; fi
    for s in "$@"; do
      CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts $PY scripts/mehf_internal_geometry_spectrum.py $C $X --internal $s --output outputs/clean_protocol_v1/grid/${arm}_${which}_f${f}_s$s.json > $LOG/grid_${arm}_${which}_f${f}_s$s.log 2>&1 || { echo "[$(date '+%F %T')] grid-extend $arm $which f$f s$s FAILED" >> $LOG/CHAIN.log; exit 1; }
    done
  done
done
echo "[$(date '+%F %T')] grid-extend f$f scales $* DONE" >> $LOG/CHAIN.log
