#!/bin/bash
# LAP arm, one lane = one fold: train (clean config) then s-grid 1.0..5.0 on fit and dev samples.
# Identical command lines to clean_protocol_lane.sh / clean_protocol_grid_extend.sh. GPU 1 only. Never touches test.
cd /home/ieb24002/MEHF
f=$1
PY=/home/ieb24002/anaconda3/envs/MDWM/bin/python
LOG=outputs/clean_protocol_v1/logs
stamp() { echo "[$(date '+%F %H:%M:%S')] $*" >> $LOG/CHAIN.log; }
mkdir -p outputs/clean_protocol_v1/grid
arm=lap
t0=$(date +%s)
stamp "train lap f$f START"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src $PY scripts/run_mehf_transfer.py --config configs/clean_f${f}_lap.yaml > $LOG/train_lap_f$f.log 2>&1
rc=$?
stamp "train lap f$f END rc=$rc wall=$(( $(date +%s) - t0 ))s"
[ $rc -ne 0 ] && { stamp "lap lane f$f ABORT"; exit 1; }
C="--checkpoint outputs/clean_protocol_v1/lap_f$f/checkpoint.pt --rigid 0.32 --impropers --restore off --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5 --angle-projection on --full-pocket-clash"
for which in fit dev; do
  if [ $which = fit ]; then X="--n-examples 96"; else X="--n-examples 160 --use-dev"; fi
  stamp "grid lap $which f$f start"
  for s in 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts $PY scripts/mehf_internal_geometry_spectrum.py $C $X --internal $s --output outputs/clean_protocol_v1/grid/lap_${which}_f${f}_s$s.json > $LOG/grid_lap_${which}_f${f}_s$s.log 2>&1
    rc=$?; [ $rc -ne 0 ] && { stamp "grid lap $which f$f s$s FAILED rc=$rc"; exit 1; }
  done
done
stamp "lap lane f$f DONE (train + grids 1.0-5.0)"
