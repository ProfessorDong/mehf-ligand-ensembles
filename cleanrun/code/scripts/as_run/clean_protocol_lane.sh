#!/bin/bash
# One lane = one fold: train ISO then HARM (clean configs), then the s-grid on the fitting sample
# (fit_examples.csv, 96 balanced) and on the dev sample (dev_examples.csv, 160 balanced).
# GPU 1 only. Three lanes (folds 1-3) run concurrently => at most 3 jobs on the card at once.
cd /home/ieb24002/MEHF
f=$1
PY=/home/ieb24002/anaconda3/envs/MDWM/bin/python
LOG=outputs/clean_protocol_v1/logs
stamp() { echo "[$(date '+%F %H:%M:%S')] $*" >> $LOG/CHAIN.log; }
mkdir -p outputs/clean_protocol_v1/grid
for arm in iso harm; do
  t0=$(date +%s)
  stamp "train $arm f$f START"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src $PY scripts/run_mehf_transfer.py --config configs/clean_f${f}_${arm}.yaml > $LOG/train_${arm}_f$f.log 2>&1
  rc=$?
  stamp "train $arm f$f END rc=$rc wall=$(( $(date +%s) - t0 ))s"
  [ $rc -ne 0 ] && { stamp "lane f$f ABORT"; exit 1; }
done
for arm in iso harm; do
  C="--checkpoint outputs/clean_protocol_v1/${arm}_f$f/checkpoint.pt --rigid 0.32 --impropers --restore off --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5 --angle-projection on --full-pocket-clash"
  for which in fit dev; do
    if [ $which = fit ]; then X="--n-examples 96"; else X="--n-examples 160 --use-dev"; fi
    stamp "grid $arm $which f$f start"
    for s in 1.0 1.5 2.0 2.5 3.0 3.5 4.0; do
      CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts $PY scripts/mehf_internal_geometry_spectrum.py $C $X --internal $s --output outputs/clean_protocol_v1/grid/${arm}_${which}_f${f}_s$s.json > $LOG/grid_${arm}_${which}_f${f}_s$s.log 2>&1
      rc=$?; [ $rc -ne 0 ] && { stamp "grid $arm $which f$f s$s FAILED rc=$rc"; exit 1; }
    done
  done
done
stamp "lane f$f DONE (train + grids)"
