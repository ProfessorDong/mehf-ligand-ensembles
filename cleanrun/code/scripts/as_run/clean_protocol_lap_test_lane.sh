#!/bin/bash
# Clean-LAP test read (read 4). usage: lane.sh <fold>. GPU 1 ONLY. Flags identical to clean_protocol_test_lane.sh.
# No skip-if-exists, no retry: a failure stops the lane and is reported.
cd /home/ieb24002/MEHF
f=$1
PY=/home/ieb24002/anaconda3/envs/MDWM/bin/python
OUT=outputs/clean_protocol_v1/test_read; LOG=outputs/clean_protocol_v1/logs/test_read_lap
EX=outputs/clean_protocol_v1/test_examples_clean.csv
FLAGS="--rigid 0.32 --impropers --restore off --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5 --angle-projection on --full-pocket-clash --examples-csv $EX --all-examples"
stamp() { echo "[$(date '+%F %H:%M:%S')] $*" >> outputs/clean_protocol_v1/logs/CHAIN.log; }
for op in primary secondary; do
  s=$($PY - $op $f <<'P'
import json,sys
op,fold=sys.argv[1:3]
j=json.load(open("outputs/clean_protocol_v1/s_star_frozen_lap.json"))
e=j[{"primary":"PRIMARY_fitting_folds","secondary":"SECONDARY_dev_folds"}[op]]["LAP"][f"fold{fold}"]
assert e["matched"]
print(repr(float(e["s_star"])))
P
)
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH=src:scripts $PY scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/clean_protocol_v1/lap_f$f/checkpoint.pt $FLAGS --internal $s --output $OUT/test_LAP_${op}_f$f.json > $LOG/test_LAP_${op}_f$f.log 2>&1
  rc=$?; [ $rc -ne 0 ] && { stamp "LAP TEST READ run $op f$f FAILED rc=$rc (see $LOG/test_LAP_${op}_f$f.log)"; exit 1; }
done
stamp "LAP test read lane f$f DONE"
