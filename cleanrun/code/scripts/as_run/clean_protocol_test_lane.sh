#!/bin/bash
# One lane of the single clean-protocol test read.  usage: lane.sh <group: clean|leaky> <fold>
# GPU 1 ONLY.  Identical flags to the grid runs (scripts/clean_protocol_lane.sh) except the example source and --all-examples.
cd /home/ieb24002/MEHF
group=$1; f=$2
PY=/home/ieb24002/anaconda3/envs/MDWM/bin/python
OUT=outputs/clean_protocol_v1/test_read; LOG=outputs/clean_protocol_v1/logs/test_read
EX=outputs/clean_protocol_v1/test_examples_clean.csv
FLAGS="--rigid 0.32 --impropers --restore off --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5 --angle-projection on --full-pocket-clash --examples-csv $EX --all-examples"
stamp() { echo "[$(date '+%F %H:%M:%S')] $*" >> outputs/clean_protocol_v1/logs/CHAIN.log; }
run() { # name checkpoint scale
  [ -f $OUT/$1.json ] && { echo "$1 exists, skipping (completed earlier)"; return 0; }
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH=src:scripts $PY scripts/mehf_internal_geometry_spectrum.py --checkpoint $2 $FLAGS --internal $3 --output $OUT/$1.json > $LOG/$1.log 2>&1
  rc=$?; [ $rc -ne 0 ] && { stamp "TEST READ run $1 FAILED rc=$rc (see $LOG/$1.log)"; exit 1; }
  return 0
}
S() { $PY - "$@" <<'P'
import json,sys
op,arm,fold=sys.argv[1:4]
j=json.load(open("outputs/clean_protocol_v1/s_star_frozen_v2.json"))
e=j[{"primary":"PRIMARY_fitting_folds","secondary":"SECONDARY_dev_folds"}[op]][arm][f"fold{fold}"]
assert e["matched"]
print(repr(float(e["s_star"])))
P
}
if [ $group = clean ]; then
  for op in primary secondary; do for arm in ISO HARM; do
    s=$(S $op $arm $f); lc=$(echo $arm | tr A-Z a-z)
    run test_${arm}_${op}_f$f outputs/clean_protocol_v1/${lc}_f$f/checkpoint.pt $s
  done; done
else
  # leaky arms at their own frozen read-1 / read-2 operating points (matched_torsion_fit.json); GS unmatched -> nearest grid 1.0
  case $f in
    1) sI=1.1090897330797107; sS=3.2452895917489304; sL=2.436334413438594;;
    2) sI=1.3085511944557766; sS=3.3291757906919934; sL=2.482539286712936;;
    3) sI=1.1070911128142067; sS=3.786814861114942;  sL=2.4483873096435924;;
  esac
  run test_LEAKY_ISO_f$f   outputs/internal_calibration_v1/full_f$f/checkpoint.pt $sI
  run test_LEAKY_GS_f$f    outputs/aniso_prior_v3/full_f$f/checkpoint.pt 1.0
  run test_LEAKY_SIGMA_f$f outputs/sigma_star_v1/full_f$f/checkpoint.pt $sS
  run test_LEAKY_LAP_f$f   outputs/laplacian_prior_v1/full_f$f/checkpoint.pt $sL
fi
stamp "test read lane $group f$f DONE"
