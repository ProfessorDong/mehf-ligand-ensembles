#!/bin/bash
# model-free controls on the post-exclusion test list, projection OFF and ON.  Control A at s_i = 4.0 (top of the controls' frozen
# grid, the point P8 names) and 5.0 (top of the arms' Amendment-7 grid); Control B has no scale.  Control A has no fold dependence
# (no weights, no fold-specific input), so one run with the fold-1 HARM config serves all folds.  GPU 1 only.
cd /home/ieb24002/MEHF
for proj in off on; do
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH=src:scripts /home/ieb24002/anaconda3/envs/MDWM/bin/python scripts/clean_protocol_test_controls.py \
    --config configs/clean_f1_harm.yaml --dims-checkpoint outputs/laplacian_prior_v1/full_f1/checkpoint.pt \
    --examples-csv outputs/clean_protocol_v1/test_examples_clean.csv --internal-scales 4.0 5.0 --projection $proj \
    --out outputs/clean_protocol_v1/test_read > outputs/clean_protocol_v1/logs/test_read/controls_$proj.log 2>&1 \
    || { echo "[$(date '+%F %H:%M:%S')] TEST READ controls $proj FAILED" >> outputs/clean_protocol_v1/logs/CHAIN.log; exit 1; }
done
echo "[$(date '+%F %H:%M:%S')] test read controls DONE (off+on)" >> outputs/clean_protocol_v1/logs/CHAIN.log
