#!/bin/bash
# model-free controls (Sigma* source, copy-the-parent), projection OFF and ON, CPU only, post-exclusion lists
cd /home/ieb24002/MEHF
f=$1
for proj in off on; do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONPATH=src:scripts /home/ieb24002/anaconda3/envs/MDWM/bin/python scripts/clean_protocol_baselines.py \
    --config configs/clean_f${f}_harm.yaml --dims-checkpoint outputs/laplacian_prior_v1/full_f1/checkpoint.pt \
    --fold-dir outputs/clean_protocol_v1/folds/f$f --fold $f --projection $proj \
    > outputs/clean_protocol_v1/logs/controls_f${f}_$proj.log 2>&1 || echo "controls f$f $proj FAILED" >> outputs/clean_protocol_v1/logs/CHAIN.log
done
echo "[$(date '+%F %H:%M:%S')] controls f$f done (off+on)" >> outputs/clean_protocol_v1/logs/CHAIN.log
