cd /home/ieb24002/MDWM_TCBB
# wait for f1 and f2, then train f3, then measure all three at the scale grid
while pgrep -f "sigma_star_full_f[12].yaml" > /dev/null; do sleep 60; done
echo "f1,f2 done $(date)"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src conda run -n MDWM --no-capture-output \
  python scripts/run_mehf_transfer.py --config configs/sigma_star_full_f3.yaml \
  > runlogs/sigma_f3.log 2>&1
echo "f3 done $(date)"
for f in 1 2 3; do for sc in 1.5 2.5 3.5; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output \
    python scripts/mehf_internal_geometry_spectrum.py \
    --checkpoint outputs/sigma_star_v1/full_f$f/checkpoint.pt \
    --rigid 0.32 --internal $sc --n-examples 96 --impropers --angle-projection on \
    --output outputs/sigma_star_v1/spectrum_f${f}_s${sc}.json > /dev/null 2>&1
done; done
echo CHAIN_DONE $(date)
