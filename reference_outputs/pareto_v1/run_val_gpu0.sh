cd /home/ieb24002/MDWM_TCBB
for f in 1 2 3; do
  echo "START ANISO f$f $(date)"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output python scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/aniso_prior_v3/full_f$f/checkpoint.pt --rigid 0.32 --internal 2.5 --examples-csv outputs/pareto_v1/val_examples.csv --all-examples --impropers --angle-projection on --restore off --output outputs/pareto_v1/val/aniso_f$f.json > /dev/null 2>&1 && echo "OK ANISO f$f $(date)" || echo "FAIL ANISO f$f $(date)"
done
echo VAL0_DONE
