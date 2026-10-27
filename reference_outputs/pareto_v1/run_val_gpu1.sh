cd /home/ieb24002/MDWM_TCBB
for f in 1 2 3; do
  echo "START ISO f$f $(date)"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output python scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/internal_calibration_v1/full_f$f/checkpoint.pt --rigid 0.32 --internal 2.0 --examples-csv outputs/pareto_v1/val_examples.csv --all-examples --impropers --angle-projection on --restore off --output outputs/pareto_v1/val/iso_f$f.json > /dev/null 2>&1 && echo "OK ISO f$f $(date)" || echo "FAIL ISO f$f $(date)"
done
echo VAL1_DONE
