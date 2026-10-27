cd /home/ieb24002/MDWM_TCBB
for f in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output python scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/internal_calibration_v1/full_f$f/checkpoint.pt --rigid 0.32 --internal 2.0 --n-examples 96 --impropers --angle-projection on --output outputs/pareto_v1/fit/iso_proj_f${f}_s2.0.json
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output python scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/internal_calibration_v1/full_f$f/checkpoint.pt --rigid 0.32 --internal 3.0 --n-examples 96 --impropers --angle-projection on --output outputs/pareto_v1/fit/iso_proj_f${f}_s3.0.json
done
echo GRID0_DONE
