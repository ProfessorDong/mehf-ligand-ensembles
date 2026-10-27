cd /home/ieb24002/MDWM_TCBB
for f in 2 3; do for sc in 2.5 3.5 4.5; do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output python scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/internal_calibration_v1/full_f$f/checkpoint.pt --rigid 0.32 --internal $sc --n-examples 96 --impropers --angle-projection on --output outputs/pareto_v1/fit/iso_proj_f${f}_s${sc}.json
done; done
echo GPU1_DONE
