cd /home/ieb24002/MDWM_TCBB
for sc in 2.5 3.0 3.5; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output python scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/aniso_prior_v3/full_f3/checkpoint.pt --rigid 0.32 --internal $sc --n-examples 96 --impropers --angle-projection on --output outputs/aniso_prior_v3/proj_f3_s${sc}.json
done
for f in 1; do for sc in 2.5 3.5 4.5; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output python scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/internal_calibration_v1/full_f$f/checkpoint.pt --rigid 0.32 --internal $sc --n-examples 96 --impropers --angle-projection on --output outputs/pareto_v1/fit/iso_proj_f${f}_s${sc}.json
done; done
echo GPU0_DONE
