cd /home/ieb24002/MDWM_TCBB
for f in 1 2 3; do for sc in 1.5 2.0; do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts conda run -n MDWM --no-capture-output python scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/aniso_prior_v3/full_f$f/checkpoint.pt --rigid 0.32 --internal $sc --n-examples 96 --impropers --angle-projection on --output outputs/aniso_prior_v3/proj_f${f}_s${sc}.json
done; done
echo GRID1_DONE
