# fitting-fold grid for one fold, started once that fold's checkpoint exists
cd /home/ieb24002/MEHF
f=$1
while [ ! -f outputs/laplacian_prior_v1/full_f$f/checkpoint.pt ] || ps -eo cmd | grep -q "[l]aplacian_full_f$f.yaml"; do sleep 60; done
echo "[$(date +%H:%M)] grid f$f start" >> outputs/laplacian_prior_v1/logs/CHAIN.log
C="--checkpoint outputs/laplacian_prior_v1/full_f$f/checkpoint.pt --rigid 0.32 --n-examples 96 --impropers --restore off --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5"
for s in 1.0 1.5 2.0 2.5 3.0 3.5 4.0; do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts /home/ieb24002/anaconda3/envs/MDWM/bin/python scripts/mehf_internal_geometry_spectrum.py $C --internal $s --angle-projection on --output outputs/laplacian_prior_v1/spectrum_f${f}_s$s.json > outputs/laplacian_prior_v1/logs/spectrum_f${f}_s$s.log 2>&1
done
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts /home/ieb24002/anaconda3/envs/MDWM/bin/python scripts/mehf_internal_geometry_spectrum.py $C --internal 2.5 --angle-projection off --output outputs/laplacian_prior_v1/noproj_f${f}_s2.5.json > outputs/laplacian_prior_v1/logs/noproj_f$f.log 2>&1
echo "[$(date +%H:%M)] grid f$f done" >> outputs/laplacian_prior_v1/logs/CHAIN.log
