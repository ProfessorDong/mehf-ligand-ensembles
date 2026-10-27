cd /home/ieb24002/MEHF
L=outputs/laplacian_prior_v1/logs/CHAIN.log
P=/home/ieb24002/anaconda3/envs/MDWM/bin/python
while [ "$(grep -c 'grid f[123] done' $L)" -lt 3 ]; do sleep 60; done
echo "[$(date +%H:%M)] all grids done; applying the matched-torsion rule (fitting folds only)" >> $L
PYTHONPATH=src:scripts $P scripts/laplacian_matched_torsion.py > outputs/laplacian_prior_v1/logs/matched_torsion.log 2>&1
for f in 1 2 3; do
  s=$($P -c "import json;e=json.load(open('outputs/laplacian_prior_v1/matched_torsion_fit.json'))['arms']['LAPLACIAN']['fold$f'];print(e['matched_torsion_scale'] if e['matched'] else e['nearest_grid_scale'])")
  m=$($P -c "import json;print(json.load(open('outputs/laplacian_prior_v1/matched_torsion_fit.json'))['arms']['LAPLACIAN']['fold$f']['matched'])")
  echo "[$(date +%H:%M)] fold $f: scale $s (matched=$m), fixed before the test read" >> $L
  echo "$f $s" >> outputs/laplacian_prior_v1/logs/test_scales.txt
done
echo "[$(date +%H:%M)] TEST READ START (LAPLACIAN arm only)" >> $L
while read f s; do
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:scripts $P scripts/mehf_internal_geometry_spectrum.py --checkpoint outputs/laplacian_prior_v1/full_f$f/checkpoint.pt --rigid 0.32 --internal $s --steps 16 --seed 20260918 --examples-csv outputs/sigma_star_test_v1/test_examples.csv --all-examples --impropers --angle-projection on --restore off --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5 --output outputs/laplacian_prior_v1/test_LAPLACIAN_f$f.json > outputs/laplacian_prior_v1/logs/test_f$f.log 2>&1 &
done < outputs/laplacian_prior_v1/logs/test_scales.txt
wait
echo "[$(date +%H:%M)] TEST READ END; files: $(ls outputs/laplacian_prior_v1/test_LAPLACIAN_f*.json | wc -l)/3" >> $L
PYTHONPATH=src:scripts $P scripts/laplacian_test_decision.py > outputs/laplacian_prior_v1/logs/decision.log 2>&1
echo "[$(date +%H:%M)] decision written rc=$?" >> $L
