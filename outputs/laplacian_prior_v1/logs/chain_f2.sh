cd /home/ieb24002/MEHF
# wait until fold 1 or fold 3 has finished (its checkpoint written and its process gone)
while true; do
  n=$(ps -eo cmd | grep -c "[l]aplacian_full_f[13].yaml")
  [ "$n" -lt 2 ] && break
  sleep 60
done
echo "[$(date +%H:%M)] starting f2 (attempt 2; attempt 1 ran out of GPU memory with three concurrent runs)" >> outputs/laplacian_prior_v1/logs/CHAIN.log
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/ieb24002/anaconda3/envs/MDWM/bin/python scripts/run_mehf_transfer.py --config configs/laplacian_full_f2.yaml > outputs/laplacian_prior_v1/logs/train_f2.log 2>&1
echo "[$(date +%H:%M)] f2 finished rc=$?" >> outputs/laplacian_prior_v1/logs/CHAIN.log
