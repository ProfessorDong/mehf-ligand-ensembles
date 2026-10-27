#!/bin/bash
# OPTIONAL. Regenerates the 100-pocket TargetDiff sample with TargetDiff's OWN released
# per-pocket sampler (seed 2021, 1000 steps, 20 molecules per pocket). Needs the TDIFF
# conda env (environment/environment_targetdiff.yml). ~6 GPU-min per pocket.
source "$(dirname "$0")/common.sh"
GPU="${GPU:-0}"; TDPY="${TDPY:-python}"
CODE="$ROOT/data/external/targetdiff/code"
OUT="${OUT:-$ROOT/data/external/targetdiff/samples}"
CFG=$(mktemp --suffix .yml)
sed "s#__CHECKPOINT__#$ROOT/data/external/targetdiff/targetdiff_pretrained_models/pretrained_diffusion.pt#" scripts/targetdiff/sampling.yml.in > "$CFG"
(cd "$CODE" && $TDPY "$ROOT/scripts/targetdiff/make_pockets10.py")
for pdb in "$ROOT"/data/external/crossdocked/pockets10/*_pocket10.pdb; do
  name=$(basename "$pdb" _pocket10.pdb)
  [ -f "$OUT/$name/sample.pt" ] && continue
  (cd "$CODE" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$CODE" $TDPY scripts/sample_for_pocket.py "$CFG" \
     --pdb_path "$pdb" --device cuda:0 --batch_size 20 --num_samples 20 --result_path "$OUT/$name")
done
$PY scripts/targetdiff/collect_targetdiff_sdf.py
