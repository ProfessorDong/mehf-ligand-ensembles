#!/bin/bash
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD/src:$PWD/scripts
while read -r cmd; do
  name=$(echo "$cmd" | grep -oE 'test_[A-Za-z_]+_f1')
  start=$(date +%s)
  /home/dong/anaconda3/envs/MEHF/bin/${cmd} > outputs/noproj_test_v1/logs/gate_${name}.log 2>&1
  echo "$name exit=$? seconds=$(( $(date +%s)-start ))"
done < outputs/noproj_test_v1/gate/commands.txt
