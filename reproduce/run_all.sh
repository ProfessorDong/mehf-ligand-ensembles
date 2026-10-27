#!/bin/bash
# Full regeneration (without the optional retraining and TargetDiff re-sampling), then verification.
set -euo pipefail
cd "$(dirname "$0")"
bash 00_split_lists.sh
bash 01_model_free.sh
bash 03_spectra.sh
bash 04_decisions.sh
bash 05_published_models.sh
source common.sh
$PY reproduce/verify.py
