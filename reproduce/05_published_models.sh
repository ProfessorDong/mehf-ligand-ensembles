#!/bin/bash
# Published-model measurements (DiffSBDD released samples; TargetDiff samples in
# data/external/targetdiff/samples_sdf). CPU only. Depends on: 01 and 03 (c_isotropy_seam).
source "$(dirname "$0")/common.sh"
D=data/external/diffsbdd/crossdocked_fullatom_cond
T=data/external/targetdiff/samples_sdf
REF=data/external/crossdocked/ref_flat
REC=data/external/crossdocked/test_set
for r in 1 10 100; do $PY scripts/published_model_isotropy_test.py --generated-dir $D --max-molecules 300 --max-per-pocket 3 --restraint $r --output outputs/published_models_v1/diffsbdd_r$r.json; done
for r in 1 100; do $PY scripts/published_model_isotropy_test.py --generated-dir $REF --max-molecules 100 --max-per-pocket 3 --restraint $r --output outputs/published_models_v1/reference_r$r.json; done
$PY scripts/published_model_isotropy_test.py --generated-dir $D --max-molecules 1000 --max-per-pocket 10 --restraint 1.0 --output outputs/paper_evidence_v1/c_diffsbdd_isotropy_n1000.json
$PY outputs/paper_evidence_v1/c_isotropy_seam.py
$PY outputs/paper_evidence_v1/e_ff_restraint_sweep.py --reference-dir $REC --receptor-root $REC
for s in 20260924 1234 7 31337 99; do
  $PY scripts/geometry_corrector_gate.py --generated-dir $D --reference-dir $REF --receptor-root $REC --max-molecules 2000 --max-per-pocket 20 --holdout --split-seed $s --output outputs/corrector_gate_v1/scaleup/holdout2000_$s.json &
  $PY scripts/geometry_corrector_gate.py --generated-dir $T --reference-dir $REF --receptor-root $REC --max-molecules 2000 --max-per-pocket 20 --holdout --split-seed $s --output outputs/published_models_v1/targetdiff/holdout2000_$s.json &
done
wait
# the final gate script adds arm G (MMFF restraint 5); reference_outputs keeps both versions
mkdir -p outputs/corrector_gate_v1/scaleup_with_r5 && cp outputs/corrector_gate_v1/scaleup/holdout2000_*.json outputs/corrector_gate_v1/scaleup_with_r5/
