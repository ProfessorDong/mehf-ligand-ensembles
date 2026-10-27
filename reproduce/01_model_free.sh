#!/bin/bash
# Section 1 measurements. No model involved; CPU only; reads the MISATO frame store.
# Depends on: 00. (The zero-parameter isotropy test needs a model spectrum and is in 04.)
source "$(dirname "$0")/common.sh"
log "isotropic-noise cost, 2,000 systems";    $PY scripts/isotropic_noise_geometry_cost.py --n-systems 2000 --seed 20260923 --output outputs/isotropic_noise_cost_v1/summary_heavy.json
log "hydrogen-constraint confound";           $PY scripts/check_hydrogen_constraint_confound.py --n-systems 600 --output outputs/isotropic_noise_cost_v1/hydrogen_check.json
log "Wilson B derivation, 600 systems";       $PY scripts/wilson_b_derivation_test.py --n-systems 600 --fd-check 60 --out outputs/wilson_b_theory_v1/wilson_b
log "exact Sigma* control, 150 systems";      $PY scripts/exact_sigma_star_noise.py --n-systems 150 --seed 20260924 --output outputs/sigma_star_v1/control.json
log "theory modules (tau_oop etc.)";          $PY outputs/theory_modules_v1/analysis.py --n-systems 150 && $PY outputs/theory_modules_v1/aggregate.py
log "Sigma* B rows: autograd vs finite diff"; $PY tests/verify_b_rows.py
log "prior-only noise profiles (ISO / Laplacian / Sigma*)"; $PY scripts/prior_noise_profiles.py
for s in a_matching_protocols a2_blaw_amplitude b_aromatic_oop b2_oop_confounds; do
  log "paper evidence: $s";                   $PY outputs/paper_evidence_v1/$s.py
done
