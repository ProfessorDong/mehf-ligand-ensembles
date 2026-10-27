#!/bin/bash
# The pre-registered decision rules and post-spectrum analyses. Depends on: 01, 03.
source "$(dirname "$0")/common.sh"
log "matched-torsion operating points (fitting folds)"; $PY scripts/matched_torsion_fit.py
log "val confirmation";                                 $PY scripts/analyse_pareto_val.py
log "test confirmation";                                $PY scripts/sigma_star_test_decision.py
log "section 2: loss term vs projection";               $PY scripts/angle_term_effects.py
log "MEHF zero-parameter isotropy test";                 $PY scripts/mehf_zero_parameter_isotropy_test.py
log "leakage audit";                                         $PY scripts/leakage_audit.py
log "Laplacian arm: matched-torsion rule";                 $PY scripts/laplacian_matched_torsion.py
log "Laplacian arm: added test comparison";                $PY scripts/laplacian_test_decision.py
