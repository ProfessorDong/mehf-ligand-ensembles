# TEST RUN PLAN — frozen before the first `test` number exists
Written 2026-09-25. `test` untouched at the time of writing.

Split: outputs/sigma_star_test_v1/test_examples.csv
  usable_examples(split="test") -> 1,456 directional examples, 66 UniProt targets.
  Target overlap with fit+dev of all NINE checkpoints verified 0.

Fixed for every run:
  --rigid 0.32 --steps 16 --seed 20260918 --all-examples --impropers
  --angle-projection on --restore off --angle-target-width 0.0
  --projection-iterations 8 --projection-angle-relaxation 0.5
These reproduce the `config_check` recorded in matched_torsion_fit.json exactly.

Per-arm scale, taken from matched_torsion_fit.json and NOT recomputed:

| arm | fold | checkpoint | --internal |
|---|---|---|---|
| ISO | 1 | outputs/internal_calibration_v1/full_f1/checkpoint.pt | 1.1090897330797107 |
| ISO | 2 | outputs/internal_calibration_v1/full_f2/checkpoint.pt | 1.3085511944557766 |
| ISO | 3 | outputs/internal_calibration_v1/full_f3/checkpoint.pt | 1.1070911128142067 |
| ANISO_v3 | 1 | outputs/aniso_prior_v3/full_f1/checkpoint.pt | 1.0 (nearest grid; UNMATCHABLE) |
| ANISO_v3 | 2 | outputs/aniso_prior_v3/full_f2/checkpoint.pt | 1.0 (nearest grid; UNMATCHABLE) |
| ANISO_v3 | 3 | outputs/aniso_prior_v3/full_f3/checkpoint.pt | 1.0 (nearest grid; UNMATCHABLE) |
| SIGMA_star | 1 | outputs/sigma_star_v1/full_f1/checkpoint.pt | 3.2452895917489304 |
| SIGMA_star | 2 | outputs/sigma_star_v1/full_f2/checkpoint.pt | 3.3291757906919934 |
| SIGMA_star | 3 | outputs/sigma_star_v1/full_f3/checkpoint.pt | 3.786814861114942 |

Outputs: outputs/sigma_star_test_v1/test_<ARM>_f<K>.json and .per_example.csv

Analysis, fixed here:
  per-example spread = max - min over the four ratios
      {bond_length, bond_angle, dihedral, planar_improper}
  per-example value  = mean of that spread over the arm's three fold checkpoints,
      on the intersection of examples all three folds produced.
  PRIMARY: median over examples, per arm.
  CONFIRMED iff  median(SIGMA_star) < median(ISO)  AND  median(SIGMA_star) < median(ANISO_v3)
      AND the 95% cluster bootstrap over UniProt (10,000 resamples, seed 20260925)
      of the paired difference SIGMA_star - ISO excludes 0.
  Reported regardless: every channel ratio per arm, internal RMSF per arm, the
      paired difference against ANISO_v3, per-fold values, realised torsion ratio.

Restart policy: a run that did not complete may be restarted with the identical
command, logged in CHAIN.log. No configuration change after the first test number.

---
## ADDENDUM A — usable-example definition, written 14:45 while the ISO runs are
## still in flight and BEFORE any test_*.json exists (verified: none on disk).

The four-channel spread is undefined for an example with no planar ring centre,
where `planar_improper_ratio` is empty. Such examples are EXCLUDED, not
imputed. The analysis set is therefore:

  examples for which all four ratios are finite in all three fold checkpoints
  of the arm, intersected across the three arms so every arm is scored on the
  identical example set.

The count kept and the count dropped are reported. This rule is mechanical, is
the only way a max-minus-min over four channels can be formed, and is fixed here
before any test value is visible.
