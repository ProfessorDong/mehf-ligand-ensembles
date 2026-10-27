# Post-hoc test re-analysis: absolute calibration endpoints (plan, written before computing)

Date: 2026-10-23. Author voice: LD project. Status: POST-HOC. The test outputs were read once
under `sigma_star_test_v1/PREREGISTRATION.md` (ISO, GS, Sigma*) and `laplacian_prior_v1/
PREREGISTRATION.md` (LAP). This analysis reads the same stored per-example CSVs
(`test_<ARM>_f<K>.per_example.csv`); it runs no model. It answers audit item E03
(MEHF_TCBB_Rigorous_Audit.md): channel spread S cannot distinguish uniform over- or
under-dispersion from accuracy.

## Endpoints (fixed here)
For each arm, fold, and example, with ratios r_c = w_gen/w_ref:
- E4 = mean over the four channels (bond, angle, torsion, out-of-plane) of |ln r_c|.
- E5 = mean over the four channels and the internal RMSF ratio of |ln r|.
- E3 = mean over bond, angle, torsion of |ln r_c| (defined on all 1,456 examples; no planar center needed).
Per-example values are averaged over the three fold checkpoints (same convention as S).
Summaries: median and IQR across examples; 90th percentile (tail).
Paired comparisons: Sigma* minus each comparator; mean and median of per-example differences;
95% percentile CI of the mean from 10,000 UniProt-cluster bootstrap resamples, seed 20260925
(the frozen test function `cluster_bootstrap`); fraction of examples with Sigma* lower.
Example sets: the 1,410 four-channel examples for E4/E5 and S; all examples with finite
bond/angle/torsion/RMSF for E3.
Sensitivity: (a) per-fold medians; (b) equal-target weighting (median over 60 targets of
per-target medians); (c) absolute widths: median reference and generated width per channel.

## Predictions (from the published channel medians, before computing per-example values)
- P1: On E4, Sigma* is NOT better than ISO (ISO lower or CI includes zero). Reason: ISO's bond,
  angle, torsion medians are close to 1; Sigma* exceeds 1 on all four.
- P2: On E5, Sigma* is better than ISO and GS (RMSF 0.38/0.43 are far from 1).
- P3: On E4 and E5, Sigma* is better than LAP.
- P4: On E3, ISO is better than Sigma*.
Every result is reported whichever way it falls.
