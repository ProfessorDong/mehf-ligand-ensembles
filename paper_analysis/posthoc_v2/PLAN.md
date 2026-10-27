# Post-hoc v2: independent checks of numbers proposed by the 2026-10-23 external audit

**Status: post-hoc.** Stored files only; nothing is run.

1. **Angle-loss and angle-projection effects, per fold.** Source: `reference_outputs/angle_term_v1/spectrum/{A,B,C,D}_f{1,2,3}.json`.
   - Measure: median bond-angle ratio divided by median internal-RMSF ratio.
   - Per fold, loss effect = mean of (B−A, D−C); projection effect = mean of (C−A, D−B). Each is expressed as a percentage of the 3-fold mean of A.
   - Output: the three per-fold values and their mean. No interval is computed.
2. **Equal-target mean paired effect**, HARM − ISO, for S, E4, E5 and E3. Source: `posthoc_v1` endpoints recomputed from the per-example CSVs.
   - Per target: the mean of per-example differences. Then the mean over targets.
   - 95% percentile CI: 10,000 target resamples, seed 20260925.
   - Also: the number of targets whose mean difference is below 0, and the share of the 1,410 examples held by the 3 largest targets.

**Predictions.** The audit values should reproduce:
- Loss −4.04 / +0.50 / −1.53, mean −1.69%. Projection −52.37 / −58.01 / −59.93, mean −56.77%.
- Equal-target E4 −0.067 [−0.102, −0.031]; S −1.348 with 60/60 targets lower; top-3 targets 852 examples (60.4%).
