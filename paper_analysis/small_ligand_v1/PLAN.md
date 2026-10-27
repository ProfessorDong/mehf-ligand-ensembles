# Post-hoc: is the small-ligand E4 pattern a ligand-size effect or a target effect?

Status: **post-hoc**, written 2026-10-23 before computing. Stored test records only; nothing is run.

**Background.** Supplement Table S3 shows that on products with fewer than 20 heavy atoms (536 examples, 11 targets), HARM has the higher E4 on 93% of examples. Size is confounded with target: 11 targets supply 536 examples.

**Cohort and quantities.** The 1,410-example cohort, using fold-averaged per-example values as in the paper.
Define ΔE4 = E4(HARM) − E4(ISO).

**Analyses.**
1. **Concentration.** For the size bin < 20, give per-target n and mean ΔE4. Also give the share of the bin's examples held by its largest two targets.
2. **Target-level test of the size pattern.** One point per target: its mean ΔE4 against its median product heavy-atom count. Report the Spearman rho over targets, with a 10,000-resample target bootstrap interval (seed 20260925). Repeat with targets weighted equally and the bin defined at the target level: targets whose median product has < 20 heavy atoms vs. ≥ 20. Report the mean of per-target mean ΔE4 in each group, and the number of targets with ΔE4 > 0.
3. **Within-target check.** For targets with at least 5 examples in both the < 20 and the ≥ 20 bins, report the within-target difference of mean ΔE4 between the bins. This is likely infeasible; report the number of eligible targets.
4. **Channel decomposition.** Per channel c, report the median of |ln r_c(HARM)| − |ln r_c(ISO)| for the < 20 and ≥ 20 bins. Also report the median planar-center count per example if it is stored; otherwise say it is not available.
5. **Sensitivity.** Recompute the example-weighted mean ΔE4 (target-bootstrap 95% CI) excluding the < 20 bin. This is descriptive; it does not replace the reported estimand.

**Predictions.**
- P1: one or two targets hold most (> 60%) of the < 20 bin.
- P2: the target-level Spearman rho is negative (HARM relatively better for larger ligands), with an interval that may include 0 because there are only 60 targets.
- P3: in the < 20 bin, the HARM penalty comes mainly from the bond-length and bond-angle channels.
- P4: excluding the bin moves the example-weighted mean ΔE4 below 0.

**Decision rule for the manuscript.**
- If the target-level analysis (2) supports a size trend (interval excludes 0), the main text may carry one post-hoc sentence pointing to the supplement.
- Otherwise the pattern stays in the supplement, described as concentrated in a few targets. It is not called a size effect.
