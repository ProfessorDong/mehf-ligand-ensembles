# Post-hoc v2 result (2026-10-23)

**Status: post-hoc.** Stored outputs only. The plan is `PLAN.md`; `sha256sum -c PLAN.sha256` passes.
Run: `python -I checks.py <MEHF root> .`, which writes `RESULT.json`.

## 1. Angle-term effects (training-fold spectrum diagnostic; s_i fixed)

Measure: median bond-angle ratio ÷ median internal-RMSF ratio. Each effect is a percentage of the fold-mean of arm A.

| Effect | f1 | f2 | f3 | Mean |
|---|---|---|---|---|
| Angle loss | −4.04% | +0.50% | −1.53% | −1.69% |
| Angle projection | −52.37% | −58.01% | −59.93% | −56.77% |

Both lines reproduce the audit's values. The angle-loss effect changes sign across folds. No interval is reported, because the six contrasts in `angle_term_effects.py` share models and data, so its Student-t intervals are not valid (R03).

## 2. Equal-target mean paired effect, HARM − ISO (test, matched-torsion s*, projection on)

Target-cluster bootstrap: 10,000 resamples, seed 20260925, 95% percentile intervals.

| Endpoint | Mean | 95% CI | Targets with HARM lower | Cohort |
|---|---|---|---|---|
| S | −1.348 | [−1.420, −1.274] | 60/60 | 1,410 four-channel examples |
| E4 | −0.067 | [−0.102, −0.031] | 47/60 | 1,410 |
| E5 | −0.184 | [−0.222, −0.141] | 50/60 | 1,410 |
| E3 | +0.107 | [+0.071, +0.142] | 16/66 | E3-defined examples |

The three largest targets (Q6XEC0 378, Q9H2K2 358, P07900 116; 852 examples) hold **60.4%** of the 1,410 examples.

**Interpretation.**
- The registered estimand weights each example equally. Under it, E4 is HARM − ISO = +0.065 [−0.085, 0.132], which does not show lower E4 for HARM.
- Under equal target weighting, the E4 effect is −0.067. That is a different question, asked after the fact, and it does not replace the registered result. The difference between the two comes from a few large targets.
- E3 favors ISO under both weightings.
- S favors HARM in every target.

**Cohort note.** Version 1 of this script computed S over 66 targets, because it dropped missing values per fold. That gave −1.225 [−1.330, −1.119]. The plan's comparison is the 1,410-example four-channel cohort, and the table above uses that cohort.
