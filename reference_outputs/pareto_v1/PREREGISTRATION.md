# Pareto confirmation on `val` — PRE-REGISTRATION
Written 2026-09-23, after all fitting-fold work and BEFORE any computation on
`val`. Nothing in this file may be edited after the first `val` run starts;
anything learned afterwards goes in a separate RESULT section below it.

## Claim under test

At matched overall dispersion, the anisotropic prior (aniso_prior_v3: absolute
physical widths, 32 Gauss-Seidel sweeps, bonds and angles structured) plus the
inference angle projection gives LOWER bond-angle over-dispersion than the
isotropic prior with the SAME projection and the SAME scale-selection rule.
Only the prior differs between the arms.

## How this configuration was arrived at, stated plainly

It was found by exploration on the fitting folds, after several failed
pre-registered attempts (cells A-G, aniso v1 void, v2, v3 without projection;
see outputs/angle_term_v1/FINDING.md and outputs/aniso_prior_v3/). The v3 +
projection combination was chosen after seeing v3's result. Everything on the
fitting folds is therefore exploratory. This test exists because of that.

## Frozen configuration

| arm | checkpoints | internal scale | selection |
|---|---|---|---|
| ANISO | outputs/aniso_prior_v3/full_f{1,2,3}/checkpoint.pt | **2.5** | fit RMSF 0.639 |
| ISO | outputs/internal_calibration_v1/full_f{1,2,3}/checkpoint.pt | **2.0** | fit RMSF 0.647 |

Scale rule, identical for both: the smallest scale on the common grid
{1.5, 2.0, 2.5, 3.0, 3.5} whose three-fold mean internal RMSF ratio on the
fitting folds is >= 0.60. Angle never entered the choice. (The isotropic arm's
grid originally lacked 2.0, which would have forced it to 2.5 at RMSF 0.797 —
an unmatched comparison. 2.0 was added to BOTH arms before this file was
written.)

Common settings: rigid 0.32, angle projection ON, restore OFF, 16 steps,
batch 4, seed 20260918, heavy atoms, `scripts/mehf_internal_geometry_spectrum.py`
with `--examples-csv outputs/pareto_v1/val_examples.csv --all-examples`
(all 1,392 directional examples, 106 UniProt targets, 696 pairs; no sampling).
The script refuses to run if the example list shares any UniProt target with a
checkpoint's training data.

## Analysis, fixed now

1. For each arm, run all three fold checkpoints on every val example. Per
   example and channel, the arm's value is the mean over the three checkpoints
   of the generated/reference spread ratio.
2. PRIMARY: per-example paired difference D = angle(ANISO) - angle(ISO),
   paired by example key. Statistic: median of D. Uncertainty: 95% cluster
   bootstrap over UniProt targets, 4,000 resamples, seed 20260918.
3. **CONFIRMED iff** median D < 0, **and** the upper CI bound < 0, **and**
   median RMSF(ANISO) >= median RMSF(ISO) - 0.05 — i.e. the angle gain is not
   bought with dispersion.
4. Otherwise **NOT CONFIRMED**, reported as such, with the fitting-fold numbers
   relabelled as unconfirmed exploration.

Reported regardless of outcome, not part of the decision: the same paired
analysis for bond length and dihedral; planar improper; each arm's per-fold
medians (so fold consistency is visible); ANISO's absolute angle against the
earlier bar [0.8, 1.35] and its RMSF against 0.60.

## Single-read rules

- `val` is read once. A run may be restarted only if it did not complete, with
  an identical command; that is logged. No configuration may change after the
  first val number is seen.
- `test` is not read.
- Bookkeeping: `val` was previously reserved by
  outputs/internal_calibration_v1/PREREGISTRATION.md for a torsion-JS claim.
  That direction was abandoned when the task was found to be saturated (the
  between-replica ceiling of 0.574); its val read never happened and is
  forfeited by this one. `val` is not being used twice.
