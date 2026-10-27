# PRE-REGISTRATION — Laplacian (HarmonicFlow-style) prior, added comparison
Written 2026-09-29, before any code for this arm exists and before any number for
it has been produced. Hashed immediately; not editable afterwards. Anything
learned later goes into RESULT.md.

## Status of this comparison

The confirmatory test read of ISO / ANISO v3 / Sigma* was completed on 2026-09-25
(`reference_outputs/sigma_star_test_v1/`). This comparison is ADDED AFTER that read,
at the request of the first author, because the closest prior-art alternative —
a harmonic prior built from the bond-graph Laplacian (EigenFold; HarmonicFlow,
Stärk et al. ICML 2024) — was missing. It is reported as a post-hoc, added
comparison, never as part of the original confirmation.

What keeps it honest:
- The Sigma* arm, its checkpoints, its operating points and its test numbers are
  FROZEN. Nothing about Sigma* is re-run, re-tuned or re-selected.
- The new arm goes through exactly the procedure the other arms went through,
  with every choice fixed here before its first number exists.

## The arm: LAPLACIAN

Local noise with covariance proportional to the pseudo-inverse of the heavy-atom
bond-graph Laplacian, independent across the three Cartesian axes:

    L = D - A   (heavy-atom covalent bond graph of the product, unweighted)
    Cov(eps) = L^+ (x) I_3     (null space = one translation per connected component, removed)

This is the HarmonicFlow / EigenFold harmonic prior, applied — like Sigma* — to the
LOCAL displacement term of MEHF's source. The draw is rescaled to the RMS amplitude
the isotropic draw would have had, exactly as Sigma* is (`_harmonic_local_noise`),
so only the SHAPE differs from ISO; the Laplacian's own global scale constant
(alpha in EigenFold) therefore drops out and is not a free parameter. Rigid
translation/rotation terms, noise independence across frames, and everything
else are unchanged. Implemented as a new default-off flag `laplacian_local_noise`,
so every existing checkpoint reproduces exactly (verified before training).

## Training

`configs/laplacian_full_f{1,2,3}.yaml` = `configs/sigma_star_full_f{1,2,3}.yaml`
with `harmonic_local_noise: true` replaced by `laplacian_local_noise: true` and the
output root changed. Same data, split, folds, seeds (train seed 2026), epochs (8),
noise scales (0.35, 3.5), objective, selection (dev_loss).

## Operating point (fitting folds only)

Same grid {1.0, 1.5, ..., 4.0}, same inference config (--rigid 0.32
--n-examples 96 --impropers --angle-projection on --restore off
--angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5,
seed 20260918, 16 steps), same rule as `scripts/matched_torsion_fit.py`
(piecewise-linear torsion crossing of 1.0, smallest crossing; UNMATCHABLE if none,
nearest grid point then used and reported as unmatched). Also the projection-off
spectrum at scale 2.5 on each fold, for the prior-vs-projection decomposition.

## Test read (once)

Three runs (one per fold checkpoint) on `outputs/sigma_star_test_v1/test_examples.csv`,
all 1,456 examples, at the fold's selected scale; per-example value = mean over the
three checkpoints; same Addendum-A example exclusion; same four-channel spread.

## Analysis, fixed now

PRIMARY: median per-example channel spread, LAPLACIAN vs SIGMA_star, paired
difference (SIGMA_star - LAPLACIAN) with a 95% UniProt cluster bootstrap
(10,000 resamples, seed 20260925), on the common example set.
SECONDARY (reported regardless): LAPLACIAN vs ISO paired; per-channel ratios;
internal RMSF; realised torsion ratio; projection-off channel ratios.

Reading of the outcome, fixed now:
- If the CI of SIGMA_star - LAPLACIAN lies below 0: Sigma* is more uniform than the
  Laplacian prior on held-out targets.
- If it contains 0: no difference is shown; the paper says so.
- If it lies above 0: the Laplacian prior is better and the paper says so.

## Predictions, stated before any number exists

P1 (noise level) A bond-graph Laplacian couples bonded atoms, so it suppresses
   bond-length distortion relative to ISO but leaves angles, torsions and
   out-of-plane coordinates much less constrained.
P2 (trained, projection off) LAPLACIAN's out-of-plane ratio stays far above
   Sigma*'s (Sigma*: 1.03 fold mean) and closer to ISO's (5.58).
P3 (test) Median spread lies between Sigma* (0.770) and ISO (1.974), and the
   primary CI lies below 0.
These are predictions, not part of the decision rule.
