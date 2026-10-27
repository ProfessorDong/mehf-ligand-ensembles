# RESULT — Laplacian (HarmonicFlow-style) prior, added comparison
Read once, 2026-09-29 13:05-13:44, under PREREGISTRATION.md (sha256 14d067ed...).
Scales fixed from fitting folds at 13:05, before the test runs started (logs/CHAIN.log).
Sigma*, ISO and ANISO v3 are the frozen confirmatory runs of 2026-09-25; nothing about
them was re-run. This is an ADDED post-hoc comparison.

## Process notes
- Fold 2 training ran out of GPU memory on the first attempt (three concurrent runs on
  one 48 GB card, GPU 0 being occupied by an unrelated service). It was restarted with
  the identical config once a slot was free (logs/train_f2_oom_attempt1.log). No
  output of the failed attempt was used.
- `scripts/prior_noise_profiles.py` (the P1 noise-level analysis) was run before it
  was hashed. It is descriptive and not part of the decision rule.
- All three folds were matchable: s* = 2.436 / 2.483 / 2.448.

## Primary (test, 1,410 examples, 60 targets)

| arm | median spread | IQR |
|---|---|---|
| MEHF (Sigma*) | 0.770 | 0.47-1.33 |
| MEHF-Laplacian | 1.352 | 0.96-2.01 |
| MEHF-iso | 1.974 | 1.64-2.45 |
| MEHF-GS | 2.527 | 2.24-2.88 |

Sigma* - Laplacian: mean -0.580, 95% UniProt cluster bootstrap [-0.750, -0.404];
Sigma* lower on 86.0% of examples. READING (pre-registered): **Sigma* is more uniform
than the Laplacian prior.**

Secondary: Laplacian - ISO: -0.572 [-0.718, -0.471]; Laplacian lower on 89.4%.
Per-channel test medians (Laplacian): bond 2.18, angle 1.96, torsion 1.10,
out-of-plane 2.62, internal RMSF 0.85. Realised torsion 1.08 / 1.02 / 1.14.

## Predictions, scored
- P1 PARTLY WRONG. Predicted: bond lengths damped, other channels much less. Measured
  (noise only, matched internal RMSF, 150 ligands): the Laplacian damps every stiff
  channel by a similar factor (bond 10.2 -> 5.8, angle 7.8 -> 4.4, torsion 2.6 -> 1.2,
  out-of-plane 6.8 -> 2.6 relative to isotropic); spread 7.92 (ISO) / 4.81 (Laplacian) /
  1.01 (Sigma*).
- P2 HALF RIGHT. Projection off, fitting folds, scale 2.5: Laplacian out-of-plane 2.56,
  far above Sigma*'s 1.03 as predicted, but closer to Sigma* than to ISO's 5.58, not
  closer to ISO as predicted.
- P3 CONFIRMED. Median spread between Sigma* and ISO; primary CI below 0.

## Observations the paper must carry
- The Laplacian prior is a strong baseline: it improves on isotropic noise by about as
  much (0.57) as Sigma* improves on it (0.58).
- It is less over-dispersed than Sigma* (internal RMSF 0.85 vs 1.49) at matched torsion.
- It degraded far less from fitting folds to test (1.27-1.33 -> 1.35) than Sigma* did
  (0.22-0.46 -> 0.77).
- Its residual distortion sits in the stiff channels (bond 2.18, angle 1.96) and in
  out-of-plane (2.62), the coordinates a graph Laplacian does not resolve.
