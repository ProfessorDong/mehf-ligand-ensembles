# aniso_prior_v3 — pre-registration, written 2026-09-23 while training, before any v3 sample exists

## What changed from v2, and why, each justified by a measurement on the NOISE

v2 (scaled tau, 8 sweeps) gave bond angle 1.162, dihedral 1.036, but bond length
0.393 (2.5x under-dispersed) and internal RMSF 0.413.

1. **tau held absolute** (`anisotropic_tau_absolute: true`). Measured on the v2
   noise at the sampling scale 1.5: bond width 0.0165 A, angle 1.08 deg — both
   about half of physical (0.0326 A, 2.774 deg). Scaled tau multiplied the
   physical width by the amplitude ratio (~0.34 at scale 1.5), so the prior
   itself squeezed the stiff coordinates at sampling time. That, not a "triple
   constraint", is the measured cause of v2's bond under-dispersion.

2. **32 sweeps** (`anisotropic_sweeps: 32`). With 8 sweeps the stiff residual is
   proportional to the starting amplitude, so absolute tau alone does nothing:
   angle 3.78 deg at scale 3.5 vs 9.57 deg at 7.0.

3. **bonds kept in the stiff set.** The first plan excluded them. Measured:
   excluding bonds leaves them at the full isotropic amplitude (0.42 A at scale
   7, a 30% stretch), the linearised angle constraint then fails, and angle
   width tracks amplitude even at 64 sweeps (9.57 -> 8.28 -> 7.56). The theory
   specified S = {bonds, angles} together; that is what is used.

Resulting noise, measured before training, 32 examples:

| sampling scale | angle | bond | dihedral |
|---|---|---|---|
| 1.5 | 2.32 deg | 0.0323 A | 7.86 deg |
| 2.5 | 2.57 deg | 0.0347 A | 10.42 deg |
| 3.5 | 2.95 deg | 0.0404 A | 13.20 deg |
| physical | 2.774 deg | 0.0326 A | 14.6 deg |

Stiff widths near physical and roughly amplitude-independent over 1.5-3.5;
the dihedral, the soft coordinate, carries the amplitude. Decoupling is partial:
at scale 7.0 the angle still reaches 5.25 deg.

## Protocol

Measure with `scripts/mehf_internal_geometry_spectrum.py`, 96 balanced FITTING
examples, three folds, rigid 0.32, at sampling internal scales 1.5 / 2.5 / 3.5.
Choose the scale by internal RMSF ONLY (smallest scale giving three-fold mean
RMSF >= 0.60; if none does, report that). Report every channel at every scale.
Angle is never a selection criterion.

## Predictions

P1  at scale 1.5, bond length ratio rises from v2's 0.393 into [0.7, 1.5].
P2  at scale 1.5, bond angle ratio in [0.8, 1.5].
P3  the bond angle ratio rises by LESS than the RMSF ratio as scale goes
    1.5 -> 3.5; concretely angle(3.5)/angle(1.5) < RMSF(3.5)/RMSF(1.5).
    This is the test of the decoupling. If the two rise in proportion, the
    learned field re-couples what the prior decoupled.
P4  THE BAR: at the RMSF-selected scale, bond angle ratio in [0.8, 1.35] AND
    internal RMSF >= 0.60, three-fold means.

Consistency checked: P4's two conditions can hold simultaneously; P3 is implied
by but does not imply P4.

Falsifiers stated now. If P3 fails, the ceiling lives in the learned velocity
field, not the prior, and no prior change can fix it. If P3 holds but no scale
reaches RMSF 0.60 within [1.5, 3.5], the decoupling is real but insufficient in
range, and the remaining gap is the inter-coordinate correlation (independent
w_k draws) that the theory's full Sigma* would supply.

## RESULT, folds 1-2 (fold 3 training)

| fold/scale | angle | bond | dihedral | RMSF |
|---|---|---|---|---|
| f1 1.5 | 3.055 | 0.731 | 1.451 | 0.603 |
| f1 2.5 | 3.166 | 0.773 | 1.665 | 0.740 |
| f1 3.5 | 3.343 | 0.833 | 1.822 | 0.890 |
| f2 1.5 | 3.098 | 0.785 | 1.648 | 0.614 |
| f2 2.5 | 3.224 | 0.840 | 1.909 | 0.754 |
| f2 3.5 | 3.451 | 0.935 | 2.119 | 0.887 |

P1 CONFIRMED (bond 0.73/0.79). P2 REFUTED (angle 3.06/3.10). P3 CONFIRMED
(1.5->3.5: RMSF x1.48, angle x1.09 f1; RMSF x1.44, angle x1.11 f2). P4 FAILS
on the angle.

Reading: prior noise angle at scale 1.5 is 2.32 deg, output angle ~8.5 deg. The
angle floor (~3x physical) is contributed by the LEARNED FIELD, not the prior,
and is nearly amplitude-independent. v2's angle of 1.16 was bought by the
scaled-tau squeeze of the sampling prior, which is also what cost it RMSF.

## FOLLOW-UP, EXPLORATORY, pre-stated before running

v3 prior (amplitude headroom) + the existing hard angle projection at inference
(8 sweeps, as in cell C). No retraining. Scales 2.5, 3.0, 3.5. Scale selected by
RMSF only: smallest scale whose two-fold mean RMSF >= 0.60.

Prediction, from cell C's measured factors (angle x0.336, RMSF x0.767):
at scale 3.5 angle ~1.12 and RMSF ~0.68; at 2.5 angle ~1.06 and RMSF ~0.57.
PASS iff at the RMSF-selected scale angle in [0.8, 1.35] and RMSF >= 0.60.

This is a configuration chosen after seeing v3's result, on the same fitting
examples. Any pass is EXPLORATORY and needs confirmation on data it was not
chosen on.
