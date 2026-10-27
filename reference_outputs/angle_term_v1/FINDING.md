# The angle term: what an inference-time correction can and cannot do

96 balanced fitting-fold examples per cell, three full-scale folds, scales
0.32/1.50 (EXPLORATORY, not a validated operating point), heavy atoms only,
median ratios of generated spread over reference spread.

## All seven operators

| cell | angle loss | angle projection | target | restore | bond angle | bond length | dihedral | internal RMSF |
|---|---|---|---|---|---|---|---|---|
| A | off | off | — | off | 4.806 | 1.335 | 1.677 | 0.648 |
| B | **on** | off | — | off | 4.305 | 1.157 | 1.464 | 0.586 |
| C | off | **on** | hard | off | 1.613 | 1.244 | 1.258 | 0.497 |
| D | **on** | **on** | hard | off | 1.407 | 1.091 | 1.111 | 0.456 |
| E | off | **on** | hard | **on** | 2.418 | 1.816 | 1.723 | 0.688 |
| F | **on** | **on** | hard | **on** | 2.085 | 1.559 | 1.504 | 0.623 |
| G | off | **on** | **soft, tau = 2.794 deg** | off | 1.699 | 1.273 | 1.261 | 0.500 |

G per fold: angle 1.809 / 1.499 / 1.789, RMSF 0.545 / 0.442 / 0.513.

## The figure of merit, and the result

Dispersion-normalised angle ratio = angle ratio / internal RMSF ratio. It is
invariant to any scalar amplitude change, so it measures selectivity: how much
of an angle improvement is real, as opposed to the whole ensemble shrinking.

| cell | normalised angle ratio |
|---|---|
| A no correction | 7.42 |
| B loss only | 7.34 |
| C hard projection | 3.24 |
| **D hard projection + loss** | **3.09** (best) |
| E hard projection + restore | 3.51 |
| F all three | 3.35 |
| G soft projection | 3.40 |

**The bar set in advance — bond angle ratio about 1 with internal RMSF at or
above 0.60 — requires a normalised ratio of 1.67 or lower. Every operator with
a projection sits at 3.1 to 3.5. Nothing moves it.**

Three separate inference-time manipulations were tried and all three failed to
change the selectivity:

* changing the target AMPLITUDE (restore, E and F): a global scalar has no
  directional structure, so it multiplies the stiff directions the projection
  just fixed by the same factor as the free ones. Measured: soft channels move
  exactly with the amplitude (dihedral 1.373 against RMSF 1.384) while the two
  stiff channels move MORE (bond angle +8%, bond length +6%), so restore is
  cosmetic for the soft coordinates and marginally counterproductive for the
  stiff ones.
* changing the target VALUE (soft target, G): driving angles to
  reference + N(0, tau^2) with the corpus-measured tau = 2.794 degrees instead
  of to reference exactly. Normalised ratio 3.40 against the hard constraint's
  3.24 — no improvement, slightly worse, on all three folds.
* adding a LOSS term (B, and the B/D contrast): main effect on the angle ratio
  -0.353 +- 0.601 with the sign flipping across folds, against a retraining
  noise floor of +0.750 measured by retraining one fold with the loss off.
  Indistinguishable from noise. The projection's main effect is -3.045 +- 0.149.

## Why the soft target could not help, in one line of arithmetic

The model generates an angle spread of 13.88 degrees where the reference is
2.77. The physical width tau is 2.794 degrees. Since tau is small compared with
what the model produces, targeting 2.794 rather than 0 removes almost the same
amount of angular amplitude. And the angle directions span 52% of the
heavy-atom degrees of freedom (bonds span 38.4%, bonds and angles together
90.8%), so removing that amplitude necessarily costs about half the dispersion
budget whichever target is used.

## The conclusion this forces

**This class of defect cannot be repaired at inference.** The model's dispersion
budget is in the wrong place — it is spending roughly half of it on angular
directions that should hold 2.8 degrees — and every operator that puts the
angles right must take that budget away, because the budget is the problem.

Putting the budget in the right place is a statement about the process that
generated it, so it has to happen during training: the noise process itself must
be anisotropic, rather than isotropic noise followed by a correction. That is
the theory report's Sigma* = kT (B^T F B)^+ prior applied as the TRAINING prior
via the O(N) Gauss-Seidel sampler, and it is the one thing in the theory recipe
that has not been tested. Adding a weighted loss term on top of isotropic noise,
which is what cell B tested, is not the same thing and does not work.

## Pre-registered predictions and how they scored

| prediction | outcome |
|---|---|
| angle loss changes little, projection does the work | CONFIRMED, with a retraining-noise floor behind it |
| theory C-PRED-1, angle in [0.7, 2.0] at cell D | 1.407, CONFIRMED |
| theory C-PRED-2, bond length in [1.2, 1.6] | 1.091, MISSED, below the band |
| theory C-PRED-3, dihedral in [0.5, 1.9] | 1.111, inside, but NOT selectively: normalised 2.59 -> 2.44 |
| theory C-PRED-4, RMSF >= 0.60 with angle about 1 | FAILED, 0.456 at cell D |
| aromatic planarity does not improve | CONFIRMED, normalised 6.23 -> 6.75 |
| restore returns RMSF to 0.60 and angle stays below 2.0 | RMSF confirmed, angle REFUTED at 2.418 |
| soft target gives angle in [0.85, 1.35] at RMSF >= 0.60 | REFUTED on both, all three folds |

One of these pre-registrations was internally inconsistent and that is recorded:
requiring the normalised ratio to hold near 3.2 AND the RMSF to return to 0.60
forces the angle ratio to 1.92 or above, which contradicts the same
pre-registration's requirement that it stay below 2.0. The measurement landed
almost exactly on that boundary (cell F: RMSF 0.623, angle 2.085). Arithmetic
consistency of a pre-registration must be checked before it is issued.
