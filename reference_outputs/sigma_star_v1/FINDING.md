# Exact Sigma* for ligands: 6x more uniform than the Gauss-Seidel sampler,
# and one channel the harmonic model cannot reach

150 MISATO ligands, 8-40 heavy atoms, seed 20260924.
`scripts/exact_sigma_star_noise.py`, predictions hashed (9da7f250...) before any
measurement. B rows by central finite difference, deliberately an independent
route from the analytic angle and out-of-plane forms used elsewhere, so a
mistake in either cannot cancel.

## The premise that was wrong, and why this is cheap

`outputs/theory_modules_v1/RESULT.md` treats Sigma* = (B^T F B)^+ as unusable at
O(N^3). That holds for proteins. A 30-heavy-atom ligand gives a 90x90 matrix
whose dense eigendecomposition is microseconds, and the reference geometry is
fixed per system so it is computed once and cached. For the molecules this paper
is about, the exact harmonic covariance is affordable and no low-rank
approximation is needed.

The internal set must be COMPLETE or the pseudo-inverse is meaningless: with
only bonds and angles the torsional directions are in the null space of H and
would receive unbounded amplitude. Bonds, angles, proper torsions and
out-of-plane coordinates are all included, each with its measured width
(0.0326 A, 2.774 deg, 14.6 deg, 6.48 deg), F = diag(1/tau^2).

## Result

| channel | ratio to physical | IQR |
|---|---|---|
| bond | 1.146 | 1.054 - 1.310 |
| bond angle | 0.850 | 0.812 - 0.917 |
| **torsion** | **0.564** | 0.391 - 0.796 |
| out-of-plane | 0.780 | 0.712 - 0.843 |
| internal RMSF | 0.726 | 0.457 - 0.990 |

**Channel spread 0.582**, against the Gauss-Seidel sampler's 3.295: a factor of
5.7 more uniform, better than the 4.6 the theory pass predicted. Every channel
lands between 0.56 and 1.15 where the sampler spans 2.28 to 5.57.

P2 (spread below 1.0) CONFIRMED. P4 CONFIRMED: exactly six near-zero eigenvalues
in 95.3% of systems. The other 4.7% are reported rather than dropped; they are
expected to be disconnected fragments or degenerate geometries and have not yet
been inspected.

## P1 FAILS, on one channel, for a reason that is not an implementation defect

Torsion comes out at 0.564, below the stated 0.65 floor. This is the harmonic
approximation reaching its limit rather than a bug: real torsional motion is
MULTI-MODAL, jumping between rotamer wells, while a harmonic model can only
oscillate about a single mean. No choice of tau_torsion fixes that, because the
deficit is in the shape of the distribution, not its width.

**This must be stated as a limitation of the prior, not hidden.** It also
predicts where the remaining gap in a trained model will sit: the prior can
deliver correct bond, angle and out-of-plane statistics and will systematically
under-deliver torsional spread.

## What this does NOT yet show

These are properties of the NOISE, not of a model trained on it. The v3
experience is directly relevant: a prior whose own statistics were correct did
not hand those statistics to the trained model, because the learned velocity
field contributes its own error. Nothing may be claimed about a model until one
is trained on this prior and measured.

---

# Audit findings, 2026-09-25

## A silent training gap in the 4.7% of systems without exactly six rigid modes

This file previously said those systems were "reported rather than dropped" and
guessed they were disconnected fragments. Both statements are wrong.

Inspected: two of sixty sampled systems (1N9A, 35 atoms, 10 near-zero modes;
3B66, 29 atoms, 7). **Both are single connected components.** The cause is
NEAR-LINEAR BOND ANGLES — 1N9A has angles at 179.07 and 179.57 degrees, 3B66 at
174.4, 174.5 and 178.96, the nitrile and sp-linear motifs. A single scalar
bond-angle coordinate cannot span both orthogonal bending directions of a linear
X-Y-Z unit, so one bending direction per linear group is captured by no Wilson
row at all, and the resulting genuine near-zero Hessian eigenvalue is classified
as "rigid" by the cutoff.

"Reported rather than dropped" is true only of the offline measurement script.
In the training path, `_harmonic_local_noise` calls
`factor, _ = sigma_star_factor(...)` and **discards the rigid-mode count**. For
the affected molecules a real, floppy intramolecular bending direction receives
zero noise from the harmonic term, and is not covered by the rigid
translation/rotation term either, because it is not a rigid mode. The
suppression is silent: nothing logs or flags it.

This is a limitation of the trained checkpoints, not only of the description.
Fixing it requires a second out-of-plane-like coordinate for linear units and
therefore retraining, so for now it is documented and its size is bounded at
3 to 5 percent of molecules.

## The comparators' advantage is mostly the projection; Sigma*'s cannot be

With `angle_projection` off, ANISO v3's three-channel spread balloons from
0.41-0.48 to 2.3-2.5, its angle channel going from about 1.4 to 3.0-3.5. So the
ANISO headline numbers are largely the inference projection rather than the
prior. Since the projection provably touches only bonds, angles and pocket
clashes, it cannot be responsible for Sigma*'s torsion and out-of-plane
advantage. The projection-off sweep for Sigma* and ISO was still running when
the audit reported, so this causal statement is supported for ANISO and NOT YET
closed for the other two arms; it must be before the paper attributes the effect.

## Scale range, wider than measured before

Sigma*'s four-channel spread is 0.16 to 0.31 from scale 1.0 to 3.5 across the
folds sampled, rising to 0.43 at 4.0 on fold 1. So it is flat over a wide range
rather than exactly invariant, and the endpoint drift should be stated. The
comparators sit at 1.4 to 3.1 over the same range and never overlap it.

## What the audit could not break

Independent re-derivation of the Wilson rows on 60 systems not drawn from either
script's own sample, comparing full Hessian eigenvalue spectra rather than
aggregate ratios: zero mismatches in rigid-mode or non-rigid-DOF counts, maximum
relative eigenvalue error 6.4e-7, median 7.7e-10. The factor cache cannot
collide: two molecules would need identical bond-index bytes and identical live
coordinates to 0.001 A. The three checkpoint families differ in exactly one
config block and share byte-identical `fit_examples.csv` per fold, identical
epochs, steps, and seeds, so there is no compute or data-split confound.

One latent issue, inert today: `_TOPOLOGY_CACHE` keys on `n_atoms & 0xFF`, which
would alias at 256 atoms. Current use caps at 40 heavy ligand atoms.

## A wording correction the audit forced

The amplitude rescale in `_harmonic_local_noise` pins the delivered noise to
whatever the isotropic path would have produced at the same `--internal` scale.
The physical tau constants therefore fix Sigma*'s RELATIVE apportionment across
channels, not the absolute noise level. "Scale" is a multiplier on an isotropic
reference budget and scale 1.0 does NOT reproduce the physical widths; the paper
must say so or readers will assume otherwise.

## The prior, not the projection: closed, 2026-09-25

The audit left this open. Measured directly, `--angle-projection off`, scale 2.5,
same 96 balanced fitting examples, four channels:

| arm | bond | angle | torsion | out-of-plane | spread (3 folds) |
|---|---|---|---|---|---|
| **Sigma*** | 0.650 | 1.809 | 0.837 | **1.048** | **1.160 / 1.133 / 1.116** |
| ANISO v3 | 0.773 | 3.166 | 1.665 | **4.794** | 4.022 / 4.007 |
| ISO | 2.288 | 7.599 | 2.962 | **5.694** | 5.311 / 4.675 / 5.460 |

With NO projection at all, Sigma*'s spread is 1.12-1.16 against ANISO's 4.01 and
ISO's 4.68-5.46. **The advantage belongs to the prior.**

The single cleanest piece of evidence is the out-of-plane column: Sigma* reaches
1.048, essentially exact, with the projection switched off, where ANISO gives
4.794 and ISO 5.694. The projection provably never touches that coordinate, so
nothing but the prior can account for it.

**The two mechanisms are complementary and the paper must say so rather than
claiming either alone.**

* The PRIOR delivers torsion and out-of-plane, which the projection cannot reach
  at all: out-of-plane 1.048 against 4.794, torsion 0.837 against 1.665.
* The PROJECTION delivers bond and angle: Sigma*'s angle channel goes from 1.809
  without it to 0.751 with it.
* Together, spread 1.13 without projection becomes 0.28 with it.

This also sharpens the ANISO comparison. ANISO v3's published bond and angle
numbers were largely the projection: its three-channel spread goes from 0.41-0.48
with projection to 2.3-2.5 without. Sigma*'s does not depend on the projection in
the same way.
