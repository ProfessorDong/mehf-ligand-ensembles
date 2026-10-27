# The isotropy prediction on a PUBLISHED model: REFUTED in its strong form

DiffSBDD released samples (`crossdocked_fullatom_cond`, 100 pockets x 100
molecules), measured 2026-09-24. Script `scripts/published_model_isotropy_test.py`
(sha256 e6c56859...), whose docstring stated the predictions before any
generated molecule was read. Heavy atoms; generated and reference both
hydrogen-free; 100/100 RDKit-sanitisable in the file checked.

## Headline: the strong prediction fails, in a specific and informative direction

Prediction P1 was that ONE sigma explains all coordinate classes, so that the
angle error follows from the bond error through ||B_angle||. After correcting
for the baseline (below), on 300 molecules:

| quantity | value |
|---|---|
| bond excess over real-ligand baseline | 0.1148 A -> implied sigma 0.0812 A |
| angle excess predicted from that sigma | 6.96 deg |
| angle excess MEASURED | **12.10 deg** |
| measured / predicted | **1.74x**, error -42.5% |

**P1 FAILS** (band was 35%). **The residual error of this published diffusion
model is NOT isotropic.** Angles are distorted about 1.7x more than an isotropic
error of the observed bond magnitude would produce; equivalently the model gets
bond lengths right better than isotropy predicts. A plausible reading, not
tested here: bond length is almost determined by atom types and bond order and
is therefore highly learnable, while angles carry more conditional freedom.

**P3 also FAILS.** In units of physical thermal width the damage is bond 3.52,
angle 4.36 — angles are worse, not bonds. The predicted ordering is wrong.

**P2 PASSES**: implied sigma 0.081 A, inside the stated 0.02-0.40 A.

**P4 PASSES, and strongly.** Mean absolute out-of-plane deviation at aromatic
centres: **generated 5.09 deg against real ligands 1.18 deg**, and the generated
value falls to 1.14 deg on relaxation, so the raw output really is non-planar.
Aromatic planarity is broken by a factor of 4.3 relative to real ligands.

## Two controls that had to run before any of this could be read

**1. The dihedral channel is not measurable this way and was dropped.**
Sweeping the positional restraint on the MMFF relaxation:

| restraint (kcal/mol/A^2) | bond (A) | angle (deg) | dihedral (deg) |
|---|---|---|---|
| 100 | 0.1115 | 8.91 | 19.05 |
| 10 | 0.1224 | 11.92 | 31.36 |
| 1 | 0.1243 | 12.76 | **47.91** |

Bond and angle plateau; the dihedral diverges as the restraint is released,
because an unrestrained minimisation rotates torsions to search for a nearby
conformer. The dihedral change therefore measures conformational search, not
local error, and is excluded from the decision. Reported so the exclusion is
visible, not silent.

**2. MMFF disagrees with real structures too, and that baseline is large.**
The identical measurement on the 100 real CrossDocked test ligands gives bond
0.0476 A and angle 4.06 deg at restraint 1. So roughly 38% of the generated
bond movement, in quadrature, is MMFF-versus-experiment disagreement rather than
model error. Every number above is the generated value with this baseline
removed in quadrature. Without the control the bond error would have been
overstated by that amount, and the isotropy test, which is a RATIO between
channels, would have been distorted in both channels.

## Raw, uncorrected numbers, for completeness

| channel | real ligands | DiffSBDD | ratio |
|---|---|---|---|
| bond change on relaxation | 0.0476 A | 0.1243 A | 2.6x |
| angle change on relaxation | 4.06 deg | 12.76 deg | 3.1x |
| aromatic out-of-plane, raw | 1.18 deg | 5.09 deg | 4.3x |

## What this does and does not establish

- It DOES establish, on a published pocket-conditioned diffusion model, that the
  generated local geometry is substantially worse than real ligands in every
  channel measured, and that aromatic planarity in particular is broken.
- It DOES refute the strong isotropy prediction. The B-matrix relation that
  holds exactly for isotropic noise does not describe a trained model's
  residual. Our own earlier measurement on MEHF is consistent with this:
  bond length was its best-preserved channel too, which had been attributed
  entirely to MEHF's explicit bond projection. DiffSBDD has no such projection
  and shows the same ordering, so that attribution is now in question and the
  MEHF write-up must be softened accordingly.
- It does NOT yet cover a second published model; only DiffSBDD's released
  CrossDocked samples were measured. TargetDiff and MolDiff checkpoints are
  downloaded but not sampled.
- 300 of the available 10,000 molecules were used (3 per pocket, 100 pockets);
  the numbers are stable across the restraint sweep but a full-scale rerun is
  cheap and should be done before publication.
