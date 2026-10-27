# The geometric cost of isotropic Cartesian noise

Measured 2026-09-23 on 2,000 MISATO ligand MD ensembles.
Script: `scripts/isotropic_noise_geometry_cost.py`.
Confound check: `scripts/check_hydrogen_constraint_confound.py`.

## Design

Two arms at **matched coordinate-level dispersion**:

* **physical** — the real MD ensemble of the ligand.
* **synthetic** — one frame of that ligand plus isotropic, independent per-atom
  Gaussian noise, whose amplitude is solved so the arm has the SAME internal
  RMSF as the physical arm. The match is exact: median achieved ratio 1.00002.

Both arms then have their internal coordinates measured the same way. The ratio
is synthetic spread over physical spread, per system, reported as a median over
systems.

No model is involved. This is a property of the noise, not of any network.

## Result, heavy atoms only

| internal coordinate | physical spread | synthetic spread | ratio | IQR |
|---|---|---|---|---|
| bond length | 0.0326 Å | 0.498 Å | **15.27** | 11.2 – 20.2 |
| bond angle | 2.77° | 27.3° | **9.77** | 7.73 – 11.2 |
| dihedral | 14.6° | 54.0° | **3.51** | 2.68 – 4.37 |

Matched internal RMSF: 0.645 Å median.

The ordering is monotone in stiffness, which is the predicted shape. Physical
thermal motion buys its coordinate dispersion almost entirely with the soft
coordinates; isotropic noise has no such preference and spends equally in every
direction, so at equal coordinate dispersion it distorts the stiff coordinates
by an order of magnitude more.

## The confound that was found and removed

The first run reported a bond-length ratio of **25.99**. That number is wrong
and is not used.

MISATO's MD constrains bonds involving hydrogen. Measured directly: the median
physical spread of any-hydrogen bonds is **2.09e-06 Å**, i.e. exactly frozen by
the constraint algorithm, giving a meaningless ratio of ~219,000 for those bonds
and inflating the all-bond figure. Heavy–heavy bonds, which the integrator
treats normally, give **15.31** on the same 600 systems and **15.27** on 2,000.

Every internal coordinate touching a hydrogen is therefore excluded. The
dihedral ratio moves the other way under this restriction, from 2.12 to 3.51,
because hydrogen-bearing dihedrals such as methyl rotations are unusually floppy
and were diluting the physical spread.

## Why this matters for 3D molecular generation

Diffusion and flow-matching models for 3D molecules perturb, and are trained to
predict, **atom coordinates**, under an isotropic Gaussian process and an
isotropic squared-error objective. They are then judged on **internal-coordinate**
statistics: bond length and bond angle distribution divergences, dihedral
distributions, and validity checks built from them.

These measurements give the exchange rate between the two. A residual error that
is isotropic in Cartesian space costs about 15x more in bond-length fidelity,
and about 10x more in bond-angle fidelity, than a physically distributed error
of the same magnitude. Equivalently: coordinate-level accuracy is nearly blind
to precisely the failure that geometric validity checks detect, and an isotropic
training objective applies no pressure to fix it.

## What has NOT been shown

* That any specific published model exhibits this in its samples. That is a
  separate measurement on released generated molecules and has not been done.
* That correcting the anisotropy improves any generative model. That is the
  method claim and it is not yet tested.
* The numbers come from one corpus (MISATO, protein-bound ligands, 100 frames
  per system). They have not been reproduced on GEOM or CrossDocked.

---

# The prediction on a real model, and how it failed usefully

Measured 2026-09-23. Script: `scripts/mehf_internal_geometry_spectrum.py`.
96 balanced fitting-fold examples, full-scale checkpoint `full_f1`, scales
(0.32, 1.50), heavy atoms only (`load_system` drops hydrogens, so MEHF never
sees them).

**Stated prediction, written in the script before running:** bond length >
bond angle > dihedral, all above 1, bond length largest by a wide margin.

**Measured:**

| internal coordinate | reference | generated | ratio | IQR |
|---|---|---|---|---|
| bond length | 0.0327 Å | 0.0457 Å | **1.41** | 1.30 – 1.54 |
| bond angle | 2.77° | 13.88° | **5.02** | 4.67 – 5.56 |
| dihedral | 13.8° | 24.9° | **1.78** | 1.26 – 2.53 |

**The prediction is wrong on the ordering and is retracted as stated.** Bond
length is the best preserved coordinate, not the worst.

## Why, and why the real answer is sharper

MEHF is not a bare isotropic Cartesian model. It carries exactly two geometry
terms and no others:

* `bond_loss_weight = 1.0` — a bond length penalty in the training objective.
* `projection_iterations = 8` — `_project_geometry`, documented in the source as
  "SE(3)-equivariant inference projection for **bonds and hard clashes**".

There is no angle term and no torsion term, in the loss or in the projection.

So the model contains its own controlled experiment:

| internal coordinate | explicitly constrained | ratio |
|---|---|---|
| bond length | **yes**, loss and projection | **1.41** |
| bond angle | **no** | **5.02** |
| dihedral | no, but physically soft | 1.78 |

The statement this supports is stronger and more actionable than the one that
was predicted:

> Under an isotropic Cartesian objective, every internal coordinate that is not
> explicitly constrained is left over-dispersed, and the damage scales with the
> physical stiffness of that coordinate. Bond angles are the stiffest
> unconstrained coordinate in this model and are the worst by a factor of three
> over the next one. Adding a single explicit term is enough to bring a
> coordinate back to near 1, as the bond length column shows.

That is a direct, cheap, falsifiable method proposal: add the angle term, and
the 5.02 should fall the way 1.41 already has.

## Caveat on the RMSF column

The internal RMSF ratio here is 0.70, not the ~1.0 that the calibration
reported, because this script measures on `load_system`'s heavy-atom block with
its own reference frame draw while the calibration used
`subspace_dispersion.block_moments`. The two are different protocols on the same
model and are not interchangeable; the geometry ratios above are all computed
within this one protocol, reference and generated alike, so the comparison is
internally consistent.

---

# RETRACTION and correction, 2026-09-23, from the independent audit

## The causal story was wrong: it is the projection, not the loss

The audit ran the ablation that separates the two candidate causes of the good
bond-length number, by overriding `projection_iterations` on the same
checkpoint and resampling. Median ratios, n=96, scales 0.32/1.50:

| projection_iterations | bond length | bond angle | dihedral |
|---|---|---|---|
| 8 (as reported) | **1.41** | 5.02 | 1.78 |
| 4 | 2.76 | 5.18 | 1.75 |
| 2 | 4.29 | 5.35 | 1.75 |
| 1 | 5.57 | 5.51 | 1.75 |
| 0 (projection off) | **7.40** | 5.78 | 1.78 |

With the projection off, the bond length ratio is 7.40 — the same range as the
bond angle (5.02 to 5.78), which has no explicit term at all. The trained
`bond_loss_weight = 1.0` term buys almost nothing on its own. The angle and
dihedral columns stay flat across every setting, which is the control that
shows the override did what it claims.

**The sentence "Adding a single explicit term is enough to bring a coordinate
back to near 1, as the bond length column shows" is retracted.** What brings it
back to near 1 is a hand-written SE(3)-equivariant geometric correction applied
at inference. Training pressure through the objective did not generalise to the
coordinate it penalises.

The corrected statement, which is stronger and less comfortable:

> Under an isotropic Cartesian objective, adding a penalty term for an internal
> coordinate does NOT fix that coordinate. What fixes it is an explicit
> geometric correction operator applied to the samples. This matters because
> adding geometry losses is the common remedy in the literature and this
> evidence says it does not work.

## The exchange rate is sensitive to the matching statistic, by about 2x

The audit built a superposition-free alternative match target (mean standard
deviation of all heavy-atom pairwise distances, exactly matched by iterative
rescale) and reran on 800 systems:

| internal coordinate | matched on internal RMSF | matched on pairwise distance |
|---|---|---|
| bond length | 15.27 | **7.02** |
| bond angle | 9.77 | **5.07** |
| dihedral | 3.51 | **1.41** (Q1 = 0.96) |

The ordering survives and every value stays above 1 at the median, so the
qualitative claim holds. The magnitudes roughly halve, and the dihedral's lower
quartile crosses below 1, meaning for a fraction of ligands isotropic noise at
this matched dispersion does not distort torsions more than physical motion.

Cause: Kabsch superposition to a mean structure is not neutral. It removes six
rigid degrees of freedom by least squares, and correlated physical motion is
more rotation-like than iid noise, so alignment attenuates the physical arm
more. Matching post-alignment RMSF therefore does not match true dispersion.

**15.27 / 9.77 / 3.51 must not be presented as "the" exchange rate.** Both
matching statistics are reported, as a range.

Related: the one-shot rescale in `noise_arm` is exact only for the
superposition-based statistic and only near the operating point. The audit
measured up to 2.6% non-linearity at large sigma, and found the alternative
statistic needs an iterative rescale (one-shot reaches only 1.065 median).
At the actual operating point the match is 1.00002 median, 0.9996 to 1.0054
over 2,000 systems, so the reported numbers are not affected.

## Ring dihedrals inflate the dihedral ratio

Splitting dihedrals by whether they lie inside a ring, 730 systems:
ring 6.20 against non-ring 2.82. Ring dihedrals are near rigid, the same
mechanism as the constrained hydrogen bonds already excluded. The write-up
noted that hydrogen-bearing dihedrals are unusually floppy but not that ring
rigidity pulls the number the other way. Both must be stated.

## Scope of the attribution sentence

"There is no angle term and no torsion term" is true of the checkpoint measured
(`full_f1`, whose saved `model_cfg` has only `bond_loss_weight`,
`clash_loss_weight`, `projection_iterations`), and must be scoped to that
checkpoint rather than to MEHF in general, because an angle mechanism is being
added to the codebase.

## Provenance of the inference scales

The scales 0.32 / 1.50 used for the model measurement came from an exploratory
per-fold grid search (`outputs/internal_calibration_v1/spread_f1.json`) and are
not a validated operating point. Any report must say so.

## What the audit could not break

Independent reimplementation from the raw store reproduced 15.273225 / 9.767662
/ 3.506791 bit-identically, and 1.409 / 5.018 / 1.776 for the model, to three
decimals. `bond_index` integrity checked on 2,000 systems: no self-bonds, no
duplicates, no out-of-bounds. Seed changed to 7: 15.06 / 9.58 / 3.48, within
1.5%. Frame thinning to every 5th and every 10th frame: no dependence. No
direction or swap error found.

Open worry the audit flagged and did not resolve: whether 1 ns, 100-frame MD
under-samples multi-modal torsion distributions, which would make the physical
dihedral spread an underestimate and inflate the dihedral ratio further.

---

# SECOND CORRECTION, 2026-09-23: the amplitude match used the wrong atom set

Found by the theory agent, verified independently
(`scripts/verify_heavy_vs_allatom_match.py`, 600 systems, same seed).

`isotropic_noise_geometry_cost.py` sets the synthetic arm's noise amplitude from
`internal_rmsf(coords)` computed on the store's `ligand_coords`, which INCLUDES
hydrogens, while every internal coordinate it measures is heavy-atom only.
Hydrogens are far more mobile, so the two are not the same number:

    all-atom internal RMSF    0.6405 A
    heavy-atom internal RMSF  0.4640 A      (ratio 0.724)

The synthetic arm's heavy atoms were therefore given about 38% more dispersion
than the physical arm's heavy atoms, and every ratio measured on heavy-atom
internals is inflated by that much.

| internal coordinate | all-atom match (published) | heavy-to-heavy match |
|---|---|---|
| bond length | 15.31 | **11.55** |
| bond angle | 9.81 | **7.88** |
| dihedral | 3.55 | **2.47** |

For a model that never sees hydrogens -- MEHF does not, `load_system` drops
them -- the heavy-to-heavy numbers are the defensible ones.

## The headline number has now moved twice, and once more under a different match

| stage | bond length ratio |
|---|---|
| first reported | 26.0 |
| minus SHAKE-frozen X-H bonds | 15.3 |
| minus the all-atom amplitude match | **11.6** |
| under the audit's superposition-free pairwise-distance match | ~7.0 |

The ordering bond > angle > dihedral survives every one of these, and every
value stays above 1 at the median. **The magnitude does not survive as a single
number.** Any claim must be stated as a range with the matching protocol named,
not as "isotropic noise costs 15x".

---

# VOID: the first anisotropic-prior run, 2026-09-23

`outputs/aniso_prior_v1/` is void and is kept only as a record of the bug.

The anisotropic training prior was implemented as Gauss-Seidel sweeps driving the
noise's bond and angle components to draws at their physical widths. Its first
measured result looked like a clean refutation: bond angle ratio 4.806 -> 5.631
(f1) and 7.483 (f2), i.e. WORSE than the isotropic baseline, with the internal
RMSF unchanged at 0.64.

That conclusion is wrong and was never reported as a result, because the control
written before it was interpreted -- measure the NOISE ITSELF, anisotropic path
on versus off, same systems and seed
(`scripts/check_structured_noise_itself.py`) -- failed:

| noise channel | isotropic draw | anisotropic draw | ratio |
|---|---|---|---|
| bond length | 0.290 A | 0.363 A | 1.25 WIDER |
| bond angle | 17.5 deg | 21.1 deg | 1.20 WIDER |
| dihedral | 26.8 deg | 40.3 deg | 1.50 WIDER |

The bond component should have been about 0.025 A. It was 15x that, and every
channel moved the wrong way, including the dihedral, which the sweeps do not
touch.

Cause: no under-relaxation. Bonds and angles are a REDUNDANT constraint system
and un-relaxed Gauss-Seidel on redundant constraints oscillates and diverges,
adding amplitude instead of removing it. The existing inference projection uses
`projection_relaxation = 0.5` for exactly this reason; the new sweeps used an
implicit 1.0.

After applying the same relaxation the control passes:

| noise channel | isotropic draw | anisotropic draw | physical |
|---|---|---|---|
| bond length | 0.290 A | **0.0445 A** | 0.0326 A |
| bond angle | 17.5 deg | **3.02 deg** | 2.774 deg |
| dihedral | 26.8 deg | **14.44 deg** | 14.6 deg |

The noise now carries essentially the physical internal-coordinate structure.

The three checkpoints in `aniso_prior_v1` were trained against the diverging
process and every number measured from them is void. Retrained as
`aniso_prior_v2`.

**The general point, which belongs in the paper's methods discussion:** the
control that measures the intervention itself, rather than its downstream
effect, was what caught this. Without it the project would have reported a
confident negative result -- "structuring the prior makes bond angles worse" --
that was an artifact of a diverging solver. A false negative is as damaging as a
false positive and is harder to notice, because it agrees with the expectation
that the simple fix will not work.

---

# THREE CORRECTIONS from the second theory pass, 2026-09-24

## C1. "276x stiffer than the typical Cartesian mode" is RETRACTED

The figure came from the first theory agent's prose and was relayed onward. It
is not reproducible and was never written to disk: nothing under
`outputs/wilson_b_theory_v1/` contains it. An independent recomputation with
three reasonable definitions of "typical mode" gives medians **78.8, 105.0 and
97.1**, with IQRs reaching about 200. 276 sits in the upper quartile of two
definitions and is the median of none.

**Supportable sentence: the stiff directions are about 100x stiffer than the
typical Cartesian mode, with a wide per-molecule spread.** Any use of 276 is
withdrawn.

## C2. "Independent w_k draws are the missing ingredient" is WRONG

This was stated repeatedly, including to the user, as the explanation for why
correct marginal widths coexist with too-low overall dispersion. It is refuted.

Under Sigma* the internal coordinates are near-UNCORRELATED: median |corr| for
pairs sharing two or more atoms is **0.041**, and 7e-15 for disjoint pairs.
There is a theorem behind it: B (B^T F B)^+ B^T = F^{-1/2} P_M F^{-1/2}, and
P_M approaches the identity for a spanning set. Drawing each w_k independently
is therefore very nearly correct, and the amortised correlation network that
seemed the natural fix is dead on arrival.

**The real defect is the COMPLEMENT, not the stiff set.** Sigma*'s soft subspace
is violently anisotropic — the top mode carries 45.4% of its trace and the top
five carry 89.2%, so 90% of the soft motion lives in about 8% of the degrees of
freedom — while the Gauss-Seidel sampler leaves that subspace WHITE. Physical
soft motion is collective; white noise of the same marginal width adds in
quadrature and cancels. Measured exchange rate **3.085x** (IQR 2.37 to 4.01): at
matched torsional width the sampler reaches only 31.5% of physical internal
RMSF, where exact Sigma* reaches 125%.

This resolves the result that had no explanation: on `val` the dihedral ratio is
1.520 (over-dispersed) while internal RMSF is 0.618 (under-dispersed) at the
same time, which no scalar amplitude can reconcile. Their ratio is 2.46, inside
the measured IQR of the exchange rate. The pure-noise measurement and the
trained model agree.

## C3. "Aromatic planarity is broken by a factor of 4.3" needs requalifying

The measurement stands as stated — DiffSBDD 5.09 deg against real CrossDocked
ligands 1.18 deg, both single conformers through the identical function — but
the INTERPRETATION overreaches, for two reasons that push the same way.

First, the thermal width of this coordinate is not 1.18. Measured on MISATO MD
ensembles it is **6.48 deg**, so a 5.09 deg deviation is not outside what a
thermally accessible structure shows. (That 6.48 is itself inflated: the centre
selection does not filter for aromaticity, so saturated-ring pucker enters it,
and an aromatic-only figure would be lower.)

Second, crystal ligand geometries are refined under planarity restraints, so
1.18 deg is partly a property of refinement rather than of molecules.

**Report the ratio as a comparison against crystal reference geometry, not as
evidence that the structures are physically impossible.** The ensemble-matched
figure from `val`, 3.845, is the defensible one where a thermal comparison is
wanted.

## What survived and is strengthened

The B-matrix law holds on the model-free measurement: one sigma explains all
three published isotropic costs (11.55 / 7.88 / 2.47), with implied sigma
0.2662 / 0.2333 / 0.2967 A, agreement within 12.4%. Note this does NOT conflict
with the DiffSBDD refutation: pure noise obeys the law, as it must; a trained
model's residual does not, which is the finding.

The derived objective weights are confirmed independently from corpus widths:
angle 0.45339 against the handed-down 0.448 (1.2%), torsion 0.016367 against
0.0157 (4.3%), and a new out-of-plane weight of 0.08309 rad^-2.
