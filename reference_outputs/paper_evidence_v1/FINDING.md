# Evidence base for sections 1 and 2

Theory agent, 2026-09-25. Predictions pre-registered in `PREDICTIONS.md`
(sha256 `8ee6d088c9909447432966ecd265c63cb09c076b7c31cd5295c12e36cbbdbfe2`),
written before any number below was computed. Nothing under
`src/mdwm/models/`, `outputs/pareto_v1/`, `outputs/sigma_star_test_v1/` or
`outputs/sigma_star_v1/` was written. The MMP corpus `test` PDB ids (218 of
them) are excluded from every MISATO measurement here.

Commands:

```
PYTHONPATH=src conda run -n MDWM python outputs/paper_evidence_v1/a_matching_protocols.py
conda run -n MDWM python outputs/paper_evidence_v1/a2_blaw_amplitude.py
conda run -n MDWM python outputs/paper_evidence_v1/b_aromatic_oop.py
conda run -n MDWM python outputs/paper_evidence_v1/b2_oop_confounds.py
# (GPU) for f in f1 f2 f3; do for pi in 0 8; do
PYTHONPATH=src conda run -n MDWM python scripts/mehf_internal_geometry_spectrum.py \
  --checkpoint outputs/internal_calibration_v1/full_${f}/checkpoint.pt \
  --rigid 0.32 --internal 1.50 --n-examples 96 --projection-iterations ${pi} \
  --impropers --output outputs/paper_evidence_v1/mehf_isotropy/proj${pi}_${f}.json
conda run -n MDWM python outputs/paper_evidence_v1/c_isotropy_seam.py
conda run -n MDWM python scripts/published_model_isotropy_test.py \
  --generated-dir data/external/diffsbdd/crossdocked_fullatom_cond \
  --max-molecules 1000 --max-per-pocket 10 --restraint 1.0 \
  --output outputs/paper_evidence_v1/c_diffsbdd_isotropy_n1000.json
conda run -n MDWM python outputs/paper_evidence_v1/e_ff_restraint_sweep.py \
  --reference-dir data/external/crossdocked/test_set \
  --receptor-root data/external/crossdocked/test_set
```

---

# REFUTATIONS FIRST

**Three things that were being said must stop being said.**

**R-a. "Isotropic noise costs 15x / 11.6x / 7x in bond length" is not a fact
about the noise.** It is a fact about the rule chosen for "matched dispersion",
and that rule is a choice of one number, sigma. Three defensible
rotation-invariant matching statistics on the same 800 systems give sigma =
0.165, 0.201 and 0.282 A and bond ratios 6.95, 8.46 and 11.63. The physics is
the same in all three.

**R-b. "The 6.48 deg out-of-plane width is inflated by saturated-ring pucker,
and an aromatic-only figure would be lower" is FALSE, and it was my own
project's caveat.** The published estimator already applies a mean-|chi| < 10 deg
pre-filter to the centres, and that filter removes EVERY hydrogen-bearing ring
centre: their median mean-|chi| is 52.5 deg, because they are tetrahedral, not
puckered-planar. Restricting to sp2 centres changes the number by 0.0%
(6.466 -> 6.466 deg) and restricting to RDKit-aromatic centres by -1.1%
(6.466 -> 6.392 deg). The caveat is withdrawn; the number does not move.

**R-c. "Force-field relaxation is destructive" is FALSE as a general claim and
must be withdrawn.** It is true of unrestrained MMFF only. Across a restraint
sweep on the same 200 DiffSBDD molecules, a restraint of 5 kcal/mol/A^2 gives
the BEST angle distribution of any arm measured (W(angle) 1.664 against
unrestrained MMFF's 1.810), moves the ligand 0.399 A, and leaves the receptor
clash rate at the raw model's own level (mean 0.64 clashes, 20% of molecules,
against raw 0.64 and 17%). There is no setting at which relaxation is both
destructive and useful; there is a trade-off curve, and the middle of it is
better than either end.

**R-d (bonus, and it cuts against our own method).** The corrector gate compared
the projection only against restraint 0 and restraint 100. Restraint 5 is a
stronger baseline than either. Same 200 molecules, same held-out split seed
20260924 as `outputs/corrector_gate_v1/holdout_20260924.json`, and my pipeline
reproduces that file's arms exactly (raw 0.0486 / 5.3339, MMFF free
0.0262 / 1.8104, MMFF restrained-100 0.0234 / 2.4256), so this is a like-for-like
comparison:

| arm | W(bond) | W(angle) | planarity | RMSD from raw |
|---|---|---|---|---|
| F soft-target projection (gate, same seed) | 0.0264 | 2.247 | 0.604 | 0.234 |
| MMFF restraint 5 (this work) | **0.0260** | **1.664** | 2.138 | 0.399 |

The restrained force field wins both distributional channels; the projection
wins planarity and pose. The gate's surviving claims (aromatic planarity, no
clashes) are unaffected, but the pose-preservation margin is a factor of 1.7,
not the factor of 6.9 that the "moves the ligand seven times less" sentence is
built on, because that sentence compares against the unrestrained arm. This
belongs to whoever owns sections 3 and 4.

---

# a. How to state the cost of isotropic noise

## What was measured

`a_matching_protocols.py`, 800 MISATO systems, heavy atoms throughout
(internals, match target and noise amplitude), test PDB ids excluded, exact
match by bisection (achieved/target = 1.00000 median in every protocol).
Physical widths on these systems: bond 0.03260 A, angle 2.786 deg,
dihedral 14.97 deg.

| matching statistic | rotation-invariant | sigma (A) | bond | angle | dihedral | bond/angle |
|---|---|---|---|---|---|---|
| per-atom RMSF after Kabsch superposition (published) | yes | 0.2823 | **11.63** | 7.97 | 2.48 | 1.458 |
| SD of |x_i - centroid| | yes | 0.2009 | **8.46** | 6.10 | 1.74 | 1.387 |
| SD of all heavy-atom pairwise distances | yes | 0.1651 | **6.95** | 5.00 | 1.41 | 1.391 |
| per-atom SD about the centroid (rotation retained) | **no** | 0.6078 | 22.15 | 11.74 | 4.97 | 1.886 |

This reproduces both published numbers independently: 11.63 against the
heavy-to-heavy 11.55, and 6.95 against the audit's 7.02.

The last row is rejected on principle and not on its number: rigid tumbling
contributes dispersion and exactly zero internal-coordinate distortion, so a
statistic that counts it inflates the noise amplitude. It is reported to show
what the range looks like if that rule is dropped (A4 CONFIRMED, predicted >13,
measured 22.15).

## Why no single matched-dispersion number is defensible, and what is

`a2_blaw_amplitude.py` sweeps sigma directly, with no matching at all, and
divides the noise-induced internal spread by the injected sigma. If
Var(q_k) = sigma^2 ||B_k||^2 is the whole story, that returns ||B_k|| and
nothing else.

| sigma (A) | implied ||B_bond|| | implied ||B_angle|| | implied ||B_dih|| | bond/angle |
|---|---|---|---|---|
| 0.02 | 1.4023 | 1.5240 | 2.0435 | 1.367 |
| 0.05 | 1.4009 | 1.5193 | 2.0528 | 1.369 |
| 0.08 | 1.3979 | 1.5142 | 2.0684 | 1.371 |
| 0.15 | 1.3866 | 1.4876 | 2.1410 | 1.384 |
| 0.20 | 1.3744 | 1.4559 | 2.2442 | 1.402 |
| 0.2823 | 1.3459 | 1.3698 | 2.4120 | 1.459 |
| 0.40 | 1.2926 | 1.2192 | 2.4757 | 1.575 |

Below sigma ~ 0.10 A the implied B-norms are flat to 1% and the first-order law
is exact; they differ from the Wilson values (1.4142 / 1.6353 / 2.1213) by
-0.8% / -6.8% / -3.8%, which is the estimator (mean over coordinates of the
per-coordinate SD) against the median-over-coordinates B-norm, not a failure of
the law. Above sigma ~ 0.15 A the law visibly bends: at 0.40 A the angle channel
is 25% below its first-order value.

**Every matched-dispersion protocol operates at sigma = 0.165 to 0.28 A, i.e.
in the bent part of the curve, and 2 to 3.5x above the residual of any real
model** (DiffSBDD's implied sigma is 0.088 A raw, 0.081 A baseline-corrected).
That is an independent reason not to headline it: the matched-dispersion
framing evaluates the law at an amplitude where the law is already 10-16% off.

Predictions: A1 partially confirmed (one sigma per protocol, but the implied
B-norms drift 2-5% between protocols because the protocols sit at different
amplitudes); A2 CONFIRMED (bond/angle 1.387-1.458 across rotation-invariant
protocols, a 5% spread against a 1.67x spread in the absolute number);
A3 CONFIRMED (sigma ratio 1.71, predicted 1.3-2.0); A4 CONFIRMED;
A5 CONFIRMED (6.95-11.63, predicted 6-14).

## The defensible sentence

> Isotropic Cartesian error of per-atom standard deviation sigma over-disperses
> internal coordinate k by sigma||B_k||/tau_k relative to its physical thermal
> width. With the measured ||B_k|| and corpus widths this is **43 thermal widths
> of bond-length error, 31 of bond angle and 8.0 of torsion per Angstrom of
> sigma**; the relation is exact to 1% for sigma below 0.1 A, which covers every
> trained model we measured.

State the matched-dispersion figure, if it is wanted at all, as a secondary
sentence with the protocol named and the range shown:

> At the coordinate dispersion of a real 1 ns thermal ensemble, isotropic noise
> over-disperses bond lengths by 7.0x (matched on rotation-invariant pairwise
> distances) to 11.6x (matched on superposed per-atom RMSF), angles by 5.0x to
> 8.0x and torsions by 1.4x to 2.5x. The ordering and the ratio between channels
> are protocol-invariant; the absolute factor is not.

**Withdrawn: any sentence of the form "isotropic noise costs Nx", with N a
single number.**

---

# b. Aromatic planarity

## What was measured

`b_aromatic_oop.py`, 400 MISATO systems, 2,666 candidate centres, test ids
excluded. Aromaticity is transferred from each ligand's deposited SMILES by
element-labelled graph isomorphism of the heavy-atom graph (354 systems matched,
39 failed, 7 had no SMILES) -- a chemical criterion with no geometric input, so
it cannot select planar centres by being planar.

With the published mean-|chi| < 10 deg pre-filter, which is part of the
published estimator:

| centre set | n centres | tau (SD, deg) | mean |chi| (deg) |
|---|---|---|---|
| published: ring atom, 3 heavy neighbours | 2047 | **6.466** | 5.358 |
| + sp2 (no hydrogen on the centre) | 2047 | **6.466** | 5.358 |
| + RDKit aromatic | 1768 | **6.392** | 5.300 |
| sp2 but not aromatic | 107 | 6.761 | 5.719 |

Without the pre-filter, the hydrogen-bearing centres are visible and are not
pucker: 489 centres, median mean-|chi| **52.5 deg**, i.e. tetrahedral. The
pre-filter removes all of them. That is why the sp2 restriction changes nothing.

Per-centre, aromatic only: SD 6.33 deg, mean |chi| **5.22 deg**, |mean chi|
0.99 deg (the ensemble mean is planar). Carbon 5.18 deg, nitrogen 6.74 deg.
`b2_oop_confounds.py`: 0 of 400 systems show a periodic-image break (max bonded
heavy-heavy distance > 2.5 A), and the first-half and second-half widths are
6.28 / 6.28 deg, so neither the store's known image-jump problem nor trajectory
drift is carrying the number.

Predictions: **B1 REFUTED** (predicted the aromatic-only width would fall to
3.5-5.5 deg; measured 6.39 deg, i.e. no change at all). B2 confirmed trivially
(sp2 and aromatic agree to 1.1%). B3's band MISSED HIGH -- I predicted an
aromatic E|chi| of 2.0-4.5 deg and it is 5.22 deg -- but B3's consequence is
confirmed and is stronger than predicted: DiffSBDD's raw output sits at
5.09/5.22 = 0.98 of the thermal mean absolute deviation at n=300 and
5.82/5.22 = 1.11 at n=1000, not 4.3x. B4 CONFIRMED (the pre-filter changes the
aromatic tau by 0.5%).

## The statistic mismatch, which is the real problem with 4.3x

DiffSBDD's 5.09 deg and the crystal reference's 1.18 deg are mean absolute
deviations from planarity of a single conformer. 6.48 deg is a frame-to-frame
standard deviation. They are not the same statistic. The like-for-like thermal
quantity is E|chi| per frame, and for real aromatic centres in 300 K MD it is
**5.22 deg**.

| object | aromatic mean |chi| | source |
|---|---|---|
| real CrossDocked ligands (crystal/docked) | 1.18 deg | published_models_v1 |
| MMFF minimum of DiffSBDD molecules | 1.14 deg (n=300), 1.83 deg (n=1000) | published_models_v1, this run |
| **real ligands, 300 K MD ensemble** | **5.22 deg** | this work |
| DiffSBDD raw output | 5.09 deg (n=300), **5.82 deg** (n=1000) | published_models_v1, this run |

## The defensible sentence

> DiffSBDD's aromatic centres are 4.3x further from planar than the
> crystallographic and docked ligands it was trained on (5.09 deg against
> 1.18 deg mean absolute out-of-plane deviation, both single conformers through
> the same function; 5.82 deg on 1,000 molecules). This is a
> distribution-matching failure against the model's own training data, not a
> statement that the structures are physically impossible: a 300 K MD ensemble
> of real aromatic centres has a mean absolute out-of-plane deviation of
> 5.22 deg, so the raw output sits within thermal reach of planarity even though
> it is far outside the distribution it was fitted to.

**Withdrawn: "aromatic planarity is broken by a factor of 4.3" used as evidence
of physical invalidity. Withdrawn: the caveat that 6.48 deg is inflated by
saturated-ring pucker.** Also note, for whoever owns section 4: a corrector that
drives this channel to 0.35-0.46 deg is pushing it BELOW the thermal width, not
towards it.

---

# c. The seam: the mechanism and the DiffSBDD refutation

## The positive control that was missing

The mechanism says an isotropic Cartesian objective applies no differential
pressure across internal coordinates. The strong form of that -- "so a trained
model's residual is isotropic" -- was refuted on DiffSBDD. It had never been
tested on a trained model where it should hold.

`c_isotropy_seam.py`. For each example, solve sigma from the BOND channel under
generated^2 = reference^2 + (sigma ||B_k||)^2 and predict the ANGLE channel.
||B_k|| is taken from the pure-noise sweep at the same amplitude and through the
identical estimator, so no theory constant enters. 1.000 means isotropic.

| arm | n | implied sigma (A) | angle measured / isotropic prediction |
|---|---|---|---|
| MEHF, projection OFF, f1 | 96 | 0.175 | **1.068** [1.021, 1.135] |
| MEHF, projection OFF, f2 | 96 | 0.151 | **1.054** [1.006, 1.111] |
| MEHF, projection OFF, f3 | 96 | 0.173 | **1.088** [1.038, 1.136] |
| MEHF, projection ON (8), f1 | 95 | 0.023 | 4.039 |
| MEHF, projection ON (8), f2 | 80 | 0.018 | 3.931 |
| MEHF, projection ON (8), f3 | 95 | 0.022 | 4.087 |
| DiffSBDD (arm A, n=1000, baseline-corrected) | 1000 | 0.081 | **1.739** |

The proj-0 arm reproduces the audit's numbers exactly (bond 7.397 / angle 5.781
/ dihedral 1.779 against the audit's 7.40 / 5.78 / 1.78). The 1.07 is not an
artifact of the quadrature assumption: under the opposite extreme, that the
generated ensemble carries no signal at all, the same statistic gives
1.079 / 1.066 / 1.095.

Predictions: **C1 CONFIRMED** (predicted < 20%, measured 5-9%);
**C2 CONFIRMED** (predicted > 2x, measured 3.93-4.09x over three folds);
**C3 CONFIRMED** (the DiffSBDD departure is in the harmful direction and holds
at n = 1000: relative error -41.0% at n=1000 against -42.5% at n=300).

## The statement that holds both

> An isotropic Cartesian objective does not make a model's residual isotropic.
> It leaves the residual's shape unmanaged. Isotropy is what that shape defaults
> to when nothing else acts on it -- measured on a trained flow model with its
> inference-time geometric correction disabled, the residual is isotropic to
> within 7% across the bond and angle channels (1.068 / 1.054 / 1.088 over three
> folds) -- and it is what the mechanism's arithmetic applies to. Departures are
> possible and are not benign. Where a model departs by design, through an
> explicit geometric correction, the departure is 4x and is the intended
> repair. Where DiffSBDD departs without one, it departs the wrong way: its
> bond lengths are better than isotropy predicts and its bond angles are 1.74x
> WORSE, so the isotropic estimate is a lower bound on the damage to the
> stiffest unprotected coordinate, not an upper bound.

The seam is therefore not a contradiction but an ordering: isotropy is the null,
the null is already expensive because ||B_k||/tau_k spans 43 : 31 : 8, and the
one published model measured is worse than the null in the channel that matters
most.

---

# d. "Adding a geometry loss does nothing"

## The evidence is much tighter than the -0.353 +- 0.601 number suggests

The raw angle ratio is a noisy quantity because the whole ensemble's amplitude
moves between folds. `angle_term_v1` defines its own figure of merit -- the
angle ratio divided by the internal-RMSF ratio, which is invariant to any
global amplitude change -- and the main effect is far cleaner on it:

| main effect on the angle channel | per-fold values | mean | 95% CI |
|---|---|---|---|
| angle loss, RAW ratio (B-A and D-C, n=6) | -1.232, +0.505, -0.775, -0.431, +0.139, -0.326 | **-0.354** | [-1.007, +0.300] |
| angle loss, DISPERSION-NORMALISED (n=6) | -0.379, +0.119, -0.014, -0.220, -0.044, -0.213 | **-0.125** | [-0.313, +0.062] |
| projection, DISPERSION-NORMALISED (n=6) | — | **-4.213** | [-4.505, -3.922] |

Base (cell A) = 4.806 raw, 7.422 normalised. So the loss moves the normalised
figure of merit by **-1.7%, CI -4.2% to +0.8%**, and the projection moves it by
**-56.8%**. The retraining-noise floor, from the two control retrains that
exist, is +0.254 and +0.108 normalised (+3.4% and +1.5%), i.e. the same size as
the effect.

**Prediction D1 REFUTED**: I predicted the design could not detect anything
below a 25% reduction. On the normalised metric the 80%-power MDE is 0.204,
which is 2.8% of base. The design is much more sensitive than I expected, and
the null is correspondingly much harder.

## But it still cannot carry a claim about the LITERATURE, for a reason that has
nothing to do with power

Two independent problems, and only the first is fixable.

**1. One weight, and the weight is small by construction.** 0.448 is the
theoretically derived Mahalanobis weight (tau_bond/tau_angle)^2.
`theory_modules_v1` 3.3 measures what that weight does to the objective's
residual-variance allocation: **4.8%**, against a factor of 79 for the full
reweighting. A near-null at that weight is the PREDICTED outcome, not a
surprise. The experiment bounds the effect of that weight; it says nothing
about a weight 10x or 100x larger, which is what a practitioner who wanted the
angles fixed would reach for.

**2. It is not what the literature does.** Checked directly in the three
reference codebases held locally:

| model | local-geometry loss term | status |
|---|---|---|
| DiffSBDD (`lightning_modules.py`) | none; an auxiliary Lennard-Jones (nonbonded) term only | no bond/angle/dihedral loss exists |
| TargetDiff (`models/molopt_score_model.py` lines 210-219) | `bond_loss`, `loss_bond_weight`, `loss_non_bond_weight` | **commented out** |
| MolDiff (`models/model.py` lines 23, 166, 191) | `bond_len_loss`, an MSE on predicted vs true bond length | implemented; `getattr(config, 'bond_len_loss', False)` and the string appears in no config under `configs/` -- **off in the released model** |

So "everyone adds a geometry loss" is false. The accurate description is the
opposite and it is more interesting: the term is the obvious remedy, two of the
three reference implementations wrote it, and none of them shipped it on. A
disabled flag is not a measured null and must not be presented as one.

## The second measurement, which is sharper than the angle one

MEHF's checkpoints carry `bond_loss_weight = 1.0` and no angle term
(`model_cfg` = `{'mean_loss_weight': 0.0, 'bond_loss_weight': 1.0,
'clash_loss_weight': 10.0, 'projection_iterations': 8}`). With the inference
projection disabled, the isotropy test in section c says the bond channel is
only **1.07x better** than the angle channel implies under isotropy -- and the
angle channel has no loss term at all. A loss term at weight 1.0 on a
coordinate bought at most 7%. The inference-time projection on the SAME
coordinate took it from 7.40 to 1.41, a factor of 5.2.

D3 CONFIRMED (predicted within 20%, measured 5-9%).

## The defensible sentence

> Two coordinates, two weights, one architecture: a bond-length penalty at
> weight 1.0 leaves its coordinate within 7% of what an unpenalised coordinate
> shows, and an angle penalty at its theoretically derived weight moves the
> dispersion-normalised angle ratio by -1.7% (95% CI -4.2% to +0.8%) against a
> retraining-noise floor of +1.5% to +3.4%. On the same model and the same
> coordinate, an explicit geometric correction applied to the samples moves the
> same quantity by -57%. We cannot generalise this to every geometry loss at
> every weight, and we do not: what the reference implementations show is that
> the term is written and then left off (TargetDiff's is commented out,
> MolDiff's is off by default, DiffSBDD has none), so the question the field has
> actually left open is not whether the remedy works but whether anyone has
> shipped it.

**What would make it general, and why it is not done here:** retrain a published
architecture -- MolDiff, which already has the switch -- with
`bond_len_loss: True` and without, and measure the normalised bond ratio. That
is a controlled, single-flag experiment on somebody else's model. MolDiff's
released training config is 110,000 iterations at batch 256; that is days of GPU
and is out of scope for this pass. **Until it is run, the claim must be scoped
to "in the one architecture where we measured it" and must not be phrased as a
claim about the literature.**

---

# e. "Force-field relaxation is destructive"

`e_ff_restraint_sweep.py`, the same 200 DiffSBDD molecules (2 per pocket, 100
pockets) as the corrector gate, held-out half of the 100 real CrossDocked test
ligands as the distributional reference, MMFF94 with a positional restraint
swept over seven values. `rmsd_internal` is the RMSD from the raw output after
optimal superposition, so it isolates the change in the molecule's own shape
from rigid drift.

| arm | W(bond) | W(angle) | planarity (deg) | RMSD (A) | RMSD after alignment | clashes, median / mean / % of molecules |
|---|---|---|---|---|---|---|
| raw model output | 0.0486 | 5.334 | 4.140 | 0 | 0 | 0 / 0.64 / 17% |
| MMFF, unrestrained | 0.0262 | 1.810 | **0.566** | **1.675** | 1.302 | 2 / 5.12 / **68%** |
| MMFF, restraint 0.2 | 0.0262 | 1.789 | 0.941 | 0.923 | 0.923 | 0 / 1.31 / 38% |
| MMFF, restraint 1 | 0.0262 | 1.758 | 1.266 | 0.630 | 0.630 | 0 / 0.92 / 26% |
| **MMFF, restraint 5** | 0.0260 | **1.664** | 2.138 | 0.399 | 0.399 | 0 / **0.64** / 20% |
| MMFF, restraint 20 | 0.0251 | 1.746 | 2.896 | 0.267 | 0.267 | 0 / 0.65 / 17% |
| MMFF, restraint 100 | 0.0234 | 2.426 | 3.172 | 0.153 | 0.153 | 0 / 0.61 / 17% |
| MMFF, restraint 1000 | 0.0203 | 4.260 | 3.655 | 0.055 | 0.055 | 0 / 0.64 / 17% |

The unrestrained arm reproduces the published numbers exactly (RMSD 1.675 A,
planarity 0.566, median 2 clashes, raw planarity 4.140, restraint 100 at 0.153 A
and 3.172 deg).

Predictions: **E1 CONFIRMED** (the general claim has to be withdrawn);
**E2 REFUTED, in the interesting direction** -- I predicted the restrained arm
would close less than 40% of the planarity gap and would lose the angle channel
to free MMFF. At restraint 5 it closes 48% of the planarity gap and it WINS the
angle channel (1.664 against 1.810). **E3 PARTLY CONFIRMED**: 1.302 A of the
unrestrained arm's 1.675 A survives optimal superposition, so 78% of the
displacement is a change of the molecule's own shape that no receptor term and
no in-pocket minimisation can prevent; but the clash penalty is not intrinsic --
it is already gone at restraint 5 without any receptor term at all.

## The defensible sentence

> Relaxing a generated ligand with a force field is a trade-off, not a
> destruction. Unrestrained MMFF repairs the geometry and moves the ligand
> 1.675 A -- 78% of it a change of the molecule's own shape -- and raises the
> fraction of molecules with a sub-2 A receptor contact from 17% to 68%. A
> positional restraint removes that cost smoothly: at 5 kcal/mol/A^2 the ligand
> moves 0.399 A, the clash rate returns to the raw model's own level, and the
> bond and angle distributions are as close to real ligands as the unrestrained
> arm gets. What the restraint cannot buy is aromatic planarity: it is 0.57 deg
> unrestrained, 2.14 deg at restraint 5 and 3.17 deg at restraint 100 against
> the raw output's 4.14, so the channel is repaired only in proportion to how
> far the pose is allowed to move.

**Withdrawn: "force-field relaxation is destructive", and any use of the
1.675 A / 2 clashes figures without the word "unrestrained" attached.** The
honest version of the section-2 claim is the last sentence above: the family is
a one-parameter trade-off between repairing the out-of-plane channel and moving
the pose, and no setting escapes it.

---

# Scope and what is not established

* MISATO, protein-bound ligands, 1 ns, 100 frames, one force field. The
  aromatic out-of-plane thermal width of 5.22 deg is a property of that MD, and
  a classical force field's improper stiffness is not independently verified
  here. It is the same MD that every tau in this project comes from, so it is
  internally consistent, but a quantum or experimental check has not been done
  and the aromatic conclusion in b rests on it.
* One published model (DiffSBDD) for every "published model" statement.
  TargetDiff and MolDiff checkpoints are on disk and were not sampled.
* The geometry-loss measurements are on one architecture (MEHF), two
  coordinates, two weights, and on fitting folds at an exploratory operating
  point (scales 0.32 / 1.50) that is not a validated one.
* The restraint sweep uses 200 of 10,000 available molecules and half of 100
  reference ligands.
