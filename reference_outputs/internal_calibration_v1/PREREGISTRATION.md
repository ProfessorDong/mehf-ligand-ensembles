# PRE-REGISTRATION — internal-subspace calibration as the claimed advance
Written 2026-09-22, BEFORE any torsion Jensen-Shannon distance is computed for
this claim, and before `val` is read for any purpose.

## 1. The mechanism that makes this claim separable from the Stage 1 failure

`MEHF._structured_noise` (src/mdwm/models/mehf.py:938) returns

    (translation + rotation) * rigid_scale + local * internal_scale

- `translation` is one vector shared by all atoms of a ligand, so it is an
  exact global translation. Torsion angles are exactly invariant to it.
- `rotation` is `omega x (x - centre)`, the generator of a rigid rotation about
  the ligand centroid. Torsions are invariant to it to first order in |omega|;
  at the scales used here (rotation_noise_radian 0.08 x rigid_scale 0.35, i.e.
  |omega| ~ 0.028 rad) the residual is O(|omega|^2) ~ 1e-3 rad and is far below
  the 10-degree torsion bin width.
- `local` is independent per-atom noise and is the ONLY term that moves torsions
  materially.

Therefore the torsion JS metric is controlled by `internal_scale` and is blind
to `rigid_scale`. Stage 1 measured that the internal scale TRANSFERS across
target-disjoint folds (between-fold spread of the internal RMSF ratio 0.050 to
0.080, against a window width of 0.30) while the rigid scale does NOT (spread
0.169 to 0.260). The subspace that fails to transfer is the subspace the metric
cannot see. This is why the Stage 1 negative result does not block this claim,
and it is also why this claim must NOT be reported as calibrating the prior as
a whole.

## 2. Hypothesis

Phase C evaluated MEHF with the ORIGINAL prior, whose measured internal RMSF
ratio at full scale is 0.363 (gate table, variant V0): the generated torsional
ensembles were roughly 2.75x too narrow. The hypothesis is that Phase C's
torsion-JS failure is caused by that under-dispersion, and that a decomposed
prior whose internal scale is moment-matched to ratio ~1.03 improves torsion JS
relative to B1 (copy the parent ensemble).

The single-temperature remedy cannot reach this state: one multiplier scales
rigid and internal together, so driving the internal ratio from 0.363 to 1.0
requires t ~ 2.75, which drives the centroid ratio from 1.767 to far outside any
window. That statement is currently an extrapolation and is being MEASURED
directly (section 6) before it is reported.

## 3. Frozen configuration — fixed here, before any JS is computed

- rigid_scale = 0.35, internal_scale = 3.5.
- Selected by AMENDMENT 2's worst-case rule applied to MOMENTS ONLY on the
  fitting folds (refined grids, 3 folds x 25 cells). No JS entered the choice.
  Guard 3 holds.
- Checkpoints: outputs/subspace_prior_v1/v1_f{1,2,3}/checkpoint.pt, unchanged.
- 50 samples per side, sampling steps as in the frozen baseline protocol,
  torsion JS on fully-mapped product torsions, identical to the protocol that
  produced the frozen baseline table (floor 0.285, B1 0.586, B4 0.555,
  B3 0.743, B2 0.800).

## 4. Primary comparison and decision rule

On `val` (696 pairs, 106 targets, never read):

- PRIMARY: the paired per-pair difference D = JS(MEHF) - JS(B1), median over
  pairs, with a 95% cluster bootstrap CI over UniProt.
- **ADVANCE DEMONSTRATED** iff median D < 0 and the 95% CI excludes 0.
- Otherwise **NOT DEMONSTRATED**, reported as such.
- Also reported unconditionally, whatever the outcome: the recovery fraction
  R = (B1 - MEHF) / (B1 - floor), the comparison against B4, and the realised
  internal RMSF ratio on `val`.

## 5. The single read of `val`

`val` is read ONCE. That single read yields BOTH pre-specified analyses:
(a) the Stage 2 moment confirmation of the parent pre-registration, and
(b) the primary JS comparison above. Both criteria are fixed before the read,
both are reported regardless of outcome, and neither is selected on the basis
of the other. `test` is not read.

## 6. Exploratory work declared in advance

- A `dev` run of the same pipeline will be made first, for plumbing and for a
  power estimate. It is EXPLORATORY and is labelled as such wherever reported.
  **The `val` run happens regardless of what the `dev` run shows.** This is
  committed here, in writing, before the `dev` run is executed, so that a poor
  `dev` result cannot be converted into a decision not to confirm.
- The single-temperature diagonal t * (rigid_0, internal_0) for the trained
  scales of each fold will be measured on the fitting folds, moments only, to
  replace the extrapolation in section 2 with a measurement.

## 7. What this claim is NOT

It is not a calibration of the prior. The rigid subspace does not transfer
across folds and the orientation dispersion ratio sits near 0.5 everywhere in
the explored plane. Any figure or sentence reporting this advance carries that
limitation with it.

## 8. AMENDMENT A — training scale, written before any JS is computed

The `v1_f*` checkpoints were trained on `max_pairs: 800` of the 3,489 available
train pairs. Evaluating an 800-pair model against the frozen baselines would
test a deliberately weakened version of the method: a negative result would not
be conclusive, and a positive one would understate it.

The claim is therefore evaluated on a FULL-SCALE retrain (all train pairs of the
primary target-disjoint split), decomposed prior, same architecture.

Stage 1 established that retraining moves the sampling-time optimum (the
training objective's own scales, 0.46-0.67 rigid and 1.88-2.49 internal, are not
where the moments want them, 0.35 and 3.0-3.5). The numeric pair (0.35, 3.5) in
section 3 is therefore provisional to the 800-pair checkpoints. What is
pre-registered is the PROCEDURE, not the number:

  After full-scale retraining, both scales are re-fixed by moment matching on
  the FITTING folds only, using AMENDMENT 2's worst-case-across-folds rule, on
  the same refined grid. Torsion JS enters no part of this. The resulting pair
  is frozen and written into this file BEFORE `val` is read.

Everything else in sections 4, 5 and 7 is unchanged. Section 6's commitment
stands: the `val` run happens regardless of what any `dev` run shows.

Recorded fact: `outputs/subspace_prior_v1/v1_f1/eval_js/` and `v1_f2/eval_js/`
exist but are EMPTY, verified 2026-09-22. No torsion JS has ever been computed
for any decomposed-prior checkpoint. Guard 3 is intact at the time of writing.

## 9. AMENDMENT B — the calibration sample was target-concentrated, 2026-09-22

Found while reading the first full-scale calibration output: `n_targets: 20`.

`scripts/run_subspace_plane.py` drew its calibration sample as
`fit_examples.head(n_examples)`. The example CSVs are written in per-target
blocks, so that slice is not a sample of the corpus, it is a sample of whichever
targets come first:

| calibration sample | distinct targets | largest single target |
|---|---|---|
| v1_f1 head(160)  (800-pair, Stage 1 basis) | 43 | 43% |
| v1_f2 head(160) | 44 | 38% |
| v1_f3 head(160) | 45 | 38% |
| full_f1 head(160) | 20 | **68%** |
| full_f2 head(160) | 20 | **68%** |
| full_f3 head(160) | 22 | **68%** |

This is the same failure mode as the degenerate fold read earlier in this
project (a dev set that was 91% one UniProt). Two consequences, both recorded
before any re-run:

1. **The full-scale calibration point measured so far is void.** The cell
   (rigid 0.70, internal 6.5) giving internal 1.041 / centroid 1.023 /
   dev 0.041 was computed on a sample that is 68% one protein. It is not
   carried forward and is not reported as a calibration result.

2. **Stage 1's headline claim is now in question.** The claim "the internal
   scale transfers across folds and the rigid scale does not" rested on a
   between-fold spread of the centroid ratio of 0.169-0.260. Each fold's
   calibration sample was dominated by one to three DIFFERENT targets
   (largest 38-43%, top three 56-59%), so part of that spread may be
   composition of the calibration sample rather than a property of the
   subspace. The claim is suspended until re-measured under balanced sampling,
   and it will be retracted in writing if the spread collapses.

**Fix.** `--example-sampling balanced` added to the script: targets are shuffled
with the run seed and walked round robin, one row each per pass. `head` is kept
as the default so every earlier run remains reproducible exactly as it was.
Measured effect on full_f1: 20 targets / 68% largest becomes 160 targets / 0.6%
largest at the same sample size.

All calibration from this point uses `--example-sampling balanced`, and any
number reported from a `head` sample is labelled as such.

## 10. FROZEN CONFIGURATION, 2026-09-22 — before `val` or `dev` JS is read

Balanced calibration (`--example-sampling balanced --n-examples 160`, ~152
targets per fold, largest target 0.6% of rows) on the FITTING folds of the three
full-scale checkpoints. Moments only. No torsion JS was computed anywhere in
this selection; guard 3 holds.

Selected by AMENDMENT 2's worst-case-across-folds rule:

**rigid_scale = 0.32, internal_scale = 2.50**

| fold | internal RMSF ratio | centroid ratio | orientation ratio | d_f |
|---|---|---|---|---|
| full_f1 | 1.107 | 1.040 | 0.685 | 0.107 |
| full_f2 | 0.928 | 0.903 | 0.570 | 0.097 |
| full_f3 | 1.060 | 1.090 | 0.694 | 0.090 |

Worst case across folds 0.107, inside the window on all three. This is the
first configuration in this project that is in-window on every target-disjoint
fitting fold. It is frozen here and is what goes to `dev` and then to `val`.

Orientation is NOT in the window (0.570 to 0.694) and section 7 stands: this is
calibration of two of the three subspaces.

## 11. RETRACTION — Stage 1's headline claim does not survive balanced sampling

Stage 1 reported, and it was relayed to the user, that "the internal subspace
calibrates with a transferable global scalar; the rigid subspace does not",
supported by between-fold spreads of 0.050-0.080 (internal) against 0.169-0.260
(centroid). That contrast was measured on `head` samples in which one target
supplied 38-43% of the rows, and it does not survive.

Under balanced sampling at the frozen cell the two spreads are comparable:

| quantity | values across folds | spread |
|---|---|---|
| internal RMSF ratio | 1.107, 0.928, 1.060 | 0.179 |
| centroid ratio | 1.040, 0.903, 1.090 | 0.187 |

Both subspaces show the same fold-to-fold variation, about 0.18, and both fit
inside the 0.30 window. **The claim that the rigid subspace fails to transfer
is retracted.** So is the claim, made in the same period, that "the fixed point
training scale = sampling optimum does not exist": the full-scale models were
trained at rigid 0.35 and their balanced optimum is rigid 0.32.

What survives is weaker and is what the paper will say: a single pair of global
scalars, moment-matched on fitting folds, places both the internal and the
centroid dispersion inside the window on all three held-out-target folds, while
the orientation subspace remains under-dispersed at roughly 0.65.

Also retracted: "orientation is stuck near 0.5 everywhere, three folds". That
number came from the same biased samples. Balanced sampling gives 0.57 to 0.69
at the frozen cell and up to 0.81 elsewhere in the region.

## 12. AMENDMENT C — the calibration target was the wrong moment, 2026-09-23

### What happened, in order, with nothing omitted

1. Section 10 froze (rigid 0.32, internal 2.50) on the Cartesian internal RMSF
   ratio and the centroid ratio.
2. The `dev` run declared exploratory in section 6 was executed on full_f1
   (239 pairs, 30 targets). **MEHF 0.6451 against B1 0.5530, B4 0.5321,
   floor 0.2587. MEHF LOSES; R = -0.313.** This is recorded here and will be
   reported whatever happens next.
3. The same output showed the diagnosis: generated torsion spread 42.98 degrees
   against a reference of 22.38, a ratio of 1.92, while the Cartesian internal
   RMSF ratio at the same configuration was 0.92. MEHF was not under-dispersed
   in torsion space, it was nearly twice too WIDE.

### The fact that justifies the change, provable without any JS

Measured on the FITTING folds only, balanced sampling, moments only
(`outputs/internal_calibration_v1/spread_f{1,2,3}.json`), at rigid 0.32:

| internal | torsion spread ratio (f1) | Cartesian internal RMSF ratio (f1) |
|---|---|---|
| 1.00 | 0.656 | 0.456 |
| 1.50 | 1.003 | 0.674 |
| 2.00 | 1.323 | 0.881 |
| 2.50 | 1.602 | 1.087 |

**The two moments cannot be matched by one scalar; they are 1.67x apart.**
Matching the Cartesian internal RMSF forces the torsion spread to 1.60.

The cause is in the prior. `local` is isotropic, independent per-atom noise. A
dihedral is read off four atoms, so uncorrelated displacement of a given RMSF
produces far more angular variance than physical internal motion of the same
RMSF, in which bond lengths and angles are near rigid and displacement is
strongly correlated. Matching a Cartesian moment therefore cannot calibrate a
torsion-space metric, and the reported metric is a torsion-space metric.

### The change

The calibration target becomes the **torsion circular spread ratio**, with the
centroid ratio unchanged as the second criterion. Both are moments, both are fit
on FITTING folds only, and torsion JS still enters no choice. Guard 3 holds in
the sense that matters: nothing is selected on the reported metric.

**Honesty about provenance.** The trigger to look at the torsion spread was the
exploratory `dev` JS in step 2. The change is therefore post-hoc in origin, and
this file says so. What keeps it admissible is that the justifying fact -- the
1.67x gap between the two moments -- is measured entirely on fitting folds and
would have been visible without ever computing a JS. It is a correction of a
specification error, not a search for a configuration that scores well.

### New frozen configuration, worst case across folds

**rigid_scale = 0.32, internal_scale = 1.50**

| fold | torsion spread ratio | centroid ratio | Cartesian internal RMSF |
|---|---|---|---|
| full_f1 | 1.003 | 1.009 | 0.674 |
| full_f2 | 0.908 | 0.862 | 0.545 |
| full_f3 | 1.069 | 0.998 | 0.641 |

Worst deviation 0.092 (spread) and 0.138 (centroid), both inside [0.85, 1.15]
on all three folds. The Cartesian internal RMSF is now 0.55-0.67 and is
reported as a known, explained mismatch, not hidden.

Sections 4, 5 and 7 are unchanged. `val` is still unread and is still read once.
