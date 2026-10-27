# RESULT of the pre-registered `val` confirmation — 2026-09-23

Pre-registration: outputs/pareto_v1/PREREGISTRATION.md (sha256 91b2190e...).
Analysis: scripts/analyse_pareto_val.py (sha256 15c249c2...). Both hashes were
re-verified unchanged after the val runs and before the analysis was executed.
All six val runs completed first time; no restarts. `test` not read.

## VERDICT: CONFIRMED

1,392 directional examples, 106 UniProt targets, every example paired.

| channel | ANISO median | ISO median | paired diff (ANISO-ISO) | 95% cluster CI | ANISO lower on |
|---|---|---|---|---|---|
| **bond angle (PRIMARY)** | **1.453** | **2.171** | **-0.724** | **[-0.763, -0.695]** | **100.0%** |
| bond length | 1.034 | 1.637 | -0.603 | [-0.628, -0.581] | 99.9% |
| dihedral | 1.520 | 1.750 | -0.167 | [-0.195, -0.134] | 91.7% |
| internal RMSF | 0.618 | 0.613 | +0.001 | [-0.003, +0.006] | 47.6% |
| planar improper | 3.845 | 4.081 | -0.233 | [-0.295, -0.174] | 75.5% (n=1327) |

Decision conditions: median D < 0 TRUE; upper CI < 0 TRUE; RMSF not traded
(0.618 >= 0.613 - 0.05) TRUE.

The dispersion is matched on val to three decimals (paired RMSF difference
+0.001, CI straddling 0), so the improvement in every internal-coordinate
channel is not bought with ensemble width.

## Fitting-fold exploration replicated on held-out targets

| arm | fitting folds (exploratory) | val (confirmatory) |
|---|---|---|
| ANISO bond angle | 1.460 | 1.453 |
| ISO bond angle | 2.177 | 2.171 |
| ANISO RMSF | 0.639 | 0.618 |
| ISO RMSF | 0.647 | 0.613 |

## Per checkpoint (fold consistency)

ANISO bond angle 1.427 / 1.467 / 1.467; ISO 2.307 / 1.873 / 2.321.
Per-checkpoint share of examples where ANISO is lower: 100.0% / 99.0% / 100.0%
(f2's ISO checkpoint is the strongest ISO fold; it wins on 1% of examples, by
at most +0.648). The pooled 100% arises from averaging three checkpoints.
Sanity: reference spreads identical across arms on every example (verified), so
the differences are entirely in the generated ensembles. Generated angle width,
degrees: ANISO 3.9-4.0, ISO 5.1-6.4, physical reference 2.72.

## What is NOT shown, and must travel with this result

- The absolute bar from earlier (angle in [0.8, 1.35] with RMSF >= 0.60) is
  STILL NOT MET: ANISO angle 1.453, i.e. generated angles remain ~1.45x the
  physical width. The confirmed claim is RELATIVE: same projection, same
  selection rule, matched dispersion, only the prior differs.
- Aromatic out-of-plane improves (-0.23) but stays ~3.8x physical, consistent
  with the derived first-order blindness of bond+angle terms to it.
- The task is ligand conformational-ensemble generation for GIVEN molecules
  (MISATO, MMP product ensembles), not de novo molecule generation. Nothing here
  yet tests generated drug-like molecules or standard SBDD metrics.
- Scales 0.32 rigid / chosen internal scales are settings selected on fitting
  folds; the val result confirms them only for this corpus.

---

# Independent audit, 2026-09-24 — claim survived; two qualifications are now binding

An audit agent attacked this result on leakage, recomputation, arm fairness,
RMSF matching, metric robustness and the 100% figure. It could not break it.
Two findings must travel with the claim and are added to the limitations above.

## Q1. About 5% of val examples are not novel CHEMISTRY, only novel pairings

The split is UniProt-disjoint by construction and that was verified empirically:
zero UniProt overlap and zero PDB-ID overlap between val and the fit/dev folds
of all six checkpoints, and no PDB maps to two UniProts anywhere in the corpus.

Going further than the pre-registration required, the audit hashed each ligand's
heavy-atom bond graph (2-round Weisfeiler-Lehman on atomic numbers and
`bond_index`, independent of PDB numbering). **21 of val's 662 PDB ligands are
chemically identical to a ligand bound to a DIFFERENT target in train**, e.g.
val 5I7Y matches train 5I88 and 6BQD. That is 70 of 1,392 examples, 5.0%.

This cannot bias the comparison, because both arms train on identical data. It
does mean `val` must not be described as held-out *chemistry*; it is held-out
*targets*. Sensitivity: dropping all 70 moves the primary median from -0.7242 to
-0.7176. The leaked examples on their own give -0.918, a LARGER effect, so they
are not manufacturing the result.

## Q2. "The angle gain is not bought with dispersion" is a POOLED statement

The pre-registered condition is defined pooled and is met (0.618 vs 0.613). It
does not hold uniformly:

- **fold 1 alone: ANISO 0.582 vs ISO 0.646, a deficit of 0.064, which exceeds
  the 0.05 tolerance.** Fold 2 goes the other way (+0.095), fold 3 is -0.016,
  and the three-fold mean happens to land at +0.001.
- 13 of 106 targets (12.3%) individually exceed the 0.05 tolerance.

The verdict stands because the rule was pre-specified at the pooled level, but
any paper sentence must say pooled, not per fold and not per target.

Also worth stating plainly rather than presenting as a happy coincidence: both
arms' scales were chosen by a rule that targets fitting-fold RMSF >= 0.60, so
the two arms landing at similar val RMSF is partly by construction. Angle never
entered scale selection, which is the part that matters.

## What the audit confirmed

- Primary median reproduces bit-identically (-0.7242064306560552) from
  independent code with a different bootstrap seed and resample count; the
  independent cluster CI [-0.7624, -0.6948] matches the reported one to three
  decimals. An iid bootstrap gives a NARROWER CI, so the reported cluster CI is
  the conservative one, not a bug.
- The two checkpoint families differ in exactly three config keys, all of them
  the anisotropic prior's own switches. Everything else is identical, including
  parameter count, epochs, steps, per-fold example and target counts, and all
  inference settings except the pre-registered `internal_scale`. The
  `angle_loss_weight` logged in aniso's history is 0.0 in both arms: a
  diagnostic, never weighted into the loss.
- Hashes match and file mtimes are strictly ordered val_examples -> prereg ->
  analysis script -> the six val runs -> result, consistent with no post-hoc
  edit. (The tree is not a git repo, so this is hash and mtime evidence only.)
- Metric robustness: excluding the 10% of examples with the smallest reference
  angle spread gives -0.7176; excluding 25% gives -0.7205. In raw degrees the
  difference is -1.975 deg with 100% of examples favouring ANISO (generated
  3.95 deg vs 5.93 deg, physical 2.72 deg). Reference spreads are bit-identical
  across arms on every example.
- The 100% is not an artifact: the per-example difference distribution runs from
  -2.842 to **-0.099**, so the weakest example is still a ~10% improvement and
  nothing sits near zero. 0 of 106 targets go the wrong way. Dropping the
  largest target (P00760, 10.5% of examples) gives -0.7233; weighting targets
  equally gives -0.7214.

## Corroboration of the mechanism, found by the audit

Correlation between the angle ratio and the RMSF ratio across examples:
**ISO r = 0.308** (angle width tracks overall dispersion, which is what
isotropic noise predicts) against **ANISO r = 0.069**. The anisotropic prior
measurably decouples the two, which is what it was designed to do.
