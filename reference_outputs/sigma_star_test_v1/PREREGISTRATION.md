# PRE-REGISTRATION — Sigma* confirmation on `test`
Written 2026-09-25, BEFORE the matched-torsion fitting-fold points exist for any
arm, and before `test` is touched for any purpose. Not editable after the first
`test` number is produced; anything learned later goes in a separate RESULT
section below.

## Why a new pre-registration and not the old rule

The previous rule picked the smallest scale whose fitting-fold internal RMSF
reached 0.60. It was written when RMSF was the binding constraint. Under the
Sigma* prior RMSF is no longer binding — it reaches 1.17 at scale 2.5 and 1.59
at 3.5 — so that rule now selects scale 1.5, where every channel sits at 0.44 to
0.73 and the model is uniformly too narrow. Reusing it would pick a bad operating
point; re-deriving it from the numbers just measured would be tuning on the
outcome. It is therefore replaced here, before the data that could tune it.

## The primary claim needs no operating point at all

**PRIMARY: the channel spread**, defined as max minus min over the four channel
ratios {bond length, bond angle, torsion, out-of-plane}, each generated width
over physical width. It is reported at EVERY scale on a fixed grid, so no scale
is selected and none can be. On fitting folds it is 0.230 / 0.277 / 0.300 at
scales 1.5 / 2.5 / 3.5, against ANISO v3's 0.415 at 2.5 and ISO's 0.537 at 2.0.

Comparison arms, all with the identical inference angle projection, identical
seeds, differing ONLY in the prior: ISO (`internal_calibration_v1`), ANISO v3
(`aniso_prior_v3`), Sigma* (`sigma_star_v1`), folds 1-3 each.

**Matched-torsion comparison.** Because the arms sit at different amplitudes, the
per-arm number carried into the primary test is its spread at the scale where
that arm's fitting-fold torsion ratio is closest to 1.0, found on the grid
{1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0} by linear interpolation of the torsion ratio.
Torsion is used because it is the dominant physical soft coordinate and because
it is not the reported quantity; the spread is.

## Decision rule on `test`

Evaluate the three arms once on `test`, each at the scale its FITTING folds
selected, on all usable test examples, three fold checkpoints per arm, the
per-example value being the mean over checkpoints.

- **CONFIRMED** iff the median per-example channel spread of Sigma* is below
  BOTH other arms', and the 95% cluster bootstrap over UniProt of the paired
  difference against ANISO v3 (the stronger comparator) excludes 0.
- Otherwise **NOT CONFIRMED**, reported as such, with the fitting-fold numbers
  relabelled exploratory.

Reported regardless of outcome and not part of the decision: every channel ratio
per arm; internal RMSF per arm; the paired difference against ISO; per-fold
values; and the realised torsion ratio on test, which shows whether the matching
transferred.

## Single-read rules

`test` is read ONCE, and it is the last unread split in this project; `val` was
spent on outputs/pareto_v1. A run may be restarted only if it did not complete,
with an identical command, logged. No configuration changes after the first
`test` number is seen. If the matched-torsion scales turn out to be outside the
grid for any arm, that arm is reported as unmatchable rather than extrapolated.

## Stated in advance, so it cannot be claimed afterwards

Sigma* is expected to be OVER-dispersed in internal RMSF at its matched-torsion
scale (fitting folds suggest roughly 1.5). The claim being tested is uniformity
across channels, NOT that every channel sits at 1.0. A uniform but uniformly
over-dispersed model would satisfy the primary criterion, and the per-channel
table is reported so that is visible rather than hidden.

---

# AMENDMENT 1 — the comparator numbers were not like-for-like, 2026-09-25
Written before `test` is read for any purpose. `test` remains untouched.

## The error

The comparator spreads quoted above, ANISO v3 0.415 and ISO 0.537, are
max-minus-min over THREE channels (bond, angle, torsion). Sigma*'s quoted
0.230 / 0.277 / 0.300 are over FOUR, including out-of-plane. Verified:
1.460 - 1.045 = 0.415 exactly and 2.177 - 1.640 = 0.537 exactly, while Sigma*'s
0.230 at scale 1.5 is 0.670 (oop) minus 0.440 (angle).

The out-of-plane values for the comparators already existed in the same JSON
files; they were simply left out of the quoted spread. Recomputed on the same
four channels:

| arm, headline scale, fold 1 | bond | angle | torsion | oop | spread (4 channel) |
|---|---|---|---|---|---|
| ISO, scale 2.0 | 1.74 | 2.35 | 1.75 | **4.23** | **2.492** |
| ANISO v3, scale 2.5 | 0.98 | 1.44 | 1.34 | **3.88** | **2.896** |
| Sigma*, scale 2.5 | 0.83 | 0.75 | 0.82 | **0.99** | **0.243** |

The cause is mechanical and was read from the source, not inferred:
`_structure_local_noise` builds Gauss-Seidel targets for bonds and angles only,
and `_project_geometry` corrects bonds, angles and pocket clashes only. Neither
touches out-of-plane, so under the ANISO and ISO priors that channel keeps the
raw isotropic residual — ANISO v3 generates 24.9 degrees of pyramidalisation
against a physical 6.48.

## The correction, and why it is recorded rather than quietly applied

**The corrected comparator spreads are ISO 2.49 and ANISO v3 2.90, not 0.537 and
0.415.** The decision rule is unchanged: Sigma*'s median per-example spread must
be below both other arms' and the paired difference against ANISO v3 must have a
95% cluster-bootstrap CI excluding 0. All three arms are computed over the same
four channels, which is what the rule always intended and what its stated
baselines failed to do.

**This correction makes the claimed margin about ten times larger, not smaller.**
That is precisely why it is written down. An error that flatters the result is
more dangerous than one that hurts it, because nothing downstream will flag it.

## Consequence for what may be claimed

The out-of-plane channel is where almost all of the comparators' spread lives.
The honest reading is not "Sigma* is uniformly better across four channels by a
small margin" but "the previous priors leave one channel catastrophically
unstructured, and Sigma* is the first that structures it, because it is the
first whose internal coordinate set is complete." The paper must say which
channel carries the margin rather than quoting the aggregate alone.

---

# AMENDMENT 2 — ANISO v3 is unmatchable; the paired test moves to ISO
Written 2026-09-25, before `test` is read. `test` remains untouched.

## What the fitting folds returned

Matched-torsion scan on the pre-registered grid {1.0 … 4.0}, three folds, four
channels, projection on, 96 balanced fitting examples
(`outputs/sigma_star_test_v1/matched_torsion_fit.json`):

| arm | fold | matched | s* | spread | internal RMSF |
|---|---|---|---|---|---|
| ISO | 1/2/3 | yes | 1.109 / 1.309 / 1.107 | 1.762 / 1.706 / 1.938 | 0.415 / 0.386 / 0.388 |
| **ANISO v3** | 1/2/3 | **NO** | — | 2.507 / 2.372 / 2.351 at nearest grid | — |
| Sigma* | 1/2/3 | yes | 3.245 / 3.329 / 3.787 | **0.216 / 0.270 / 0.460** | 1.635 / 1.484 / 1.585 |

**ANISO v3's torsion ratio never reaches 1.0 anywhere on the grid** — its minimum
is 1.098 / 1.241 / 1.119 at scale 1.0, the bottom of the grid. It is therefore
UNMATCHABLE under this file's own rule, which forbids extrapolation.

## The consequence, and the change

The decision rule named ANISO v3 as the comparator for the paired
cluster-bootstrap test, because it was the stronger of the two. It cannot serve:
there is no scale at which its torsion ratio equals Sigma*'s, so no matched
comparison against it exists.

**The paired bootstrap test therefore runs against ISO**, the only comparator
that matches. Everything else in the rule is unchanged: Sigma*'s median
per-example four-channel spread must be below both other arms', and the 95%
UniProt cluster bootstrap of the paired difference **against ISO** must exclude
zero. ANISO v3 is still evaluated on `test` and still has to lose on the median,
but at its nearest grid scale, and that is reported as an unmatched comparison.

This is a weaker test than the one originally written, because ISO is the weaker
comparator. It is recorded as a weakening rather than presented as equivalent.

## Also recorded before the read

Sigma*'s matched-torsion scale is 3.2 to 3.8, where its internal RMSF is 1.48 to
1.64 — over-dispersed by about 50%. That was flagged in the original file as
expected and is now quantified. The claim remains uniformity across channels,
not that every channel sits at 1.0, and the per-channel table will show the
over-dispersion rather than hide it.

Fold 3 is Sigma*'s worst fold (spread 0.460 against 0.216 and 0.270), so the
fitting-fold spread is not as tight across folds as the single-fold numbers
suggested.
