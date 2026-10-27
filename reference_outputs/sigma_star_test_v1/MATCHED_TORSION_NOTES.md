# Matched-torsion fitting-fold scales: result, provenance, and two things the
# pre-registration's own numbers do not survive contact with

Produced 2026-09-25 by `scripts/matched_torsion_fit.py` (rule hashed in
`matched_torsion_fit.rule.sha256` BEFORE any grid point was read by it) from the
grid points listed in `matched_torsion_fit.json`. **`test` was not read, opened
or referenced at any point.** Every number here is on fitting folds only
(`full_f{1,2,3}/fit_examples.csv`), which is what
`mehf_internal_geometry_spectrum.py` reads by default.

## The headline the caller needs before deciding anything about `test`

**ANISO v3 is UNMATCHABLE on all three folds.** Its fitting-fold torsion ratio is
already ABOVE 1.0 at the bottom of the pre-registered grid -- 1.098, 1.241, 1.119
at scale 1.0 for folds 1, 2, 3 -- and increases monotonically with scale, so the
interpolant never crosses 1.0 anywhere on {1.0 ... 4.0}. Under the
pre-registration's own instruction ("If the matched-torsion scales turn out to be
outside the grid for any arm, that arm is reported as unmatchable rather than
extrapolated") ANISO v3 has no matched-torsion scale.

ANISO v3 is also the arm the decision rule names as **the stronger comparator**,
the one the 95% cluster bootstrap of the paired difference is computed against.
**As written, the primary decision rule therefore cannot be executed in full.**
This is reported first, before the numbers that did come out, because it is a
decision the caller has to make and not one this run may make for them.

## Matched-torsion scales and spreads

Spread = max minus min over the four channel ratios
{bond length, bond angle, torsion, out-of-plane}, each linearly interpolated at
s* from the two bracketing grid points.

| arm | fold | s* (torsion ratio = 1.0) | channel spread at s* | internal RMSF at s* |
|---|---|---|---|---|
| ISO | 1 | 1.109 | 1.762 | 0.415 |
| ISO | 2 | 1.309 | 1.706 | 0.386 |
| ISO | 3 | 1.107 | 1.938 | 0.388 |
| ANISO v3 | 1 | UNMATCHABLE | -- | -- |
| ANISO v3 | 2 | UNMATCHABLE | -- | -- |
| ANISO v3 | 3 | UNMATCHABLE | -- | -- |
| **Sigma\*** | 1 | 3.245 | **0.216** | 1.635 |
| **Sigma\*** | 2 | 3.329 | **0.270** | 1.484 |
| **Sigma\*** | 3 | 3.787 | **0.460** | 1.585 |

Each arm's torsion ratio is monotone increasing in scale on this grid and crosses
1.0 exactly once where it crosses at all, so the smallest-crossing tie-break
never had to be used.

**The pre-registration's stated-in-advance expectation held.** It said Sigma* was
expected to be over-dispersed in internal RMSF at its matched-torsion scale,
"roughly 1.5". Measured: 1.635 / 1.484 / 1.585. That was written down before
these scales existed and it is right.

At its matched-torsion scale ANISO v3 has no spread to report, but the shape of
its grid makes the answer insensitive to the question: its four-channel spread is
2.51, 2.37, 2.35 at scale 1.0 and 3.12, 3.11, 3.09 at scale 4.0, because its
out-of-plane ratio sits at 3.4-4.3 at every scale. Whatever scale it were matched
at, inside or outside the grid, its spread would be an order of magnitude above
Sigma*'s 0.22-0.46. This is stated as context, not as a substitute for the test
the pre-registration specifies.

## The reuse control, run first and passed

24 of the 63 grid points already existed (ANISO v3 `proj_f*_s{1.5,2.0,2.5,3.0,3.5}`
from 2026-09-23, Sigma* `spectrum_f*_s{1.5,2.5,3.5}` from 2026-09-25 01:34-01:54)
and were reused rather than recomputed, as instructed. Reusing a file produced by
an earlier session is itself an assumption -- that it was produced with the same
seed and the same inference switches -- so it was measured rather than assumed.
`outputs/aniso_prior_v3/proj_f1_s2.5.json` was recomputed from scratch with
today's command and compared to the stored file:

| check | result |
|---|---|
| example keys, in order | identical |
| every `*_reference` column | identical |
| bond / angle / torsion / oop / RMSF ratio medians | agree to 1e-6 |
| largest per-example torsion-ratio difference | 9.6e-7 |

So the stored points were produced with the same seed (20260918), the same
example selection and the same switches, and are interchangeable with freshly
computed ones. Command and output kept in the run log; the recomputed copy was
written outside the repository so it cannot be mistaken for a grid point.

**Not reused:** `outputs/aniso_prior_v3/spectrum_f*_s*.json`. Those were run with
`angle_projection = False` and this comparison requires it ON for every arm.

39 points were computed today: ISO all 21, ANISO v3 at 1.0 and 4.0 (6), Sigma* at
1.0, 2.0, 3.0 and 4.0 (12). Every point, reused or new, carries its configuration
in `matched_torsion_fit.json` under `config_check`, and all 63 read
rigid 0.32, angle projection on, restore off, angle target width 0.0,
8 projection iterations, relaxation 0.5, 96 examples, 16 steps.

## The second thing: the pre-registration's own background numbers do not
## reproduce under the definition the pre-registration gives

The pre-registration says the channel spread is "max minus min over the four
channel ratios {bond length, bond angle, torsion, out-of-plane}", and then quotes,
as fitting-fold background, ANISO v3 0.415 at scale 2.5 and ISO 0.537 at scale
2.0. Computed from the grid points under that four-channel definition, with the
angle projection on for every arm, the means over the three folds are:

| arm, scale | quoted in PREREGISTRATION | four-channel (its own definition) | three-channel, out-of-plane dropped |
|---|---|---|---|
| ANISO v3, 2.5 | 0.415 | **2.853** | 0.448 |
| ISO, 2.0 | 0.537 | **2.468** | 0.541 |
| Sigma\*, 1.5 | 0.230 | **0.231** | 0.073 |
| Sigma\*, 2.5 | 0.277 | **0.277** | 0.090 |
| Sigma\*, 3.5 | 0.300 | 0.315 | 0.300 |

The two Sigma* values at 1.5 and 2.5 match the four-channel figure to three
decimals. The ANISO v3 and ISO values match the THREE-channel figure, the one
with the out-of-plane channel dropped, to 8% and 1%. The out-of-plane channel is
exactly where ISO and ANISO v3 are worst (ratios 2.2-5.9 and 3.4-4.3) and where
Sigma* is not, so the comparison as quoted in the pre-registration compares a
four-channel Sigma* spread against three-channel comparator spreads.

**The direction of the error is against Sigma*, not for it.** Under the
pre-registration's own four-channel definition the gap is roughly 0.23-0.28
against 2.47-2.85, about tenfold, not the twofold the quoted numbers suggest. It
is reported anyway, because a background number that does not reproduce is a
defect whether or not it flatters the claim, and because the caller should know
before reading `test` that the prose and the definition in the same document do
not describe the same quantity.

Nothing above changes the pre-registered primary metric, the decision rule or the
grid. No configuration was altered. The only open question this run creates is
the one it must not answer by itself: what to do about a comparator arm that the
pre-registration's own matching rule cannot place.

## Exploratory, and NOT part of any decision

Because "unmatchable" for ANISO v3 means only "no crossing inside the grid", the
same measurement was also run at scales 0.25, 0.50 and 0.75 for the three ANISO
folds, written as `outputs/aniso_prior_v3/EXPLORATORY_subgrid_proj_f*_s*.json`.
These are outside the pre-registered grid, they are labelled exploratory in their
own filenames, they are not read by `matched_torsion_fit.py`, and they may not be
used to place ANISO v3 for the `test` comparison. They exist so the caller can see
whether ANISO v3 is unmatchable in kind or merely unmatchable on this grid.
