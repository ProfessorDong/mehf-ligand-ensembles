# TEST RESULT — read once, 2026-09-25, 16:44
Decision rule applied verbatim from PREREGISTRATION.md (+ Amendments 1, 2) and
TEST_RUN_PLAN.md (+ Addendum A), all hashed before any test number existed.
Produced by `scripts/sigma_star_test_decision.py`, itself hashed with zero
`test_*.json` on disk. Raw: `TEST_RESULT.json`.

## VERDICT: CONFIRMED

Scored 1,410 of 1,456 test examples over 60 of 66 UniProt targets. The 46
dropped examples have no planar ring centre, so a four-channel spread is
undefined for them (Addendum A). Six targets contributed no scorable example.

## Primary: median per-example four-channel spread (lower is better)

| arm | median | IQR | fold 1 / 2 / 3 |
|---|---|---|---|
| ANISO v3 (unmatched, nearest grid) | 2.527 | 2.241 – 2.876 | 2.601 / 2.474 / 2.515 |
| ISO | 1.974 | — | 2.008 / 1.851 / 2.032 |
| **Sigma\*** | **0.770** | 0.471 – 1.330 | 0.740 / 0.758 / 0.804 |

Paired cluster bootstrap over UniProt, 10,000 resamples, seed 20260925:

| contrast | mean diff | 95% CI | excludes 0 | examples favouring Sigma* |
|---|---|---|---|---|
| Sigma* − ISO (the pre-registered test) | **−1.152** | **[−1.370, −1.021]** | yes | **98.8%** |
| Sigma* − ANISO v3 (unmatched, reported only) | −1.623 | [−1.916, −1.486] | yes | 99.6% |

All three decision conditions hold: median below both comparators, CI against
ISO excludes zero, direction favourable.

## Reported regardless of outcome

Per-channel median ratio (generated width / physical width) and internal RMSF:

| arm | bond | angle | torsion | out-of-plane | internal RMSF |
|---|---|---|---|---|---|
| ISO | 1.017 | 1.241 | 1.071 | **2.872** | 0.379 |
| ANISO v3 | 1.040 | 1.329 | 1.207 | **3.477** | 0.427 |
| Sigma\* | **1.657** | 1.398 | 1.155 | 1.894 | **1.488** |

Realised torsion ratio, i.e. whether the matching transferred:
ISO 1.052 / 0.997 / 1.109 — transferred.
Sigma* 1.096 / 1.108 / 1.192 — transferred, slightly over-matched on fold 3.
ANISO v3 1.168 / 1.187 / 1.195 — still above 1.0, as its unmatchability implied.

## What degraded, stated because it is against the method

Sigma* is the ONLY arm whose spread got materially worse off the fitting folds.

| arm | fitting folds | test | ratio |
|---|---|---|---|
| ISO | 1.762 / 1.706 / 1.938 | 1.974 | ~1.05 |
| ANISO v3 | 2.507 / 2.372 / 2.351 | 2.527 | ~1.05 |
| **Sigma\*** | **0.216 / 0.270 / 0.460** | **0.770** | **1.7 – 3.4** |

This is not a statistic artefact: the test spread-of-medians is 0.739 against
the median-of-spreads 0.770, so the two statistics agree, and the fitting-fold
values are spread-of-medians computed the same way.

Where it went: on the fitting folds Sigma*'s bond ratio was 1.18 / 1.26 / 1.46
and out-of-plane 1.22 / 1.27 / 1.42. On test they are 1.657 and 1.894. Both
drifted up together, which is what an operating point chosen high on the
amplitude curve (s* = 3.25 to 3.79) does under a distribution shift. **The
fitting-fold figure 0.216 must not be quoted as the method's number. 0.770 is.**

The claim that survives is the relative one, which is what was pre-registered:
at matched torsional width, Sigma* is 2.6x more uniform across the four channels
than isotropic and 3.3x more uniform than the previous anisotropic prior, on
unseen targets, and it is the only arm that brings out-of-plane below 2.

## Absolute standing, not claimed and not hidden

Sigma* is over-dispersed: internal RMSF 1.49 and bond 1.66 against a physical
1.0. It is uniformly too wide rather than mis-apportioned, which is the property
the pre-registration said was being tested and said would look like this.
No channel of any arm sits at 1.0 on all four.
