# Post-hoc absolute-calibration endpoints: result

Plan: `PLAN.md` (sha256 954d60fd…26ccf, written and hashed before computing).
Script: `posthoc_endpoints.py`; output: `RESULT.json`. Reads only the stored per-example test CSVs.
Check: the script reproduces the frozen median spreads exactly (1.9741 / 2.5273 / 1.3525 / 0.7697).
Status: POST-HOC re-analysis of test outputs that were read once under pre-registration.
The 95% intervals are conditional on the trained checkpoints. They come from the frozen
UniProt-cluster bootstrap: 10,000 resamples, seed 20260925, applied to the mean of
per-example paired differences.

The endpoints are three mean absolute log width errors:
- **E4:** over the four channels.
- **E5:** over the four channels and the internal RMSF ratio.
- **E3:** over bond length, bond angle and torsion, on all examples.

Per-example values are averaged over the three folds, and the summary statistic is the median over examples.

| Endpoint (n) | ISO | GS | LAP | Σ* | Σ*−ISO mean [CI] | Σ*−GS | Σ*−LAP |
|---|---|---|---|---|---|---|---|
| S (1,410) | 1.974 | 2.527 | 1.352 | 0.770 | −1.152 [−1.370, −1.021] | −1.623 [−1.916, −1.486] | −0.580 [−0.750, −0.404] |
| E4 (1,410) | 0.432 | 0.495 | 0.687 | 0.500 | +0.065 [−0.085, 0.132] | +0.009 [−0.160, 0.089] | −0.182 [−0.248, −0.154] |
| E5 (1,410) | 0.529 | 0.558 | 0.623 | 0.491 | −0.041 [−0.211, 0.055] | −0.064 [−0.238, 0.020] | −0.122 [−0.188, −0.075] |
| E3 (1,456; 66 targets) | 0.228 | 0.239 | 0.591 | 0.446 | +0.224 [0.099, 0.283] | +0.210 [0.073, 0.270] | −0.136 [−0.181, −0.107] |

The table gives median values. In the paired columns, a negative value favors Σ*.

**Fraction of examples on which Σ* has the lower value:**

| Endpoint | vs. ISO | vs. LAP |
|---|---|---|
| E4 | 41.8% | 89.8% |
| E5 | 61.8% | 76.5% |
| E3 | 14.2% | 82.7% |

**Per-fold medians.** The per-fold medians agree with the fold-averaged ones, which is consistent across folds.
- E4: Σ* 0.482 / 0.485 / 0.545; ISO 0.439 / 0.414 / 0.443.

**Equal-target weighting** (median of the 60 target medians):

| Endpoint | ISO | GS | LAP | Σ* |
|---|---|---|---|---|
| E4 | 0.402 | 0.480 | 0.578 | **0.292** |
| E5 | 0.538 | 0.567 | 0.526 | **0.315** |
| S | 1.902 | 2.443 | 1.205 | 0.515 |

The E4 ordering of Σ* and ISO therefore depends on the weighting: Σ* does better on the median target.

**Absolute widths** (generated vs. reference medians):

| Channel | Reference | ISO | Σ* |
|---|---|---|---|
| Out-of-plane | 6.27° | 17.9° | 12.0° |
| Bond | 0.0318 Å | 0.0324 Å | 0.0523 Å |
| Angle | 2.61° | 3.30° | 3.64° |
| Torsion | 14.2° | 15.9° | 19.5° |
| Internal RMSF | 0.413 Å | 0.156 Å | 0.618 Å |

**Predictions:**
- **P1 SUPPORTED.** Σ* is not better than ISO on E4.
- **P2 NOT SUPPORTED.** On E5, Σ* has the lower median, but the paired CIs include zero against both ISO and GS.
- **P3 SUPPORTED.** Σ* is better than LAP on E4 and E5.
- **P4 SUPPORTED.** ISO is better than Σ* on E3.

**Reading.** The harmonic source improves cross-channel balance and the out-of-plane and
internal-RMSF channels. It does not improve the average absolute width error over the four
channels relative to isotropic noise, because it over-disperses bond, angle and torsion. It
does beat the Laplacian prior on every calibration endpoint. This narrows the paper's claim
from "more accurate" to "better-balanced, with a different error profile".
