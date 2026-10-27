# RESULT_LAP -- clean MEHF-LAP read (FOURTH read of the test split)

**Provenance, stated plainly.** This is the FOURTH read of this test split. The decision to train MEHF-LAP was taken AFTER the
clean ISO/HARM test numbers (primary Delta -1.0345, secondary -0.9398) were known (LAP_DECISION.txt, sha256 f9bcc97d...). It was
not taken blind. Section 0 of the pre-registration already discounts the nominal 95% intervals for three reads and Amendment 3
raised the comparison count to four; this read adds LAP at two operating points, so the nominal intervals below should be read
as materially wider than printed. This result may not be used to revise the ISO/HARM headline (Tier A, already fixed).

Command: `scripts/clean_protocol_lap_test_lane.sh {1,2,3}` (3 lanes, GPU 1 only), then
`scripts/clean_protocol_lap_test_decision.py`. Flags identical to the ISO/HARM read. Operating points from
`s_star_frozen_lap.json` (sha256 44d43078...). Read start 2026-10-25 08:38:01, end ~09:57:35; 6 runs, 12 files (6 json + 6
per-example csv), 0 failures, 0 restarts. The analysis script was hashed (CHAIN.log) before any LAP test output existed and ran
on its first invocation without modification. Helper functions imported from `clean_protocol_test_decision.py` (405cc82b..., unchanged).
Data: same 1,406 usable examples from 59 targets (0 additional dropped for LAP; all 1,452 rows produced per fold).
Full numbers: `TEST_RESULT_LAP.json`.

## LAP at the two operating points (medians over the 1,406 examples; per-example values averaged over 3 fold checkpoints)

| | PRIMARY (fit s*) | SECONDARY (dev s*) |
|---|---|---|
| s* per fold | 3.385 / 2.573 / 3.044 | 3.290 / 2.882 / 3.112 |
| bond / angle / torsion / oop ratio | 2.311 / 2.080 / 1.100 / 2.673 | 2.385 / 2.151 / 1.137 / 2.748 |
| median spread S (IQR) | **1.382** (0.937-1.986) | **1.420** (0.963-2.045) |
| per-fold median S | 1.493 / 1.256 / 1.386 | 1.461 / 1.382 / 1.414 |
| median psi | 0.733 | 0.732 |
| internal RMSF ratio (median) | 0.937 | 0.968 |
| realised torsion ratio per fold (pooled) | 1.218 / 0.983 / 1.108 (1.100) | 1.189 / 1.091 / 1.133 (1.137) |
| full-pocket clash, pooled pair fraction | 5.2e-10 | 1.0e-9 |
| examples with any clash / outside-crop clash | 0.02% / 0 | 0.05% / 0 |
| fit/dev spread at s* (in-sample, mean) | 1.330 | 1.326 |

Clash column is floor-saturated (see RESULT.md, the 1.2 A threshold is never reached by real trajectories); it is not a quality
ranking and is reported to satisfy the A4.2 gate.

Same examples, clean arms from the completed read: Sigma* (HARM) median S 0.796 / 0.872 (psi 0.539 / 0.522, RMSF 1.54 / 1.75);
ISO 1.902 / 1.904 (psi 1.169). LAP sits between them in spread, and between them in psi.

## Pre-registered contrast: MEHF(Sigma*) minus MEHF-LAP, per-example spread, paired, UniProt cluster bootstrap (10,000, seed 20260925)

| operating point | mean difference | 95% CI | median diff | Sigma* lower on | per-fold means |
|---|---|---|---|---|---|
| PRIMARY | **-0.531** | [-0.675, -0.386] | -0.599 | 85.6% | -0.642 / -0.437 / -0.514 |
| SECONDARY | **-0.477** | [-0.603, -0.353] | -0.550 | 84.1% | -0.517 / -0.444 / -0.469 |

n = 1,406 examples, 59 targets in both. Both intervals exclude zero and the point estimates are negative in every fold. For
context, LAP minus ISO is -0.504 [-0.650, -0.403] (primary) and -0.463 [-0.617, -0.352] (secondary), i.e. LAP is itself a
large improvement over ISO, and Sigma* is a further, similar-sized improvement over LAP. The reproduced Sigma*-ISO values
(-1.0345, -0.9398) match the completed read exactly, a check that the pairing and example sets are the same.

## Blocking conditions for the LAP arm
- **B4 (prior art): DOES NOT FIRE** at either operating point. Clean LAP median spread 1.382 vs Sigma* 0.796 (primary) and 1.420
  vs 0.872 (secondary); LAP is above Sigma* by 0.59 and 0.55. The proposed prior beats the published graph-Laplacian source on this
  metric once the pocket leak is removed, under the stated 4-read discount. (This is a spread/amplitude-matched-torsion result,
  with the caveats of Amendment 5 about the metric; Sigma* also over-disperses internal RMSF, 1.54 / 1.75, while LAP is at
  0.94 / 0.97 -- a reader should see that next to the spread.)
- **B1 (amplitude premise): does not fire** for LAP at either point. Per-fold realised torsion ratio 1.218 / 0.983 / 1.108
  (primary) and 1.189 / 1.091 / 1.133 (secondary), all inside [0.85, 1.25]; pooled 1.100 / 1.137. Fold 1 primary at 1.218 is the
  closest to the edge. LAP is therefore matched, unlike HARM at its dev points, and is a fair comparator.
- B2 and B3 are not re-evaluated for LAP (they are defined on Sigma*); the existing Sigma* determinations stand.

## Scoring the recorded prediction (LAP_DECISION.txt, confidence 0.7)
Prediction: clean LAP between ISO and Sigma* on spread, nearer to Sigma* than the original 1.352 suggested, point estimate
1.1-1.4, and Sigma* minus LAP negative with an interval excluding zero.
- Between ISO and Sigma*: **CONFIRMED** at both points (0.796 < 1.382 < 1.902; 0.872 < 1.420 < 1.904).
- Sigma* minus LAP negative, CI excluding zero: **CONFIRMED** at both points.
- Point estimate in 1.1-1.4: **primary 1.382 inside; secondary 1.420 outside by 0.02.** A 0.02 overshoot is a miss on the band as
  written, and I score it as such rather than rounding it in.
- "Nearer to Sigma* than the original protocol's 1.352 suggested": **WRONG.** Clean LAP (1.382 / 1.420) is slightly higher than
  the leaky LAP (1.352), not lower. The Sigma*-LAP gap is unchanged: 0.582 under the original protocol (1.352 - 0.770) against
  0.586 (primary) and 0.548 (secondary) now. LAP, like Sigma*, moved slightly up; only ISO moved down (1.974 -> 1.902). The
  decision file's mechanism (leak mildly flattering wide-ensemble arms, penalising narrow ones, so LAP would move toward
  Sigma*) is not supported.
Overall: HALF RIGHT to MOSTLY RIGHT. Direction, ordering and significance hold; one numerical band is missed at the secondary
point and the stated mechanism is not supported.

## Other observations
- LAP's fitting-fold to test degradation is small: fit spreads at s* 1.147-1.504 (mean 1.330) vs test median 1.382 (ratio ~1.04),
  like ISO and unlike Sigma*, which again degrades more (fit 0.39-0.46 vs test 0.80, ratio ~1.9). This is consistent with the
  P4 account that the degradation is a property of where Sigma* operates.
- LAP's channel balance is poor in the oop, bond and angle channels (2.3 / 2.1 / 2.7) with torsion at 1.1, i.e. its spread is
  driven by the stiff channels; Sigma*'s corresponding ratios are 1.68 / 1.41 / 1.98 (primary).
- The Tier of the ISO/HARM headline is unchanged by this read. No blocking condition fires for LAP.
