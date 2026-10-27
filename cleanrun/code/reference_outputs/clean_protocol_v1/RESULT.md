# RESULT — clean-protocol replication of the Sigma* test result (third and final read of the test split)

**First line: the clean protocol replicates in direction but not cleanly in the Tier. The PRIMARY operating point lands in Tier R by
0.013 (point estimate -1.0345 against a cut of -1.021); the SECONDARY operating point lands in Tier A (-0.940). Under Amendment 3 the
headline is the worse of the two, so it is Tier A, labelled operating-point-sensitive, and it is BLOCKED at the secondary operating point
by B1 (HARM torsion ratio too high) and B3 (copy-the-parent with projection ON beats HARM on spread there). B2 does not fire. B4 is not
applicable. The published -1.152 attenuates to -1.035 (primary) / -0.940 (secondary); the attenuation is itself significant
(DiD +0.117 [+0.075,+0.156] and +0.212 [+0.146,+0.283]).**

Read once, 2026-10-24 13:53-19:23 (30 inference runs, 0 failures, 0 restarts), under `PREREGISTRATION.md`
(sha256 3863992a30606022fa11736e86104faa4bcf6c6d36d76273a9f678e4ae9f4daf, 2,066 lines, verified before the read). Operating points from
`outputs/clean_protocol_v1/s_star_frozen_v2.json` (sha256 dabcd5f3...), all 12 cells matched, none extrapolated. Analysis script
`scripts/clean_protocol_test_decision.py` sha256 405cc82b...cb4e9a9e1236595d66159e3fd65383624 was hashed before the first test command and run
exactly once; its output is `TEST_RESULT.json` next to this file. Nothing was re-run, rescaled or re-selected after a number was seen. The third
read of this split: nominal 95% intervals carry the family-wise cost of section 0 and Amendment 3 (four comparisons, roughly 80% intervals).
Third read means this is the last one; the split is spent.

## Process notes
- A previous launch (12:09) never started and produced nothing (CHAIN.log). This session is the only read.
- Examples: 1,452 listed (726 pairs, 65 targets; the frozen read-1 list minus the 28 excluded systems, by PDB id,
  `test_examples_clean.csv` sha256 5756bdbb...). **Usable after the Addendum-A rule (all four ratios finite in all three folds of both clean
  arms): 1,406 examples from 59 targets** (46 dropped for lacking an approximately planar centre; read 1 had 1,456 -> 1,410 from 60 targets).
  Every fold run produced 1,452 rows.
- GPU 1 only. Another user's process [REDACTED, see REDACTIONS.md] (up to 28 GB) was on GPU 1 for part of the session; not ours, not touched.
- T7 asserted at analysis time: every clean run resolved `pocket_selection=frame0`, `harmonic_tau_set=train_only_large`, and the scale in the
  JSON equals the frozen s*; every leaky run resolved `min_over_frames`.
- **LAP was never trained under the clean protocol and `LAP_DECISION.txt` was never written** (section 3.1 required it at the moment the queue
  was fixed). B4 is conditional on LAP being trained, so it is reported not applicable; leaky LAP (frozen median 1.352 on the same examples) is
  cross-protocol only. I did not create the file now, since that would backdate a decision.
- Fresh inference of the four leaky arms (needed only for the clash secondary) does NOT reproduce the frozen per-example CSVs exactly: medians agree
  to 0.0005-0.019 (ISO 1.954 vs 1.973, GS 2.527 vs 2.527, LAP 1.361 vs 1.352, Sigma* 0.765 vs 0.770) but single examples differ by up to 1.06-1.72.
  DIAGNOSED 2026-10-24: the cause is batch-composition dependence of the sampling seed, and the live-vs-frozen source plays no part. `build_batches`
  (`run_subspace_plane.py:70-120`) groups consecutive rows into batches of 4 and seeds one generator per batch from the FIRST example's key, so an
  example's draw depends on which example leads its batch and on the other members' atom counts. The clean list drops 4 of the 1,456 rows (at frozen
  positions 347, 675, 1075, 1403), which shifts every later batch boundary. Measured consequences: examples whose batch is unchanged reproduce the
  frozen CSV to <=4.6e-7 (321 of 396 exactly equal, SIGMA fold 1), and every example in a changed batch moves, median |d| 0.12 on the bond-length
  ratio. Re-running the SAME checkpoint on the SAME examples at batch size 4 reproduces the fresh CSV to 7e-7, while batch sizes 3 and 1 disagree with
  it by median 0.04-0.09 and max 0.52-0.82 -- the same magnitude as fresh-versus-frozen. The frozen per-example values were therefore never
  reproducible against any other draw; this is a property of the sampler, not a regression, and the pipeline is deterministic given a fixed list,
  order and batch size. The Tier is untouched (leaky spreads come from the frozen CSVs, A1.4.2). The leaky clash rates are a different draw of the
  same model, not the identical draw. The primary contrast is protected: clean HARM and clean ISO share the clean list, hence the same batches and
  seeds, so the pairing uses common random numbers. The cross-protocol DiD pairs frozen leaky draws against fresh clean ones and therefore carries an
  unpaired-draw noise component that the example-level bootstrap does not model; it was already diagnostic-only, and it was not recomputed.

## Primary contrast (HARM = clean Sigma*, minus ISO; per-example mean over 3 folds, UniProt cluster bootstrap, 10,000 resamples, seed 20260925)

| operating point | N / targets | median HARM | median ISO | mean paired diff | 95% CI | HARM lower on | per-fold diff | Tier |
|---|---|---|---|---|---|---|---|---|
| PRIMARY (fitting-fold s*) | 1,406 / 59 | 0.796 (IQR 0.489-1.382) | 1.902 (1.577-2.378) | **-1.0345** | [-1.2630, -0.9175] | 98.1% | -1.117 / -0.917 / -1.070 | **R** (by 0.013) |
| SECONDARY (dev-fold s*) | 1,406 / 59 | 0.872 (0.578-1.470) | 1.904 (1.582-2.380) | **-0.9398** | [-1.1693, -0.8404] | 96.9% | -0.895 / -1.026 / -0.899 | **A** |

Tier R needs point <= -1.021, CI excluding 0, median below the comparator: met at primary with 0.013 to spare; the CI's far edge (-1.263) is inside
read 1's interval but its near edge (-0.918) is not <= -1.021. Section 6 keys the Tier on the point estimate, so R is the literal reading; I flag how
close to the cut it is. Amendment 3 case 2 was excluded pre-read (max |s*_dev - s*_fit| = 0.855, HARM fold 2); the two operating points land in
different Tiers, so **case 4 applies: headline = worse = Tier A, "operating-point-sensitive"**, and the manuscript must say the magnitude depends on
whether the amplitude was matched in-sample or on a target-disjoint fold.

## Per-arm table (medians over the 1,406 usable examples; ratios are fold-averaged per example then medianed)

| arm @ op | bond | angle | torsion | oop | S (spread) | psi | torsion per fold | internal RMSF | full-pocket clash (pooled pair frac / frame rate / examples with >=1) |
|---|---|---|---|---|---|---|---|---|---|
| ISO @ primary | 1.110 | 1.302 | 1.080 | 2.875 | 1.902 | 1.169 | 1.135 / 0.951 / 1.132 | 0.407 | 0 / 0 / 0 |
| **HARM @ primary** | 1.676 | 1.409 | 1.136 | 1.983 | **0.796** | 0.539 | 1.201 / 1.029 / 1.186 | 1.542 | 5.5e-8 / 4.7e-4 / 1.8% |
| ISO @ secondary | 1.106 | 1.297 | 1.075 | 2.873 | 1.904 | 1.169 | 1.047 / 1.101 / 1.062 | 0.406 | 0 / 0 / 0 |
| **HARM @ secondary** | 2.042 | 1.629 | **1.256** | 2.176 | **0.872** | 0.522 | **1.295** / 1.203 / **1.258** | 1.748 | 1.2e-7 / 1.0e-3 / 3.7% |

(`n_outside_frame0_crop` clash, the M3 region: HARM 4.2e-9 pair fraction at primary, 1.4e-8 at secondary; ISO 0.)
Leaky arms on the same 1,406 examples (frozen CSVs): ISO 1.973, GS 2.527 (unmatched), LAP 1.352, Sigma* 0.770. Clean HARM is 0.026 WORSE than leaky
Sigma* (0.796 vs 0.770) and clean ISO is 0.071 BETTER than leaky ISO (1.902 vs 1.973): the attenuation of the gap comes from both ends, mostly ISO.
HARM is over-dispersed (internal RMSF 1.54 / 1.75) and ISO under-dispersed (0.41); matched torsion does not match overall amplitude.

## Model-free controls, same table, same 1,406 examples (CUDA; Control A has no fold dependence so one run serves all folds)

| control | proj | torsion ratio | matched? | S | psi | bond / oop | internal RMSF | full-pocket clash pair frac / frame rate / examples>=1 |
|---|---|---|---|---|---|---|---|---|
| A: Sigma* source, s_i = 4.0 | OFF | 0.683 | **unmatched** | 1.084 | 0.885 | 1.427 / 1.233 | 0.920 | 2.7e-8 / 2e-4 / 1.1% |
| A, s_i = 4.0 | ON | 0.664 | **unmatched** | 0.587 | 0.703 | 0.661 / 1.182 | 0.889 | 0 / 0 / 0 |
| A, s_i = 5.0 | OFF | 0.836 | **unmatched** (just) | 1.572 | 0.985 | 1.829 / 1.515 | 1.150 | 1.5e-7 / 1.3e-3 / 5.6% |
| A, s_i = 5.0 | ON | 0.802 | **unmatched** | 0.667 | 0.645 | 0.851 / 1.431 | 1.098 | 3e-9 / 1e-5 / 0.1% |
| B: copy-the-parent | OFF | 0.861 | matched | 3.356 | 1.550 | 2.365 / 1.158 | 1.072 | 3.5e-5 / 0.147 / 54.1% |
| B | ON | 0.877 | matched | **0.836** | 0.750 | 1.154 / 1.188 | 0.920 | 4.8e-6 / 0.029 / 20.6% |

Control A is unmatchable on the grid, at 5.0 as at 4.0 (the 5.0 point was an addition on my part, decided before the read and logged in CHAIN.log,
because Amendment 7 does not say whether the extended grid covers the controls; both are reported).

## Blocking conditions (evaluated at each operating point separately, A3.3 item 6)

- **B1 — FIRES at the secondary operating point, not at the primary.** Realised test torsion ratio, HARM: primary 1.201 / 1.029 / 1.186 (inside
  [0.85,1.25]); **secondary 1.295 / 1.203 / 1.258, folds 1 and 3 above 1.25** (and the pooled median 1.256). Under the all-rows JSON medians fold 1
  still fires (1.281) and fold 3 does not (1.239). ISO is inside the window at both. **The anticipated trigger did not occur**: HARM fold 2 at the
  primary s* came out at 1.029, not 0.844. B1 fired on the opposite side, at the other operating point. Consequence per section 6: HARM at the
  secondary operating point is reported as unmatched (over-amplitude), exactly as GS was in read 1.
- **B2 — does not fire** at either operating point. Clean HARM's spread does not improve on leaky Sigma* (0.796 / 0.872 vs 0.770), so the first
  condition fails; independently its full-pocket clash did not exceed the tolerance. Pinned definition (A4.3, pooled pair fraction): leaky tolerance
  8.7e-8 (four arms; same for three), clean HARM minus leaky Sigma* = -3.2e-8 (primary) / +3.3e-8 (secondary); frame-rate version tolerance 6.5e-4,
  worsening -1.8e-4 / +3.7e-4. No variant fires. (The 11:43 "B2 does not fire" was on fit/dev samples; this is the test version.)
- **B3 — FIRES at the secondary operating point (projection ON, Control B), not at the primary.** Control B with projection ON is matched
  (torsion 0.877 in [0.85,1.25]) and its median spread 0.836 is below clean HARM's secondary 0.872, so B3(a) fires; at the primary HARM's 0.796 is
  below 0.836 and it does not. The margin is 0.036 in one direction and 0.040 in the other: the verdict flips on which operating point HARM sits
  at. Control A is unmatched at both s_i and both projection states, so (b) applies: it fires only if psi < HARM's (0.703 / 0.645 vs 0.539 / 0.522),
  so it does not. Per Amendment 5 section 5.5(d) a matched control winning on spread fires B3 regardless of psi; psi (HARM 0.52-0.54 vs 0.750) and
  clash (Control B ON: frame rate 0.029 and 20.6% of examples vs HARM 0.0005-0.001 and 1.8-3.7%) say HARM is the better allocator and the
  safer one, but those are not routes for B3 not to fire. Restatement requirement of section 6 applies: the spread may not be presented as evidence
  that MEHF generates good ensembles; the control rows stay in the main table.
- **B4 — not applicable** (LAP not trained).

## Tier
Primary R, secondary A. **Headline = Tier A (operating-point-sensitive)**; blocked by B1 and B3 (both secondary-only). No other tier is invented.
What the reading licenses: the clean protocol reproduces a large, significant HARM-minus-ISO spread reduction (about -0.94 to -1.03, 96-98% of
examples) that is about 10-18% smaller than the published -1.152, and the near-tie with projection-ON copy-the-parent (0.836 vs 0.796 / 0.872)
means the spread alone cannot be offered as evidence of ensemble quality.

## Diagnostics the pre-registration asked for
- DiD (clean minus leaky Delta, same 1,406 examples, same bootstrap): primary **+0.117 [+0.075, +0.156]** (clean -1.0345 vs leaky -1.1513);
  secondary +0.212 [+0.146, +0.283]. Diagnostic only; it contains retraining variance (the only prior estimate is +1.5-3.4% on a channel ratio),
  which cannot be separated with three folds and no replicate seeds. The sign says the leak was, if anything, flattering the method by about 10%.
- Fitting-fold to test degradation (headline median / fitting spread at s*): HARM primary 1.78 / 1.72 / 2.04 (mean-fit ratio 1.84), secondary
  1.71 / 1.63 / 1.76; ISO primary 0.99 / 1.21 / 1.02 (mean-fit 1.07), secondary 1.14 / 0.98 / 1.12.
- Crop diagnostics (no model; `outputs/clean_protocol_v1/test_read/crop_recall.json`), all 3,791 post-exclusion systems: frame-0 vs all-frames crop
  overlap mean 0.816 / median 0.823 / min 0.375; recall of all-frames contact pairs (< 4 A at any frame) by the frame-0 crop: mean 0.916, median 0.930,
  min 0.377, 33.8% of systems below 0.90; atom-level recall mean 0.864 (60.0% below 0.90). Test-list systems: 0.918 pair / 0.863 atom.

## Predictions P1-P10, scored against the test numbers
- **P1a REFUTED, P1b WRONG** (both scored before the read, A7.6: bond/angle widths move about 2% and up to 4.5%, bound 0.035 > 0.02). Not
  re-scorable on test: the old-tau retrain was never run (A2.4). Stands.
- **P2 CONFIRMED** (primary): Delta -1.0345 in [-1.45,-0.85]; CI [-1.263,-0.918] excludes 0; HARM median 0.796 in [0.65,1.10]; ISO 1.902 in
  [1.75,2.20]. Secondary also inside every band (-0.940, 0.872, 1.904). Confidence 0.85 was fair; the Tier-R margin is thin.
- **P3 WRONG.** Predicted DiD <= 0 (gap widens, 0.55). Measured DiD +0.117 [+0.075,+0.156]: the gap moved toward zero, significantly, at
  both operating points. I have no tested mechanism for the direction; B2 shows no fence was lost in clash terms.
- **P4 CONFIRMED.** HARM degrades 1.84x (> 1.3) while ISO's mean-fit ratio is 1.07 (< 1.15). Caveat: ISO fold 2 alone is 1.21 on the
  per-fold version; on the spread-of-medians version ISO is 0.97-1.05 and HARM 1.74-2.12.
- **P5 WRONG** (as already pre-scored from dev, A5.3). Control A ON spread 0.587 (s_i 4.0) / 0.667 (5.0), below the [0.8,1.6] band and below clean
  HARM (0.796), not above it. OFF at 5.0 (1.572) is inside the band but P5 specified projection ON.
- **P6 WRONG as pre-scored, with a test footnote.** OFF: 3.356, outside [1.0,2.5], and it does not beat ISO. ON: 0.836 inside [0.5,1.4] and
  beats ISO; it is above HARM at the primary operating point and below HARM at the secondary. The pre-read WRONG stands because the OFF half was
  missed on band and ordering; the ON half was hit.
- **P7 WRONG.** Recall of contact pairs by the frame-0 crop is mean 0.916 (not < 0.90; the atom-level 0.864 would have satisfied it, but the text says
  pairs). And the clash clause fails: ISO is 0 vs 0 (not higher), clean HARM at primary is LOWER than leaky Sigma* (5.5e-8 vs 8.7e-8), higher
  only at the secondary (1.2e-7).
- **P8 CONFIRMED.** Control A ON stays unmatched (torsion 0.664 at 4.0, 0.802 at 5.0, both < 0.85) and its psi (0.703 / 0.645) stays above clean
  HARM's (0.539 primary / 0.522 secondary).
- **P9 WRONG.** Control B ON psi is 0.750, BELOW trained ISO's 1.169 (it is 1.550 only with projection OFF). After projection copy-the-parent is a
  better allocator than isotropic noise, not a worse one.
- **P10 CONFIRMED.** Projection cuts Control A's bond ratio by about half (1.427 -> 0.661 at 4.0; 1.829 -> 0.851 at 5.0) and the torsion ratio by only
  3-4%, so psi falls 0.885 -> 0.703 (4.0) and 0.985 -> 0.645 (5.0). This is the number the pre-registration most wanted: **the parameter-free projection
  does a large share of the allocation work** (Control A ON psi 0.64-0.70 vs HARM 0.52-0.54 vs ISO 1.17) -- the learned flow improves psi further, by
  about 0.11-0.16, but the "no model-free control reaches the arm's allocation" reading holds only for psi, not for spread.

Tally: CONFIRMED P2, P4, P8, P10; WRONG P1b, P3, P5, P6, P7, P9; REFUTED P1a. Four of ten confirmed.

## Observations the paper must carry
- The effect replicates but shrinks, and which Tier you get depends on which operating point you believe (R at fitting-fold s*, A at dev-fold s*).
  HARM's torsion curve is 2.8-3.7x flatter than ISO's at its crossing, so a small fit/dev difference moves HARM's s* by up to 0.855.
- Spread is gameable by amplitude (A5.4) and here also by the projection: projection-ON copy-the-parent reaches 0.836 with no model, no training and a
  matched torsion ratio. It loses on psi (0.750 vs 0.52-0.54) and on clashes (28-61x HARM's clash frame rate), which is the case for reporting psi and
  clash beside spread, not a reason to drop the spread.
- HARM clashes (1.8-3.7% of examples) and ISO never does; Sigma*-family ensembles are ~4x wider (internal RMSF 1.5-1.7 vs 0.41).
- ISO is under-dispersed (internal RMSF 0.41) at matched torsion; HARM over-dispersed (1.5-1.7). Matched torsion is not matched amplitude.

## Release items (A6.5)
PREREGISTRATION sha256 above; draft-1 hash a51668b7... (recipe in the document); three test reads: 2026-09-25 (sigma_star_test_v1), 2026-09-29
(laplacian_prior_v1), 2026-10-24 (this); `excluded_systems.json` (28); `tau_train_only_large.json`; `s_star_frozen_v2.json` (fit and dev values);
analysis script hash taken with zero `test_*.json` on disk (405cc82b...); `LAP_DECISION.txt`: absent (see Process notes). Per-example CSVs,
configs and logs: `outputs/clean_protocol_v1/test_read/` (63 files), `outputs/clean_protocol_v1/logs/test_read/` (35 files), `READ_CONFIG.json`.
