# Reproducibility and leakage audit

Performed 2026-09-25, after the single `test` read, on the original working tree
(`MDWM_TCBB`). Every result was regenerated from the released checkpoints and
data into a separate directory and compared with the stored record, which is
shipped unchanged in `reference_outputs/`. No stored file was modified.

Comparison: `reproduce/verify.py` (byte equality, else every number within a
relative tolerance and every non-path string equal).

## 1. What was regenerated and how it compared

| group | files | result |
|---|---|---|
| Decision rules re-applied to the stored per-example data: `TEST_RESULT.json`, `matched_torsion_fit.json`, `VAL_RESULT.json` | 3 | byte-identical |
| **Model evaluations from the released checkpoints**: fitting-fold grids, no-projection controls, angle-term factorial, val (6 x 1,392), test (9 x 1,456) | 173 JSON + 173 CSV | 171 of 173 match within 1e-5 (largest relative deviation 1.5e-6 in any summary value, 8.7e-6 in any per-example value); 2 differ, see section 3, item 4 |
| **Test decision re-applied to the regenerated test inference** | 1 | verdict CONFIRMED; every reported number identical to 6 decimals |
| Model-free measurements: Wilson B (600 systems), isotropic-noise cost (2,000), hydrogen confound (600), exact Sigma* control (150), tau / theory modules (150), zero-parameter isotropy test, section-1 evidence scripts a, a2, b, b2 | 16 | byte-identical (one differs only in the recorded store path) |
| Sigma* implementation check: autograd vs finite-difference B, 60 systems | 1 | byte-identical |
| Published models: DiffSBDD isotropy (restraint 1 / 10 / 100, n = 300; n = 1,000), reference ligands (1 / 100), isotropy seam, force-field restraint sweep | 8 | identical |
| Corrector gates, 2,000 molecules x 5 split seeds x {DiffSBDD, TargetDiff} | 10 | identical |
| Section-2 effects (loss term vs projection, retraining floor) from the angle-term spectra | 1 | reproduces the published -1.7% [-4.2%, +0.8%], -56.8%, +3.4% / +1.5% |
| val / test example lists from the corpus | 2 | byte-identical |
| TargetDiff pockets (`pockets10/`, 100 PDB + index) from TargetDiff's own code | 101 | byte-identical |
| MMP corpus rebuilt from census + store | 3 | 6,280 / 6,281 pairs identical; see 3.3 |

**Package self-test.** The assembled package was run from its own folder
(stages 00, 01, 03 on a subset plus the regenerated spectra, 04, 05) and
`reproduce/verify.py` returned: 108 files byte-identical, 330 within tolerance,
0 different, 0 not regenerated (exit 0). Path handling, stage ordering and the
decision scripts therefore work outside the original tree.

Model evaluations differ from the stored values only at the level of float32 GPU
reduction order (relative differences of order 1e-7); every median, quantile and
decision is unchanged.

Commands recovered during the audit (not previously recorded) are now fixed in
`reproduce/`: Wilson B used `--fd-check 60`; the DiffSBDD isotropy tests used
`--max-molecules 300 --max-per-pocket 3`; the soft-target cells G and H of the
angle-term factorial used `--angle-target-width 0.048766`, which the script
version of the time did not write into its output. Older result files also lack
a few configuration fields that the current script records; `verify.py` treats a
field absent from the reference as a schema addition, never a missing result.

## 2. Leakage audit

| check | result |
|---|---|
| UniProt targets shared between train / val / test | 0 / 0 / 0 |
| PDB entries shared between splits | 0 / 0 / 0 |
| Checkpoints trained on anything but `train` | none (all 14: `split=train`; fit and dev examples both inside `train`; fit/dev share no target) |
| Checkpoint selection | `dev_loss` on a grouped fold inside `train` |
| val / test example keys or pair keys appearing in any other file under `outputs/`, `configs/`, `scripts/` | none, apart from the corpus definition and one pre-split baseline file (below) |
| Pre-registration hashes vs current files | all match their last recorded hash |
| Inference-time targets (projection angles, bond targets) | computed from the input pose `product_init`, never from the scored reference frames |
| Scored quantity | generated-width / reference-width ratios; copying the input pose would score 0, so the input frame cannot be exploited |

**Disclosed, with measured effect:**

1. **The four physical widths tau** were measured on random MISATO systems that
   include val/test ligands (tau_bond: 36 of 600 systems; tau_angle and
   tau_torsion: 148 of 2,000; tau_oop: 12 of 130). Re-estimated without them:
   tau_bond +0.02%, tau_angle +0.26%, tau_torsion +0.40%, tau_oop -0.03%. The
   theory-modules script meant to exclude test systems filtered on MISATO's own
   split labels rather than this project's target-disjoint split, so the
   exclusion was ineffective (29 of 491 test PDBs caught).
2. **Chemical overlap across the protein split**: 37 of 491 test ligand systems
   (31 distinct molecules) have a chemically identical ligand in `train`, bound
   to a different protein. Removing every test example whose parent or product
   ligand occurs in train (63 of 1,410): Sigma* 0.804, ISO 1.977, ANISO 2.536;
   paired vs ISO -1.141 [-1.363, -1.004]. The verdict is unchanged.
3. **`outputs/mmp_transfer_baselines_v1/baselines_dev.csv`** (2026-09-18, earlier
   project) scores non-learned baselines on 76 examples that later fell into
   val/test. It predates the target-disjoint split, uses a different metric and
   no MEHF model, and nothing in this work reads it.
4. **"Fitting folds" are in-sample.** The exploratory operating-point scans use
   each checkpoint's own `fit_examples.csv`, i.e. its training examples. This does
   not touch val/test, but it is why the fitting-fold spreads of Sigma*
   (0.22-0.46) are optimistic relative to test (0.77).

## 3. What is not reproduced, and why

1. **Training was not re-run in this audit** (by decision). The configs, seeds and
   data are released; `reproduce/02_train.sh` retrains into `outputs_retrain/`.
   GPU training uses non-deterministic float reductions, so retrained weights
   will not be bit-identical. A partial retrain started before the audit was
   stopped after one epoch: ISO fold 1 epoch-1 losses matched the original to
   0.04% (fit) and 0.8% (dev), within the project's measured retraining-noise
   floor of 1.5-3.4%.
2. **200-molecule corrector gates** (`corrector_gate_v1/{holdout_*,with_r5,gate*}.json`)
   were produced by an earlier version of `geometry_corrector_gate.py` (the
   pre-registered hash was `97c83628...`, the final file is `ab25b456...`; the
   fitted angle classes changed from 27 to 26). The final code gives slightly
   different numbers on the same 200 molecules, with the same qualitative
   conclusion. The 2,000-molecule scale-up, which uses the final code, reproduces
   exactly and is what should be cited. `verify.py` lists these files as historical.
3. **Corpus rebuild is load-sensitive at one pair.** The MCS search for
   1FKF -> 1A7X (56 mapped atoms) takes 9.5-10.0 s against a 10 s timeout; under
   heavy CPU load it times out and the pair drops out. It is a `train` pair, so
   val/test are unaffected. Use the shipped corpus.
4. **Soft-target cell G, folds 1 and 2** (`angle_term_v1/spectrum/G_f1`, `G_f2`)
   differ from the final code by up to 0.9% (fold 2 angle ratio 1.499 vs 1.513).
   They are the two earliest soft-target runs; fold 3, run a minute later, and
   every later soft-target run reproduce exactly. This is consistent with their
   predating the change that draws the soft-target offsets from the seeded
   sampler generator (documented in `mehf.py`); without version history it cannot
   be proven. Effect: cell-G mean angle ratio 1.699 -> 1.703, normalised
   selectivity 3.40 -> 3.41. The section-2 effects use cells A-D and are unaffected.
5. **TargetDiff re-sampling is not bit-reproducible.** Re-running TargetDiff's own
   released sampler with the identical command and seed on one pocket changed 16
   of 20 molecules (up to 0.43 Å; some atom types), consistent with
   non-deterministic GPU scatter operations over 1,000 diffusion steps in the
   upstream code. The exact samples analysed are shipped, and every TargetDiff
   gate result reproduces exactly from them; a fresh re-sample is a different
   draw from the same model. Whether the TargetDiff conclusions hold on an
   independent re-sample has not been tested (about 10 GPU-hours).
6. **Exploratory and superseded records that are not regenerated**: the stage-1
   scale-calibration probes of `internal_calibration_v1` (fitting folds only,
   labelled exploratory in the files, sampling mode not recorded; the provenance
   of rigid = 0.32), earlier Jensen-Shannon evaluations, and the superseded
   ANISO v2 prior. `verify.py` lists them as historical.
7. **Voided / retracted records** kept for transparency: `aniso_prior_v1/`
   (diverging sampler), `isotropic_noise_cost_v1/summary.json` (first run, bond
   ratio 25.99 before the SHAKE confound was removed), smoke tests.

## 4. Changes made when packaging (code behaviour unchanged)

- Seven scripts and one shell script had hard-coded `/home/...` roots; they now
  resolve the root from their own location. Path-only edits.
- New: `scripts/make_split_example_lists.py`, `scripts/angle_term_effects.py`,
  `scripts/targetdiff/{make_pockets10.py, collect_targetdiff_sdf.py, sampling.yml.in}`,
  `tests/verify_b_rows.py` (previously session-local scripts; each verified to
  reproduce its recorded output), `reproduce/`.

## 5. Corrections to earlier statements found during the audit

- The training objective is not only the isotropic CFM regression: it also has a
  bond-length term (weight 1.0) and a pocket-clash term (weight 10.0), identical
  in all three arms (angle weight 0). The prior is the only difference between arms.
- The projection-only corrector (arm D) damages aromatic planarity on DiffSBDD as
  well as on TargetDiff (DiffSBDD, 2,000 molecules: 4.58 -> 10.84 deg), so this
  failure is not specific to models that are already near-planar.

## 6. Added after the audit (2026-09-29)

- `scripts/leakage_audit.py` packages the leakage checks of section 2 (previously
  run ad hoc); its output `reference_outputs/leakage_audit_v1/audit.json` reproduces
  every number quoted there. It runs in `reproduce/04_decisions.sh`.
- `reference_outputs/corrector_gate_v1/scaleup_with_r5/`: the 2,000-molecule DiffSBDD
  gate regenerated with the final script, which adds arm G (MMFF, restraint 5). Arms
  A-F are identical to `scaleup/`; the paper's DiffSBDD gate numbers come from here.
- `paper/`: the manuscript, whose numbers and figures are generated from `reference_outputs/`.

## 7. Added comparison: graph-Laplacian prior (2026-09-29)

`reference_outputs/laplacian_prior_v1/`: a fourth prior (bond-graph Laplacian,
HarmonicFlow / EigenFold style) added after the confirmatory test read, under its own
pre-registration hashed before any code existed (PREREGISTRATION.md, RESULT.md).
New default-off flag `laplacian_local_noise` in `mehf.py`; existing checkpoints were
re-evaluated after the change and match the record (2 identical, 2 within tolerance).
Three checkpoints (`outputs/laplacian_prior_v1/full_f*`, now 17 in total, all
train-only per the updated leakage audit), 27 evaluations (added to
`reproduce/spectra_commands.txt`, 200 in total), `scripts/laplacian_matched_torsion.py`
(runs the hashed matching rule unchanged), `scripts/laplacian_test_decision.py`,
`scripts/prior_noise_profiles.py`. Result: MEHF (Sigma*) 0.770 vs Laplacian 1.352,
paired -0.580 [-0.750, -0.404].
