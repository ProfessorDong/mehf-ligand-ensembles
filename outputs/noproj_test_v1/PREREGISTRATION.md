# PREREGISTRATION: held-out evaluation of the source priors without inference-time projection

Written 2026-10-13 by LD, before any run of this study. No `noproj_test_v1` result file
exists when this file is hashed (`PREREGISTRATION.sha256`). The decision script
`scripts/noproj_test_decision.py` is hashed at the same time.

## Why this study exists

The submitted draft's Abstract claims that, with inference-time projection disabled,
the harmonic prior reduces the out-of-plane width ratio from 5.58 to 1.03 relative to
the isotropic prior. Provenance audit (`MEHF_TCBB_Liang/PROVENANCE_NOPROJ_CLAIM.md`) found:

1. The numbers come from 96 TRAINING examples per fold (`fit_examples.csv`), at a fixed
   internal scale 2.5, not from val/test or from the matched-torsion operating point.
2. "Projection disabled" was `--angle-projection off`. That turns off only the angle step.
   The bond-length step and the pocket-clash push still ran for 8 iterations
   (`src/mdwm/models/mehf.py`, `_project_geometry`). No truly projection-free
   evaluation exists in the record.
3. No held-out (val or test) evaluation without projection exists.

This study measures the claim on the held-out test targets, under both meanings of
"no projection".

## Status relative to the confirmatory test

The test split was read once, on 2026-09-25, under `reference_outputs/sigma_star_test_v1/`.
This is a SECOND, POST-HOC read of the same test examples. It follows the Laplacian
precedent (`reference_outputs/laplacian_prior_v1/`). It does not alter or re-decide the
confirmatory verdict, and the manuscript will label it as added after the confirmatory read.
No checkpoint is retrained, and no operating point is re-selected.

## Fixed design

- Checkpoints: the 12 released checkpoints, unchanged. ISO = `internal_calibration_v1/full_f{1,2,3}`;
  GS = `aniso_prior_v3/full_f{1,2,3}`; LAP = `laplacian_prior_v1/full_f{1,2,3}`;
  SIG (Σ*) = `sigma_star_v1/full_f{1,2,3}`.
- Examples: `outputs/sigma_star_test_v1/test_examples.csv`, `--all-examples` (1,456).
- Every other flag is identical to the frozen test commands in
  `reproduce/spectra_commands.txt` (lines 137-145, 186, 195, 204): `--rigid 0.32 --steps 16
  --impropers --restore off --angle-target-width 0.0 --projection-angle-relaxation 0.5`,
  default seed 20260918.
- Operating points:
  - **S\***: each arm's and fold's frozen test operating point from those commands.
    ISO 1.10909/1.30855/1.10709; GS 1.0 (unmatchable, nearest grid); SIG 3.24529/3.32918/3.78681;
    LAP 2.43633/2.48254/2.44839.
  - **S2.5**: internal scale 2.5 for every arm. This is the setting of the draft's
    fitting-fold analysis.
- Projection conditions:
  - **C0**, full projection: the frozen confirmatory files. Not re-run, except for the
    environment gate below.
  - **C1**, angle step off: `--angle-projection on` → `off`, with `--projection-iterations 8`.
    Bond and clash steps are retained, which is identical to the draft's "no projection".
  - **C2**, no projection: `--projection-iterations 0`, so `_project_geometry` returns the
    flow output unchanged.
- Runs: 4 arms × 3 folds × {C1, C2} × {S\*, S2.5} = 48 evaluations, written to
  `outputs/noproj_test_v1/{C1,C2}_{Sstar,S2.5}_{ARM}_f{K}.json/.per_example.csv`.

## Environment gate (must pass before any of the 48 runs)

The original environment (torch 2.12, 2 × A6000) is not available. This machine has torch
2.11 on an RTX 4090. Before any new run, re-run two frozen C0 test evaluations,
`test_SIGMA_star_f1` and `test_ISO_f1`, with their exact commands into
`outputs/noproj_test_v1/gate/`. **Gate passes** if every channel median and the internal-RMSF
median in each re-run JSON is within 0.5% relative of the frozen value. The largest
per-example relative deviation is reported either way. If the gate fails, no new run
proceeds and the study is reported as not performed.

## Statistics (fixed)

The functions of `scripts/sigma_star_test_decision.py` are imported unchanged:
- per-example values are averaged over the three folds;
- examples are restricted to those scorable in every arm;
- the paired difference is the mean over examples;
- the 95% CI is a percentile cluster bootstrap over UniProt targets, 10,000 resamples,
  seed 20260925;
- the median of per-example spread is reported with IQR.

## Hypotheses

- **P1 (primary).** Under C2 at S\*, Σ* has a lower out-of-plane width ratio than ISO and
  than LAP: mean paired difference < 0, with the 95% cluster-bootstrap CI excluding 0, for
  both contrasts. Reported as SUPPORTED only if all four conditions hold.
- **P1b.** As P1, for the four-channel spread.
- **P2 (directional, size reported).** Under C1 at S2.5, Σ*'s median out-of-plane ratio on
  test exceeds the fitting-fold value 1.03, because every Σ* channel degraded off the
  fitting folds.
- **P3.** For Σ* at S\*, the median out-of-plane ratio changes by less than 10% relative
  between C2 and C0.
- **P4.** At S\*, the relative change in median out-of-plane ratio from C2 to C0 is
  larger in magnitude for ISO than for Σ*.

C0 and C2 use the same seed, and with `angle-target-width 0` the projection draws no
random numbers. The flow samples are therefore identical, so C0 − C2 is the exact
per-example effect of projection.

## Reporting rule

Every number above is reported regardless of outcome: in `RESULT.md`, in §IV-D of the
manuscript, and, for any arm or contrast that is unfavorable to Σ*, in the text and not
only in tables. The Abstract may carry a no-projection statement only if P1 is SUPPORTED,
and then only with the test-split C2 numbers. If P1 is not supported, the Abstract carries
no no-projection claim, and the failure is stated in §IV-D.
