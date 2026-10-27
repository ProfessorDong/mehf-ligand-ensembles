# MEHF: an exact harmonic prior for flow matching of ligand conformational ensembles

Code, checkpoints and the complete result record for the MEHF paper.

**The problem.** Diffusion and flow-matching models for 3D molecules add isotropic
Gaussian noise to Cartesian coordinates and are trained with an isotropic squared
error, but they are judged on internal coordinates. Through the Wilson B matrix,
`Var(q_k) = sigma^2 ||B_k||^2`, isotropic noise costs about 43 / 31 / 8 physical
thermal widths per Angstrom of sigma on bond lengths / bond angles / torsions: it
spends its budget in the stiffest directions.

**The method.** MEHF (`MappedEditHierarchicalFlow`) is an SE(3)-equivariant
conditional flow-matching model. This work replaces the local part of its source
distribution with the molecule's own harmonic covariance

    Sigma* = (B^T F B)^+ ,   F = diag(1 / tau_k^2)

over a COMPLETE internal-coordinate set (bonds, angles, proper torsions,
out-of-plane), rescaled to the isotropic amplitude so only the shape changes.
The same draw is used in training and sampling. The interpolation, regression
target, loss and network are unchanged.

| where | what |
|---|---|
| `src/mdwm/models/harmonic_prior.py` | `Sigma*` factor (autograd Wilson B rows, eigendecomposition, cache) |
| `src/mdwm/models/mehf.py` → `_harmonic_local_noise` | the prior inside `_structured_noise`, used by `_cfm_loss` and `sample` |
| `configs/sigma_star_full_f{1,2,3}.yaml` | `harmonic_local_noise: true`; identical otherwise to the ISO / ANISO configs |

## Headline result (test split, read once, pre-registered)

Median per-example four-channel spread (max − min over the generated/physical
width ratios of bond, angle, torsion, out-of-plane; lower = more uniform), at
matched torsional width, 1,410 examples / 60 UniProt targets never used for
training or selection:

| prior | spread | paired vs ISO, 95% UniProt-cluster bootstrap |
|---|---|---|
| isotropic (ISO) | 1.974 | — |
| anisotropic Gauss-Seidel (ANISO v3, unmatchable) | 2.527 | — |
| **Sigma\*** | **0.770** | **−1.152 [−1.370, −1.021]**, 98.8% of examples |

Pre-registration, amendments and the full result: `reference_outputs/sigma_star_test_v1/`.
Known limits are stated there and in `REPRODUCIBILITY.md`; in particular Sigma* is
over-dispersed in absolute terms (internal RMSF ratio 1.49) and it is the only arm
that degraded between the fitting folds (0.22–0.46) and test (0.77).

## Layout

```
src/mdwm/            model, data, training, evaluation code (package name kept from the parent project)
scripts/             every entry point used for a reported number
configs/             training configs of the 14 released checkpoints
outputs/             inputs the pipeline reads: released checkpoints (outputs/<arm>/full_f*/),
                     the MMP corpus, the MISATO MMP census, the frozen val/test example lists.
                     Regenerated results are written here too.
reference_outputs/   the frozen record: every result, FINDING, PREREGISTRATION and RESULT file
reproduce/           staged regeneration scripts and verify.py
tests/verify_b_rows.py   autograd vs finite-difference check of the Sigma* implementation
data/                MISATO frame store and third-party data (see "Data")
environment/         conda exports of the two environments used
```

## Install

```bash
conda env create -f environment/environment_mehf.yml       # env "MDWM": python 3.10, torch 2.12 (cu130), rdkit 2026.3.3
conda env create -f environment/environment_targetdiff.yml # env "TDIFF": only for re-sampling TargetDiff (stage 06)
export PYTHONPATH=$PWD/src:$PWD/scripts
```

Hardware used: 2 x NVIDIA RTX A6000, 64 CPU cores.

## Data

| path | what | source |
|---|---|---|
| `data/MISATO/extracted/frames100_all_v1/` | 16,536 ligand MD systems, 100 frames each, 8 Å pocket (7.1 GB) | built from MISATO `MD.hdf5` (Zenodo record 7711953, DOI 10.5281/zenodo.7711953) by `scripts/misato_store_pipeline.sh`; manifest and hashes in `data/MISATO/manifest/`, build notes in `data/MISATO/README.md` |
| `outputs/misato_mmp_census_v1/` | matched-molecular-pair census of MISATO | `scripts/misato_census_*.py` (queries RCSB) |
| `outputs/mmp_transfer_corpus_v1/` | 4,913 pairs / 517 targets, atom mappings | `scripts/build_mmp_ensemble_transfer_corpus.py` |
| `data/external/diffsbdd/` | DiffSBDD released samples (`crossdocked_fullatom_cond`) | DiffSBDD authors' release |
| `data/external/crossdocked/` | CrossDocked test set, 100 reference ligands, `pockets10/` | CrossDocked2020; `pockets10/` from `scripts/targetdiff/make_pockets10.py` |
| `data/external/targetdiff/` | TargetDiff code + pretrained weights (MIT), our 100-pocket sample | TargetDiff authors' release; sample from `reproduce/06_targetdiff_sampling.sh` |

The third-party data are bundled so this folder runs offline. Check each
source's licence before redistributing them; `.gitignore` excludes them.

## Reproduce

```bash
bash reproduce/run_all.sh          # stages 00, 01, 03, 04, 05, then verify.py
```

| stage | what | cost (observed) |
|---|---|---|
| `00_split_lists.sh` | val / test example lists | seconds |
| `01_model_free.sh` | Wilson B theory, isotropic-noise cost, Sigma* control, tau measurements, section-1 evidence | ~45 min CPU |
| `02_train.sh` (optional) | retrain the 14 checkpoints into `outputs_retrain/` | ~70 min each, 1 GPU |
| `03_spectra.sh` | all 173 model evaluations (fitting grids, val, test, ablations) | ~3 h, 2 GPUs (`GPUS=0,1 LANES=6`) |
| `04_decisions.sh` | matched-torsion operating points; val and test decision rules; loss-vs-projection effects; zero-parameter isotropy test | ~1 min |
| `05_published_models.sh` | DiffSBDD / TargetDiff isotropy tests and corrector gates (2,000 molecules x 5 seeds each) | ~20 min CPU |
| `06_targetdiff_sampling.sh` (optional) | re-sample TargetDiff on the 100 pockets | ~10 GPU-h |

`python reproduce/verify.py` compares everything under `outputs/` with
`reference_outputs/` and reports IDENTICAL / WITHIN_TOL / DIFFERENT / NOT_REGENERATED.
Expected after `run_all.sh`: DIFFERENT 0 and NOT_REGENERATED 0 (on the audit machine:
108 identical, 330 within tolerance). Model evaluations match to about 1e-6
relative (float32 GPU reduction order), everything else byte-for-byte. Superseded
and exploratory records are listed separately as historical; see `REPRODUCIBILITY.md`.

## Splits and leakage controls

- Split by UniProt target (70/15/15 by pair count, seed 20260918): train 3,489
  pairs / 345 targets, val 696 / 106, test 728 / 66. No UniProt target and no PDB
  entry is shared between splits.
- All 14 checkpoints train on `train` only; checkpoint selection uses a grouped
  dev fold inside `train`. Operating points are chosen on fitting folds.
- `val` and `test` were each evaluated once, after hashed pre-registrations
  (`reference_outputs/pareto_v1/`, `reference_outputs/sigma_star_test_v1/`).
- Disclosed: the four physical widths tau were measured on random MISATO
  systems, 6–9% of which belong to val/test; excluding them moves tau by at most
  0.4%. 7.5% of test ligand systems have a chemically identical ligand in train
  (different protein); excluding them leaves the test verdict unchanged. Details
  in `REPRODUCIBILITY.md`.

## Before public release

- choose a licence for this code;
- decide whether to rename the `mdwm` package to `mehf` (checkpoints store only
  tensors and config dicts, so a rename does not break them);
- replace bundled third-party data with download instructions if their licences
  do not allow redistribution.
