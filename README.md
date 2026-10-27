# MEHF: ligand conformational ensembles by flow matching with chemistry-aware noise

Code, trained checkpoints, and the complete result record for

> T. Ding, D. Xiang, and L. Dong, "Generating Ligand Conformational Ensembles by Flow Matching
> with Chemistry-Aware Noise," submitted to *IEEE Transactions on Computational Biology and
> Bioinformatics*.

MEHF (Mapped-Edit Hierarchical Flow) is an E(3)-equivariant conditional flow-matching model. It
generates a ligand's conformational ensemble in its protein pocket. The model is conditioned on the
atom-mapped molecular-dynamics (MD) trajectory of a matched-molecular-pair analogue, and it samples
coarse to fine.

The paper replaces the local part of the source distribution with a Wilson-matrix harmonic prior,
Σ\* = (BᵀFB)⁺. Here F holds separate empirical widths for bond lengths, bond angles, proper torsions
and out-of-plane coordinates. The sampled source is the truncated factor of Σ\*, rescaled to the
realized amplitude of an isotropic draw, so it is not exactly Gaussian.

Data: MISATO protein–ligand MD (Siebenmorgen et al., 2024; Zenodo record
[7711953](https://zenodo.org/records/7711953), CC BY 4.0).

## Two protocols, one repository

| | Original protocol (paper Tables II–III) | Clean-protocol replication (paper Section IV-D, Table IV) |
|---|---|---|
| Where | repository root | [`cleanrun/`](cleanrun/README.md) |
| Pocket atoms | 96 nearest the ligand by minimum distance over all 100 target frames; frame-1 coordinates | 96 nearest the ligand in frame 1 |
| Widths τ | 0.0326 Å, 2.77°, 14.6°, 6.48° (samples included some validation and test systems) | re-estimated on 2,330 training-split systems |
| Arms | MEHF-ISO, -GS, -LAP, -HARM | MEHF-ISO, -LAP, -HARM, plus two model-free controls |
| Test examples | 1,410 (60 targets) | 1,406 (59 targets); 28 systems excluded |

The `cleanrun/` code is the root code with two options added (`pocket_selection`, `harmonic_tau_set`).
Their defaults reproduce the original protocol. `cleanrun/` sits inside this checkout and reaches the
shared data and records through relative symlinks.

## Results, as pre-registered

Medians of the per-example channel spread S (max − min of the four generated/reference width ratios;
lower means more balanced) on held-out test targets, with projection on and torsion-matched operating
points. Paired differences are means with 95% UniProt-target bootstrap intervals.

| | ISO | GS | LAP | HARM | HARM − ISO |
|---|---|---|---|---|---|
| Original protocol | 1.974 | 2.527\* | 1.352 | **0.770** | −1.152 [−1.370, −1.021] |
| Replication, fitting-fold s\* | 1.902 | — | 1.382 | **0.796** | −1.035 [−1.263, −0.918] |
| Replication, development-fold s\* | 1.904 | — | 1.420 | **0.872** | −0.940 [−1.169, −0.840] |

\*MEHF-GS could not be torsion-matched.

The paper reports these together with what they do not show:
- **Balance, not accuracy.** On the four-channel absolute log width error, HARM is not better than ISO.
- **Over-dispersion.** HARM is over-dispersed, with an internal RMSF ratio of 1.49–1.75.
- **Spread alone is not evidence of ensemble quality.** Copying the analogue's motion and applying the
  same projection, with no model, reaches S = 0.836.
- **Four test reads.** The test split has been read four times, each under its own hashed
  pre-registration, so nominal intervals overstate confidence. The clean MEHF-LAP arm was trained after
  the clean ISO/HARM results were known.

Records of each read:
- the original read: [`reference_outputs/sigma_star_test_v1/`](reference_outputs/sigma_star_test_v1/)
- the Laplacian read: [`reference_outputs/laplacian_prior_v1/`](reference_outputs/laplacian_prior_v1/)
- the two replication reads: [`cleanrun/frozen/`](cleanrun/frozen/) and [`cleanrun/expected/`](cleanrun/expected/)

## Layout

```
src/mdwm/           model, data loading, training, evaluation (package name kept from the parent project)
scripts/            every entry point behind a reported number
configs/            training configs of the released original-protocol checkpoints
outputs/            pipeline inputs (MMP corpus, census, example lists) and checkpoints (outputs/<run>/full_f*/)
reference_outputs/  frozen result record: PREREGISTRATION, RESULT, FINDING files and per-example CSVs
reproduce/          staged scripts that regenerate the original-protocol results
tests/              autograd vs finite-difference check of the Wilson B rows
environment/        conda environment exports
data/               MISATO store build notes; the store itself is not committed (see data/README.md)
cleanrun/           the clean-protocol replication (own README, code, configs, checkpoints, records)
paper_analysis/     post-hoc analyses behind the paper's supplementary material
docs/               the original run's README, kept as a record
CHECKSUMS.sha256    hashes of the original-protocol tree, including the data store
```

## Install

```bash
conda env create -f environment/environment_mehf.yml   # env "MDWM": Python 3.10, PyTorch 2.12
conda activate MDWM
export PYTHONPATH=$PWD/src:$PWD/scripts                # for cleanrun/, its reproduce scripts set this
```

`environment/environment_targetdiff.yml` is only needed to re-sample TargetDiff (reproduce stage 06).

## Data

See [`data/README.md`](data/README.md). In short:
- The MISATO frame store (7.1 GB, 16,536 systems) is rebuilt from MISATO's `MD.hdf5` by
  `scripts/misato_store_pipeline.sh`.
- `sha256sum -c CHECKSUMS.sha256` verifies the rebuilt store, along with every other file of the
  original record.
- The per-example results, splits, matched pairs and atom maps are committed here.
- Third-party data (CrossDocked pockets, DiffSBDD and TargetDiff samples) are not redistributed.

## Reproducing

- **Original protocol.** `reproduce/run_all.sh` runs the stages in order; `reproduce/common.sh` documents
  the settings.
- **Replication.** Follow [`cleanrun/README.md`](cleanrun/README.md). Its steps 1–2 reproduce
  bit-for-bit in minutes without a GPU.
- **GPU nondeterminism.** Retraining is not bit-reproducible because GPU reductions are nondeterministic.
- **Per-example values depend on the batch.** The sampler seeds each batch of four examples from its
  first member, so a single example's values depend on the example list, its order and the batch size.
  Compare summary statistics unless all three are identical.

## Known discrepancies between code comments and the paper

The code is published as it ran. Where its comments and docstrings disagree with the paper, the paper
is correct:
- **"Exact Σ\*".** The sampled source is the truncated, amplitude-rescaled factor α·Lξ, not exactly
  N(0, Σ\*).
- **"SE(3)-equivariant".** The network is E(3)-equivariant.
- **Null modes.** Comments in `harmonic_prior.py` (lines 14–22) and `scripts/exact_sigma_star_noise.py`
  say pseudoinverse null modes get unbounded amplitude. Discarded modes receive zero variance.
- **Centroid.** `mehf.py` says the projection preserves the centroid exactly. Because of the per-atom
  divisor, it does not.
- **"Gauss–Seidel".** The GS sampler updates all bonds, then all angles, simultaneously within each
  sweep; it is not Gauss–Seidel in the strict sense. In addition, κ and s_i scale only its initial
  isotropic draw.
- **Null-mode rate.** Comments citing 4.7% of ligands with extra null modes come from 150 ensemble-mean
  structures. At first-frame poses the rate is 7 of 3,819 corpus ligands (0.18%); see
  `paper_analysis/null_modes_v1/`.

## License

- **Code:** MIT; see [LICENSE](LICENSE).
- **Data derived from MISATO** (per-example metrics, manifests, splits): CC BY 4.0, as MISATO itself.
  Please cite MISATO.
- **Third-party models and data:** used under their own licenses and not redistributed.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite MISATO:

> T. Siebenmorgen et al., "MISATO: machine learning dataset of protein–ligand complexes for
> structure-based drug discovery," *Nat. Comput. Sci.*, vol. 4, pp. 367–378, 2024.
