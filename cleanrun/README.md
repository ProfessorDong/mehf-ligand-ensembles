# cleanrun — reproducing the clean-protocol supplementary experiments

Everything needed to reproduce the MEHF clean-protocol run of 2026-10-23/25:
the pocket leak fix, the train-split-only widths, the three retrained arms, the
two model-free controls, and the two test reads.

Nothing here is a copy of the main results. The original (leaky-protocol) run
lives in `../reference_outputs/` and is untouched by any of this.

---

## 0. What you need that is not in this folder

- **conda env `MDWM`.** There is no env called `MDWM_GPU`; using that name fails
  silently and produces no output. Captured versions: Python 3.10.12,
  torch 2.12.0+cu130 (CUDA 13.0), numpy 2.2.6.
- **A GPU with ~12 GB free.** Peak is 11.8 GB per training run. Three concurrent
  runs fit on a 48 GB card; four do not, reliably.
- **A stock MEHF checkout** — the folder as it was *before* this supplementary
  task. `cleanrun` is not fully standalone: it needs that tree for the MISATO
  store (7.6 GB) and for the corpus and reference inputs the scripts read
  (`outputs/mmp_transfer_corpus_v1/pairs.csv`,
  `reference_outputs/isotropic_noise_cost_v1/`, `reference_outputs/theory_modules_v1/`).
  By default `cleanrun` expects to sit **inside** that checkout, at `MEHF/cleanrun/`,
  and resolves everything by relative symlinks. Nothing it runs writes into the
  stock tree: regenerated output goes to `cleanrun/code/outputs/clean_protocol_v1/`.

  The code, however, must be **cleanrun's, not the stock tree's**. The stock
  `src/` has no `pocket_selection` and would silently use the leaky
  over-all-frames pocket crop. `reproduce/_common.sh` sets `PYTHONPATH`
  accordingly and `preflight` refuses to start if the wrong loader is importable,
  naming the file it found.

## 1. Layout

| Path | What it is |
|---|---|
| `code/src`, `code/scripts` | the code as it stood for this run, including the `pocket_selection` option, the train-only tau set and the exclusion handling |
| `configs/` | `clean_f{1,2,3}_{iso,harm,lap}.yaml` — the nine training configs |
| `inputs/data` | symlink to the MISATO store (16,536 npz) |
| `inputs/excluded_systems.json` | the 28 excluded systems and the derivation |
| `inputs/split_counts_clean.json` | post-exclusion split counts |
| `inputs/tau_train_only_large.json` | the four widths re-estimated on 2,330 training-split systems |
| `inputs/folds/` | the regenerated per-fold fit and dev example lists |
| `inputs/test_examples_clean.csv` | the 1,452-row test list (1,406 usable after the planar-centre filter) |
| `frozen/` | the pre-registration, its hash history, `LAP_DECISION.txt`, and the frozen s* JSONs |
| `checkpoints/` | the nine trained checkpoints (iso/harm/lap × folds 1–3) |
| `expected/` | what this run produced: `RESULT.md`, `TEST_RESULT.json`, the LAP equivalents, all per-example CSVs, and the B2 clash outputs |
| `logs/` | every log, including `CHAIN.log` — the chain of custody, read this first |
| `reproduce/` | numbered scripts, below |

## 2. Order

Each script prints what it is about to do and refuses to clobber existing
output. `reproduce/00_verify_inputs.sh` is safe to run at any time.

| Step | Script | Cost | Needs GPU |
|---|---|---|---|
| 0 | `00_verify_inputs.sh` | seconds | no |
| 1 | `01_exclusions.sh` | ~3 min | no |
| 2 | `02_tau.sh` | seconds | no |
| 3 | `03_train.sh` | ~15 GPU-hours | yes |
| 4 | `04_grid_and_sstar.sh` | ~3 h | yes |
| 5 | `05_controls.sh` | ~1 h | light |
| 6 | `06_test_read.sh` | ~1.5 h | yes |
| — | `verify.py` | ~1 min | no |

Steps 1 and 2 reproduce exactly and are the cheapest way to confirm the setup is
sound. Step 3 is the expensive one; if you only want to check the *analysis*,
skip it and use `checkpoints/`.

## 3. What reproduces exactly, and what does not

**Exactly, bit-for-bit:** the exclusion list, the split counts, the widths, and
every decision re-applied to stored per-example data.

**To float32 GPU reduction order (~1e-7):** model evaluations re-run from the
released checkpoints, *provided* you keep the same example list, in the same
order, at the same batch size.

**Not at all, and this is the trap:** individual per-example rows, if you change
the list, reorder it, or change `--batch-size`. The sampler seeds one generator
per batch of four from the **first example's key in that batch**, so an example's
noise draw depends on which example leads its batch and on the other members'
atom counts. Dropping a single row shifts every later batch boundary. Measured:
unchanged batches reproduce to 4.6e-7 and a same-settings re-run to 7e-7, but
batch sizes 3 and 1 disagree with batch size 4 by a median of 0.04–0.09 and a
maximum of 0.8 on the width ratios. Medians over the full list are stable to
within 0.02.

So: **compare summary statistics, not individual rows**, unless you are running
the identical list at the identical batch size. Retraining will not reproduce
the checkpoints bit-identically either — GPU training uses non-deterministic
reductions.

## 4. The test reads

`06_test_read.sh` is gated deliberately. This test split has now been read
**four** times: the original confirmatory read, the added Laplacian arm, the
clean ISO/HARM read, and the clean LAP read. Each was governed by a
pre-registration written and hashed beforehand.

`frozen/PREREGISTRATION.md` has sha256
`3863992a30606022fa11736e86104faa4bcf6c6d36d76273a9f678e4ae9f4daf`. Draft 1,
hashed before any training began, is still verifiable inside it:

```
head -638 frozen/PREREGISTRATION.md | head -c -1 | sha256sum
# a51668b7af5d8ba9ec4243597fbcf1046ba9e72bebd7ee6c661cad5590ff322e
```

`frozen/LAP_DECISION.txt` records, in its own words, that the decision to train
the Laplacian arm was taken *after* the ISO/HARM test numbers were known.

## 5. Headline, for checking you got the same thing

On 1,406 usable examples from 59 targets, at the primary (fitting-fold)
operating point:

| Arm | Spread S | psi | internal RMSF |
|---|---|---|---|
| MEHF-ISO | 1.902 | 1.169 | 0.41 |
| MEHF-LAP | 1.382 | 0.733 | 0.94 |
| MEHF (Sigma*) | 0.796 | 0.539 | 1.54 |

Sigma* − ISO = −1.0345, 95% CI [−1.263, −0.918], lower on 98.1% of examples.
Sigma* − LAP = −0.531 [−0.675, −0.386].

Blocking conditions B2 and B4 do not fire. B1 and B3 fire at the secondary
operating point only. The headline tier is **A (attenuated)** under the
pre-registration's downgrade-only rule.

---

## 6. Frozen artifacts and their hashes

| File | sha256 (first 16) | What it is |
|---|---|---|
| `frozen/PREREGISTRATION.md` | `3863992a30606022` | final, governed both test reads |
| (draft 1, inside it) | `a51668b7af5d8ba9` | hashed **before any training**; recover with the `head -638` recipe above |
| `frozen/LAP_DECISION.txt` | `f9bcc97d18dad82e` | records that the LAP arm was chosen *after* the ISO/HARM numbers were known |
| `frozen/s_star_frozen_v2.json` | `dabcd5f38db88b6b` | the operating points used for the ISO/HARM read |
| `frozen/s_star_frozen_v1_superseded.json` | `f92d7f95722f8786` | the earlier freeze on the grid to 4.0, kept because Amendment 7 rests on comparing the two |
| `frozen/s_star_frozen_lap.json` | `44d4307807d01d52` | the operating points used for the LAP read |

The Amendment 7 argument is checkable from these two freezes alone: all eight
cells that already matched on the shorter grid are **bit-identical** in the
longer one, and each nine-point torsion curve has the old seven-point curve as
an exact prefix. The grid extension only added measurements; it did not move a
crossing.

## 7. Known gaps

- The Gauss–Seidel arm was not retrained under this protocol, so it has no entry
  here.
- `04_grid_and_sstar.sh` regenerates the ISO/HARM freeze. The LAP freeze comes
  from `code/scripts/clean_protocol_sstar_lap.py`, run the same way.
- `06_test_read.sh` drives one lane as written. The full read is three folds ×
  two operating points; see `code/scripts/clean_protocol_test_lane.sh` and
  `clean_protocol_lap_test_lane.sh` for how the lanes were driven.

---

## 8. Giving this to someone else

Hand them **both**: a stock MEHF checkout *and* this `cleanrun` folder placed
inside it as `MEHF/cleanrun/`. Then:

```
cd MEHF/cleanrun
PY=/path/to/mdwm-env/bin/python ./reproduce/00_verify_inputs.sh
PY=/path/to/mdwm-env/bin/python ./reproduce/01_exclusions.sh
PY=/path/to/mdwm-env/bin/python ./reproduce/02_tau.sh
```

Steps 1 and 2 take minutes, need no GPU, and reproduce **bit-for-bit** — they
are the cheapest proof that the setup is sound before committing GPU time.

### What has actually been tested, and what has not

Be aware of the difference before you promise anyone a clean run.

| | Status |
|---|---|
| `00_verify_inputs.sh` | run, passes |
| `01_exclusions.sh` | run end to end; regenerates the same 28 systems, byte-identical |
| `02_tau.sh` | run end to end; regenerates the same four widths |
| evaluation chain (the inner command of steps 4 and 6) | smoke-tested on CPU with cleanrun's code, a cleanrun checkpoint and the stock store; output records `pocket_selection = frame0` and `harmonic_tau_set = train_only_large`, confirming the clean loader was used |
| `03_train.sh`, `04`, `05`, `06` in full | **not run from this folder.** The commands are the ones recorded in `code/scripts/as_run/`, which did produce `expected/`, but the portable wrappers have not been executed end to end |

All of the above was tested on the machine the run happened on. It has not been
tested on a second machine or against a freshly downloaded MEHF checkout.

Steps 3–6 need a GPU. Step 3 can be skipped entirely by using the nine
checkpoints in `checkpoints/`.

If `cleanrun` must live somewhere other than inside the MEHF checkout, set
`MEHF_DATA=/path/to/MEHF/data` and repoint the read-only symlinks under
`code/outputs/` and `code/reference_outputs/` to that checkout.

### If something looks wrong

- `WRONG CODE ON THE PATH` — the guard did its job; `PYTHONPATH` is being
  overridden somewhere. Unset it and re-run.
- `store has N npz, expected 16536` — `MEHF_DATA` points at the wrong tree, or
  the store is incomplete.
- Per-example rows differ but medians agree — expected; see section 3. Compare
  with `reproduce/verify.py`, which compares summary statistics by design.
