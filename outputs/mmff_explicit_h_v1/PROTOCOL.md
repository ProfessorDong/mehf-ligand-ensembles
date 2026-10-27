# Pre-specified protocol: MMFF94 baselines rerun with explicit hydrogens

Written 2026-10-13. This file was written and hashed into `PROTOCOL.sha256` before any
corrected (explicit-H) arm was run at scale.

Disclosure: before this file was written, a 30-molecule smoke test was run on DiffSBDD
(seed 7) only to check that the code executes. Only the Minimize return codes were looked
at (all 0). No metric from the smoke test was inspected.

## Defect being corrected

The audit (`MEHF_TCBB_Liang/audit/B_numbers_IVE_IVF.md` §3) found a defect in the
`tab:ff` and `tab:gates` MMFF arms. They run RDKit MMFF94 on hydrogen-stripped ligands:
`removeHs=True`, no `AddHs`, a restraint on every heavy atom, `Minimize(maxIts=2000)`,
and the convergence flag is never checked. MMFF94 atom typing and its energy terms
assume explicit hydrogens, so this baseline is physically invalid.

## Step 0 (already done; it passed before this file was written)

The original `e_ff_restraint_sweep.py` was run unchanged on a scratch copy. The only edit
was the path fix needed to import the byte-identical copy of `geometry_corrector_gate.py`
(sha256 `ab25b456...718b`). It reproduces `reference_outputs/paper_evidence_v1/e_ff_restraint_sweep.json`
on all 8 arms and all fields, to within floating-point rounding (relative difference about
1e-15).

**Data defect found during Step 0.** In the extracted directory
`MEHF/data/external/crossdocked/test_set/`, 35 of the 200 files are zero bytes, including
some `*_rec.pdb` receptors and some reference `.sdf` files. The original sweep crashes on
them. The zip is intact (`unzip -t` reports no errors; sha256 `e89c5890...24fb`).
Therefore every run here uses a fresh extraction of `test_set.zip` into a scratch
directory. Nothing under `MEHF/data` is modified. With the clean extraction, Step 0
reproduces exactly. The gate runs below must also reproduce the reference records for the
original arms, which serves as the check on the receptor data.

## Corrected protocol (the ONLY change is inside the MMFF arms)

The following are identical to the original:
* the molecule files;
* the reading step (`SDMolSupplier(removeHs=True, sanitize=True)`);
* the selection rules, including the original rule that a molecule is kept only if
  heavy-atom MMFF setup succeeds. This keeps the molecule set identical to the
  manuscript's;
* the reference split;
* the metric code (`geometry`, `rms_deviation`, `distributional`, `summarise`, the RMSD
  definition, and clashes against receptor heavy atoms below 2.0 Å);
* the restraint values.

Each MMFF arm is changed as follows (`code/mmffh.py: relax_explicit_h`):
1. `Chem.AddHs(mol, addCoords=True)` is applied to the heavy-atom molecule.
2. `MMFFGetMoleculeProperties` (default MMFF94) and `MMFFGetMoleculeForceField` are built
   on the H-complete molecule. If either returns None, or AddHs raises an error, the
   molecule is recorded as **failed** for that arm, and the failure count is reported.
3. For k > 0, `MMFFAddPositionConstraint(i, 0.0, k)` is applied to **heavy atoms only**.
   Hydrogens are free.
4. `Minimize(maxIts=2000)` is called, and its return code is **recorded**: 0 means
   converged, 1 means not converged.
5. `Chem.RemoveHs` is applied. The script checks that the heavy-atom count and the element
   order are identical to the input. A mismatch is recorded as a failure.
6. The result is evaluated with the original code on the heavy-atom coordinates.

As in the original, the ligand is minimized **alone, in vacuo, with no receptor atoms**.
The correction does not change this, and it remains a limitation of the MMFF baseline.
Clashes are still counted afterward against the receptor.

The non-MMFF arms (raw, projection D, projection + out-of-plane E, and the soft target F)
are recomputed with the original code and are not changed.

The original heavy-atom arms are recomputed alongside the new arms in the same run. This
gives a side-by-side comparison and a byte-level reproduction check against the reference
records. Their convergence codes are also recorded.

## Runs

(a) **tab:ff sweep.** DiffSBDD, 200 molecules, max 2 per pocket, split seed 20260924.
Reference dir and receptor root are the clean test_set. k ∈ {0, 0.2, 1, 5, 20, 100,
1000}. Script: `code/sweep_explicit_h.py`.

(b) **tab:gates.**
* DiffSBDD (`data/external/diffsbdd/crossdocked_fullatom_cond`) and TargetDiff
  (`data/external/targetdiff/samples_sdf`).
* `--max-molecules 2000 --max-per-pocket 20`, holdout split.
* Reference dir `data/external/crossdocked/ref_flat`; receptor root is the clean test_set.
* Split seeds 20260924, 1234, 7, 31337, 99.
* Explicit-H MMFF arms: free (HB), k=5 (HG), k=100 (HC). Script: `code/gate_explicit_h.py`.

The reported table value is the **mean over the five splits of each per-split value**:
* the median over molecules, for Angle (angle_rms), Plan., and RMSD;
* the pooled class-weighted W1, for W1.

This was confirmed before this file was written: those means of the reference records
reproduce the printed tab:gates entries (for example, DiffSBDD MMFF free gives 5.482 /
1.695 / 0.629 / 1.579).

## Pre-specified rules

* **Failures.** A molecule whose explicit-H MMFF setup fails stays in the run, because
  selection is unchanged. It contributes no rows to that explicit-H arm, so the arm is
  summarized over the molecules for which it succeeded. The n is reported. If any arm has
  more than 0 failures, a sensitivity analysis is also reported with all arms restricted
  to the common subset.
* **Non-convergence.** The fraction of molecules with Minimize code 1 is reported per arm.
  Non-converged molecules are **kept** in the primary numbers, as in the original, which
  also kept them. If more than 5% of molecules fail to converge at maxIts=2000 in any
  explicit-H arm of a run, that run is repeated with an extra variant at maxIts=10000 for
  k=0 and k=5 only (`--extra-maxits 10000`). The variant is reported as a sensitivity
  analysis. The primary maxIts=2000 numbers are not replaced.
* **Reporting.** All numbers are reported, whichever way they land. There is no
  cherry-picking of seeds, restraints, or subsets.

## Questions that RESULT.md must answer (stated in advance)

Q1. Does unrestrained explicit-H MMFF still move ligands substantially? The original was
1.675 Å (sweep), 1.58 Å (DiffSBDD gate), and 1.34 Å (TargetDiff gate). Does it still
raise the clash rate above raw (sweep: 17% raw vs 68%)?

Q2. Does the restraint trade-off persist? Increasing k should monotonically reduce RMSD
and clash rate while giving up planarity and W1 angle, with an interior optimum of
W1 angle near k=5.

Q3. Is projection's W1 angle still worse than MMFF's at k=0 and k=5? Projection has 3.33
(DiffSBDD) and 3.14 (TargetDiff), against MMFF free at 1.70 / 1.89 and k=5 at 1.63 / 1.80.
Is projection's Angle (RMS vs. fitted ideal) still far better than MMFF's?

Q4. Planarity. Does MMFF still flatten the aromatic rings (free, about 0.6–0.8°)? And is
projection + out-of-plane (0.32 / 0.30°) still below every MMFF arm?
