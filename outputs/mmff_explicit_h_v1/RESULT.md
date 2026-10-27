# Result: MMFF94 baselines rerun with explicit hydrogens

Run on 2026-10-13. Everything ran on CPU, with RDKit 2026.03.3 in the MEHF conda env.

The protocol was fixed in advance in `PROTOCOL.md`. Its sha256 and the sha256 of each
script are in `PROTOCOL.sha256`, and both were written before the corrected arms were run.
All code is in `code/`. The runs used scratch copies whose sha256 values are identical to
those files.

Nothing under `MEHF/` outside this directory was modified.

## 0. Reproduction checks (all passed)

| Check | Result |
|---|---|
| Original `e_ff_restraint_sweep.py`, run unchanged (path fix only), vs. `reference_outputs/paper_evidence_v1/e_ff_restraint_sweep.json` | n = 200 in both. All 8 arms and all fields agree to a relative difference of about 1e-15, so every printed digit is identical. |
| Original heavy-atom arms recomputed inside the corrected sweep run (`sweep/sweep_explicit_h.json`) vs. the same reference JSON | max rel. diff 2.9e-15, 0 mismatches |
| Original gate arms A–G recomputed inside the corrected gate runs vs. `reference_outputs/corrector_gate_v1/scaleup_with_r5/` (DiffSBDD) | 5 seeds, 245 fields, max rel. diff 9.8e-13, 0 mismatches, n = 2000 |
| Same check vs. `reference_outputs/published_models_v1/targetdiff/` (TargetDiff) | 5 seeds, 245 fields, max rel. diff 1.2e-12, 0 mismatches, n = 1811 |

The table quantity is the mean over the five splits of each per-split value:
* the median over molecules, for Angle, Plan., and RMSD;
* the pooled class-weighted W1, for W1.

This is confirmed: those means of the reference records reproduce every printed
`tab:gates` entry.

**Data defect found in Step 0.** It must be fixed in the package, though not by this run.
* In `MEHF/data/external/crossdocked/test_set/`, 35 of 200 files are zero bytes. These
  include some receptor `*_rec.pdb` files and some reference ligand `.sdf` files.
* With this directory, the original sweep crashes (`OSError: Invalid input file`). The gate
  would also silently drop receptors, which changes the clash statistics.
* `test_set.zip` itself is intact (`unzip -t` reports no errors).
* All runs here use a fresh extraction of the zip in a scratch directory. With that
  extraction, everything reproduces exactly.
* `sha256sum -c MEHF/CHECKSUMS.sha256` independently flags the same 35 files.
* The checksum check also reports 1,609 mismatches under `data/external/targetdiff/samples/`
  and 21,735 under `data/MISATO/manifest/`. Neither is used by these runs. The TargetDiff
  gate reads `targetdiff/samples_sdf/`, and its exact reproduction (below) confirms that
  this input is intact.
* The `paper/` files listed in the checksum file are missing.
* Recommendation: re-extract `test_set.zip` in the package, and investigate the other
  checksum failures before archiving.

## 1. What changed

The MMFF arms are the only change, in `code/mmffh.py: relax_explicit_h`. Each arm runs these
steps:
1. `AddHs(addCoords=True)`.
2. MMFF94 setup on the H-complete molecule.
3. A harmonic position restraint k on **heavy atoms only**.
4. `Minimize(maxIts=2000)`, with the return code recorded.
5. `RemoveHs`, followed by a check that heavy-atom count and element order are unchanged.
6. Evaluation with the original metric code.

The molecules, the selection rule, the reference splits, the metrics, and the restraint
values are all identical to the original.

The ligand is still minimized **alone, in vacuo**, as in the original. The receptor enters
only through the clash count after minimization. This remains a limitation of the MMFF
baseline and should be stated in Methods.

## 2. Convergence and failures

| Run | Arm(s) | n | Minimize code 0 (converged) | Not converged | Failed |
|---|---|---|---|---|---|
| Sweep (200) | explicit-H, all 7 k | 200 | 200 | 0 (0%) | 0 |
| Sweep (200) | original heavy-atom, all 7 k | 200 | 200 | 0 (0%) | 0 |
| DiffSBDD gate (2000) | explicit-H free / k=5 / k=100 | 2000 | 1998 | 0 (0%) | **2** |
| DiffSBDD gate (2000) | original B / G / C | 2000 | 2000 | 0 (0%) | 0 |
| TargetDiff gate (1811) | explicit-H free / k=5 / k=100 | 1811 | 1811 | 0 (0%) | 0 |
| TargetDiff gate (1811) | original B / G / C | 1811 | 1811 | 0 (0%) | 0 |

Every arm converges within 2,000 iterations, so the 5% non-convergence trigger did not
fire. The pre-specified maxIts=10000 variant was therefore **not run**. The original
protocol's unchecked flag turns out to be harmless here, because every original minimization
also returned 0.

The 2 DiffSBDD failures are the same two molecules in every split:
* `3gs6-...-docked-4_gen.sdf` #5, SMILES `O=C1O[C@H]2[C@@H](COCC[C@@H](O)[P@SP1H]2=O)S1`;
* `4tqr-...-min-0_gen.sdf` #5, SMILES `O=[P@SP2H](OC[C@@H]1CCCN1CO)OOP(=O)(O)O`.

Both contain a P–H on a phosphorus with non-tetrahedral (SP) stereochemistry. RDKit's
`RemoveHs` refuses to strip that hydrogen, so the heavy-atom order check fails. MMFF setup
itself did not fail.

As pre-specified, these molecules contribute no rows to the explicit-H arms, which have
n = 1998. The common-subset sensitivity analysis (`gates/diffsbdd_common_subset/`)
restricts all arms to those 1,998 molecules. No table quantity changes by more than 0.004. The largest change is projection +
out-of-plane Angle, 1.2774 → 1.2806. All rounded table entries are unchanged, except that
the W1 for MMFF free, original (1.6952 → 1.6950) sits on a rounding boundary.

## 3. Table `tab:ff`: original (heavy-atom only) vs. explicit-H

DiffSBDD, 200 molecules, split seed 20260924. Column layout is the same as `tab:ff`:
RMSD (Å), W1 bond (Å), W1 angle (°), planarity (°), Clash (% of molecules with at least one
receptor heavy-atom pair below 2.0 Å).

| | RMSD orig | RMSD +H | W1 bond orig | W1 bond +H | W1 angle orig | W1 angle +H | Planarity orig | Planarity +H | Clash orig | Clash +H |
|---|---|---|---|---|---|---|---|---|---|---|
| Raw output | 0.000 | 0.000 | 0.0486 | 0.0486 | 5.33 | 5.33 | 4.14 | 4.14 | 17 | 17 |
| MMFF, restraint 0 | 1.675 | **1.433** | 0.0262 | **0.0219** | 1.81 | **1.60** | 0.57 | **0.98** | 68 | **65** |
| MMFF, restraint 0.2 | 0.923 | **0.958** | 0.0262 | **0.0218** | 1.79 | **1.60** | 0.94 | **1.40** | 38 | **44** (43.5) |
| MMFF, restraint 1 | 0.630 | **0.721** | 0.0262 | **0.0217** | 1.76 | **1.59** | 1.27 | **1.59** | 26 | **31** (30.5) |
| MMFF, restraint 5 | 0.399 | **0.463** | 0.0260 | **0.0211** | 1.66 | **1.63** | 2.14 | **2.54** | 20 (19.5) | **22** |
| MMFF, restraint 20 | 0.267 | **0.299** | 0.0251 | **0.0194** | 1.75 | **1.77** | 2.90 | **2.97** | 18 (17.5) | **18** |
| MMFF, restraint 100 | 0.153 | **0.175** | 0.0234 | **0.0166** | 2.43 | **2.33** | 3.17 | **3.41** | 17 | **17** |
| MMFF, restraint 1000 | 0.055 | **0.061** | 0.0203 | **0.0197** | 4.26 | **4.06** | 3.65 | **3.75** | 18 (17.5) | **18** (17.5) |

* Values in parentheses are the unrounded percentages. The manuscript rounds 19.5 to 20 and
  17.5 to 18.
* Supplementary quantities for the explicit-H runs:
  * Free-MMFF Kabsch-aligned internal RMSD is 1.201 Å (original 1.302 Å).
  * Free-MMFF mean clashes per molecule are 4.62 (original 5.12; raw 0.64).
  * Minimum receptor distances are in the JSON.

## 4. Table `tab:gates`: original vs. explicit-H

Values are the five-split means. Column layout is the same as `tab:gates`: Angle (°, RMS
deviation from fitted ideal), W1 (°, angle Wasserstein), Plan. (°), RMSD (Å). The
non-MMFF rows are recomputed and identical to the original.

**DiffSBDD (2,000; explicit-H MMFF rows n = 1,998)**

| Correction | Angle | W1 | Plan. | RMSD |
|---|---|---|---|---|
| Raw output | 12.13 | 5.79 | 4.58 | 0.00 |
| MMFF, free (orig, heavy-atom only) | 5.48 | 1.70 | 0.63 | 1.58 |
| **MMFF, free (explicit H)** | **5.62** | **1.58** | **1.08** | **1.44** |
| MMFF, restraint 100 (orig) | 7.63 | 2.73 | 3.67 | 0.16 |
| **MMFF, restraint 100 (explicit H)** | **7.41** | **2.54** | **3.71** | **0.18** |
| MMFF, restraint 5 (orig) | 5.75 | 1.63 | 2.08 | 0.40 |
| **MMFF, restraint 5 (explicit H)** | **5.85** | **1.63** | **2.26** | **0.46** |
| Projection | 1.16 | 3.33 | 10.12 | 0.23 |
| Projection + out-of-plane | 1.28 | 3.27 | 0.32 | 0.23 |

**TargetDiff (1,811)**

| Correction | Angle | W1 | Plan. | RMSD |
|---|---|---|---|---|
| Raw output | 9.63 | 4.56 | 1.76 | 0.00 |
| MMFF, free (orig) | 5.36 | 1.89 | 0.83 | 1.34 |
| **MMFF, free (explicit H)** | **5.41** | **1.71** | **1.35** | **1.31** |
| MMFF, restraint 100 (orig) | 7.11 | 2.86 | 2.77 | 0.12 |
| **MMFF, restraint 100 (explicit H)** | **6.88** | **2.52** | **3.24** | **0.15** |
| MMFF, restraint 5 (orig) | 5.51 | 1.80 | 2.09 | 0.35 |
| **MMFF, restraint 5 (explicit H)** | **5.58** | **1.74** | **2.43** | **0.43** |
| Projection | 1.17 | 3.14 | 7.90 | 0.20 |
| Projection + out-of-plane | 1.23 | 3.09 | 0.30 | 0.20 |

Additional gate quantities, not in the table: the five-split mean of the fraction of
molecules with at least one receptor clash, and W1 bond (Å).

| | DiffSBDD clash % | DiffSBDD W1 bond | TargetDiff clash % | TargetDiff W1 bond |
|---|---|---|---|---|
| Raw | 18.5 | 0.0573 | 7.0 | 0.0350 |
| MMFF free orig / +H | 69.8 / 65.8 | 0.0229 / 0.0160 | 56.3 / 56.2 | 0.0227 / 0.0162 |
| MMFF k=5 orig / +H | 22.4 / 25.2 | 0.0227 / 0.0153 | 9.6 / 12.0 | 0.0225 / 0.0158 |
| MMFF k=100 orig / +H | 20.2 / 20.6 | 0.0215 / 0.0150 | 7.3 / 7.5 | 0.0228 / 0.0150 |
| Projection / + out-of-plane | 19.4 / 19.4 | 0.0244 / 0.0241 | 7.4 / 7.4 | 0.0242 / 0.0240 |

The split-to-split SD of the explicit-H MMFF entries is ≤ 0.10° for Angle and ≤ 0.32° for
W1. Plan. and RMSD do not vary across splits for the MMFF arms, because those coordinates do
not depend on the split. Per-split SDs are in `SUMMARY.json`.

## 5. Do the manuscript's qualitative conclusions change?

**Q1. Yes in magnitude, no in substance.** Unrestrained explicit-H MMFF still moves
ligands substantially, but less than reported:
* sweep 1.433 Å instead of 1.675 Å;
* DiffSBDD gate 1.44 Å instead of 1.58 Å;
* TargetDiff gate 1.31 Å instead of 1.34 Å.

Most of that displacement (1.20 Å) survives alignment. The clash rate is still raised
about 4×: 17% → 65% in the sweep (originally 68%), 18.5% → 65.8% for DiffSBDD, and
7.0% → 56.2% for TargetDiff.

**Q2. Yes in part.** As k increases, RMSD and clash rate still fall monotonically, and
planarity is still progressively given up (0.98° → 3.75°), so the restraint trade-off
persists. However, the W1-angle "sweet spot" at k = 5 largely disappears. With explicit
hydrogens, W1 angle is flat at 1.59–1.63° for k = 0 to 5 (minimum 1.592° at k = 1) and only
degrades beyond k = 20. Originally, W1 angle improved from 1.81° to 1.66° between k = 0 and
k = 5. Any manuscript sentence implying that a moderate restraint improves the angle
distribution over free MMFF should be dropped. What survives is that k ≈ 5 keeps nearly
free-MMFF angle quality at about one third of the displacement.

**Q3. No change.**
* Projection's W1 angle is still clearly worse than every low-k MMFF arm: DiffSBDD 3.33
  vs. 1.58 (free) and 1.63 (k = 5); TargetDiff 3.14 vs. 1.71 and 1.74.
* Projection's RMS deviation from the fitted ideal angles is still far lower: 1.16–1.28°
  vs. 5.4–7.4° for MMFF.
* Projection's displacement is still smaller than free or k = 5 MMFF: 0.20–0.23 Å vs.
  0.43–1.44 Å.

**Q4. Changes in magnitude only.** With explicit hydrogens, MMFF flattens aromatic rings
less: free MMFF gives 1.08° and 1.35° instead of 0.63° and 0.83°, and the reference ligands
give 1.18°. Projection + out-of-plane (0.32° and 0.30°) remains below every MMFF arm.

**Other changes.**
* Explicit hydrogens make MMFF's W1 bond better at every restraint: sweep 0.017–0.022 Å
  vs. 0.020–0.026 Å, and gates 0.015–0.016 Å vs. 0.022–0.023 Å.
* MMFF free and k = 5 now beat projection (0.024 Å) on W1 bond by a wider margin.

**Numbers in the text that must be updated** (manuscript lines 1420–1429):
* "1.675 Å" becomes 1.433 Å.
* "from 17% to 68%" becomes "from 17% to 65%".
* "0.399 Å and … 2.14°" for k = 5 becomes 0.463 Å and 2.54°.
* All MMFF rows of `tab:ff` and `tab:gates` change as shown above.
* The Methods section should state the following: explicit hydrogens added with
  coordinates; restraint on heavy atoms only; minimization in vacuo without receptor atoms;
  all minimizations converged within 2,000 iterations; 2 of the 2,000 DiffSBDD molecules
  were excluded from the MMFF arms because of hypervalent P–H.

## 6. Files

All paths are relative to `MEHF/outputs/mmff_explicit_h_v1/`.

| File | Content |
|---|---|
| `PROTOCOL.md`, `PROTOCOL.sha256` | Pre-specified protocol and hashes of the protocol and code |
| `code/mmffh.py` | Original and corrected MMFF relaxation functions |
| `code/sweep_explicit_h.py` | tab:ff sweep with both arm families |
| `code/gate_explicit_h.py` | tab:gates per split with both arm families |
| `code/gate_explicit_h_common_subset.py` | Common-subset sensitivity, triggered by the 2 failures (diff vs. gate script: excludes failed molecules from all arms) |
| `code/geometry_corrector_gate_COPY_unmodified.py` | Byte-identical copy of `MEHF/scripts/geometry_corrector_gate.py` |
| `code/summarize.py` | Reproduction checks and table aggregation |
| `sweep/sweep_explicit_h.json` (+ `_per_molecule.json`) | 200-molecule sweep, all arms, with Minimize codes per molecule |
| `gates/diffsbdd/holdout2000_<seed>.json` (+ per-molecule) | DiffSBDD gate, 5 seeds |
| `gates/targetdiff/holdout2000_<seed>.json` (+ per-molecule) | TargetDiff gate, 5 seeds |
| `gates/diffsbdd_common_subset/` | Common-subset sensitivity, 5 seeds |
| `SUMMARY.json` | Reproduction results, five-split means and SDs, and the full sweep |
| `logs/` | Run logs |

`step0/step0_sweep_repro.json` and `step0/step0.log` hold the Step 0 output of the
unmodified original sweep. The content is identical to the reference JSON, as reported
in §0.
