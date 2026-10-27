# Post-hoc: matched-aggregation aromatic planarity reference (audit R04)

**Status: post-hoc**, written 2026-10-23 before computing. Stored input files only; no model is run.

**Question.** The dashed line in Fig. 6 (1.18 deg) is a mean pooled over 291 aromatic centers
(`reference_outputs/published_models_v1/reference_r1.json`). The bars are medians over molecules
of each molecule's mean |chi|. What is the reference under the bars' aggregation?

**Inputs.** The 100 CrossDocked reference ligands, `MEHF/data/external/crossdocked/ref_flat/*.sdf`,
read as in both source scripts: `SDMolSupplier(removeHs=True, sanitize=True)`, first molecule per file.

**Observable.** Identical to `scripts/geometry_corrector_gate.py::geometry` and
`published_model_isotropy_test.py::improper`. Centers are aromatic atoms with exactly 3 heavy
neighbors (j; a, b, c in RDKit neighbor order). chi = |asin(u . n)|, where u is the unit vector
a - j and n is the unit normal of (b - j) x (c - j).

**Outputs.**
- (a) The pooled mean over centers. This must reproduce 1.1832 with n = 291, as a check.
- (b) The median over molecules with at least one center of the per-molecule mean |chi|, with
  the number of molecules.
- (c) The IQR of (b).

**Predictions.** (a) reproduces 1.183. (b) lies in [0.5, 1.18], because per-molecule means are
right-skewed. The corrected-arm values (0.32 DiffSBDD, 0.30 TargetDiff) stay below (b).

**Use.** If (b) is computed, the paper reports the reference on the same aggregation as the bars
and labels it post-hoc. The pooled 1.18 is kept only as a stated alternative.
