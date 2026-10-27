# Post-hoc data analysis: how much does trajectory-derived pocket selection change the pocket input?

Date: 2026-10-23. **Status: post-hoc, data-only.** No model runs.

**Motivation.** Audit 2026-10-23, finding M01:
- MEHF's pocket consists of the 96 candidate atoms with the smallest minimum ligand distance over all 100 MD frames.
- The candidates are all protein heavy atoms within 8 Å of any ligand atom in any frame.
- A prospective, first-frame-only rule would instead rank atoms by their frame-0 distance.

**Data.** `MEHF/data/MISATO/extracted/frames100_all_v1/<PDB>.npz` for every test system: the 491 products or parents in `sigma_star_test_v1/test_examples.csv`.
- Arrays used: `pocket_coords` (frames × candidates × 3), `pocket_min_distance_over_frames`, and `ligand_coords` (all ligand atoms, as in the extractor).
- No system in the store was truncated at the 2,000-atom cap. The candidate set therefore contains every protein heavy atom within 8 Å at frame 0.

**Quantities, per test system** (only systems with more than 96 candidates are cropped):
- U = the 96 atoms used, by smallest over-frame minimum, as in `mmp_transfer.py`.
- F = the 96 atoms by smallest frame-0 distance among candidates.
- Check that the 96th frame-0 distance is ≤ 8 Å, so that F equals the global first-frame rule.
- Overlap = |U ∩ F| / 96.
- For atoms in U \ F: their frame-0 distance, and how far they rank under the frame-0 rule.

**Summaries:** the median and minimum overlap, the fraction of systems with overlap = 1, and the number of systems with fewer than 96 candidates (no crop, so no effect).

**Prediction.** High overlap (median ≥ 0.85) but rarely complete. The atoms that differ lie near the boundary, at 5–8 Å at frame 0. This prediction is reported either way.

**Use.** The result quantifies the extent of the input dependency in the manuscript. It does not measure the effect on model outputs; that needs a rerun, which is on the UConn list.
