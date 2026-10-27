# Post-hoc analyses behind the paper

These analyses re-read stored outputs. None runs a trained model, and none reads a test result that
did not already exist.

Each folder has a `PLAN.md` whose SHA-256 (in `PLAN.sha256`) was recorded before the analysis ran,
plus the script and its `RESULT.json` / `RESULT.md`. They are labeled post hoc in the paper.

In the usage lines:
- `<MEHF root>` is the repository root;
- `<cleanrun dir>` is `cleanrun/`.

Check a plan hash from inside its folder, e.g. `cd pocket_v1 && sha256sum -c PLAN.sha256`. Two
folders (`null_modes_v1`, `ref_contact_v1`) recorded the hash with the folder prefix, so check those
from here: `sha256sum -c null_modes_v1/PLAN.sha256`.

| Folder | Paper item |
|---|---|
| `posthoc_v1/` | Absolute log width errors E3, E4, E5 (Table III) |
| `posthoc_v2/` | Per-fold effects, equal-target summaries, angle-loss effects by fold (Section IV-C; Supplementary Tables S1, S2, S5) |
| `pocket_v1/` | Overlap of the all-frame and first-frame pockets (Section III-A; Supplementary Fig. S3) |
| `planarity_ref_v1/` | Reference-ligand planarity on the same aggregation as the bars, 0.64° (Section IV-F) |
| `small_ligand_v1/` | Small-ligand E4 pattern, with its decision rule (Supplementary Section S4) |
| `supplement_v1/` | Supplementary tables and figures; `VALUES.json` holds every printed value |
| `null_modes_v1/` | Ligands with more than six near-zero modes: 7 of 3,819 (Section IV-G) |
| `ref_contact_v1/` | Closest ligand–pocket approach in the reference MD: median 2.55 Å, minimum 2.31 Å (Section IV-D) |
| `clean_protocol_check_v1/` | Independent recomputation of every replication number from the per-example CSVs (`recompute.py`); like-for-like fitting-to-test balance (`degradation.py`); Supplementary Table S9 (`make_replication_table.py`) |
| `boot.py`, `tab.py` | Summaries for the in-sample projection decomposition (Section IV-E; Supplementary Fig. S4) |
