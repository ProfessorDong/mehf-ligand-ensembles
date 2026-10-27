# ref_contact_v1 — closest ligand–pocket approach in the reference MD (post hoc, LD 2026-10-26)

Question. The replication's clash column counts ligand heavy atoms within 1.2 Å of a pocket
heavy atom. Kirk's fork states that the reference trajectories never come that close
(minimum separation median 2.55 Å, never below 2.4 Å, in 40 systems). No file backs this up.

Method. For every system in a status=='ok' corpus pair, read the MISATO store:
- take the ligand heavy atoms and the stored pocket heavy atoms (the full candidate pool);
- compute the minimum heavy–heavy distance over all 100 frames.
Report the median, 1st percentile and minimum over systems, and the number of systems below
2.0 Å and below 1.2 Å.

No model, CPU only.

Decision rule. The paper may state that the reference trajectories never reach 1.2 Å only if
0 systems fall below 1.2 Å. It then quotes the measured minimum.
