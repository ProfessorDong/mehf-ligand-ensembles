# Result: trajectory-derived vs first-frame-only pocket (post-hoc, data only)

Plan: `PLAN.md` (sha256 c0ebf1dd…70ae7, hashed before computing). Script: `pocket_overlap.py`. Output: `RESULT.json`.

**Scope.**
- 491 test systems. All 491 have more than 96 candidates, so every one is cropped.
- For 489 systems the 96th frame-0 distance is ≤ 8 Å. For these, F is exactly the global first-frame rule.
- For 2 systems (5EFC, 6MO9) it is not, so F there is restricted to the stored candidates.

**Overlap |U ∩ F| / 96**, all systems:

| Statistic | Value |
|---|---|
| Median | 0.833 (16 of 96 atoms differ) |
| IQR | 0.792–0.875 |
| 10th–90th percentile | 0.74–0.90 |
| Minimum | 0.56 |
| Complete overlap | 0 systems |

Restricting to the 489 exact-rule systems gives the same median (0.833).

**Atoms used but not in the first-frame set.** In total, 8,220 atoms over 491 systems:
- Their median frame-0 distance to the ligand is 6.0 Å.
- Their median frame-0 rank is 134.

They are boundary atoms that come within range of the ligand later in the trajectory.

**Prediction.** P: median overlap ≥ 0.85, rarely complete. **Not met**: the median is 0.83, and overlap is never complete.

**Implication.** The pocket input of every evaluated model depends on the target's future trajectory for about one-sixth of its atoms. The effect on model outputs is not measured. That needs first-frame preprocessing and retraining, which is on the UConn request list.
