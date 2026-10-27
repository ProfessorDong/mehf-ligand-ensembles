# Planarity reference on matched aggregation: result (post-hoc, 2026-10-23)

- Plan: `PLAN.md`, sha256 `6afdf46c…202c0ca7`, hashed before computing.
- Script: `planarity_ref.py`.
- Run: `python -I planarity_ref.py MEHF/data/external/crossdocked/ref_flat .`

| Statistic | Value |
|---|---|
| Pooled mean over centers (check) | 1.1832° over 291 centers; reproduces `reference_r1.json` |
| **Median of per-molecule mean \|χ\|** | **0.641°** (66 of 100 molecules have ≥ 1 center) |
| IQR of the per-molecule means | 0.311°–1.469° |

**Predictions.**
- (a) The check reproduces the record: met.
- (b) The matched value lies in [0.5, 1.18]: met (0.64).
- (c) The corrected arms (0.32° and 0.30°) lie below (b): met. Both are near the 25th percentile of the reference distribution.

**Consequences for the paper (R04).**
1. The line in Fig. 6 moves to 0.64°, the aggregation the bars use. The pooled 1.18° is kept in the caption as the alternative.
2. Both corrected arms remain below the reference median. The over-flattening is real but modest, because 0.30–0.32° lies inside the reference IQR.
3. **Reversal.** Under the pooled reference, TargetDiff's corrected value looked farther from the reference than the raw one (|0.30 − 1.18| = 0.88 vs |1.76 − 1.18| = 0.58). Under the matched reference it is closer (0.34 vs 1.12). The manuscript's sentence "for TargetDiff, the corrected value is farther from the reference than the raw output" was an artifact of mismatched aggregation and is removed.
4. The comparison is still between different molecule populations. The reference ligands are docked or minimized poses, not the generated molecules. The comparison is context, not a per-molecule accuracy score.
