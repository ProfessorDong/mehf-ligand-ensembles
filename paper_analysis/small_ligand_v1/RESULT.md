# Small-ligand E4 pattern: result (post-hoc, 2026-10-23)

- Plan: `PLAN.md`, sha256 `705bf119…0fd0f3` (`PLAN.sha256`), written before computing.
- Script: `small_ligand.py`. Output: `RESULT.json`.
- Cohort: 1,410 examples, fold-averaged.
- ΔE4 = E4(HARM) − E4(ISO).

**Check.** The all-example mean ΔE4 is +0.065 [−0.085, 0.132]. This reproduces the main-text value exactly.

| Analysis | Result | Prediction |
|---|---|---|
| 1. Concentration | Q6XEC0 (341) and Q9H2K2 (140) hold **89.7%** of the 536 examples with < 20 heavy atoms. Both have mean ΔE4 > 0 (+0.19, +0.38). | P1 met |
| 2. Target level | Spearman ρ (target median size vs target mean ΔE4) = **−0.13 [−0.41, 0.16]** over 60 targets. Small-ligand targets (6): mean ΔE4 = **+0.003**, 2/6 positive. Other targets (54): −0.075, 11/54 positive. | P2: sign as predicted; the interval includes 0 |
| 3. Within target | 6 targets have ≥ 5 examples in both bins. In **all 6**, the small-product examples have the higher ΔE4 (+0.09 to +0.43). | — (descriptive; 6 targets) |
| 4. Channels (median of HARM − ISO in \|ln r\|) | < 20: bond +0.67, angle +0.34, torsion +0.21, out-of-plane −0.16. ≥ 20: bond +0.22, angle +0.09, torsion +0.10, out-of-plane −0.63. Planar-center counts are not stored. | P3 met |
| 5. Excluding < 20 | Example-weighted mean ΔE4 = −0.037 [−0.107, 0.029] | P4 met (below 0; the interval includes 0) |

**Decision (plan rule).** The target-level interval includes 0, so this is not reported as a size effect. It stays in the supplement, described as concentrated in two targets. The within-target pattern (6/6) is reported as a hypothesis for a pre-registered test. No main-text sentence is added. The main text already says the example-weighted E4 result "is driven by a few large targets", and that remains accurate.

**Interpretation, limited to what the numbers show.** Where HARM loses on E4 for small products, it loses through over-dispersed bond lengths and angles. Its out-of-plane gain is also smaller there.

**Hypothesis (untested).** A single global amplitude s*, selected per fold checkpoint, is too large for small ligands.
