# Supplement v1: descriptive summaries of stored records (post-hoc)

Status: **post-hoc, descriptive.** Written 2026-10-23, before any value below was computed. Stored files only; nothing is run or inferred. No new inferential claims and no new test-set decision rules.

Script: `make_supplement.py <MEHF root> <out dir>`. Outputs: TikZ/LaTeX snippets and `VALUES.json`.

1. **Fig. S1, operating-point calibration** (fitting folds = training data).
   - Fold-mean median channel ratio vs s_i for ISO, GS, LAP and HARM.
   - Sources: `sigma_star_test_v1/matched_torsion_fit.json` and `laplacian_prior_v1/matched_torsion_fit.json`.
   - Marks the per-fold s* values.
2. **Table S2, per-fold test medians.**
   - Cohort: the 1,410 examples with all four ratios finite in all three folds of every arm. Assert n = 1410.
   - For each arm and fold: the four channel medians, internal RMSF, and the median per-example S.
   - Check: the paper's per-fold S (HARM 0.740/0.758/0.804, ISO 2.008/1.851/2.032) and the fold-averaged median S (0.770, 1.974).
3. **Fig. S2, target level.** Per-target mean S, ISO vs HARM, marked by target size. Descriptive only.
4. **Table S4, heterogeneity of the HARM − ISO effect** (fold-averaged, 1,410 cohort).
   - Bins of product heavy-atom count (<20, 20–29, 30–39, ≥40) and quartiles of the MCS fraction.
   - Per bin: n examples, n targets, median ΔS, % with ΔS < 0, median ΔE4, % with ΔE4 < 0.
   - No tests or intervals: the bins are not pre-registered.
5. **Table S3** collects existing results (`posthoc_v1`, `posthoc_v2`): example- vs target-weighted effects.
6. **Table S5** collects width provenance from `leakage_audit_v1/audit.json`.
7. **Fig. S3** is the histogram of the pocket overlap from `analysis/pocket_v1/RESULT.json`.
8. **Table S6** collects the angle-term per-fold effects from `posthoc_v2`.

Prediction for the checks in item 2: they reproduce to 3 decimals.
