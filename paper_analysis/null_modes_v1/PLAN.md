# null_modes_v1 — how many ligands have more than six near-zero modes? (post hoc, LD 2026-10-26)

Question. The paper says 4.7% (7 of 150 systems) of ligands have more than six near-zero
Hessian modes. That number comes from `scripts/exact_sigma_star_noise.py`. That script
builds B by finite differences, around the ensemble-MEAN structure, for systems with 8–40
heavy atoms. Kirk's fork reports 0.195% (12 of 6,144 fold molecules), with no file behind it.

Method. Use the model's own construction, `mdwm.models.harmonic_prior.sigma_star_factor`,
with cutoff 1e-6:
- coordinates: frame-0 heavy atoms of each MISATO store system;
- bonds: the store's bond index, restricted to heavy atoms.
Population: every system that occurs in a status=='ok' pair of
`outputs/mmp_transfer_corpus_v1/pairs.csv`. Report:
- n_rigid (the count of discarded modes) per system;
- the fraction with n_rigid > 6, overall and by split.

No model, no inference, no test outputs. CPU only.

Decision rule.
- If the corpus-wide fraction differs from 4.7% by more than a factor of 2, the paper
  states the corpus-wide fraction from this analysis, labeled as computed on the reference
  poses used by the model.
- Otherwise the text keeps 4.7% and adds the corpus-wide value.
