# Section-4 module analysis: PRE-REGISTERED PREDICTIONS
Written 2026-09-24 by the theory agent, BEFORE any computation in this directory.
Nothing below may be edited after `analysis.py` is first executed; results go in
RESULT.md. Hash in PREDICTIONS.sha256.

Corpus: MISATO `data/MISATO/extracted/frames100_all_v1`, restricted to systems
whose PDB is NOT in the corpus `test` split
(`outputs/mmp_transfer_corpus_v1/pairs.csv`, 218 test PDB ids excluded).
Heavy atoms only, matching every earlier measurement.

Reference numbers taken as given from prior work (not recomputed here):
tau_bond 0.0326 A, tau_angle 2.774 deg = 0.04841 rad, tau_dihedral 14.6 deg
= 0.2548 rad; heavy-to-heavy isotropic costs 11.55 / 7.88 / 2.47; val ANISO
bond 1.034, angle 1.453, dihedral 1.520, RMSF 0.618, planar improper 3.845.

--------------------------------------------------------------------------
## T1. Wilson B-vector norms, and the out-of-plane null-space claim

P-T1a  ||B_bond||^2 median = 2.000 +- 0.001 A^-2 (exact: two opposed unit vectors).
P-T1b  ||B_angle||^2 median in [2.00, 2.60] A^-2 (analytic 2.02 for sp3 at
       109.5 deg / 1.52 A, 2.59 for sp2 at 120 deg / 1.39 A). Point: 2.30.
P-T1c  ||B_oop||^2 median in [5.0, 7.0] A^-2 (analytic 6.21 for an ideal planar
       sp2 centre). The improper is therefore the LARGEST-norm stiff coordinate,
       ~2.4x the angle and ~3.1x the bond.
P-T1d  NULL-SPACE CLAIM. The normalised out-of-plane B-vector at a planar centre
       lies in the null space of the bond+angle B-matrix to first order.
       Measured as the squared projection onto rowspace(B_bond,angle):
       median <= 0.05. FALSIFIED if median > 0.20, in which case the "bond+angle
       are first-order blind to planarity" claim is wrong and is reported first.
P-T1e  Second-order scaling. Displacing the central atom by delta along the plane
       normal changes bond lengths by ~delta^2/(2r). Ratio of the bond-length
       change at delta=0.2 A to that at delta=0.1 A: 4.0 +- 0.3. Same ratio for
       the bond-angle change: 4.0 +- 0.5.

## T2. How much weight does a reweighted objective actually move?

Metric: for a symmetric PSD weight operator W on Cartesian displacements,
A(W) = (stiffest bond-stretch direction's Rayleigh quotient) / (median Rayleigh
quotient over random unit directions in the non-rigid subspace).

P-T2a  A(Sigma*^-1) = A(B^T F B) in [100, 500]. Point: 276 (the figure the
       theory report quotes; this is an independent recomputation of it).
P-T2b  A(I + 0.448 * B_angle^T B_angle), i.e. the cell-B objective, in
       [1.2, 2.5]. Point: 1.6.
P-T2c  A(I) = 1.000 exactly (arithmetic control; if it is not 1.000 the code is
       wrong and everything in T2 is void).
CONSEQUENCE IF CONFIRMED: adding an angle term at 0.448 changes the objective's
directional weighting by a factor of order 1.6, while the full Mahalanobis
reweighting changes it by a factor of order 276 -- roughly 170x more. The two are
not the same intervention and cell B's null result does not bear on the full one.

## T3. The missing correlation (weakness iii)

Sigma* = (B^T F B)^+ on the non-rigid subspace, f_k = 1/tau_k^2 (kT = 1).
Diagonal sampler: eps' = eps + B_S^+ (w - B_S eps), w_k ~ N(0, tau_k^2)
INDEPENDENT -- the exact fixed point of the converged Gauss-Seidel sweep.

P-T3a  Sigma* is effectively low rank: top-1 eigenvalue >= 25% of trace, top-5
       >= 60% of trace, and the rank needed for 90% of trace is <= 0.25*(3N-6).
P-T3b  Internal-coordinate correlation under Sigma* is SHORT-RANGED in the bond
       graph: median |corr| for coordinate pairs sharing >=2 atoms >= 0.20;
       median |corr| for pairs >= 3 bonds apart <= 0.05.
       CONSEQUENCE IF CONFIRMED: a k-hop message-passing amortiser can represent
       the correlation; a dense/global object is not required.
P-T3c  The diagonal sampler under-disperses relative to Sigma* at matched stiff
       widths: RMSF(diag)/RMSF(Sigma*) in [0.55, 0.85].
       STRONG FORM: the measured val internal-RMSF ratio 0.618 lies inside that
       band, i.e. the 0.62-vs-1.0 gap is explained by the missing correlation
       rather than by model capacity. FALSIFIED if 0.618 is outside the band.
P-T3d  A rank-r additive correction along Sigma*'s top modes recovers >= 80% of
       the missing internal RMSF at r <= 8.

## T4. Planarity budget

P-T4a  Physical out-of-plane spread on real MISATO MD, same protocol as
       `out_of_plane_degrees` std: median in [1.0, 2.5] deg. Point 1.5 deg.
P-T4b  Planar-centre OOP directions as a share of 3N-6: in [8%, 20%].
       CONSEQUENCE: the dispersion cost of adding the improper to the stiff set
       is at most ~sqrt(1-0.20) = 0.89 of current RMSF in the worst case, and
       (because the prior with absolute tau did not cost RMSF at all on val,
       0.618 vs 0.613) predicted actual cost <= 0.03 in the RMSF channel.

## T5. Graph-free stiff set (weakness / question C)

Soft gate g_ij = exp(-(d_ij - d0_ij)^2 / (2 s^2)) on element-pair covalent-radius
sums d0, s = 0.12 A, replacing the discrete bond graph.
P-T5a  Bond-recovery F1 against the true `bond_index` at Cartesian noise sigma:
       >= 0.98 at sigma = 0.05 A; >= 0.90 at sigma = 0.15 A; <= 0.60 at
       sigma = 0.50 A.
       CONSEQUENCE IF CONFIRMED: the gate is reliable exactly where the stiff
       prior matters (late in sampling) and collapses to isotropy early, which is
       the required behaviour for a de novo base distribution.

## T6. Does the B-norm law actually explain the measured isotropic costs?

Under isotropic Cartesian noise of scale sigma, Var(q_k) = sigma^2 ||B_k||^2, so
the measured over-dispersion ratio for channel k is sigma ||B_k|| / tau_k, with
ONE sigma common to all channels.
P-T6   The sigma implied by the bond channel (11.55) and by the angle channel
       (7.88) agree to within 25%. The dihedral channel (2.47) DISAGREES by more
       than 25%, because torsions are anharmonic and multi-modal and ring
       torsions are near-rigid, so the harmonic relation should not hold there.

--------------------------------------------------------------------------
## Design predictions that this directory CANNOT test, stated for the record

D1  Full Mahalanobis objective (weights bond 1.0, angle 0.448, torsion 0.0157),
    trained with the anisotropic prior: by reverse-water-filling the residual
    variance allocation becomes D_k proportional to tau_k^2, so the three channel
    ratios CONVERGE to a common value. Predicted: bond 1.03 -> 1.10-1.35,
    angle 1.45 -> 1.05-1.30, dihedral 1.52 -> 1.15-1.40, spread between the three
    channels (max-min) falls from the current 0.49 to below 0.20. The primary
    angle channel improves by 0.15-0.40 and bond length gets WORSE by 0.07-0.30.
    A result where all three move DOWN together is evidence of extra capacity,
    not of the reweighting, and must be controlled against a retrained isotropic
    -objective arm.
D2  Improper added to the prior's stiff set at tau_oop from T4a, plus an improper
    term in the inference projection: planar improper 3.845 -> 1.2-1.8; bond
    angle changes by less than +-0.10 (the coordinate is orthogonal, T1d); RMSF
    changes by less than 0.03.
D3  Rank-8 learned correction to the diagonal sampler: internal RMSF 0.618 ->
    0.78-0.95 with bond angle changing by less than +-0.10.

--------------------------------------------------------------------------
## ADDENDUM, same sitting, still BEFORE any computation in this directory

Added after reading `_cfm_loss` / `_geometry_losses` (confirming the objective is
Cartesian L2 flow matching plus ADDITIVE bond/angle/clash penalties, which is
what T2 assumes) and after noticing that the val result is over-dispersed in the
dihedral channel (1.520) and UNDER-dispersed in the RMSF channel (0.618) at the
same time. Those two cannot both be fixed by a scalar amplitude, so they are a
direct signature of a covariance-SHAPE error.

Define the collectivity ratio of a covariance Sigma:
    eta(Sigma) = internal RMSF(Sigma) / mean torsional width(Sigma)
i.e. how much Cartesian dispersion a covariance buys per unit of torsional
change. Physical motion is expected to be collective (correlated torsion changes
that add coherently through lever arms); independent draws add in quadrature.

P-T3e  eta(Sigma*) / eta(Sigma_diag) in [1.5, 3.0], i.e. the true covariance buys
       1.5x-3x more Cartesian dispersion per unit of torsion than the independent
       -w sampler does.
       SHARP FORM: the deficit the val numbers require is 1.520/0.618 = 2.46
       (re-scaling the ANISO sampler until its dihedral ratio is 1.0 would leave
       RMSF at 0.618/1.520 = 0.407, so closing to RMSF 1.0 needs a factor 2.46).
       CONFIRMED if the measured eta ratio's interval contains 2.46;
       REFUTED, and the missing correlation is then NOT the explanation of the
       RMSF gap, if the measured eta ratio is below 1.3.
