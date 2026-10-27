# Section-4 module analysis — RESULT

Predictions: `PREDICTIONS.md`, sha256
`6be50c92ff393929d5d88a9718d6427b5eee5253d39dc3134e97599bf2dcb721`, written and
hashed before any number below was computed.

Commands, in order:

```
PYTHONPATH=src conda run -n MDWM --no-capture-output python \
  outputs/theory_modules_v1/analysis.py --n-systems 150 \
  --output outputs/theory_modules_v1/results.json
PYTHONPATH=src conda run -n MDWM --no-capture-output python \
  outputs/theory_modules_v1/aggregate.py
```

150 MISATO systems, 8–40 heavy atoms, seed 20260924. The 218 PDB ids carrying a
`test` label in `outputs/mmp_transfer_corpus_v1/pairs.csv` were excluded before
sampling. No model is involved anywhere: every number is a property of real
ligand geometry, of the Wilson B-matrix built at the MD-mean structure, or of a
numpy re-implementation of `_structure_local_noise` (32 sweeps, relaxation 0.5,
same targets). `src/mdwm/models/` and `outputs/pareto_v1/` were not touched.

---

# 1. REFUTATIONS, first and plainly

**R1. The physical out-of-plane width is 6.48 deg, not ~1.2 deg. P-T4a is
refuted by a factor of 4.3, and this damages the published-models planarity
claim.**

| | value |
|---|---|
| predicted tau_oop | 1.0–2.5 deg, point 1.5 |
| **measured tau_oop (150 systems, 130 with planar centres)** | **6.48 deg**, IQR 6.13–6.96 |

This is the thermal width, over a 1 ns MD ensemble, of exactly the coordinate
`mehf_internal_geometry_spectrum.out_of_plane_degrees` measures at exactly the
centres `improper_centres` selects — the same definition the val `planar
improper` channel uses. `outputs/published_models_v1/FINDING.md` compares
DiffSBDD's 5.09 deg to "real ligands 1.18 deg", where the 1.18 deg is a
**mean absolute deviation measured on static crystal structures**. Against a
thermal ensemble of the same coordinate, 5.09 deg is *inside* the physical
width. The two numbers are not the same statistic and the "aromatic planarity is
broken by a factor of 4.3" sentence cannot stand as written; it needs either an
ensemble-matched reference or an explicit statement that the reference is a
restrained crystallographic model. Caveat in the other direction: my centres are
ring atoms with three heavy neighbours and are **not filtered for aromaticity**,
so saturated-ring pucker inflates 6.48; an aromatic-only measurement would be
lower. Either way the comparison as published is not like-for-like.

**R2. The pre-registered null-space statistic came back at 1.000, the worst
possible value. P-T1d is refuted as stated.** The statistic is also the wrong
one and I say so rather than quietly substituting another:

| | value |
|---|---|
| out-of-plane B-vector, squared projection onto rowspace(B_bond, B_angle) | **1.000** (IQR 1.000–1.000) |
| a random non-rigid direction, same statistic (control) | 0.935 |
| dim rowspace(B_bond, B_angle) / (3N-6) | 0.927 |

With 93% of the non-rigid space already spanned, "is this direction in the span"
carries almost no information. The underlying claim survives in its exact form
and is verified to machine precision on an ideal planar ring
(`ideal_ring_theorem` in the script):

| quantity, exact planar ring | value |
|---|---|
| max over bonds of \|B_bond · B_oop\| | **0.0** (exactly) |
| max over angles of \|B_angle · B_oop\| | **0.0** (exactly) |
| squared rowspace projection | 2.13e-32 |

and holds approximately on real MD-mean geometries, where centres sit 5–7 deg
off planar: median over systems of the largest cosine between the normalised
out-of-plane direction and any bond B-vector is **0.028**, and any angle
B-vector **0.140**.

**R3. Internal coordinates are essentially UNCORRELATED under Sigma*. P-T3b is
refuted, and the premise of question A as posed is wrong.**

| pair class | predicted median \|corr\| | measured |
|---|---|---|
| sharing >= 2 atoms | >= 0.20 | **0.041** (IQR 0.020–0.059) |
| disjoint atom sets | <= 0.05 | 6.6e-15 (p95 0.010) |

This is a theorem, not an accident. With M = F^{1/2} B,

    B Sigma* B^T = B (B^T F B)^+ B^T = F^{-1/2} P_M F^{-1/2}

where P_M projects onto the column space of M. For a coordinate set that spans,
P_M is close to the identity on the coordinates that matter, so the internal
coordinates come out near-independent with variances ~1/f_k. **Drawing w_k
independently is therefore NOT the defect.** The defect is elsewhere (section 3).

**R4. My anisotropy factor does not reproduce the 276 quoted in the theory
report, under any of three definitions. P-T2a is refuted at the median.**

| definition (150 or 40 systems) | median | IQR |
|---|---|---|
| median_random(e'Sigma* e) / median_bond-stretch(e'Sigma* e) | **78.8** | 39–140 |
| tr(Sigma*)/(3N-6) / median_bond-stretch(e'Sigma* e) | 105.0 | 58–199 |
| tr(Sigma*)/(3N-6) / (tau_bond^2 / 2) | 97.1 | 53–184 |

276 sits inside the upper quartile of two of these but not at any median. I could
not find the file that states 276, so I cannot check whether its definition
differs. **The paper should state whichever definition it uses and the number
that goes with it; "the bond direction is ~100x stiffer than a typical Cartesian
mode, with a wide per-molecule spread" is what I can support.**

**R5. Three smaller misses.** P-T1b ||B_angle||^2 predicted [2.00, 2.60],
measured **2.674** (above the band). P-T3d predicted a rank-<=8 correction
recovers >= 80% of the missing RMSF; measured **~50%**. P-T5a predicted the
graph gate's F1 falls to <= 0.60 at sigma = 0.50 A; measured **0.626**,
marginally above. P-T2b predicted the cell-B objective's anisotropy at
[1.2, 2.5]; measured **1.048**, i.e. even weaker than predicted — this one cuts
in the argument's favour and is flagged for that reason.

---

# 2. Scorecard

| id | prediction | measured | verdict |
|---|---|---|---|
| P-T1a | \|\|B_bond\|\|^2 = 2.000 +- 0.001 | 2.000 (IQR 2.000–2.000) | CONFIRMED |
| P-T1b | \|\|B_angle\|\|^2 in [2.00, 2.60] | 2.674 | MISSED (high) |
| P-T1c | \|\|B_oop\|\|^2 in [5.0, 7.0] | **6.526** (IQR 6.15–7.07) | CONFIRMED |
| P-T1d | oop rowspace projection <= 0.05 | 1.000 | **REFUTED** (wrong statistic; exact form holds, R2) |
| P-T1e | 2nd-order ratio 4.0 +- 0.3 / 0.5 | ideal ring 3.979 / 3.881; real geometry 3.00 / 2.71 | CONFIRMED exactly planar, MISSED off-planar |
| P-T2a | A(Mahalanobis) in [100, 500] | 78.8 | **REFUTED** at the median (R4) |
| P-T2b | A(cell-B) in [1.2, 2.5] | **1.048** | MISSED (low) |
| P-T2c | A(L2) = 1.000 | 1.000 | CONTROL PASSES |
| P-T3a | top1 >= 25%, top5 >= 60%, rank90 <= 0.25 | **45.4% / 89.2% / 0.081** | CONFIRMED, strongly |
| P-T3b | corr shared>=2 >= 0.20 | 0.041 | **REFUTED** (R3) |
| P-T3c | RMSF(diag)/RMSF(Sigma*) in [0.55, 0.85] | 0.315/1.253 = **0.251** | REFUTED — the deficit is LARGER than predicted |
| P-T3d | rank <= 8 recovers >= 80% of missing RMSF | ~50% | MISSED |
| P-T3e | eta ratio in [1.5, 3.0], sharp value 2.46 | **3.085**, IQR 2.37–4.01 | band MISSED (high); sharp value 2.46 INSIDE the IQR |
| P-T4a | tau_oop in [1.0, 2.5] deg | **6.48 deg** | **REFUTED** (R1) |
| P-T4b | oop DOF share in [8%, 20%] | 8.33% (IQR 5.8–10.0) | CONFIRMED (bottom edge) |
| P-T5a | F1 >= 0.98 / >= 0.90 / <= 0.60 | **0.987 / 0.945 / 0.626** | 2 of 3 CONFIRMED, third marginal |
| P-T6 | bond-vs-angle sigma agree < 25%; dihedral disagrees > 25% | 12.4% and **11.4%** | first CONFIRMED, second **REFUTED** — and this is a positive result |

---

# 3. The measurements

## 3.1 One sigma explains all three measured isotropic costs (T6)

Under isotropic Cartesian noise, Var(q_k) = sigma^2 ||B_k||^2, so the measured
over-dispersion ratio of channel k is sigma ||B_k|| / tau_k with a single sigma.
Using the measured B-norms and the corpus widths against the published
heavy-to-heavy costs 11.55 / 7.88 / 2.47:

| channel | \|\|B_k\|\| | tau_k | measured ratio | implied sigma (A) |
|---|---|---|---|---|
| bond | 1.4142 | 0.0326 A | 11.55 | 0.2662 |
| bond angle | 1.6353 | 0.04841 rad | 7.88 | 0.2333 |
| dihedral | 2.1213 | 0.25482 rad | 2.47 | 0.2967 |

**All three agree on one sigma to within +-12.4%.** I predicted the dihedral
would break the law by more than 25% and it does not. A three-line harmonic
argument with no free parameters beyond one sigma reproduces the whole measured
spectrum. This is the strongest single piece of theory the project has and it
should open section 4.

## 3.2 The Mahalanobis weights, recomputed (T6, weights)

w_k = (tau_bond / tau_k)^2, from the corpus widths:

| coordinate | recomputed | value in hand | agreement |
|---|---|---|---|
| bond (A^-2) | 1.0 | 1.0 | — |
| bond angle (rad^-2) | **0.45339** | 0.448 | 1.2% |
| torsion (rad^-2) | **0.016367** | 0.0157 | 4.3% |
| **out-of-plane (rad^-2)** | **0.08309** | — (new) | from tau_oop = 6.48 deg |

The handed-down weights are confirmed. The out-of-plane weight is new and is the
number D would need.

## 3.3 What a reweighted objective actually moves (T2)

The residual-variance allocation implied by an objective with weight operator W
is W^+ (reverse water-filling: minimising sum_i w_i D_i at fixed capacity gives
D_i = theta / w_i). The statistic is variance along a typical non-rigid
direction over variance along a bond-stretch direction; W = I gives exactly 1.

| objective | A_var | IQR |
|---|---|---|
| plain L2 (what MEHF trains now) | **1.000** | — |
| L2 + 0.448 x angle penalty (cell B) | **1.048** | 1.011–1.077 |
| full Mahalanobis, W = B^T F B | **78.8** | 39.1–139.7 |

**Ratio 75.2.** Adding the angle term at its theoretical weight moves the
objective's variance allocation by **4.8%**. Replacing the objective moves it by
a factor of **79**. They differ by about 75x. Cell B's null result says nothing
about the full reweighting.

Restricting the same statistic to the angle-bend directions specifically
(typical direction over angle-bend direction, 40 systems) gives the sharper
version, and with it a refutation:

| objective | A_var on angle directions | implied angle-channel residual SD factor |
|---|---|---|
| L2 + 0.448 x angle penalty | 2.571 (IQR 2.42–2.74) | **0.624** |
| full Mahalanobis | 86.50 (IQR 54.8–145.0) | **0.108** |

**The water-filling model over-predicts cell B by about 4x, and this is a
refutation that has to be carried forward.** Water-filling says the cell-B
objective should have cut the angle-channel residual by 38%; `angle_term_v1`
measured a main effect of -0.353 +- 0.601 on a base of 4.806, i.e. a 10% cut
(CI -11% to +20%). The log-space realisation efficiency is **0.23**, CI roughly
[0, 0.47]. Reverse water-filling assumes the model reaches the rate-distortion
optimum; a finite-capacity trained network does not, so **water-filling is an
upper bound on what any reweighting buys, and the measured discount is ~4x**.
Every prediction in B is stated after applying that discount.

## 3.4 Sigma* is low rank (T3a)

| quantity | median | IQR |
|---|---|---|
| top-1 eigenvalue / trace | **0.454** | 0.385–0.534 |
| top-5 eigenvalues / trace | **0.892** | 0.850–0.914 |
| rank for 90% of trace, as a fraction of 3N-6 | **0.081** | 0.058–0.117 |

Roughly five modes carry ~89% of the physical dispersion of a drug-sized ligand.

## 3.5 The real defect: collectivity, not correlation (T3c, T3e)

eta(Sigma) = internal RMSF / torsional width — how much Cartesian dispersion a
covariance buys per degree of torsional change.

| arm | eta | IQR |
|---|---|---|
| physical MD | 0.03082 | 0.0242–0.0379 |
| Sigma* (harmonic) | 0.03986 | 0.0267–0.0540 |
| Gauss-Seidel sampler, independent w | **0.01012** | 0.0084–0.0113 |
| **eta(physical) / eta(Gauss-Seidel)** | **3.085** | 2.37–4.01 |

Equivalently, rescaled so every arm has the physical torsional width:

| arm | internal RMSF / physical internal RMSF | IQR |
|---|---|---|
| Sigma* | **1.253** | 0.96–1.65 |
| Gauss-Seidel sampler | **0.315** | 0.25–0.39 |

**At matched torsional width, the exact harmonic covariance reproduces the
physical internal RMSF (1.25, straddling 1), while the current sampler reaches
31% of it.** The val arithmetic independently demands a factor of 2.46
(1.520 / 0.618 read the other way: rescaling ANISO to dihedral 1.0 would leave
RMSF at 0.407) and 2.46 sits inside the measured IQR 2.37–4.01. The pure-noise
and trained-model numbers agree.

And at matched **internal RMSF**, the channel profile of each covariance:

| channel | Sigma*-shaped residual | current Gauss-Seidel residual |
|---|---|---|
| bond | 1.521 | 2.739 |
| bond angle | **1.047** | 2.279 |
| torsion | 0.798 | 2.351 |
| out-of-plane | 1.354 | **5.574** |
| **spread (max - min)** | **0.723** | **3.295** |

A residual whose covariance is Sigma* is **4.6x more uniform across channels** at
the same overall dispersion. This is the reverse-water-filling claim, measured.

## 3.6 A rank-r correction is nearly free in the channels that matter (T3d)

Additive correction along the top eigenvectors of (Sigma* - Cov(Gauss-Seidel)),
at sigma = 0.20, as multiples of the uncorrected sampler:

| r | RMSF | bond width | angle width | torsion width |
|---|---|---|---|---|
| 1 | **x1.650** | x1.081 | x1.020 | x1.001 |
| 2 | **x1.883** | x1.147 | x1.043 | x1.001 |
| 4 | x2.013 | x1.216 | x1.071 | x1.006 |
| 8 | x2.077 | x1.242 | x1.095 | x1.011 |
| 16 | x2.054 | x1.265 | x1.116 | x1.013 |

**r = 2 buys 82% of the total gain.** Cost at r = 2: +4.3% on the angle channel,
+0.1% on torsion, +14.7% on bond (the one channel that pays, and the one already
protected by the inference projection). It saturates by r = 8.

## 3.7 The improper, added to the stiff set of the same sampler (T1c, T4)

Wilson out-of-plane bend at centre j with neighbours a (out of plane), b, c:

    sin(chi) = (u_b x u_c) . u_a / sin(phi_bc)

    s_a = (1/r_a) [ (u_b x u_c)/(cos chi sin phi) - tan(chi) u_a ]
    s_b = (1/r_b) [ (u_c x u_a)/(cos chi sin phi)
                    - (tan chi / sin^2 phi)(u_b - cos phi u_c) ]
    s_c = (1/r_c) [ (u_a x u_b)/(cos chi sin phi)
                    - (tan chi / sin^2 phi)(u_c - cos phi u_b) ]
    s_j = -(s_a + s_b + s_c)

Verified against central finite differences of `out_of_plane_degrees` itself:
median relative error **6.3e-11**. At planarity the tan terms vanish, all four
vectors are parallel to the plane normal with magnitudes 1/(r sin phi), and the
row is the centroid-preserving umbrella mode — so the operator is
SE(3)-equivariant and centroid-preserving exactly as the bond and angle steps are.

| quantity | value |
|---|---|
| \|\|B_oop\|\|^2 | **6.526** (vs angle 2.674, bond 2.000) |
| DOF share of the out-of-plane directions | 8.33% of 3N-6 |
| tau_oop | 6.48 deg |

Measured effect of adding it to the Gauss-Seidel stiff set, 130 systems,
sigma = 0.20, ratios against each system's own physical width:

| channel | without improper | **with improper** |
|---|---|---|
| out-of-plane | 3.534 | **0.965** |
| bond angle | 1.333 | **0.971** |
| torsion | 1.375 | **0.979** |
| internal RMSF (absolute) | 0.1826 | 0.1573 (**-13%**) |

The uncorrected 3.534 reproduces the val `planar improper` figure of 3.845 from
noise alone, which is the control that says this pure-noise pipeline is measuring
the same thing the trained model does. Adding the improper takes the channel to
**0.965**, and improves the angle and torsion channels as a side effect, at a
13% dispersion cost that a scale re-tune recovers (the sweep is monotone in
sigma).

## 3.8 The graph-free soft gate (T5)

g_ij = 0.5 (1 - tanh((d_ij - (r_i + r_j + 0.30)) / 0.15)), no discrete graph.

| Cartesian noise sigma (A) | bond-recovery F1 | Q1 | total soft mass / true bond count |
|---|---|---|---|
| 0.00 | 0.986 | 0.927 | 1.027 |
| 0.05 | **0.987** | 0.924 | 1.020 |
| 0.15 | **0.945** | 0.890 | 1.039 |
| 0.30 | 0.791 | 0.755 | 0.961 |
| 0.50 | 0.626 | 0.560 | 0.845 |
| 1.00 | 0.324 | 0.261 | 0.571 |

The gate is near-exact where the stiff prior matters and its total mass decays to
zero as noise grows, so a stiffness built from it **anneals to isotropy on its
own** with no schedule to design. That is the property de novo generation needs.

---

# 4. Answers

## A. The missing correlation: the question was wrong, and the right version has a cheap answer

**The premise fails.** Internal coordinates are near-independent under Sigma*
(R3, median |corr| 0.041), because B (B^T F B)^+ B^T = F^{-1/2} P_M F^{-1/2} and
P_M is close to the identity for a spanning coordinate set. Drawing w_k
independently is essentially correct. An amortised network that predicts
correlated w from the molecular graph would be learning a near-diagonal target —
**that option is rejected on the measurement**, and no one should build it.

The real defect is in the **complement**, not the stiff set. The Gauss-Seidel
sampler pins bonds and angles and leaves the remaining ~7% of directions, plus
everything the stiff set does not reach, at the *isotropic* amplitude sigma. But
Sigma*'s soft subspace is violently anisotropic: **one mode carries 45% of the
trace and five carry 89%** (3.4). Physical motion is collective — correlated
torsion changes that add coherently through lever arms — so it buys Cartesian
dispersion cheaply; white noise in the same subspace adds in quadrature and buys
it dearly. Measured exchange rate: **3.09x** (3.5). That single number explains
the paper's most awkward pair of results, dihedral over-dispersed at 1.520 and
internal RMSF under-dispersed at 0.618 at the same time, which no scalar
amplitude can reconcile.

So weakness (iii) should be restated: **the prior has the right marginals and the
wrong low-rank structure.** Two concrete options:

**A1 — learned low-rank correction to the diagonal-G sampler (recommended).**
After the sweeps, add `eps += sum_{m=1..r} z_m * v_m(x)`, z_m ~ N(0,1), with the
r modes produced by a small equivariant head from the reference geometry, and
orthogonalised against the stiff B-rows so it cannot re-inflate them. Cost:
O(rN) per draw, r = 2. Parameters: one vector head, a few thousand weights. No
change to the sweep. Measured payoff (3.6): **RMSF x1.88 at +4.3% on the angle
channel and +0.1% on torsion.** Applied to the val operating point that is
0.618 -> ~1.16 on the RMSF channel if the gain transfers; conservatively, closing
most of the 0.62-to-1.0 gap. The obvious control is a **random** rank-2
correction of matched amplitude, which should give the RMSF and lose the channel
selectivity.

**A2 — closed-form low-rank, no learning at all.** The top eigenvectors of
(B^T F B)^+ are computable by a few inverse-power iterations against the
Gauss-Seidel operator already in the code, at O(N) per iteration and ~20
iterations for r = 2. Zero new parameters. Strictly cheaper than A1 and it is the
correct control for it: if A2 matches A1, the head is decoration and must be cut.

**Cheapest: A2.** It is arithmetic, not learning. A1 only earns its place if it
beats A2.

## B. Should the anisotropy go in the objective? Yes, and the full reweighting is a different intervention from the cell-B term

**Three separate arguments, two of them measured.**

*1. The change of variables is currently half done.* With Sigma* = L L^T, the
substitution y = L^{-1} x turns (anisotropic prior in x, L2 loss in x) into
(isotropic prior in y, Mahalanobis loss in y weighted by Sigma*^-1). The
statistically consistent object is **anisotropic prior AND Mahalanobis loss**,
which is exactly plain flow matching in whitened coordinates. The method as
shipped whitens the prior and leaves the loss unwhitened, so the objective is
still pulling the residual back towards Cartesian isotropy while the prior pushes
it away. That is the structural reason the angle channel stopped at 1.453 rather
than 1.0, and it is the cleanest sentence section 4 can carry.

*2. Sigma*^-1 is the harmonic strain energy of the error.*

    ||r||^2_{Sigma*^-1} = r^T (B^T F B) r / kT = sum_k f_k (B_k . r)^2 / kT
                        = sum_k (dq_k / tau_k)^2

The Mahalanobis flow-matching loss is the thermal energy cost of the prediction
error, in units of kT. Every internal coordinate is scored in units of its own
thermal width. That is the right notion of "wrong" for a molecule and it is
derived, not asserted.

*3. The cell-B term and the full reweighting differ by 34–75x, measured.* A_var
overall: L2 **1.000**, cell B **1.048**, full Mahalanobis **78.8**; restricted to
the angle directions, cell B **2.571** against Mahalanobis **86.50**, a ratio of
**33.6** (3.3). They are not the same intervention by one and a half orders of
magnitude, so cell B's null does not bear on the full reweighting.

**Quantitative prediction, with the discount applied.** Reverse water-filling
gives D_k = theta / w_k, so with w_k = 1/tau_k^2 the residual standard deviation
in every internal coordinate becomes proportional to its physical width and all
channel ratios converge. The covariance-level version of exactly that claim is
measured (3.5, last table): a Sigma*-shaped residual at matched RMSF gives
bond 1.52 / angle 1.05 / torsion 0.80 / oop 1.35, spread **0.723**, against the
current sampler's 2.74 / 2.28 / 2.35 / 5.57, spread **3.295**.

But water-filling over-predicted cell B by 4x (3.3), so the trained-model
prediction must be discounted at the measured log-space efficiency 0.23,
CI [0, 0.47]:

| efficiency | angle-channel SD factor | val angle 1.453 becomes |
|---|---|---|
| 1.00 (water-filling exact — not believed) | 0.108 | 0.16 |
| **0.23 (cell-B point estimate)** | **0.593** | **0.86** |
| 0.47 (cell-B CI upper) | 0.367 | 0.53 |
| 0.00 (cell-B CI lower) | 1.000 | 1.45 |

**Stated prediction: bond angle 1.453 -> [0.70, 1.35], most likely ~0.86; bond
length 1.034 -> [1.05, 1.40], i.e. WORSE; dihedral 1.520 -> [0.95, 1.40];
max-min spread across the three 0.486 -> below 0.25.** Note that the central
estimate lands *below* 1, i.e. the full weights risk **over**-constraining the
angle channel into under-dispersion. That is the concrete reason to ship the
tempered family rather than gamma = 1 alone.

The falsifiable signature is *convergence*, not uniform improvement. If all three
channels move down together that is extra capacity, not the reweighting, and the
control is a retrained arm with the same schedule and an L2 objective.

**The DiffSBDD refutation is the argument for doing this.** A trained model's
residual is anisotropic in a shape the architecture chose — DiffSBDD's angle
error is 1.74x what isotropy predicts from its bond error — and an L2 objective
exerts no control over that shape at all. The prior sets the noise; only the
objective sets the residual. Weakness (iv) is precisely a statement that the
residual's anisotropy is currently unmanaged, and the objective is the only place
to manage it.

**Two honest caveats, both mechanistic.** (i) The targets in the stiff
directions are themselves uncertain at the tau scale (MD sampling noise, force
field, 1 ns). The variance-optimal weight under target noise delta^2 is
1/(tau_k^2 + delta_k^2), not 1/tau_k^2, so the full weights over-weight exactly
the directions whose labels are least reliable. Ship a tempered family
w_k = (tau_bond/tau_k)^(2 gamma) with gamma in {0, 0.5, 1} and select gamma on
fitting folds; gamma = 0 is the current objective and gamma = 1 the full one.
(ii) A_var ~ 79 is a badly conditioned regression. Whiten the *target* rather
than weighting the *loss* — train on L^{-1} r with a plain L2 — which is the same
objective with a well-scaled gradient.

## C. De novo generation without a bond graph: the anisotropy cannot live in the prior, which makes B mandatory

**The blocking fact, derived.** In flow matching the prior is the t = 1 base
distribution. For de novo generation there is no molecule at t = 1 — atom
positions are the thing being generated — so there is no B-matrix, no Sigma*, and
**the anisotropic prior is not definable**. This is not an engineering obstacle
to route around; it is a statement that the ensemble-transfer setting (given
molecule, given graph) and the de novo setting are structurally different, and
that for de novo the anisotropy must live in the **path** or the **objective**.
Question B therefore is not an optional extra: it is the only route to the
paper's target application. Section 4 should be organised around that.

Four options, with costs.

**C1 — time-dependent stiffness from a distance gate, no graph (recommended).**
Replace the discrete bond list with soft gates on interatomic distance,
g_ij(x_t) = sigma((r_i + r_j + 0.30 - d_ij)/0.15), and build
F(x_t) = sum_ij g_ij f_bond B_ij^T B_ij + (angle/improper terms gated by
g_ij g_jk). Everything is smooth in coordinates and defined at every t. At high
noise no pair is near bonding distance, the gates vanish, and the metric returns
to the identity **automatically**, which is exactly the boundary condition the
base distribution requires. Measured (3.8): F1 0.987 at sigma = 0.05, 0.945 at
0.15, and total gate mass falling to 0.57 of the true bond count at sigma = 1.0.
Cost: one N x N distance matrix per step — for 30 heavy atoms, 435 pairs and
about 50 gated triples with a neighbour cutoff. Negligible against the network.
Risk, to be stated: the gates make the metric a function of x_t, so the loss is
no longer a fixed quadratic form and the change of variables in B.1 holds only
instantaneously. Use a stop-gradient on F(x_t).

**C2 — joint atom/bond generation (MolDiff-style discrete bond diffusion).** The
graph becomes a latent variable present at every t, so B is exact rather than
gated. Mechanistically clean and it also fixes the connectivity-validity problem
independently. Cost: an extra discrete head, an extra noise process, and a
retraining. Justified, but it is a second paper's worth of machinery.

**C3 — two-stage: isotropic until t < t*, then switch on the gated metric.**
Cheapest to implement, but t* is a hyperparameter with no principle behind it and
C1 gets the same behaviour for free with a continuous gate. Not recommended, and
worth stating as the obvious thing that C1 supersedes.

**C4 — keep the prior anisotropic, restrict the claim to conditional ensemble
generation.** Honest and already validated. It is the paper's current scope and
is the fallback if C1 does not land.

## D. Aromatic planarity

B-vector written and finite-difference verified in 3.7; the Gauss-Seidel
experiment in the same section takes the out-of-plane channel from 3.534 to
**0.965** and improves the angle and torsion channels as a side effect, at a 13%
dispersion cost at fixed sigma. Structural reasons it is the right coordinate to
add:

* it is the largest-B-norm stiff coordinate, ||B||^2 = **6.526** against angle
  2.674 and bond 2.000, so isotropic noise damages it most — measured, it is the
  worst channel in both the sampler (3.534) and the trained model (val 3.845);
* it is **exactly orthogonal** to every bond and angle B-vector at planarity
  (R2), so it is a genuinely new direction rather than a fourth constraint on a
  system that already has two, and Gauss-Seidel does not have to fight the
  existing ones;
* it costs only **8.3%** of the degrees of freedom.

**The number the paper reports must change, though (R1): tau_oop is 6.48 deg, not
1.2 deg, and the DiffSBDD planarity comparison is crystal-versus-ensemble.** The
fix is still worth making — the channel really is 3.8x over-dispersed against its
own thermal width — but the "planarity is broken by 4.3x" framing does not
survive and should be replaced by the ensemble-matched 3.845.

## E. Ranking, and the one thing to do first

Effect is on a measured val channel; cost is implementation plus compute.

| rank | addition | measured effect | cost | effect / cost |
|---|---|---|---|---|
| **1** | **improper in the prior's stiff set** | planar improper 3.53 -> **0.965** on the noise; angle and torsion also improve | ~40 lines mirroring the existing angle block; 3 retrains | **highest** |
| 2 | closed-form rank-2 correction (A2) | RMSF **x1.88** at +4.3% angle, +0.1% torsion | ~30 lines, no new parameters, 3 retrains | high |
| 3 | tempered Mahalanobis objective, gamma in {0, 0.5, 1} | channel spread 0.486 -> < 0.20 predicted; the only fix for weakness (iv) | one loss change plus a 3-point sweep; 9 retrains | high, but 3x the compute and the least certain |
| 4 | distance-gated graph-free metric (C1) | unblocks de novo; no channel moves in the current setting | moderate; needs a new de novo evaluation | strategic, not incremental |
| 5 | learned rank-r head (A1) | only what it beats A2 by | a head plus a control arm | do only after 2 |

**Do the improper first.** It is the largest measured effect on a channel the
paper already reports, on a coordinate that is provably orthogonal to everything
the method already constrains, it reuses the Gauss-Seidel machinery verbatim, it
needs no new hyperparameter beyond tau_oop which is now measured, and — the part
that matters for this project — it comes with its own control: because the
out-of-plane direction is exactly orthogonal to the bond and angle rows at
planarity, the bond and angle channels **must not move much**, and if they do,
the implementation is wrong. That is the same class of intervention-level control
that caught the v1 divergence.

Then the rank-2 correction, which is the only cheap thing that addresses the
internal-RMSF channel — the one number that has not moved at all across the whole
programme.

## F. Components considered and REJECTED, with the reason

* **An amortised network predicting correlated w from the graph.** Rejected on
  R3: the target is near-diagonal (median |corr| 0.041), so the network would
  learn a diagonal and the parameters would be wasted. This is the option that
  looked most attractive a priori and the measurement kills it.
* **Bigger backbone / DiT scaling.** No mechanism that is specific to the defect.
  Reverse water-filling says extra capacity lowers theta uniformly, so every
  channel improves in proportion and the *dispersion-normalised* ratios — the
  figure of merit `angle_term_v1` already established — do not move. DiffSBDD is
  a large model and sits at 5.09 deg out of plane.
* **Consistency models, distillation, few-step samplers.** Inference cost only.
  Distillation regresses onto a conditional mean over the reverse path, which
  removes high-frequency structure preferentially; predicted effect is to raise
  the stiff-channel ratios, i.e. the wrong direction. Reject with a sign.
* **Higher-order equivariant features (e3nn l >= 2), fancier equivariant
  backbones.** The existing operators are already SE(3)-equivariant and verified
  centroid-preserving; no measured failure is an equivariance failure. Weak
  conditional only: an l = 1 feature is a natural carrier for the plane normal at
  an sp2 centre, so if D is implemented as a *learned* correction rather than the
  closed-form B-vector, it would need one. Since D is closed-form, it does not.
* **Latent diffusion / VAE compression.** Actively harmful and derivably so: the
  decoder is trained with a Cartesian L2 reconstruction loss, which is exactly the
  isotropic objective this paper argues against, so it would reintroduce the
  defect at the decoder where the prior cannot reach it. Predicted: stiff channels
  return to their unconstrained values, ~4–5.
* **RL / DPO with a force-field reward, geometry reward shaping.** This is a loss
  term by another name, and `angle_term_v1` measured that class: loss terms do not
  generalise to the coordinate they penalise, projections and priors do. Predicted
  effect ~0 on the normalised ratio.
* **Test-time scaling, best-of-n with a geometry filter.** Selection, not
  generation; buys a logarithmic improvement for linear compute, and cannot fix a
  dispersion budget that is in the wrong place — every sample has it.
* **Minibatch-OT / rectified-flow couplings.** No mechanism for anisotropy on its
  own, because OT straightens paths in the L2 metric, which is the wrong metric
  here. Becomes a free and principled upgrade *conditional on* B: once the loss is
  Mahalanobis, the coupling should be OT under the same metric. Listed as a
  dependent, not a candidate.
* **Mamba / SSM backbones, and architecture swaps generally.** No derivation ties
  them to any measured channel.
