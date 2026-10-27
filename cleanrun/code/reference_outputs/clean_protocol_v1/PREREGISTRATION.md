# PRE-REGISTRATION — clean-protocol replication of the $\Sigma^{*}$ test result

Written 2026-10-23, before any code for this replication exists, before any arm of it has
been trained, and before any number from it exists. Hashed immediately; not editable
afterwards. Everything learned later goes into `RESULT.md` in this folder.

All figures quoted below for the *existing* protocol are read from the frozen record
(`reference_outputs/sigma_star_test_v1/TEST_RESULT.md`, `.../laplacian_prior_v1/RESULT.md`,
`.../leakage_audit_v1/audit.json`). None of them is recomputed here, and nothing in this
document was produced by touching the test split.

---

## 0. THIS IS THE THIRD READ OF THE TEST SPLIT. Read this section before anything else.

The test split of the MMP ensemble-transfer corpus (728 pairs, 66 UniProt targets,
1,456 directional examples) will have been evaluated **three times** when this
replication finishes:

| # | date | what was read | pre-registration |
|---|---|---|---|
| 1 | 2026-09-25 | confirmatory read of MEHF-ISO / MEHF-GS (ANISO v3) / MEHF-$\Sigma^{*}$ | `sigma_star_test_v1/PREREGISTRATION.md` (+ Amendments 1, 2) |
| 2 | 2026-09-29 | added MEHF-LAP arm, post-hoc by construction | `laplacian_prior_v1/PREREGISTRATION.md` |
| 3 | this document | retrained ISO / $\Sigma^{*}$ (/ LAP) under the clean protocol | this file |

**The split is therefore no longer a clean single-read instrument, and this document does
not pretend otherwise.** The honest inferential position:

- **Read 1** is the only read whose nominal $\alpha$ is the stated one. Its decision rule,
  its example list and its analysis script were hashed before any test number existed.
- **Read 2** was declared post-hoc in its own pre-registration and is reported in the
  manuscript as an added comparison, not as part of the confirmation.
- **Read 3** adds a third family-wise opportunity. With three reads of one split, the
  nominal 95% intervals should be read as roughly 85% intervals if one insists on a
  crude family-wise correction ($1-0.05\times3 = 0.85$), and any result whose interval
  only just excludes 0 must not be presented as confirmatory. **The $\Sigma^{*}-$ISO
  contrast is not in that regime** (point estimate $-1.152$, nearest interval edge
  $-1.021$, i.e. the effect is about 8.8 times the near-side interval half-width), which
  is the single reason a third read is worth spending at all. If the clean-protocol
  effect lands anywhere near the significance boundary, **the pre-registered reading
  below is "WEAK", not "confirmed"** (§6).

**Why a third read is nevertheless defensible here, and what makes that claim checkable:**

1. **The protocol change was decided for a reason independent of any result.** The pocket
   crop was being selected with a statistic computed over all 100 trajectory frames —
   `pocket_min_distance_over_frames` — i.e. using the very frames the model is asked to
   generate. That is a leak on its face, irrespective of whether it helps or hurts. The
   manuscript already prints the audit bullet *"Projection targets and all other
   conditioning come from the input pose"* (Section "Leakage and Reproducibility Audit"),
   and under the all-frames crop **that printed sentence is false.** The change makes an
   already-published claim true; it is not a search for a better number.
2. **The direction of the fix was fixed before any new number existed.** There is exactly
   one defensible replacement (rank by distance at the conditioning frame, frame index 0,
   the same pose the model is conditioned on), it was already implemented as
   `pocket_selection="frame0"` in `mmp_transfer.pocket_ranking_distance`, and it was
   chosen without reference to its effect. There is no second candidate that could have
   been picked after seeing results.
3. **The only pre-change measurement taken was of the *inputs*, not of the outputs.** On a
   60-system sample, the two crops overlap by mean **0.823**, median 0.833, minimum
   0.604, and **no system has an identical set** — roughly **17 of 96** pocket atoms
   change per system, up to 38 in the worst case. That number was measured from training
   and store metadata, contains no model output and no test metric, and could not have
   told anyone which way the primary contrast would move.
4. **The old numbers are not being replaced quietly.** Whatever this read returns, the
   leaky-protocol figures stay in the record and in the paper's audit, and the
   difference between the two protocols is reported per arm (§5.3). If the previously
   published $-1.152$ turns out to have been partly leak-driven, that is reported as
   such, with the magnitude, even if the clean result still confirms.

**What would have made a third read indefensible, and is therefore forbidden here:** any
change to the architecture, the training budget, the split, the metric, the operating-point
rule, the bootstrap procedure, or the example list. All are frozen (§2). A fourth read is
not available: after this one, the test split is spent for this project and any further
question must be answered on a new corpus.

---

## 1. What changes

Exactly two things change relative to the frozen protocol. Both are closures of
methodological gaps; neither is a modelling improvement.

### Change 1 — pocket conditioning is selected at the conditioning frame only

The pocket is cropped to `max_pocket_atoms = 96`. The ranking statistic changes:

$$
\text{old (leaky)}:\quad d_j^{\min}=\min_{f=0..99}\ \min_i \|\mathbf p_j^{(f)}-\mathbf x_i^{(f)}\|
\qquad\longrightarrow\qquad
\text{new (clean)}:\quad d_j^{0}=\min_i \|\mathbf p_j^{(0)}-\mathbf x_i^{(0)}\|
$$

and the kept set is the 96 smallest, stable argsort, as before. `pocket_selection:
frame0` in the data config; the code path already exists
(`src/mdwm/data/mmp_transfer.py:374-389`).

**What was and was not leaking, stated precisely, because it bounds the whole analysis:**

- The pocket **coordinates** handed to the model were already frame 0 only
  (`mmp_transfer.py:494`, `pocket_coords = product["pocket_coords"][0]`). No pocket
  coordinate from frames 1–99 ever entered the model, under either protocol.
- So the leak is **set-membership only**: *which* 96 atoms, not where they are. It carries
  no per-frame values and, being one crop shared by all 50 generated frames, it **cannot
  encode time ordering** — only a time-aggregated envelope of where the ligand goes.
- That bound cuts both ways and the unflattering half must be said: the primary metric is
  itself a **time-aggregated width**. A time-invariant envelope cue is exactly the kind of
  information a width metric can consume, and exactly the kind a dynamics metric could not.
- The leaked set enters in **three** places, not one:
  (i) the pocket embedding and the equivariant ligand–pocket messages;
  (ii) the **training clash loss, weight 10**, the largest weight in the objective, over
  all generated frames (`mehf.py:1607-1623`, `condition.pocket_coords` is the crop);
  (iii) the **inference projection's clash push** (`mehf.py:1925-1937`), again on the crop.
  Places (ii) and (iii) are where a differential interaction with the prior is plausible;
  see §4.

### Change 2 — the four widths $\tau$ are re-estimated on training-split systems only

$\tau_{\text{bond}}, \tau_{\text{angle}}, \tau_{\text{torsion}}, \tau_{\text{oop}}$ are
re-estimated with every validation and test system excluded, and passed **explicitly
through the config** rather than taken from the module constants in
`src/mdwm/models/harmonic_prior.py:32-35`. The checkpoint must record the four values it
was trained with.

Expected to be close to a no-op, and the reason is exact rather than empirical. Writing
$F=\operatorname{diag}(\tau_{c(k)}^{-2})$,

$$
\Sigma^{*}(cF)=\big(c\,B^{\top}FB\big)^{+}=c^{-1}\Sigma^{*}(F),\qquad
L\mapsto c^{-1/2}L,\qquad
\alpha=\frac{(\mathbb E\|\bm\epsilon_{\mathrm{iso}}\|^2)^{1/2}}{(\mathbb E\|L\bm\xi\|^2)^{1/2}}\mapsto c^{1/2}\alpha,
$$

so the realised draw $\bm\epsilon=\alpha L\bm\xi$ is **exactly invariant** under a common
rescaling of all four widths. Only the three independent *ratios* of the $\tau$ can
matter. The audit's recomputation without val/test moved them by $+0.02\%$ (bond),
$+0.26\%$ (angle), $+0.40\%$ (torsion), $-0.03\%$ (oop); after removing the common mode
the residual spread is $0.43\%$, so no diagonal entry of $F$ changes by more than
$\approx(1.0043)^{2}-1=0.86\%$. This is a derivation, not a hope, and it becomes a unit
test (§7, T4).

Note two asymmetries, recorded so they cannot be discovered later:
MEHF-ISO contains **no** $\tau$, so Change 2 cannot affect it at all; MEHF-LAP is
amplitude-matched and $\tau$-free as well. Change 2 therefore touches only the
$\Sigma^{*}$ arm. MEHF-GS *does* use absolute widths and is **not** being retrained (§3).

---

## 2. What is held fixed

Held fixed, bit-for-bit where possible, and any deviation voids the replication:

| | value |
|---|---|
| corpus / mappings | `outputs/mmp_transfer_corpus_v1/pairs.csv`, `atom_mappings.json` |
| store | `data/MISATO/extracted/frames100_all_v1` |
| split | by UniProt target, 3,489 / 696 / 728 pairs; 345 / 106 / 66 targets; 0 target and 0 PDB overlap |
| test example list | `reference_outputs/sigma_star_test_v1/test_examples.csv`, 1,456 directional examples, unchanged |
| architecture | 1.44 M params, hidden 96, 3 equivariant + 2 temporal layers, coarse stride 5, $T=50$ |
| objective | CFM (coarse + fine) + bond ($\lambda_b=1$) + clash ($\lambda_c=10$, $d_c=1.2$ Å); **no angle term** |
| source | rigid 0.5 Å / 0.08 rad, $\sigma_0=0.06$ Å, $(s_r,s_i)=(0.35,3.5)$, fine fraction 0.25 |
| budget | 8 epochs ($\approx$12,000 steps), AdamW, $3\times10^{-4}\to10^{-5}$ cosine, batch 4, cap 16,000 pairs |
| folds / seeds | 3 target-disjoint fit/dev folds, `fold_seed 20260918`, `train seed 2026`, checkpoint by lowest dev loss |
| inference | `--rigid 0.32 --steps 16 --seed 20260918 --impropers --angle-projection on --restore off --angle-target-width 0.0 --projection-iterations 8 --projection-angle-relaxation 0.5` |
| metric | per-example four-channel width ratios {bond, angle, torsion, out-of-plane}; spread $=\max_c r_c-\min_c r_c$; per-example value $=$ mean over the arm's three fold checkpoints |
| usable examples | Addendum A rule verbatim: all four ratios finite in all three folds of every clean arm; dropped examples counted, never imputed |
| statistics | 95% cluster bootstrap over UniProt, 10,000 resamples, **seed 20260925** (same seed as read 1, so the resample structure is shared and the cross-protocol comparison in §5.3 is paired) |

**The training budget is frozen deliberately.** Retraining is an opportunity to
accidentally give the method more compute; 8 epochs is not negotiable, and a run that
crashes is restarted with an identical command and logged, never extended.

---

## 3. Arms

| arm | prior | retrained under the clean protocol? |
|---|---|---|
| MEHF-ISO | isotropic $\mathcal N(0,\sigma_0^2 I)$ | **yes**, folds 1–3 |
| MEHF-HARM ($\Sigma^{*}$), *the method* | $(B^\top F B)^{+}$, complete coordinate set | **yes**, folds 1–3 |
| MEHF-LAP | $L^{+}\otimes I_3$, bond-graph Laplacian | **yes if compute allows**, folds 1–3, declared before the read (§3.1) |
| MEHF-GS (ANISO v3) | Gauss–Seidel, bonds + angles | **no** — carried over, cross-protocol only |

Configs: `configs/clean_{iso,harm,lap}_f{1,2,3}.yaml`, each identical to its frozen
counterpart (`internal_cal_full_f*`, `sigma_star_full_f*`, `laplacian_full_f*`) with
`data.pocket_selection: frame0` added, the four `model.tau_*` stated explicitly, and the
output root changed. A diff of more than those lines voids the replication.

### 3.1 Recorded weakening of the decision rule, before the read

Read 1's rule required $\Sigma^{*}$'s median spread to be below **both** comparators'.
MEHF-GS is not being retrained, so under the clean protocol there is no like-for-like GS
column. **The ordering requirement therefore applies only to arms retrained under the
clean protocol** (ISO, and LAP if it is trained); the GS comparison is reported as
cross-protocol and explicitly labelled as such. This is a weaker test than read 1's and
is recorded here as a weakening, in the same spirit as Amendment 2 of read 1, rather than
presented as equivalent.

**The LAP decision must be made and logged before the test read, not after.** Write
`LAP_DECISION.txt` in this folder (`trained` / `not trained`, with the reason and the
timestamp) at the moment the training queue is fixed. If LAP is trained, it is scored; if
it is not, its leaky-protocol number 1.352 is quoted as cross-protocol, exactly like GS.
Deciding this after seeing $\Sigma^{*}$'s clean number would be arm selection.

---

## 4. Is the pocket leak common-mode? A reasoned answer, not an assertion

**The argument to be audited.** Every prior arm — ISO, GS, LAP, $\Sigma^{*}$ — received
*identical* pocket conditioning, built by the identical all-frames rule. Therefore the
leak cannot by itself favour $\Sigma^{*}$ over ISO, and the published contrast is safe.

**What is airtight in it.** Two things, and they are worth keeping:

1. *No selection asymmetry.* Nobody chose a different crop for different arms. The leak is
   not an instance of the method being tuned on information withheld from the baseline.
2. *No additive mechanism.* If the leak's benefit entered each arm's spread as an additive
   constant $\delta$, it would cancel exactly in the paired difference
   $\text{spread}_{\Sigma^{*}}-\text{spread}_{\text{ISO}}$.

**Why it is not airtight.** The conclusion requires the leak's effect to be additive and
arm-independent, and three mechanisms break that. "Identical input" does not imply
"identical effect" in a nonlinear model whose *other* inputs differ by construction.

**M1 — the metric is multiplicative in amplitude, and the arms differ in shape-spread.**
Write each arm's channel ratios as a level times a shape, $r_c \approx \bar r\,u_c$ with
$\sum_c u_c = 4$. Then

$$
\text{spread} = \bar r\,(\max_c u_c - \min_c u_c),
$$

so the *same* fractional improvement in amplitude calibration $\bar r$ buys an absolute
spread reduction proportional to the arm's shape non-uniformity. On the frozen test read
ISO's shape is badly non-uniform (oop 2.87 against bond 1.02) while $\Sigma^{*}$'s is
nearly flat (1.90 / 1.40 / 1.16 / 1.66). If the leak helps the model set the motion
envelope — and an envelope cue is precisely what a time-invariant contact set supplies —
it reduces ISO's spread by $(\max u-\min u)_{\text{ISO}}\big/(\max u -\min
u)_{\Sigma^{*}}$ times more, in absolute units, than it reduces $\Sigma^{*}$'s. From the
frozen per-channel medians that factor is
$\frac{(2.872-1.017)/1.550}{(1.894-1.155)/1.526}=\frac{1.197}{0.484}=2.5$, where $1.550$
and $1.526$ are the two arms' mean ratios. That is differential, and its sign here **favours the method**: removing
the leak should widen the gap, not shrink it. The sign flips only if the leak made models
*over*-disperse rather than calibrate.

**M2 — the operating point is re-selected per arm from leaky fitting-fold curves.**
$s^{*}$ is the torsion-ratio-crosses-1 point, found on $\{1.0,\dots,4.0\}$. ISO crosses at
1.11–1.31, $\Sigma^{*}$ at 3.25–3.79, high on the amplitude curve where the frozen read
already recorded fragility: $\Sigma^{*}$'s spread went 0.216/0.270/0.460 on fitting folds
to 0.770 on test, a $1.7$–$3.4\times$ degradation, while ISO and GS moved $\approx1.05\times$.
A leak-induced shift in the torsion-vs-scale curve translates into a larger $\Delta s^{*}$
for the arm whose curve is flatter near its crossing, and into a larger $\Delta$spread for
the arm sitting where spread moves fastest with scale. Both of those are $\Sigma^{*}$.
Sign: **unknown**. Magnitude: the fold-to-fold scatter of $\Sigma^{*}$'s fitting-fold
spread (0.216 to 0.460) is the right order for how much this channel can move, i.e. up to
$\sim0.24$ spread units — about 20% of the headline $-1.152$.

**M3 — the leak lives inside the clash loss (weight 10) and the clash projection, and
those are amplitude-gated.** Because $d_j^{\min}\le d_j^{0}$ atom-by-atom, the leaky crop
systematically *adds* atoms that are far at frame 0 but are approached later, and *drops*
atoms at frame-0 ranks near the cut that are never approached. The added atoms are exactly
the ones the ligand would clash with *if it moved where it really moves*. So the leaky
crop is not just an input feature: it is a **target-informed barrier inside the loss**, a
fence drawn around the true trajectory channel. How hard that fence pushes depends on how
wide the generated ensemble is, and $\Sigma^{*}$'s ensembles are about four times wider
than ISO's (internal RMSF 1.49 vs 0.379 on the frozen read). The clash term is therefore
active far more often for $\Sigma^{*}$, so $\Sigma^{*}$ consumed more of the leak than ISO
did. Sign on the spread: each clash push is a Cartesian displacement, and by the
Section-II Wilson law a Cartesian displacement lands disproportionately in the stiff
channels (43 and 32 physical widths per Å for bond and angle), so **more pushes inflate
bond and oop ratios and increase the spread**. Removing the leak removes well-aimed pushes
and may make $\Sigma^{*}$'s spread look *better* for a bad reason — fewer clash
corrections because the fence has holes in it. This mechanism points at the method and it
has a measurable signature, which is why §5.2 adds a clash-rate-against-the-full-pocket
secondary. Without that secondary this replication could buy a prettier spread by losing
the fence, and nobody would see it.

**One mechanism that is *not* available, stated so it is not invoked later.** The leak
cannot act through the pocket *geometry*: frame-0 coordinates were used under both
protocols. And it cannot act through frame-to-frame dynamics: one crop serves all 50
frames. Any differential story must run through set membership and an amplitude or
envelope channel.

**A bound on how much of the published effect could possibly be leak-mediated.** The
prior's channel-balancing advantage is demonstrable with **no pocket input and no learned
model at all**: drawing noise from each prior around 150 reference poses, matched on
internal RMSF, gives spread 7.92 (ISO), 4.81 (Laplacian), **1.01 ($\Sigma^{*}$)**. That
measurement contains zero pocket information, so the mechanism the paper claims exists
independently of pocket conditioning. It does not prove the *trained* arms' gap is
leak-free — a trained model is not its prior — but it makes "the whole $-1.152$ was the
leak" implausible, because the leak would have to erase about **89%** of the effect (from
$1.152$ down to below the $0.131$ near-side interval half-width) for the reading to flip.

**Conclusion and confidence.**

- The common-mode argument is **sound as a defence against selection asymmetry and against
  an additive artifact, and insufficient as a defence in general.** Confidence that it is
  insufficient as stated: **high, ~0.85.** M1 and M3 are mechanisms, not possibilities;
  only their magnitudes are unknown.
- The *net* sign is genuinely undetermined: M1 favours the method, M3 points against it,
  M2 is unsigned. I decline to claim the leak was harmless, and I decline to claim it was
  decisive.
- Confidence that the clean protocol leaves the primary contrast negative and
  interval-excluding-0: **moderately high, ~0.85** — driven by the 89% erasure that would
  be required and by the pocket-free noise-only measurement, discounted by M2's
  operating-point fragility, which is the one channel that could move $\Sigma^{*}$ by
  several tenths.
- **This raises the stakes of the replication rather than lowering them.** If M3 is real,
  the clean protocol may *improve* $\Sigma^{*}$'s spread while *worsening* its physical
  plausibility. That outcome must not be reported as a win; §5.2 and §6 pre-commit the
  reading.

---

## 5. Analysis, fixed now

### 5.1 PRIMARY

Median per-example four-channel spread per clean arm, and the paired difference

$$
\Delta = \mathrm{median}\big(\text{spread}_{\Sigma^{*}}\big)-\mathrm{median}\big(\text{spread}_{\text{ISO}}\big)
$$

reported as the mean paired difference with a 95% UniProt cluster bootstrap
(10,000 resamples, seed 20260925) on the common example set, plus the fraction of
examples on which $\Sigma^{*}$ is lower. Identical definition to read 1. For reference,
read 1 gave $\Delta=-1.152$, CI $[-1.370,-1.021]$, 98.8% of 1,410 examples from 60
targets, with medians $\Sigma^{*}$ 0.770 / ISO 1.974 / GS 2.527 / LAP 1.352.

### 5.2 SECONDARY, reported regardless of the primary and not part of the decision

1. Per-channel median ratios (bond / angle / torsion / oop) per clean arm.
2. Internal RMSF ratio per clean arm (the over-dispersion that read 1 recorded as 1.49
   for $\Sigma^{*}$ and 0.379 for ISO).
3. Realised test torsion ratio per arm and fold — whether the matching transferred.
4. Fitting-fold $\to$ test degradation per arm, the quantity on which $\Sigma^{*}$ was the
   only arm that got materially worse.
5. **Clash rate against the FULL available pocket** (all `n_pocket_available` atoms, not
   the 96-atom crop): fraction of generated ligand-heavy/pocket-heavy pairs below 1.2 Å,
   and the per-example count, for every clean arm **and**, recomputed from the frozen
   leaky checkpoints, for every leaky arm. This is the M3 detector of §4. It is the only
   secondary whose failure can block a headline (§6).
6. Crop diagnostics, no model needed: overlap of the frame-0 crop with the all-frames crop
   on the full corpus (mean / median / min, extending the 60-system sample), and the
   recall of true all-frames contacts (pairs below 4 Å at any frame) achieved by the
   frame-0 crop.
7. Per-fold values for everything above.

### 5.3 The cross-protocol difference-in-differences, declared as diagnostic only

$$
\mathrm{DiD}=\big(\Delta_{\Sigma^{*}-\mathrm{ISO}}\big)_{\text{clean}}-\big(\Delta_{\Sigma^{*}-\mathrm{ISO}}\big)_{\text{leaky}}
$$

computed on examples common to both protocols (the frozen 1,410 intersected with the
clean usable set; the counts kept and dropped are reported), with the same cluster
bootstrap and the same seed so the resample structure is shared.

**The DiD is a diagnostic and not a test, for a reason that must be stated in advance.**
The clean and leaky arms are independent retrains, so the DiD contains retraining
variance as well as the leak effect, and this project has only one estimate of that
floor: the angle-term factorial's retrain control, which put pure retraining noise at
$+1.5\%$ to $+3.4\%$ on a channel ratio. With three folds and no replicate seeds the two
cannot be separated cleanly. Consequently:

- the **decision rests on the clean-protocol primary alone** (§6);
- the DiD is reported with the retrain-noise caveat attached, in both directions;
- if $|\mathrm{DiD}|$ exceeds roughly three times what retraining noise could plausibly
  produce **and** its sign says the leak was flattering the method, that is reported
  prominently in the manuscript's audit section **even if the clean primary still
  confirms**.

### 5.4 The two model-free controls

Neither needs retraining. Both are evaluated on the same test examples, at the same
matched-torsion amplitude convention where an amplitude exists, and **each in two
variants: with the inference projection OFF and with the identical 8-sweep projection
ON**, because the projection is the shared machinery and reporting only one variant would
make the controls incomparable to the arms.

**Control A — $\Sigma^{*}$ source sampled directly, no learned flow.** Draw
$\bm\epsilon=\alpha L\bm\xi$ around $\mathbf x_{\text{ref}}$ (plus the same rigid terms at
$s_r=0.32$), 50 independent frames per example, with $s_i$ set by the same
torsion-matching rule on the fitting folds. **This control has no pocket input, no
training, and no learned parameters**, so it is the matched control for the failure mode
this group keeps hitting: an apparent conditioning effect that is really a static,
shape-only effect. It answers "how much of the channel balance is the source, and how
much is the flow?"

**Control B — copy-the-parent, no model at all.** The ensemble is
$\mathbf x^{(f)}=\mathbf x_{\text{prod}}^{(0)}+\mathbf d^{(f)}$ where $\mathbf d$ is
exactly the `parent_motion_prior` the model receives
(`mmp_transfer.py:477-479`): the mapped parent displacement relative to the parent block's
first frame, after one Kabsch superposition of the parent block mean onto product frame 0,
and **exactly zero on added atoms**. No amplitude is fitted; there is nothing to fit. This
control answers "what does the conditioning channel give if you just use it as the
answer?" and it is the known-strong baseline family from the earlier MMP work, where
copy-the-parent sat at the crystallographic floor.

Reading of the controls is fixed in §6. They are reported for the clean protocol; the
leaky-protocol versions are not needed because neither control uses the pocket crop
except through the projection's clash push, which is reported for both.

---

## 6. Reading of the outcome, fixed in advance

Applied to the clean-protocol primary, in this order, and no other tier may be invented
afterwards.

**Tier R — REPLICATED.** $\mathrm{median}(\Sigma^{*}) <$ every retrained comparator's
median, the 95% CI of $\Delta$ excludes 0, and the point estimate is $\le -1.021$ (inside
or beyond read 1's interval). Reading: the clean protocol reproduces the published effect;
the clean numbers become the paper's numbers and the leaky ones move to the audit.

**Tier A — ATTENUATED BUT POSITIVE.** CI excludes 0, correct sign, point estimate in
$(-1.021,\,-0.30]$. Reading: the effect is real but smaller than published. **The clean
numbers replace the published ones everywhere, including the abstract**, the "2.6 times
more uniform" ratio is recomputed from the clean medians, and the attenuation is stated as
attenuation, with the DiD and the retrain-noise caveat.

**Tier W — WEAK.** CI excludes 0, correct sign, $|\Delta| < 0.30$. Reading: direction
holds, magnitude is far smaller than published. Given that this is the third read of the
split (§0), a barely-excluding interval here is **not** reported as a confirmation; the
manuscript says the direction replicates and the magnitude does not, and the headline
becomes the per-channel table (specifically the out-of-plane channel, which is where read
1 showed the margin actually lives) rather than the aggregate spread.

**Tier N — NOT REPLICATED.** CI contains 0, or the sign reverses, or
$\mathrm{median}(\Sigma^{*})$ is not below a retrained comparator's. Reading: the claim is
withdrawn. The clean protocol becomes the paper's result, the published $-1.152$ is
reported as having depended on the all-frames pocket crop, and the paper's contribution is
restated as the analysis of Sections II–III plus a negative replication. No further
retraining, no further scales, no fourth read.

**Independent blocking conditions** — these can downgrade a Tier but never upgrade one:

- **B1 (amplitude premise).** If any retrained arm's realised test torsion ratio falls
  outside $[0.85,1.25]$, the matched-amplitude premise failed for that arm and it is
  reported as unmatched, exactly as GS was in read 1.
- **B2 (the fence).** If clean $\Sigma^{*}$'s spread improves relative to leaky
  $\Sigma^{*}$ **and** its full-pocket clash rate (§5.2.5) worsens by more than the
  largest clash-rate difference seen among the three leaky arms, then M3 is operating and
  the spread improvement was bought by losing a target-informed barrier. The improvement
  is reported as an artifact of the fix and **is not claimed as a gain**.
- **B3 (the controls).** If either Control A or Control B, with the projection ON, attains
  a median spread **below** clean $\Sigma^{*}$'s, then the primary metric does not require
  the learned flow. The paper may then still claim what it actually tested — that among
  variants of one model at matched torsional width, the source distribution determines how
  noise is allocated across channels — but it may **not** present the spread as evidence
  that MEHF generates good ensembles, and the control's number must appear in the same
  table as the arms, not in an appendix. This is a restatement requirement, not a
  withdrawal: the arms-only comparison is a valid test of the prior either way.
- **B4 (prior art).** If MEHF-LAP is trained and its clean median spread is at or below
  clean $\Sigma^{*}$'s, the contribution loses to the published Laplacian prior and the
  paper says so plainly, whatever the ISO contrast does.

---

## 7. What the implementer must build, and the tests it must pass

No training code is specified here beyond the configs; the architecture and loop are
unchanged. The following are requirements, and each one is a test that can fail.

**Shapes** (per example, after the clean crop; $N$ = ligand heavy atoms, $T=50$):
`coords (T,N,3) float32`, `product_init (N,3)`, `parent_motion_prior (T,N,3)`,
`product_to_parent (N,) int64` with $-1$ on added atoms, `product_features (N,6)`,
`edit_features (N,5)`, `pocket_init (96,3) float32`, `pocket_features (96,2)`,
`n_pocket_available` scalar recorded per example. Batched: `(B,T,N_max,3)` with
`atom_mask (B,N_max)` and `pocket_mask (B,96)`; clash pairs capped at 16,000 per batch.

**T1 — leak poison test (decisive).** Overwrite the store field
`pocket_min_distance_over_frames` with NaN in an in-memory copy and build examples with
`pocket_selection="frame0"`. Every built example must be **bit-identical** to the
un-poisoned build and contain no NaN. The same build with
`pocket_selection="min_over_frames"` must raise or produce NaN. If T1 does not
distinguish the two, the fix is not in effect.

**T2 — frame-only test.** Zero (or randomise) frames 1–99 of `ligand_coords` and
`pocket_coords` in an in-memory copy. With `frame0` selection the *crop index set* must be
unchanged; with `min_over_frames` it must change on at least 55 of the 60 sample systems.

**T3 — crop is a rigid-motion invariant.** Apply a common random $R\in SO(3)$, $\mathbf t$
to ligand and pocket at every frame. The selected index set must be identical, and the
four channel ratios must be identical to $10^{-10}$ (they are internal coordinates). Also
assert generated-ensemble equivariance end-to-end to $10^{-5}$ relative, and permutation
equivariance under a consistent relabelling of ligand atoms together with `bond_index`,
`product_to_parent` and the internal-coordinate enumeration.

**T4 — $\tau$ scale invariance (makes P1 a test, not a hope).** Multiply all four $\tau$
by $c=2$ and by $c=0.5$. The realised $\bm\epsilon=\alpha L\bm\xi$ must agree with the
$c=1$ draw to $10^{-10}$ relative, for a fixed $\bm\xi$. If it does not, the
amplitude-matching $\alpha$ is not implemented as specified and Change 2 is not a no-op.

**T5 — frozen-record reproduction, before any new training starts.** The 60-system crop
overlap must reproduce mean 0.823 / median 0.833 / min 0.604, and at least one frozen
leaky checkpoint must reproduce its recorded fitting-fold spread to the tolerance already
established in the release audit ($1.5\times10^{-6}$ relative). A pipeline that cannot
reproduce the old numbers cannot be trusted to produce new ones.

**T6 — config diff guard.** Assert that each `clean_*` config differs from its frozen
counterpart only in `data.pocket_selection`, the four `model.tau_*`, and
`outputs.root`. Assert the four $\tau$ actually used are recorded inside the checkpoint.

**T7 — eval inherits the clean selection.** `mehf_internal_geometry_spectrum.py` reads
`pocket_selection` from the checkpoint's own data config
(`scripts/mehf_internal_geometry_spectrum.py:210`). Assert at run time that the resolved
value is `frame0` for every clean run and `min_over_frames` for every carried-over leaky
run, and write it into the output JSON's `config_check`. A clean checkpoint evaluated
under the leaky crop is a silent protocol violation and this assertion is the only thing
standing between us and it.

**Order of operations, binding.** (1) T1–T6 pass. (2) Re-estimate $\tau$ on train only and
freeze them into the configs. (3) Train the arms. (4) Run the matched-torsion scan on
fitting folds and **freeze $s^{*}$ into a JSON before any test command is issued**,
logged with a timestamp in `CHAIN.log`, exactly as read 1 did. (5) Hash the analysis
script with zero `test_*.json` on disk. (6) Read test once. (7) Write `RESULT.md`,
scoring every prediction in §8 including the wrong ones.

---

## 8. Predictions, stated before any number of this replication exists

These are predictions. **None of them is part of the decision rule in §6.** Each is
scored in `RESULT.md` as CONFIRMED / HALF RIGHT / WRONG, in the manner of
`laplacian_prior_v1/RESULT.md`, where one of three predictions was confirmed, one was half
right and one was partly wrong.

**P1 — Change 2 is a no-op, and nearly exactly so.** The re-estimated $\tau$ move the
clean fitting-fold channel ratios by $<1\%$ each and the clean test median spread by
$<0.02$ absolute, relative to a $\Sigma^{*}$ run with the old $\tau$ under the same clean
crop. *Reason:* $\bm\epsilon=\alpha L\bm\xi$ is exactly invariant under a common rescaling
of the four widths (§1), so only the $0.43\%$ residual ratio spread can act, and it enters
$F$ quadratically at $\le0.86\%$. *Risk:* low. This is close to a derivation; if it fails,
the amplitude matching is not implemented as the paper describes, which is itself worth
finding out. Confidence 0.95.

**P2 — the primary survives the clean protocol.** Clean $\Delta_{\Sigma^{*}-\text{ISO}}$
lies in $[-1.45,-0.85]$ with a 95% CI excluding 0; clean median spreads land at
$\Sigma^{*}\in[0.65,1.10]$ and $\text{ISO}\in[1.75,2.20]$. *Reason:* the leak is
set-membership only, ~17 of 96 atoms, with no coordinates and no time ordering; and the
prior's channel-balancing mechanism is measurable with no pocket input and no model at all
(noise-only spread 1.01 vs 7.92). For the reading to flip, the leak would have to account
for ~89% of the published effect. Confidence 0.85.

**P3 — the risky one: the clean gap moves AWAY from zero, not toward it.**
$\mathrm{DiD} \le 0$, i.e. $|\Delta_{\text{clean}}| \ge |\Delta_{\text{leaky}}| = 1.152$.
*Reason:* M1 of §4 — spread $\approx$ level $\times$ shape non-uniformity, and ISO's shape
non-uniformity is 2.5 times $\Sigma^{*}$'s, so an envelope cue that improves amplitude
calibration flatters ISO about 2.5 times more in absolute spread; taking it away should
cost ISO more than $\Sigma^{*}$. *Why it can lose:* M3 points the other way and may be
larger — $\Sigma^{*}$'s ensembles are ~4 times wider, so $\Sigma^{*}$ consumed more of the
leaked fence through the weight-10 clash term, and taking it away could cost $\Sigma^{*}$
more. **I put this at roughly 0.55 and expect to be told which way by the data.** This is
close to a coin flip and is stated as one.

**P4 — $\Sigma^{*}$ is again the only arm that degrades materially from fitting folds to
test.** Its fitting-fold-to-test spread ratio exceeds 1.3 while ISO's stays below 1.15.
*Reason:* the degradation is a property of sitting at $s^{*}\approx3.2$–$3.8$, high on the
amplitude curve, which is set by the prior and not by the pocket protocol. *Risk:*
moderate — if the operating-point channel M2 is the dominant leak route, cleaner
conditioning could stabilise the torsion-vs-scale curve and shrink the degradation, in
which case this is WRONG and the diagnosis of read 1's degradation changes. Confidence 0.7.

**P5 — Control A (the raw $\Sigma^{*}$ source, no flow, no pocket) is in the same league
as the trained method on the primary, and I predict it does not beat it.** With projection
ON, Control A's median spread lands in $[0.8,1.6]$: below clean ISO, above clean
$\Sigma^{*}$. *Reason:* the pocket-free noise-only measurement already gives 1.01 at
matched internal RMSF, so most of the channel balance is the source, not the flow; but the
raw source has no mechanism to respect the pocket, the mapped edit, or a 50-frame envelope,
and the trained arm's projection and velocity should improve the oop and bond channels
further. *Why it can lose, and what it would mean:* if Control A comes in **below** 0.770,
the primary metric is largely achievable with no learned model, blocking condition B3
fires, and the paper's headline must be restated as a prior comparison with the control in
the same table. Confidence 0.6 — this is the control I most expect to be surprised by, and
it is the one that matters most for the user's recurring failure mode.

**P6 — Control B (copy-the-parent) beats ISO but not $\Sigma^{*}$ on the primary.** With
projection OFF its median spread lands in $[1.0,2.5]$; with projection ON, in
$[0.5,1.4]$. *Reason, worked rather than asserted:* for coordinates among mapped atoms the
control reproduces real MD internal widths, so those ratios sit near 1. But added atoms
are frozen while the mapped core carries the parent's full rigid wander (0.7–2.9 Å
displacement RMS by the corpus's own measurement), so every bond and angle spanning the
attachment acquires a width of order 1 Å / tens of degrees. With ~1–2 attachment bonds
of ~30 and ~5 attachment angles of ~50, the channel means inflate to roughly
$(29\times0.033+1.0)/30 \approx 0.065$ Å (bond ratio ~2) and
$(45\times2.77+5\times38)/50 \approx 6.3^{\circ}$ (angle ratio ~2.3), against torsion and
oop near 1 — a spread of order 1.3. The projection, whose targets are the frame-0 bonds
and angles, should repair most of that inflation and could push the spread below 0.770.
*Why it can lose:* my attachment-count arithmetic is a back-of-envelope over the corpus's
edit-size distribution, and the whole prediction turns on it; if edits are larger than
assumed the raw spread is much worse than 2.5, and if the projection is more effective
than assumed the projected spread drops below 0.5 and B3 fires. Confidence 0.5 on the
bands, 0.7 on the ordering "beats ISO".

**P7 — the fence has holes, and they are visible.** The frame-0 crop's recall of
true all-frames contacts (pairs below 4 Å at any frame) is below 0.90, and every clean arm
shows a higher full-pocket clash rate than its leaky counterpart. *Reason:*
$d_j^{\min}\le d_j^{0}$ atom-by-atom, so the frame-0 crop must omit some atoms the ligand
really approaches; the clash term can only fence what it is shown. *Consequence if
confirmed:* it does not weaken the primary, but it means the clean protocol is honest
rather than strictly better conditioning, and the manuscript must say so in one sentence.
Confidence 0.8.

---

## 9. Single-read rules and provenance

- The test split is read **once** under this document, and it is spent afterwards for this
  project. `docs`/release notes record the date it was spent.
- A run may be restarted only if it did not complete, with an identical command, logged in
  `CHAIN.log` with the reason (read 1 and read 2 both logged such a restart; read 2's fold
  2 OOM is the precedent and no output of the failed attempt was used).
- No configuration change, no scale change, no arm addition and no metric change after the
  first clean `test` number is seen. If something is discovered before the read that
  requires a change, it is written as a dated **Amendment** in this file, hashed again,
  with the test split untouched — the mechanism read 1 used twice.
- The analysis script is hashed with zero `test_*.json` on disk, and its hash recorded in
  `CHAIN.log` before the first test command.
- `RESULT.md` scores every prediction in §8, including the ones that come out wrong, and
  reports every secondary in §5.2 whether it flatters the method or not.
- The sha256 of this file is taken once it is final, by the agent that runs the code, and
  recorded next to it as `PREREGISTRATION.sha256`. It was deliberately **not** computed
  while the document was being drafted.

## 10. Consequences for the manuscript, whatever the outcome

Recorded now so they are not negotiated after the result:

1. The audit bullet "Projection targets and all other conditioning come from the input
   pose" is **replaced** by an explicit statement of the old crop rule, the 0.823 overlap,
   the fix, and the clean numbers.
2. The experimental-setup paragraph gains one sentence defining the crop rule, because at
   present the paper says only "up to 96 pocket atoms" and the rule was not stated.
3. Section V gains the two model-free controls of §5.4 in the same table as the arms.
4. The count of test reads (three) and this document's hash are part of the release.

---
---

# AMENDMENT LOG

DRAFT 1 of everything above was hashed at 2026-10-23 18:53, sha256
`a51668b7af5d8ba9ec4243597fbcf1046ba9e72bebd7ee6c661cad5590ff322e`, recorded in
`outputs/clean_protocol_v1/logs/CHAIN.log`, **before any clean-protocol training began**
and before any clean-protocol number existed. Nothing above this line has been edited, and
that is verifiable: draft 1 had no trailing newline, so appending gave its last line one,
and the committed text is recovered byte-for-byte by

```
head -638 PREREGISTRATION.md | head -c -1 | sha256sum
#   a51668b7af5d8ba9ec4243597fbcf1046ba9e72bebd7ee6c661cad5590ff322e
```

which has been checked. That one added byte is the only difference between the text above
and the hashed draft.
Every change below is an amendment in the style of `sigma_star_test_v1`'s Amendments 1
and 2: dated, with its reason, recorded rather than applied quietly, and written while
the test split is still untouched.

| # | date | what it changes | why |
|---|---|---|---|
| 1 | 2026-10-23 | 28 pool-limited systems excluded from train/dev/val/test; §2 split and example-list rows revised | the provable-identity argument for the frame-0 crop fails on them |
| 2 | 2026-10-23 | $\tau$ re-estimated on the largest available post-exclusion train-only population; P1's confidence revised and P1 made scorable without a tenth training run | the small-sample attempt was within one bootstrap SE and cannot be adopted as evidence of a no-op |
| 3 | 2026-10-23 | $s^{*}$ reported two ways (fitting-fold PRIMARY, dev-fold SECONDARY) with a downgrade-only reading rule | the primary operating point is selected in-sample; the dev variant is in-support and must not become a second shot at significance |
| 4 | 2026-10-23 | B2 adopted verbatim; the full-uncropped-pocket clash rate promoted from optional to a required secondary with a reporting gate | without it the fix can buy a prettier spread by losing a target-informed barrier (§4, M3) |
| 5 | 2026-10-23 | P5 and P6 scored against the dev measurements of the two model-free controls; B3 repaired to be amplitude-controlled; a new amplitude-free statistic $\psi$ defined for reading the controls only | the measured controls expose that the primary metric is amplitude-gameable and that Control A is unmatchable on the frozen grid |
| 6 | 2026-10-23 | §10 retargeted to the authoritative submission source `paper/MEHF_TCBB/main.tex` (1,667 lines, Sections I–V, no appendices), as a line-keyed edit list E1–E13; three facts established in it (the 96-atom cap is stated nowhere; the projection's clash step is undisclosed and the stated relaxation does not cover it; a model-free control at lines 1577–1583 already beats the method on spread) and IV-D's prior/projection separation assessed | the submission version is what is being submitted; §10 as drafted edited text that is not in it, and described the edits abstractly |
| 7 | 2026-10-24 | noise-scale grid extended $[1.0,4.0]\to[1.0,5.0]$ after the first grid run; §2's grid row revised; §A3.3 case 2 excluded on the measured $s^{*}$; P1a REFUTED and P1b WRONG by its own bound; silent-fallback audit recorded as resolved | HARM's torsion ratio topped out below 1 in 4 of 12 cells, leaving the primary on two folds and the secondary on none; the matched-amplitude premise cannot be kept on the frozen range |

Amendments 1–4 and 6 implement decisions 1–5 taken by the first author at
2026-10-23 ~18:5x and logged in `CHAIN.log`. **Amendments 5 and 7 are the two made in
response to data**, and each states up front what was seen before it was written:
Amendment 5 is forced by the dev measurements of the two model-free controls and repairs a
rule this document already stated (§A5.5), and Amendment 7 widens a candidate range that
draft 1 froze, which is the most serious kind of change this document can carry and is
argued rather than asserted in §A7.2. **No amendment was made after a test number existed,
because no test number exists**: under all seven, the test split is untouched.

---

# AMENDMENT 1 — the pool-limited systems are excluded everywhere, 2026-10-23
Written before any clean-protocol arm is trained and before `test` is read for any
purpose under this document. `test` remains untouched.

## A1.1 Why an exclusion is needed at all, stated as the derivation it is

Change 1 (§1) replaces the all-frames crop statistic by the frame-0 one. For the clean
protocol to mean what it says, the 96 atoms it selects must be *the 96 protein heavy atoms
nearest the ligand at frame 0* — not merely the nearest 96 of whatever the store happens
to have kept. The store does not keep the whole protein: for each system it keeps the
protein heavy atoms within $R=8$ Å of the ligand at **any** stored frame, capped at 600
atoms, ranked by the all-frame minimum distance.

Write $d^0_j$ for the frame-0 distance of pool atom $j$ to its nearest ligand **heavy**
atom — exactly what `mmp_transfer.pocket_ranking_distance(..., "frame0")` computes — and
$m_j$ for its all-frame minimum distance. Let $D_{96}$ be the 96th smallest $d^0$ over the
pool. Any protein atom *outside* the pool satisfies

$$
d^0_j \;\ge\; m_j \;>\; B, \qquad
B=\begin{cases} R=8\ \text{Å} & \text{pool untruncated}\\[2pt]
\min\!\big(R,\ \max_{j\in\text{pool}} m_j\big) & \text{pool truncated at the 600 cap,}\end{cases}
$$

the first inequality because a frame-0 distance is one of the frames the minimum is taken
over, and the second because the pool is built by thresholding (and, when truncated, by
ranking) $m_j$. Hence

$$
D_{96}\le B \;\Longrightarrow\; \text{(frame-0 top-96 of the \emph{pool})}=\text{(frame-0 top-96 of the \emph{whole protein})},
$$

because every omitted atom is strictly farther at frame 0 than the 96th kept one. This is
an identity, not an approximation: for systems satisfying it, the clean crop is *provably*
what the manuscript will claim it is. For systems violating it the stored pool is a
confound of unknown sign — the crop is the nearest 96 *of a set that was itself chosen with
all-frame information*, which is a residue of exactly the leak Change 1 exists to remove.

The raw 132.8 GB `MD.hdf5` is no longer on disk, so re-extracting a frame-0-sufficient pool
was not an option. Exclusion is the only clean move.

## A1.2 The list: the first author's 20, and the 8 the criterion adds

The first author's decision names 20 systems:

> 1O0F 1RGK 1Z6D 1ZZ1 2A3B 2EA2 2Q93 2YBS 4IKS 4JKW 4JX9 4U54 4YXO 5EFC 5FLS 5U2F 5W7U 5X4Q 6HSZ 6MO9

with the population statistics median $D_{96}=5.14$ Å, $p_{99}=7.18$ Å, worst case 6MO9 at
25 Å. Those three figures are computed with distances to **all** ligand atoms, hydrogens
included. `pocket_ranking_distance` ranks on ligand **heavy** atoms only, and a
heavy-atom distance is $\ge$ the corresponding all-atom distance, so the heavy-atom
$D_{96}$ is the quantity the identity above must be applied to. Recomputed that way over
all 3,819 systems of the `status == "ok"` rows
(`outputs/clean_protocol_v1/excluded_systems.json`): median $D_{96}=5.684$ Å,
$p_{99}=7.765$ Å, 6MO9 at 26.07 Å (its pool has **zero** atoms within 8 Å of the ligand at
frame 0), 33 systems truncated at the 600 cap.

The criterion $D_{96}>B$ flags **28** systems, a strict **superset** of the 20
(`missing_vs_requester_list: []`). The eight extra are

> 1XNZ 2Q94 3L0V 4A4E 4A4G 4DE3 4G0P 4KZA

with $D_{96}$ between 8.000 Å (4A4E, 4KZA — on the boundary) and 8.38 Å (4G0P). They are
near-misses, and that is exactly why they must go: the stated *reason* for the decision is
the provable-identity property, and on these eight it is false. Keeping them would leave
the manuscript's crop sentence true "except for eight systems", which is the shape of
disclosure this project has already resolved not to make.

**Binding rule.** The excluded set is the **union**, i.e. the 28 systems above, written to
`outputs/clean_protocol_v1/excluded_systems.json` and read from there by every config. A
pair is dropped **in both directions** if *either* member is excluded. This is a
strengthening of the first author's decision in the direction its own reason demands; it is
recorded here rather than assumed, and if the first author prefers exactly 20, this
amendment must be superseded before training and the residual non-identity on the eight
must appear in the manuscript's crop sentence.

## A1.3 The counts, and which ones are authoritative

Exclusion is applied **after** the split and fold assignment: the target$\to$split and
target$\to$fold maps remain those of the pre-exclusion corpus (`fold_seed 20260918`), and
rows are then dropped. Re-splitting would change the test set wholesale and void every
comparison with read 1; that is forbidden.

| | before | after (28 excluded) | after (20 only, not used) |
|---|---|---|---|
| usable pairs, total | 4,913 | **4,830** | 4,842 |
| train pairs / targets | 3,489 / 345 | **3,417 / 342** | 3,426 / 343 |
| val pairs / targets | 696 / 106 | **687 / 105** | 690 / 105 |
| test pairs / targets | 728 / 66 | **726 / 65** | 726 / 65 |
| test directional examples | 1,456 | **1,452** | 1,452 |
| `status == ok` rows dropped | — | 93 of 6,230 (1.49%) | 75 of 6,230 (1.20%) |
| targets lost entirely | — | 5 (3 train, 1 val, 1 test: Q9QLL6) | — |

**The post-exclusion counts are authoritative.** They supersede the following rows of §2,
which are hereby revised (every other row of §2 stands):

- split: **3,417 / 687 / 726 pairs; 342 / 105 / 65 targets**; 0 target and 0 PDB overlap;
- test example list: `reference_outputs/sigma_star_test_v1/test_examples.csv` **filtered to
  the 1,452 directional examples whose pair survives the exclusion**, written out once as
  `clean_protocol_v1/test_examples_clean.csv` with its own sha256 in `CHAIN.log` before any
  test command. The filter is a subset operation on a frozen list, applied by PDB
  identifier, and it is fully determined by `excluded_systems.json`;
- statistics: the UniProt cluster bootstrap now resamples **65** test targets rather than
  66 (and the usable-example count after Addendum A will be at most 1,452, reported
  exactly, against read 1's 1,410 from 60 targets). Whether the number of *usable*
  clusters falls from 60 to 59 depends on whether the lost test target Q9QLL6 survived
  Addendum A in read 1; that is read off the frozen per-example CSVs at analysis time and
  reported, and is not looked up now.

The loss is 1.49% of pairs and 0.27% of test examples. It is not a power concern: read 1's
contrast was 8.8 near-side interval half-widths from zero.

## A1.4 Consequences inside this document

1. **$\tau$ is re-estimated on the post-exclusion train split** (Amendment 2). Order is
   binding: exclusions $\to$ $\tau$ $\to$ training.
2. **The frozen leaky arms are re-scored on the reduced example set by subsetting their
   existing per-example CSVs**, not by new inference. Those CSVs
   (`reference_outputs/sigma_star_test_v1/test_*_f*.per_example.csv`,
   `laplacian_prior_v1/.../*.per_example.csv`) already contain a value per test example, so
   the §5.3 difference-in-differences on the common set costs no new test information and
   no new test run. Only the new full-pocket clash-rate secondary (§5.2.5, Amendment 4)
   requires a fresh inference pass with the frozen checkpoints, and it is executed inside
   the single test session.
3. **The two model-free controls (§5.4) must be recomputed on the post-exclusion example
   lists.** The dev measurements already in hand
   (`outputs/clean_protocol_v1/baselines/`) were built from the **pre-exclusion** fold
   directories (`examples_from: outputs/laplacian_prior_v1/full_f*`), so every number
   quoted from them in Amendment 5 is labelled pre-exclusion and none of them is the
   reported control value.
4. **T5 is extended**: in addition to reproducing the frozen 60-system overlap statistics,
   the implementer asserts that `excluded_systems.json` regenerates bit-identically from
   the store and that the 28-system list is reproduced, including the empty
   `missing_vs_requester_list`.
5. **T8 (new) — exclusion is total.** Assert that no example built for any split of any
   clean arm, and no example in `test_examples_clean.csv`, `fit_examples.csv`,
   `dev_examples.csv` or either control's example list, has `pdb_a` or `pdb_b` in the
   28-system list. A single leaked pool-limited system voids the replication.
6. **The manuscript dataset table changes.** Edit E1 of the revised §10 (Amendment 6).

---

# AMENDMENT 2 — $\tau$ is re-estimated on the largest available train-only population, and P1's confidence is revised downward, 2026-10-23
Written before any clean-protocol arm is trained. `test` remains untouched.

## A2.1 What the first author decided, and what the earlier attempt actually showed

Change 2 (§1) is retained, with its population enlarged: the four widths are re-estimated
on **training-split systems only, on the largest population the measurement scripts can
process**, after the Amendment 1 exclusions.

The attempt already on disk (`outputs/clean_protocol_v1/tau_train_only.json`) is **not
adoptable as evidence of a no-op**, and the reason is its sample size, not its direction:

| channel | current constant | train-only | rel. change | $N_{\text{train}}$ | bootstrap SE of the median | change $/$ SE |
|---|---|---|---|---|---|---|
| bond (Å) | 0.032637 | 0.0324517 | $-0.57\%$ | 73 | $0.00022$ ($0.68\%$) | $0.8$ |
| angle (deg) | 2.774 | 2.76892 | $-0.18\%$ | 247 | $0.0098$ ($0.35\%$) | $0.5$ |
| torsion (deg) | 14.600 | 14.16218 | $\mathbf{-3.00\%}$ | 247 | $0.485$ ($3.42\%$) | $0.9$ |
| oop (deg) | 6.480 | 6.41557 | $-0.99\%$ | 15 | $0.146$ ($2.28\%$) | $0.4$ |

Every change is inside about one bootstrap SE of its own median. A $-3.00\%$ torsion shift
with a $3.42\%$ SE is a statement about the sample, not about the population, and adopting
it as either "a real shift" or "a no-op" would be reading noise.

**The 0.4% figure in the manuscript is not this measurement.** The audit bullet and
Section V's robustness paragraph say "re-estimating the widths without those systems
changes no width by more than 0.4\%"; that came from the leakage audit's *without val/test*
recomputation, whose population is the whole random MISATO draw minus a handful of val/test
ligands — about 580 of 600 systems for bond, 1,850 of 2,000 for angle/torsion. The
train-split-only population is a different and far smaller set (73 / 247 / 247 / 15,
because most MISATO systems are not in the MMP corpus at all). The two are not
interchangeable, and conflating them would be the "$\le 0.4\%$" claim doing work it never
earned. **Recorded consequence: prediction P1 is being tested against a noisier estimator
than its own reasoning assumed**, and the manuscript sentence must be rewritten (edit E6 of
the revised §10).

## A2.2 The pre-committed estimator, so it cannot be chosen afterwards

- **Population.** Every system whose PDB entry is in the **post-exclusion train split**,
  for every channel, over the maximum population each source scan can be extended to.
  The stopping rule is **exhaustion of that population**, not a precision target, and
  explicitly not "stop when the change looks small". $N$ per channel is reported whatever
  it is.
- **Estimator.** Unchanged from `tau_train_only.json`: median over systems of the
  per-system mean per-coordinate standard deviation, heavy atoms only, out-of-plane
  restricted to centres with reference mean deviation $<10^\circ$. A bootstrap SE of each
  median is reported next to each value.
- **Val, test, and the 28 excluded systems contribute nothing**, to any channel, by
  construction and by assertion (test T6 extended).
- The four values are written into the configs explicitly and recorded inside every
  checkpoint (§1, T6). The $\tau$ JSON is hashed in `CHAIN.log` **before** training starts;
  it is never re-estimated after a training or evaluation number exists.
- $\tau_{\text{oop}}$ is the weakest channel ($N_{\text{train}}=15$) and stands to change
  most. That is stated now so that a large oop shift is not later presented as a surprise.

## A2.3 What the enlarged sample can and cannot do to $\Sigma^{*}$ — the algebra, redone

§1's invariance is unaffected by sample size: with $F=\operatorname{diag}(\tau_{c(k)}^{-2})$,
$\Sigma^{*}(cF)=c^{-1}\Sigma^{*}(F)$ and $\alpha\mapsto c^{1/2}\alpha$, so
$\bm\epsilon=\alpha L\bm\xi$ is **exactly** invariant under a common rescaling and only the
three independent ratios can act. What *has* changed is the size of the residual, i.e. the
part of the four shifts that survives removal of the common mode:

$$
\delta=(-0.57,\,-0.18,\,-3.00,\,-0.99)\%,\qquad \bar\delta=-1.19\%,\qquad
\delta-\bar\delta=(+0.62,\,+1.01,\,-1.81,\,+0.20)\%
$$

a residual **spread of 2.82 percentage points**, against the $0.43\%$ that §1 quoted from
the audit's without-val/test recomputation — a factor $6.5$ larger. Since
$F_{kk}=\tau^{-2}$, the induced relative change in the diagonal of $F$ is $-2(\delta-\bar\delta)$
to first order, i.e. $(-1.24,\,-2.02,\,+3.62,\,-0.40)\%$: a spread of **5.6 percentage
points** across channels, not $\le 0.86\%$. P1's bound, which was the whole basis of its
0.95, no longer holds at that value.

Three facts limit the damage, and they are the reason I keep the prediction's
decision-relevant half:

1. **A perturbation of $F$ enters the widths at half strength.** The channel widths are
   $\sqrt{\operatorname{diag}(B\Sigma^{*}B^{\top})}$, and $\Sigma^{*}=(B^{\top}FB)^{+}$, so
   a $5.6\%$ differential perturbation of $F$ moves widths by $O(2.8\%)$, not $O(5.6\%)$.
2. **The matched-torsion rule absorbs the common mode exactly and most of the
   torsion-channel mode by construction.** $s^{*}$ is re-selected so that the realised
   torsion ratio crosses 1. Any change in $\tau$ that mainly rescales the draw is removed
   by $s^{*}$; what survives into the reported spread is only the *shape* change, which is
   second order.
3. **The spread is a difference of ratios**, so a uniform level change partly cancels in it
   (§4, M1: $\text{spread}=\bar r\,\psi$ with $\psi$ the shape non-uniformity; a pure level
   change moves the spread proportionally, and the level is what $s^{*}$ pins).

## A2.4 P1 is made scorable without a tenth training run

P1 as written compares against "a $\Sigma^{*}$ run with the old $\tau$ under the same clean
crop". That run is **not in the training queue** (nine runs: ISO/HARM/LAP $\times$ 3 folds,
three concurrent on GPU 1), and adding it would spend budget that §2 freezes. P1 is
therefore scored, with no new training, on three things, all of which are specified now:

- **S1 — the measured $\tau$ themselves.** Report the four values, their $N$, their
  bootstrap SEs, the common mode $\bar\delta$ and the residual $\delta-\bar\delta$, and the
  implied $\le\!2\,\max|\delta-\bar\delta|$ bound on the diagonal of $F$.
- **S2 — a noise-only, CPU-only, zero-training A/B of the draw.** For the same 150 reference
  poses used by `prior_noise_profiles.py`, the same molecular graphs and the **same fixed
  Gaussian $\bm\xi$**, form $\bm\epsilon_{\text{old}}=\alpha_{\text{old}}L_{\text{old}}\bm\xi$ and
  $\bm\epsilon_{\text{new}}=\alpha_{\text{new}}L_{\text{new}}\bm\xi$ and report the four
  channel widths of each and their per-channel relative differences. This measures exactly
  the quantity P1a asserts — how much the $\tau$ change moves the channel allocation of the
  draw — with the flow, the pocket and the training removed, and it cannot be confounded by
  retraining noise. **P1a is scored on S2**, at the $<1\%$ threshold P1 states.
- **S3 — T4.** The exact scale-invariance unit test (§7, T4) still has to pass; it is what
  makes the common-mode half of the argument a test rather than a hope.

P1b — "the clean test median spread moves by $<0.02$ absolute" — becomes **unscorable as a
direct measurement** and is scored as a *prediction about S2 propagated through M1*: if S2's
largest per-channel relative width difference is $w$, then at fixed shape the predicted
spread change is $\le \bar r\,\psi\,w \approx 0.77\,w$ at the frozen operating point, and
P1b is scored CONFIRMED if that bound is below 0.02 (i.e. $w<2.6\%$) and WRONG otherwise,
with the caveat that this is a bound and not an observation printed in the result.

## A2.5 Do I still stand behind P1, and at what confidence

P1's text above is unchanged and stands as the pre-draft-1 prediction. My revised
confidence, recorded here before the enlarged $\tau$ exist:

- **P1a (each clean fitting-fold channel ratio moves $<1\%$): 0.55**, down from 0.95. The
  derivation that justified 0.95 used a $0.43\%$ residual spread from the wrong population.
  With the only train-only estimate in hand the residual spread is $2.82\%$, which bounds
  the width change at $\sim2.8\%$, not $<1\%$. If the enlarged sample pulls the torsion
  shift back toward the common mode, P1a is comfortable; if $-3\%$ is real, P1a fails. I
  genuinely do not know which, and 0.55 is the honest number. **The enlarged sample is as
  likely to refute P1a as to confirm it, and that is the point of enlarging it.**
- **P1b (the decision-relevant half — the clean test median spread is insensitive at the
  0.02 level): 0.85.** I do stand behind this. It is protected by $s^{*}$ re-selection
  absorbing the level, by the half-power transmission from $F$ to widths, and by the spread
  being a difference. A $2.8\%$ width change at the frozen operating point propagates to
  $\approx0.02$ spread units — right at the stated threshold, which is why 0.85 and not
  0.95.
- **P1 as the conjunction it is literally written as: 0.5.**
- **P1's framing as "close to a derivation" is withdrawn.** It is a derivation only for the
  common mode. The ratios are an empirical input, and their uncertainty is now the dominant
  term. Whichever way it lands, this is recorded as a correction to my own reasoning, not
  as a surprise in the data.

---

# AMENDMENT 3 — $s^{*}$ is reported two ways, and a disagreement is read downgrade-only, 2026-10-23
Written before any clean-protocol arm is trained and before `test` is read. `test` remains
untouched.

## A3.1 The two operating points

- **PRIMARY — unchanged.** The torsion-matching rule of `scripts/matched_torsion_fit.py`
  applied to the **fitting folds**: piecewise-linear crossing of torsion ratio $=1$ on the
  grid $\{1.0,1.5,\dots,4.0\}$, smallest crossing, UNMATCHABLE if none (then the nearest
  grid point, reported as unmatched). This is read 1's rule on read 1's kind of data, and
  it is kept precisely so that the clean result is comparable with the frozen record and so
  that this amendment is not protocol-shopping.
- **SECONDARY — new.** The identical rule applied to the **target-disjoint dev folds** of
  each fit/dev split, reported alongside.

The rationale on the record: the fitting folds are the data the model trained on, so the
primary operating point is selected in-sample, and a reader is entitled to know whether the
headline depends on that. The dev folds are in-support and never touch val or test.

**One honesty item that must not be lost later.** The dev folds are *not* held out: §2
selects each checkpoint by **lowest dev loss**. The dev $s^{*}$ is therefore selected on
data used for model selection, and it is *less* in-sample than the fitting-fold $s^{*}$ but
not clean. It must never be described as "the held-out operating point". The only
genuinely out-of-sample operating point would be one selected on val, and val is spent
(`outputs/pareto_v1`).

## A3.2 Why this does not become a second shot at significance

Both $s^{*}$ values are computed on train-split data, frozen together into the **same**
JSON, and hashed in `CHAIN.log` **before any test command is issued** (§7, step 4). If they
coincide on a fold, that fold has one test run; if they differ, that fold has two, both
executed inside the **single** test session with no decision taken between them. So this
remains one read of the split in the sense that matters — one hashed plan, no adaptive
choice — but it is **two comparisons**, and §0's crude family-wise bookkeeping must be
updated: with four comparisons on one split ($1-0.05\times4=0.80$), nominal 95% intervals
should be read as roughly 80% intervals if one insists on the crude correction. The
downgrade-only rule below is what stops the extra operating point from being a free extra
attempt at a negative interval.

## A3.3 The reading rule, fixed now

Stated in full because deciding it after seeing the two columns is exactly the failure this
document exists to prevent.

1. **The Tier of §6 is computed from the PRIMARY (fitting-fold) $s^{*}$ run, and only from
   it.** The dev-$s^{*}$ run never sets the Tier.
2. **Insensitive case.** If, for every arm and fold, $|s^{*}_{\text{dev}}-s^{*}_{\text{fit}}|\le 0.5$
   (one grid spacing) **and** the two $\Delta$ estimates' 95% intervals overlap, the
   operating point is declared insensitive, that fact is reported in one sentence, and the
   primary numbers are the paper's numbers.
3. **Disagreement, same Tier.** If the two runs land in the same Tier but differ
   materially, both $\Delta$ values and both CIs are printed in the same table, and
   $\max_{\text{arm,fold}}|s^{*}_{\text{dev}}-s^{*}_{\text{fit}}|$ is reported as a measure
   of operating-point fragility — the quantity §4's M2 identified as the one channel that
   could move $\Sigma^{*}$ by several tenths.
4. **Disagreement, different Tier — downgrade only.** If the dev-$s^{*}$ run lands in a
   **strictly worse** Tier (ordering $R > A > W > N$), the **reported headline is the worse
   of the two Tiers**, labelled "operating-point-sensitive", and the manuscript states in
   the Experiments section that the magnitude of the result depends on whether the
   amplitude was matched in-sample or on a target-disjoint fold. If the dev-$s^{*}$ run
   lands in a **better** Tier, that is reported as a secondary observation and **the
   headline stays at the primary Tier**; a better dev-$s^{*}$ result may never be promoted.
   This mirrors §6's existing asymmetry: a secondary can downgrade a Tier and never upgrade
   one.
5. **Tier N is absorbing.** If either run gives Tier N, the claim is withdrawn under §6's
   Tier N reading. There is no "it replicates at one of two operating points".
6. **Blocking conditions B1–B4 are evaluated separately at each operating point**, and a
   block fired at either one blocks the headline. In particular B1's realised-torsion
   window $[0.85,1.25]$ is checked for both.
7. **Nothing else is re-selected on dev.** The dev folds set $s^{*}$ and nothing more: not
   the architecture, not the budget, not the arm set, not the metric, not the $\tau$.

---

# AMENDMENT 4 — B2 is adopted, and the full-pocket clash rate is a required secondary, 2026-10-23
Written before any clean-protocol arm is trained. `test` remains untouched.

## A4.1 Adoption

Blocking condition **B2** of §6 is **adopted verbatim** as the first author's decision:

> **B2 (the fence).** If clean $\Sigma^{*}$'s spread improves relative to leaky $\Sigma^{*}$
> **and** its full-pocket clash rate worsens by more than the largest clash-rate difference
> seen among the three leaky arms, then M3 is operating and the spread improvement was
> bought by losing a target-informed barrier. The improvement is reported as an artifact of
> the fix and **is not claimed as a gain**.

## A4.2 The clash-rate secondary is promoted from "reported regardless" to required, with a gate

§5.2 item 5 was listed among the secondaries. It is now a **requirement with a reporting
gate**:

> **If the full-uncropped-pocket clash rate is missing for any clean arm, the primary may
> not be reported.** The analysis script raises rather than printing a spread without it.

The reason is §4's M3 and is worth restating in one line: the leaky crop was a
*target-informed barrier inside a weight-10 loss term and inside the inference projection*,
so removing it can make $\Sigma^{*}$'s spread look better by removing well-aimed clash
pushes. Without this secondary the replication could buy a prettier number by losing the
fence and nobody would see it. A required secondary with a gate is the only form of this
check that cannot be quietly dropped when it is inconvenient.

## A4.3 The definition, pinned now so it cannot be chosen later

- **Pair set.** All (generated ligand **heavy** atom, pocket **heavy** atom) pairs over
  **all `n_pocket_available`** stored pool atoms — *not* the 96-atom crop, and not only the
  crop's complement.
- **Threshold.** $d_{\mathrm c}=1.2$ Å, the training clash distance, and no other. No
  second threshold is introduced; a sensitivity sweep over thresholds after the fact would
  be a search.
- **Reported quantities,** per arm, per fold, and pooled: (i) the fraction of such pairs
  below 1.2 Å, pooled over all 50 generated frames and all examples; (ii) the median
  per-example count of violating pairs; (iii) the fraction of examples with at least one
  violation; (iv) the same three quantities restricted to pocket atoms **outside** the
  frame-0 crop, which is the subset M3 predicts the clean arms lose their fence on.
- **Which arms.** Every clean arm, **and** every frozen leaky arm (ISO, GS, LAP,
  $\Sigma^{*}$), recomputed from the frozen checkpoints. This metric was never computed in
  read 1 or read 2, so the leaky side requires one fresh inference pass; it is executed
  inside the single test session, at each leaky arm's own frozen operating point, on the
  post-exclusion example list.
- **Both model-free controls** are scored on it too, in both projection variants.
  Control A has **no pocket input of any kind**, and copy-the-parent has none either, so
  either control may be physically impossible while scoring well on spread. A control that
  fires B3 while clashing more than every arm is reported with both numbers side by side.
- **B2's tolerance, pinned.** "The largest clash-rate difference seen among the three leaky
  arms" is
  $\max_{a\ne b\in\{\text{ISO},\text{GS},\text{LAP},\Sigma^{*}\}_{\text{leaky}}}
  \big|\text{rate}_a-\text{rate}_b\big|$, using quantity (i), computed on the same
  post-exclusion test example set, each arm at its own frozen operating point. Fixed here
  so the comparator set is not chosen after the numbers exist.
- **P7 is scored on quantity (i) and on the recall statistic of §5.2.6**, as written.

---

# AMENDMENT 5 — the two model-free controls, scored; and B3 is repaired to be amplitude-controlled, 2026-10-23
Written before any clean-protocol arm is trained and before `test` is read. `test` remains
untouched. **This is the one amendment made in response to numbers**, so §A5.4 states
exactly which numbers, where they came from, and why the change is a repair of a rule this
document already stated rather than a new rule chosen to suit the data.

## A5.1 What was measured, and under exactly what configuration

Both controls of §5.4 have been run on the **dev folds** (`outputs/clean_protocol_v1/
baselines/`, `scripts/clean_protocol_baselines.py`). The configuration matters more than
the numbers:

- **dev folds of the train split** — never val, never test;
- **pre-exclusion** example lists (`examples_from: outputs/laplacian_prior_v1/full_f*`), so
  Amendment 1 will change them slightly;
- **projection OFF** for both controls. The script states it: *"No angle projection, no
  flow."* §5.4 requires **both** variants (OFF and ON); only the OFF variant exists;
- Control A drawn with the **train-only $\tau$** (`tau_set: train_only`), which is itself a
  small piece of reassurance about Change 2 — nothing degenerated;
- scored with the imported metric code of `mehf_internal_geometry_spectrum.py`, so the
  channel definitions are the arms';
- Control A on the frozen scale grid $\{1.0,\dots,4.0\}$; Control B has no scale.

Medians of the per-example channel ratios, and — computed here from the per-example CSVs —
the **median per-example four-channel spread**, which is the primary metric's actual
definition (median of spreads, not spread of medians):

| control | fold | bond | angle | torsion | oop | RMSF | median per-example spread | median $\psi$ |
|---|---|---|---|---|---|---|---|---|
| B copy-the-parent | 1 | 3.970 | 2.274 | 0.890 | 1.041 | 1.008 | **3.077** | 1.506 |
| B copy-the-parent | 2 | 3.812 | 2.153 | 0.863 | 1.000 | 1.028 | **2.947** | 1.479 |
| B copy-the-parent | 3 | 4.238 | 2.396 | 0.842 | 1.006 | 0.951 | **3.453** | 1.585 |
| A $\Sigma^{*}$ source, $s_i=3.5$ | 1 | 0.945 | 0.660 | 0.417 | 0.568 | 0.746 | **0.515** | 0.819 |
| A $\Sigma^{*}$ source, $s_i=3.5$ | 2 | 1.254 | 0.875 | 0.537 | 0.757 | 0.788 | **0.676** | 0.813 |
| A $\Sigma^{*}$ source, $s_i=3.5$ | 3 | 1.030 | 0.719 | 0.407 | 0.614 | 0.631 | **0.554** | 0.850 |
| A $\Sigma^{*}$ source, $s_i=4.0$ | 1/2/3 | 1.131/1.495/1.229 | 0.765/1.014/0.833 | 0.474/0.608/0.462 | 0.646/0.860/0.697 | 0.852/0.901/0.721 | **0.638/0.847/0.687** | 0.871/0.866/0.901 |

where $\psi$ is defined in §A5.5. The first author's quoted pair — copy-the-parent
3.97 / 2.27 / 0.89 / 1.01 and source 0.945 / 0.660 / 0.417 / 0.746 — is fold 1 dev of each,
the source row at $s_i=3.5$.

## A5.2 P6 (copy-the-parent) — **WRONG**, on the band and on the ordering

P6 predicted, projection OFF, a median spread in $[1.0,2.5]$, and "beats ISO but not
$\Sigma^{*}$" at confidence 0.7 on the ordering.

Measured, projection OFF: **2.95 / 3.08 / 3.45**, i.e. $18$–$38\%$ **above** the top of my
band. And it does not beat ISO: 3.08 against ISO's frozen test 1.974. That comparison
mixes splits and projection states, so the decisive version is the amplitude-free one:
copy-the-parent's $\psi$ is **1.48–1.59** against trained ISO's $\hat\psi=1.197$. **Copy-the-parent
allocates width across channels even less uniformly than isotropic noise does**, and that
conclusion needs no amplitude convention at all, because its internal RMSF ratio is
$0.95$–$1.03$ — it is already at the right overall dispersion. P6 is WRONG, and not
marginally.

**Where my arithmetic failed, since P6 was explicitly "worked rather than asserted".** I
predicted bond $\approx2$ and angle $\approx2.3$ from a model in which only the $\sim1$–$2$
bonds and $\sim5$ angles *spanning the attachment* inflate, the rest sitting near 1.
Angle came in at 2.15–2.40, essentially exactly as predicted. Bond came in at 3.81–4.24,
about **twice** my estimate, and $3.97\times0.0327\ \text{Å}=0.130$ Å of mean bond width is
far too much to be carried by one or two bonds out of thirty — the whole core is distorted,
not just the attachment.

The mechanism I missed is that **a rigid motion of the parent is not a rigid motion of the
product**. If parent atom $a$ maps to product atom $i$ with offset
$\mathbf e_i=\mathbf x_i-\mathbf y_a$, then transferring the parent's frame-$f$ rigid
displacement gives

$$
\mathbf x_i\ \mapsto\ \mathbf x_i+\big(R_f\mathbf y_a+\mathbf t_f-\mathbf y_a\big)
=\underbrace{R_f\mathbf x_i+\mathbf t_f}_{\text{rigid on the product}}\;-\;\underbrace{(R_f-I)\,\mathbf e_i}_{\text{non-rigid residual}},
$$

so every mapped atom acquires a residual of magnitude $\approx\theta_f\|\mathbf e_i\|$ with
$\theta_f$ the parent's frame-to-frame rotation angle. These residuals differ between
bonded neighbours whenever their offsets differ, so they enter bond lengths directly:
$\theta\sim0.1$ rad and $\|\mathbf e\|\sim1$ Å give $\sim0.1$ Å of bond-length width, which
is the measured 0.13 Å. The parent's rigid wander (0.7–2.9 Å by the corpus's own
measurement) does not cancel; it is converted into product internal distortion by the
mapping, which is not an isometry. The single Kabsch superposition of the parent block
*mean* onto product frame 0 removes the mean rotation only, not the per-frame ones.

That also explains the shape: bond is hit hardest ($\psi$'s max channel), torsion barely at
all (0.84–0.89, slightly *under*-dispersed), which is the opposite allocation to the one
MEHF is claimed to produce.

## A5.3 P5 (the raw $\Sigma^{*}$ source) — band **WRONG**, and my stated reason was wrong too

P5 predicted, projection ON, a median spread in $[0.8,1.6]$: below clean ISO, above clean
$\Sigma^{*}$, confidence 0.6.

Measured, projection OFF, dev: **0.515 / 0.676 / 0.554** at $s_i=3.5$, and
**0.638 / 0.847 / 0.687** at $s_i=4.0$. Both sets sit **below** the floor of my band, and
both sit below or around the frozen $\Sigma^{*}$ test median of 0.770 on two of three
folds. Taken at face value this is the result that fires B3 and restates the paper's
headline.

**It should not be taken at face value, and the reason is in the protocol, not in a
post-hoc defence.** The pre-registered convention is *matched torsional width*, and
Control A cannot be matched:

> **Control A's torsion ratio never reaches 1.0 anywhere on the frozen grid.** Its maximum
> is 0.474 / 0.608 / 0.462 at $s_i=4.0$, the top of the grid. It is UNMATCHABLE under this
> document's own rule, which forbids extrapolation — the mirror image of MEHF-GS in read 1's
> Amendment 2, which was unmatchable from the other side.

At $s_i=3.5$ the control is dispersed at **0.42–0.54 of the data's torsional width** and
**0.63–0.79 of its internal RMSF**. Its small spread is bought by being small. The
decomposition §4 already fixed makes this exact: with $r_c=\bar r\,u_c$ and $\sum_c u_c=4$,

$$
\text{spread}=\bar r\,\psi,\qquad \psi\equiv\max_c u_c-\min_c u_c,
$$

so spread is the product of an **amplitude** and a **shape**. Control A's $\psi$ is
**0.81–0.85** at $s_i=3.5$ against the trained $\Sigma^{*}$ arm's $\hat\psi=0.485$: in
amplitude-free terms the raw source is $1.7\times$ **less** uniform than the trained arm,
while reporting a smaller spread. That is the gameability the first author's history
predicts, arriving on schedule.

**And the shape gets worse as the amplitude rises, which is derivable.** A bond length is a
nonlinear function of coordinates: with bond vector $\mathbf d$ and displacement difference
$\bm\delta$,

$$
\|\mathbf d+\bm\delta\|=\|\mathbf d\|+\delta_\parallel+\frac{\delta_\perp^{2}}{2\|\mathbf d\|}+O(\delta^3),
$$

so the bond-length width picks up a term growing like the **square** of the Cartesian
amplitude, while a torsion at these amplitudes ($\le7^\circ$ of $14.6^\circ$) is still in
its linear regime. Measured log-log exponents of the median ratio against $s_i$ on the
seven grid points confirm both regimes: **bond $1.14$–$1.19$, torsion $0.975$–$0.998$**.
Hence $\psi$ rises monotonically with $s_i$ across the whole grid (0.588 at $s_i=1.0$ to
0.871 at $s_i=4.0$, fold 1), and raising the control to the matched point must make it
*less* uniform, not more. Solving the fitted curves for torsion $=1$ (off-grid,
$s_i\approx6.6$–$8.6$) gives, per fold,

| fold | $s_i$ at torsion 1 | bond | angle | torsion | oop | RMSF | spread of medians | $\hat\psi$ |
|---|---|---|---|---|---|---|---|---|
| 1 dev | 8.55 | 2.67 | 1.70 | 1.00 | 1.37 | 1.82 | 1.67 | 0.991 |
| 2 dev | 6.57 | 2.60 | 1.70 | 1.00 | 1.41 | 1.48 | 1.60 | 0.953 |
| 3 dev | 8.61 | 2.94 | 1.86 | 1.00 | 1.49 | 1.55 | 1.94 | 1.065 |

**This extrapolation is a diagnostic and is not, and may not become, a reported control
value** — §6 and read 1's Amendment 2 forbid extrapolating an unmatchable arm, and that
prohibition applies to my own convenience as much as to anyone else's. Its role is only to
show which side of 0.770 the honest comparison falls on: $1.60$–$1.94$, not $0.52$–$0.68$.
The extrapolation-free version of the same statement is the monotonicity of $\psi$, which is
measured on-grid at seven points and needs no fit.

**P5's verdict, honestly.** The band $[0.8,1.6]$ is **WRONG** for the configuration
measured — the control reports 0.52–0.85 — and the ordering clause ("above clean
$\Sigma^{*}$") is **not yet scorable** (projection ON, test, and post-exclusion examples are
all missing) but survives only under an amplitude-controlled reading that P5 did not state.
I grade it **WRONG**, not HALF RIGHT, for a specific reason: the clause that decides
whether B3 fires is the numeric one, and I got it wrong on the unflattering side.

**My stated *reason* for P5 was also wrong, and that matters more than the band.** I wrote
that "most of the channel balance is the source, not the flow", citing the pocket-free
noise-only spread of 1.01. The $\psi$ comparison says the opposite: source $0.81$–$1.07$,
trained arm $0.485$. The source does **not** get most of the way. I had the right direction
for the wrong reason, which is the kind of correctness that does not transfer.

## A5.4 What the two controls together license the paper to claim — and what they do not

**Licensed, on the measurements in hand.**

1. **Neither model-free control attains a favourable channel allocation at a comparable
   amplitude.** Copy-the-parent is at the right dispersion (RMSF 0.95–1.03) and allocates
   worse than isotropic noise ($\psi$ 1.48–1.59 vs 1.197). The raw source is more uniform
   than isotropic noise but, at any amplitude on the grid and increasingly so toward the
   matched point, less uniform than the trained arm ($\psi$ 0.59–0.87 on grid, $\hat\psi\approx1.0$
   extrapolated to matched, vs 0.485).
2. **The apparently excellent spread of the raw source is an amplitude artifact**, and it is
   caught by a control the protocol already required rather than by a defence invented
   afterwards. This is a reportable methodological finding in its own right: on this metric,
   an under-dispersed sampler scores well, so **spread may only ever be read at matched
   amplitude**. It also means the manuscript's sentence "the spread … is insensitive to
   amplitude" is false as written (edit E5, Amendment 6).
3. **The conditioning channel is not the answer by itself.** Using the parent's transferred
   motion *as* the prediction — the same tensor the model receives — produces the worst
   allocation of anything measured, for a mechanism now identified (§A5.2). The paper may
   say that the transferred parent motion is not a substitute for the generated ensemble, and
   it may say why.
4. **There is positive evidence that the learned part changes the allocation, not merely its
   scale.** At the same $s_i=3.5$, source versus the trained $\Sigma^{*}$ arm: bond
   $0.945\to1.66$ ($1.8\times$), angle $0.660\to1.40$ ($2.1\times$), torsion
   $0.417\to1.15$ ($2.8\times$), oop $0.568\to1.89$ ($3.3\times$). The amplification is not
   uniform, so this is not a gain knob; in shape terms the dominant channel **moves** from
   bond ($u=1.46$ in the source) to out-of-plane ($u=1.24$ in the arm) and $\psi$ falls from
   0.815 to 0.485. Note the direction of that last fact carefully: the trained arm's
   out-of-plane over-dispersion is produced by the learned part, not inherited from the
   prior, whose raw oop here is *under*-dispersed at 0.57.
5. **The split comparison is conservative, which is worth one sentence.** The controls are
   measured on dev and the arm numbers quoted against them are from test, the split on which
   $\Sigma^{*}$ degraded by $1.7$–$3.4\times$. The control is being given the easier split
   and still loses, so the ordering is not an artifact of where each was measured.

6. **Independent confirmation, found later the same day, raises my confidence in this
   reading.** The submission manuscript itself already prints a flow-free draw from
   $\Sigma^{*}$ around 150 reference poses — bond 1.15, angle 0.85, torsion 0.56, oop 0.78
   (`paper/MEHF_TCBB/main.tex:1577–1583`) — whose spread is $1.15-0.56=0.59$, **below the
   method's headline 0.770**, and whose $\hat\psi=0.707$ is **above** the method's 0.485.
   That is this amendment's diagnosis reproduced exactly, by an independent measurement,
   written down by someone who was not looking for it: the raw source wins on spread by
   being under-dispersed and loses on the amplitude-free statistic. It is recorded here as
   evidence for the reading and in Amendment 6 §A6.2 (F6) as a required manuscript repair.

**NOT licensed. These are the claims the controls do not support, written down now so they
cannot be slid in later.**

1. **They do not establish that the learned flow is doing the work.** The measured controls
   have **no projection**; the arms have eight sweeps of it, and the projection pulls bonds
   and angles toward frame-0 targets — precisely the channel in which the source is worst
   (bond $u=1.46$). A large part of the $0.815\to0.485$ improvement could be the
   parameter-free projection rather than the learned velocity. The correct ablation is
   **source $+$ projection** versus **flow $+$ projection**, and it does not exist yet.
   Until the projection-ON variants of §5.4 are run on the same test examples, the honest
   statement is "no model-free control reaches the arm's allocation", not "the flow is doing
   the work".
2. **They say nothing about the paper's actual claim.** The claim under test is
   $\Sigma^{*}$ versus ISO — a comparison between two priors inside the same model. Both
   controls are silent on it: neither is an ISO-side control, and a world in which both
   controls lose and $\Sigma^{*}$ ties ISO is entirely consistent with these numbers.
3. **They say nothing about ensemble quality.** Uniform allocation is not accuracy. A model
   that is uniformly wrong scores perfectly on spread. The controls bound how much of a
   *uniformity* metric is reachable without learning; they cannot certify that the arm's
   ensembles resemble MD.
4. **They say nothing about physical plausibility.** Neither control has any pocket term —
   Control A has no pocket input at all, and copy-the-parent can drive the ligand into the
   protein. A control that beats an arm on spread while clashing is not a baseline that
   beats the method; hence Amendment 4 scores both controls on the full-pocket clash rate.
5. **They are dev, pre-exclusion, one scale grid, three folds, projection off.** None of
   these numbers is a reported control value. Every reported control value comes from the
   single test session on the post-exclusion list in both projection variants.
6. **$\psi$ for the frozen arms is a proxy.** $\hat\psi$ below is computed from *published
   channel medians*, which is a spread-of-medians normalised by a mean-of-medians, not a
   median of per-example $\psi$. The frozen per-example test CSVs were deliberately not
   opened to improve it. For the clean arms, $\psi$ is computed per example.

## A5.5 B3 is repaired: the control comparison becomes amplitude-controlled

**What changed and when.** B3 as drafted reads "if either Control A or Control B, with the
projection ON, attains a median spread **below** clean $\Sigma^{*}$'s, then the primary
metric does not require the learned flow". The dev measurements above show that as written
it can fire on an artifact — a control that is half as dispersed as the data wins on spread
— and that Control A is **UNMATCHABLE** on the frozen grid, a case §5.4's sentence "at the
same matched-torsion amplitude convention where an amplitude exists" gestures at but does
not resolve. Both facts were discovered on dev, before training, before any test number,
and the repair is to the *stated* amplitude-matching convention, not to the decision rule
for the arms. The primary metric, the primary contrast, the Tier definitions and the arm
comparison are **unchanged**.

**New definition — $\psi$, for reading the controls only.** Per example, with the four
channel ratios $r_c$,

$$
\psi \;=\; \frac{\max_c r_c-\min_c r_c}{\tfrac14\sum_c r_c}\,,
$$

the per-example spread divided by the per-example mean ratio; report its median. $\psi$ is
invariant to a common rescaling of all four ratios and so cannot be improved by being
under- or over-dispersed. **$\psi$ never replaces the primary metric and never enters a Tier
decision.** It enters in exactly one place, B3.

**B3, as amended:**

> **B3 (the controls), amended 2026-10-23.** For each control, in each projection variant,
> report the median spread, the median $\psi$, the realised torsion ratio and the internal
> RMSF ratio. Then:
>
> - **(a) matched case.** If a control's realised test torsion ratio lies in $[0.85,1.25]$
>   (B1's window) **and** its median spread is below clean $\Sigma^{*}$'s, **B3 fires** and
>   §6's restatement requirement applies in full.
> - **(b) unmatchable case.** If a control cannot be matched on the frozen grid, it is
>   evaluated at the nearest grid point and reported as **unmatched** (read 1's Amendment 2
>   precedent, applied to a control). A control's spread measured outside the window
>   $[0.85,1.25]$ **cannot fire B3 on spread**; instead **B3 fires if the unmatched
>   control's median $\psi$ is below clean $\Sigma^{*}$'s median $\psi$.**
> - **(c) reported either way.** Whether or not B3 fires, every control's spread, $\psi$,
>   torsion ratio, RMSF ratio and full-pocket clash rate appear **in the same table as the
>   arms**, never in an appendix, with the unmatched label where it applies. A control that
>   reports a lower spread while being out of the amplitude window is reported *as that* —
>   lower spread, unmatched amplitude, $\psi$ given — and not as a tie.
> - **(d) no escape via $\psi$.** If a control is matched and wins on spread, B3 fires
>   regardless of $\psi$. $\psi$ is a route for B3 to fire, never a route for it not to.

**Why (d) is there.** Without it, the amendment would be a device for making an
inconvenient control unable to win. With it, the amendment can only ever *add* ways for the
controls to block the headline: a matched control wins on spread as before, and an unmatched
control can now also win on $\psi$. That asymmetry is what makes this a repair rather than
protocol-shopping, and it is checkable against the frozen draft-1 text above.

**Predicted, now, so the repair is itself falsifiable.**

- **P8.** On test, with the projection ON, Control A remains unmatchable on the frozen grid
  (its realised torsion ratio at $s_i=4.0$ stays below 0.85) and its median $\psi$ stays
  **above** clean $\Sigma^{*}$'s. Confidence 0.7. *Why it can lose:* the projection acts
  hardest on bond and angle, the control's two worst channels, and could cut its $\psi$ by
  more than it cuts the arm's. If P8 is wrong, B3(b) fires and §6's restatement applies.
- **P9.** Control B's median $\psi$ on test stays above trained ISO's, i.e. copy-the-parent
  remains a worse allocator than isotropic noise even after projection. Confidence 0.8.
- **P10.** The projection reduces Control A's bond ratio by more than it reduces its torsion
  ratio, so the projection-ON control's $\psi$ is below its projection-OFF $\psi$ at the
  same $s_i$. Confidence 0.85. This is the quantity that decides how much of the arm's
  uniformity is attributable to the projection rather than the learned velocity, and it is
  the number I most want from this run.

**Reference values of $\hat\psi$ from the frozen record** (published channel medians;
proxy, see §A5.4 item 6), recorded now so the comparison cannot be assembled later:

| arm (frozen, leaky, test) | bond | angle | torsion | oop | mean | spread | $\hat\psi$ |
|---|---|---|---|---|---|---|---|
| MEHF-ISO | 1.02 | 1.24 | 1.07 | 2.87 | 1.550 | 1.85 | 1.197 |
| MEHF-GS (unmatched) | 1.04 | 1.33 | 1.21 | 3.48 | 1.765 | 2.44 | 1.382 |
| MEHF-LAP | 2.18 | 1.96 | 1.10 | 2.62 | 1.965 | 1.52 | 0.774 |
| MEHF ($\Sigma^{*}$) | 1.66 | 1.40 | 1.15 | 1.89 | 1.525 | 0.74 | **0.485** |

---

# AMENDMENT 6 — §10 retargeted to the submission version, 2026-10-23
Written before any clean-protocol arm is trained. `test` remains untouched.

**Revised the same day, before this file was hashed**, against the authoritative submission
source, which was put on disk after A6 was first drafted:
`/home/ieb24002/MEHF/paper/MEHF_TCBB/main.tex` (1,667 lines; `reference.bib`,
`IEEEtran.cls`, `figures/`). Every edit below carries its line numbers in that file. The
file is **not** edited by this document; it is only cited.

## A6.1 Which document is being edited

§10 above was drafted against the **Overleaf master** (`paper/Overleaf/main.tex`), which
carries Appendices A–D. The submission version is `paper/MEHF_TCBB/main.tex`:

| | line |
|---|---|
| abstract | 48–50 |
| I Introduction | 60 |
| II Methods | 84 (II-A Framework 86, II-B Velocity network 182, II-C Hierarchical CFM 318, II-D Rigid/local source 457, II-E Wilson prior 585, **II-F Inference-Time Geometric Projection 779**) |
| III Experimental Setup | 877 (III-A Dataset 879, III-B Source-prior variants 935, III-C Metrics 984, III-D Protocol & statistics 1038, III-E Implementation 1070) |
| IV Results and Discussion | 1102 (IV-A 1103, IV-B 1142, IV-C Comparison of source priors 1210, **IV-D Separating prior and projection 1358**, IV-E Limits of post-generation correction 1405, IV-F Robustness and Limitations 1551) |
| V Conclusion | 1600; Acknowledgments 1639 |
| appendices | **none** |

§10's four items are superseded by §A6.4; §10's text stands unedited as the draft-1 record.

## A6.2 Six facts established in the submission source

F1–F3 are the three the coordinator verified, restated with line numbers; F4–F6 are what I
found checking them, and F4 and F5 are corrections to A6's own first draft.

**F1 — the 96-atom pocket cap is never stated, and neither is the fact that the pocket is
cropped at all.** The only occurrence of "96" in the manuscript is the hidden dimension at
line 1073. The pocket enters as *"the surrounding protein-pocket atoms"* (line 93),
*"protein-pocket information"* (lines 95, 156, 169, 188, 390), and *"relative vectors
between ligand atoms and protein-pocket atoms"* (lines 278–281), with no cardinality, no
radius and no selection rule anywhere. This is **larger than the "selection rule
undisclosed" framing** of A6's first draft: a reader has no way to know a crop exists, and
would reasonably assume the full pocket. It is therefore a reproducibility defect as well
as a disclosure one — a reimplementation that put the *whole* pocket into the weight-10
clash loss and into the projection would not reproduce any number in the paper.

**F2 — the projection's clash step is absent, and the current wording misdescribes what
moves the atoms.** II-F (779–873) derives exactly two corrections, the bond step (806–826,
with under-relaxation $\gamma_{\mathrm b}$) and the angle step (828–862, with
$\gamma_{\mathrm a}$), and then states at **865–868**:

> *"In the experiments, the projection is repeated for eight iterations with an
> under-relaxation factor of $0.5$. The procedure acts directly on bond lengths and bond
> angles but does not explicitly constrain torsions or out-of-plane coordinates."*

The code performs a **third** step in every iteration (`src/mdwm/models/mehf.py:1901–1937`;
defaults at 616 `projection_relaxation=0.5`, 617 `projection_angle_relaxation=0.5`, 632
`projection_clash_relaxation=1.0`, 603 `clash_distance_angstrom=1.2`): every ligand heavy
atom closer than $d_{\mathrm c}=1.2$ Å to its **nearest cropped-pocket heavy atom** is moved
radially outward along the separation vector, at **relaxation 1.0 — full strength**, against
0.5 for bonds and angles. Two distinct errors follow. (i) *"acts directly on bond lengths
and bond angles"* is not merely incomplete; it misdescribes the mechanism, because a radial
Cartesian push is not an internal-coordinate operation at all. (ii) The single stated
under-relaxation factor **does not cover the clash step**, so even a reader who guessed the
step existed would have the wrong strength for it — and it is the strongest of the three.

**F3 — "clash" appears in the submission only as the training loss and in a baseline
table.** Lines 439 and 442 (the objective, $\mathcal L_{\mathrm{clash}}$ *"penalizes short
ligand--protein distances"*), 1082 and 1084 ($\lambda_{\mathrm{clash}}=10$), and the
DiffSBDD force-field discussion at 1422–1447. The word never appears in II-F. There is no
sentence anywhere in the submission from which a reader could infer that the projection
touches the pocket.

**F4 — recorded as a correction to A6's first draft.** The audit bullet *"Projection targets
and all other conditioning come from the input pose"* is absent from the submission, as is
**any** mention of pre-registration, hashing, leakage, or the number of test reads: `grep -i
"pre-regist|preregist|hash|leak"` on `MEHF_TCBB/main.tex` returns nothing. The submission's
only leakage-adjacent statements are *"No protein target or PDB entry is shared across the
three splits"* (919–921), *"The held-out validation and test sets are not used for checkpoint
selection"* (1092–1093), and the $\tau$ sentence at 1553–1558. So the submission does not
make the false claim because it makes no claim — the divergence stands as recorded in A6's
first draft, and it makes E9 below load-bearing rather than tidying.

**F5 — the submission does not contain the false amplitude sentence of the Overleaf
master.** III-C's spread definition (1013–1027) says only *"Because the channel spread
measures cross-channel balance rather than absolute agreement with the reference, the
individual width ratios are reported together with $S$"*, which is defensible, and III-D
(1040–1048) states the matched-torsion rule including *"If a prior does not reach this
target within the tested range, the nearest evaluated scale is used without
extrapolation"* — the submission's own precedent for Amendment 5's handling of Control A.
E6 below is therefore an **addition**, not a correction. This supersedes the first draft's
E5, which quoted the master.

**F6 — the submission already prints a model-free control that beats the method on the
primary metric, and does not notice.** Lines **1577–1583**:

> *"Second, the harmonic prior is locally unimodal. Direct sampling from $\Sigma^{*}$ around
> 150 reference poses gives width ratios of 1.15 for bond lengths, 0.85 for bond angles,
> 0.56 for torsions, and 0.78 for out-of-plane coordinates. The relatively small torsional
> width reflects the fact that real ligand torsions can occupy multiple rotamer wells,
> whereas a local harmonic model describes a single basin."*

Those four numbers have spread $1.15-0.56=\mathbf{0.59}$, **below the method's headline
0.770**, and $\hat\psi=0.59/0.8325=\mathbf{0.707}$, **above the method's 0.485**. This is
Amendment 5's diagnosis reproduced exactly, in the paper's own text, from an independent
measurement: a flow-free draw from the source wins on spread while being under-dispersed
(torsion 0.56) and loses on the amplitude-free statistic. The paper interprets the small
torsion ratio as unimodality — which it also is — and does not notice that it is the
amplitude artifact that makes the spread look good. **As the submission stands, a referee
who divides 1.15 by 0.56 concludes that the method loses to its own prior sampled without
the model.** That must be repaired in text (E7), and it raises my confidence in Amendment
5's reading, because the number was produced by someone not looking for it.

## A6.3 The two consequences, worked through

### A6.3.1 M3 is invisible in the published description, and what that implies

§4's mechanism M3 is a four-link chain: the pocket crop was chosen with a statistic over all
100 trajectory frames $\to$ the crop is the reference set of the inference clash push at
relaxation 1.0 $\to$ a radial Cartesian displacement of magnitude $\delta$ lands
disproportionately in the stiff channels, by the paper's own Wilson-law numbers (abstract
line 49: 43, 32 and 8.0 physical widths per Å for bond, angle and torsion) $\to$ the
four-channel spread moves. **Links 2 and 3 are unreconstructible from the submission**: F2
and F3 establish that the text contains no statement that the projection touches the pocket,
and F1 that it contains no statement that a crop exists. The consequence is not that M3 is
wrong; it is that a reader of the published paper **cannot audit it**, and neither could a
referee.

Four things follow for how the correction must be written up, and they are design
constraints on the edit list rather than preferences.

1. **E3 is prerequisite to everything else.** Disclosing the crop rule (E2), the frame-0
   fix (E9) and the full-pocket clash rate (E12) is uninterpretable while the projection is
   described as acting on bond lengths and bond angles. The method text must be fixed first
   and the disclosures hung off it, not the reverse.
2. **This is a Section II-F correction, not an audit note.** The error is in the
   description of the method, at 865–868, and in a formula block that is missing an
   equation. In TCBB terms that is a revision of the submitted method section — a v2 — and
   it cannot be discharged by a sentence in a protocol paragraph or by the repro package.
3. **The crop rule must be stated where the projection can see it.** E2's sentence belongs
   in II-A (90–97) *and* must be cross-referenced from II-F, because the projection's
   barrier is the cropped pocket and not the pocket; placing the cap only in III-E would
   leave the method section still unable to support M3.
4. **The full-pocket clash rate has to appear in the paper, not only in the repro
   package.** It is the only quantity by which a reader can check that the clean protocol
   did not buy a prettier spread by losing a target-informed barrier (Amendment 4, B2).
   A fix that removes an undisclosed leak through an undisclosed mechanism, and reports no
   number by which the removal can be checked, is not a disclosure.

### A6.3.2 Does IV-D's separation still hold? **No, not as stated — for two independent reasons**

IV-D (1358–1390) and the Conclusion (1621–1625) state the separation as:

> *"The harmonic source prior determines how local fluctuation amplitude is distributed
> before and during flow transport, whereas geometric projection modifies selected stiff
> coordinates after generation. The two mechanisms are therefore complementary rather than
> interchangeable."* (1384–1390)

> *"…source shaping and post-generation projection play complementary roles: the harmonic
> prior controls the allocation of local fluctuation amplitude, whereas projection primarily
> corrects selected stiff geometric coordinates."* (1621–1625)

**Reason 1 — the projection is not a stiff-coordinate operator, and the paper's own
Section II says so.** The clash step moves atoms radially in Cartesian space at relaxation
1.0. Decompose such a displacement $\bm\delta$ on a bond $\mathbf d$: the bond length
changes by $\delta_\parallel+\delta_\perp^{2}/(2\|\mathbf d\|)+O(\delta^3)$, and by the
Wilson relation the paper itself puts in its abstract, an unstructured Cartesian
displacement of 1 Å appears as 43, 32 and 8.0 physical widths in bond, angle and torsion
respectively. So the clash step **injects** stiff-channel distortion by exactly the
mechanism the paper's central argument identifies, while the bond and angle sweeps
**remove** it. The two have opposite signs in the same channels. The description *"modifies
selected stiff coordinates"* is therefore not an incomplete list but a wrong characterisation
of one of the three steps, and the paper refutes it with its own Section II. Note the
direction of the resulting mis-attribution, since it should not be overstated: IV-D assigns
the whole ON-minus-OFF change (e.g. MEHF's bond-angle ratio $1.79\to0.74$, lines 1379–1383)
to the bond and angle sweeps, whereas it is the *sum* of those sweeps and a counteracting
clash term. The sweeps are therefore acting somewhat **more** strongly than IV-D implies,
and there is a residual clash-induced inflation IV-D does not account for. This does not
flip the ON/OFF contrast; it invalidates the labels on its two branches.
   The one statement in IV-D that survives intact is 1373–1375, *"the out-of-plane channel,
   which is not directly modified by the bond- and angle-based projection"* — true of the
   two internal-coordinate steps, and the clash step's effect on out-of-plane is indirect,
   which is the sense the sentence already carries. It should still be qualified with
   "directly" doing explicit work.

**Reason 2 — under the published protocol the "post-generation correction" branch was
partly conditioned on the evaluation trajectory.** The clash step's reference set was a crop
selected by $\min_{f=0..99}$ over the very frames the model is asked to generate (§1). So
IV-D's second branch was not a pose-derived geometric correction; it contained a
**trajectory-defined** term, and the separation "source (before generation) versus geometry
(after generation)" quietly had a third ingredient that belonged to neither category. This
is an *independent* reason the separation fails, it has nothing to do with the spread metric,
and — unlike Reason 1 — it is repaired **in fact** by Change 1 rather than only in text:
after the fix the clash barrier is defined at frame 0, i.e. from the input pose, and the
branch becomes what IV-D says it is.

**What may still be claimed.** The ON/OFF comparison of IV-D remains a valid measurement of
*with versus without the whole projection block*, and the projection-OFF numbers (1368–1377:
spread 1.12–1.16 for the harmonic prior against 3.46–3.55 / 4.00–4.02 / 4.67–5.46, and the
out-of-plane ratio 1.03 against 2.56 / 4.83 / 5.58) are untouched by both reasons, because
with the projection disabled there is no clash step and no pocket in the comparison at all.
**The prior-versus-projection decomposition's projection-OFF half is the part of IV-D that
survives unqualified**, and it is also the half that carries the paper's strongest claim. The
ON half must be relabelled as "the full projection block, comprising bond, angle and
pocket-clash steps", and the complementarity sentence must be rewritten to name three
mechanisms rather than two.

## A6.4 The edits, keyed to `paper/MEHF_TCBB/main.tex` by line

Items marked **[unconditional]** are required whatever the clean read returns. Line numbers
are those of the 1,667-line source cited in A6.1; they will drift as edits are applied, so
each item also quotes its anchor text. House style keeps cross-references handwritten, so
every "Section~IV-D"/"Table~\ref" touched here must be re-checked by hand afterwards.

**E1 [unconditional] — dataset counts, five places plus a table.** Replace the pre-exclusion
counts of Amendment 1 §A1.3 everywhere they appear:
- **886–892**, *"yielding 4,913 undirected MMP pairs across 517 UniProt targets"* $\to$ 4,830 / 512;
- **Table I, 894–915**: rows become Training 342 / 3,417 / 6,834; Validation 105 / 687 / 1,374; Test 65 / 726 / 1,452; Total 512 / 4,830 / 9,660;
- **928–933** and **1050–1053**, both instances of *"1,410 directed test examples from 60 held-out protein targets"*, $\to$ the clean usable count and cluster count;
- **abstract 49**, *"On 1,410 examples from 60 held-out protein targets"*; **1224**, *"On 1,410 test examples from 60 held-out protein targets"*; **1614–1615** in the Conclusion; **Table IV caption 1267–1272** if it repeats them;
- **1088–1090**, *"approximately 5,950 and 1,030 directed examples"* $\to$ the post-exclusion fitting/dev counts.
Add after 892 one sentence: *"Twenty-eight MISATO systems are excluded from all splits
because the stored pocket candidate pool cannot provably represent the frame-0 crop defined
below; this removes 83 of 4,913 pairs (1.7%)."* State in the paper that the post-exclusion
counts are authoritative.

**E2 [unconditional] — state that the pocket is cropped, the cap, and the rule (F1).** At
**93**, *"and the surrounding protein-pocket atoms"*, add: *"the pocket being represented by
at most 96 protein heavy atoms, namely those with the smallest distance to any ligand heavy
atom at the conditioning frame, with their coordinates taken from that frame."* Cross-
reference it from **II-F (779)** and from **III-E (1070–1101)**, per A6.3.1 item 3. Three
things must be said, not one: that a crop exists, its cardinality, and its rule.

**E3 [unconditional] — the projection has three steps (F2); prerequisite to E9 and E12.**
In II-F, after the angle block ends at **862**, add a third correction with its own equation:
for ligand heavy atom $i$ and its nearest pocket heavy atom $\mathbf p_{k(i)}$ in the
96-atom crop,

$$
\Delta\mathbf x_{t,i}^{\mathrm{clash}}
=\gamma_{\mathrm{c}}\,
\max\!\big(0,\ d_{\mathrm c}-\|\mathbf x_{t,i}-\mathbf p_{k(i)}\|\big)\,
\frac{\mathbf x_{t,i}-\mathbf p_{k(i)}}{\|\mathbf x_{t,i}-\mathbf p_{k(i)}\|},
\qquad d_{\mathrm c}=1.2\ \text{Å}.
$$

**E4 [unconditional] — the relaxation factors (F2 item ii).** Rewrite **865–868**. The
single *"under-relaxation factor of $0.5$"* becomes
$\gamma_{\mathrm b}=\gamma_{\mathrm a}=0.5$ and $\gamma_{\mathrm c}=1.0$, with the asymmetry
stated: the clash step is applied at full strength. Replace *"acts directly on bond lengths
and bond angles"* with a correct characterisation — two internal-coordinate steps plus one
Cartesian pocket-clash step — and keep *"does not explicitly constrain torsions or
out-of-plane coordinates"*, which remains true.

**E5 [unconditional] — the separation claim (A6.3.2).** Rewrite **1384–1390** and
**1621–1625** to name three mechanisms: the source shapes the distribution before
generation; the bond and angle sweeps pull stiff internal coordinates toward their
input-pose values; the clash step pushes atoms in Cartesian space away from the cropped
pocket and, by the Wilson relation of Section II, that push itself lands disproportionately
in the stiff channels. Relabel the ON branch of IV-D as "the full projection block". Qualify
**1379–1383** (*"Enabling projection primarily changes the bond-length and bond-angle
channels"*) as the net effect of two counteracting terms. **1368–1377** (the projection-OFF
numbers) needs no change and should be identified as the part of the decomposition that is
unaffected, since with projection disabled no clash step and no pocket enter the comparison.

**E6 [unconditional] — an amplitude-free statistic in III-C (F5: addition, not correction).**
After the spread definition at **1013–1027**, add
$\psi=(\max_c r_c-\min_c r_c)/(\tfrac14\sum_c r_c)$ and one sentence: $S$ is the product of
an overall amplitude and a shape, so $S$ is compared only at matched torsional width — the
rule III-D already states at **1040–1048** — and $\psi$ is reported alongside $S$ for every
arm and control so that an under-dispersed sampler cannot score well on $S$ alone. Add the
$\psi$ column to Table IV (**1266–1304**).

**E7 [unconditional] — repair the Limitations control (F6).** **1577–1583** must report, next
to the four width ratios, their channel spread 0.59, the method's 0.770, the realised torsion
ratio 0.56 and the resulting $\hat\psi=0.707$ against the method's 0.485, and must say in one
sentence that the draw's low spread reflects its low amplitude rather than better channel
balance. The unimodality point is correct and stays. Without this edit the paragraph reads as
the method losing to its own prior, and the arithmetic is one division away.

**E8 [unconditional] — the $\tau$ sentence.** **1553–1558**, *"re-estimating the four
empirical internal-coordinate fluctuation widths after excluding validation and test systems
changes each value by less than 0.4\%"*, is replaced by the Amendment 2 measurement:
re-estimated on **training-split systems only**, with $N$ per channel, the bootstrap SE of
each median, the common mode and the residual ratio spread, and the S2 noise-only A/B. The
distinction must be printed — the $\le0.4\%$ figure came from a *without val/test*
recomputation over a much larger population, which is a weaker exclusion than
train-split-only — and the two numbers must never stand side by side as if they were the
same measurement.

**E9 [unconditional] — a protocol and leakage paragraph, since there are no appendices
(F4).** Insert after **1068** (end of III-D), carrying five items: (i) no UniProt target and
no PDB entry shared across splits — already at 919–921, cross-referenced not repeated;
(ii) all checkpoints trained on the training split only, selection on a target-disjoint fold
inside it — already at 1086–1093; (iii) **the pocket crop is selected at the conditioning
frame; a previous version of the protocol selected it by the minimum distance over all 100
trajectory frames, i.e. using frames the model is asked to generate; the two crops overlap
by mean 0.823 (median 0.833, minimum 0.604, about 17 of 96 atoms differing), and the pocket
coordinates were taken from frame 0 under both**; (iv) the widths $\tau$ are estimated on
training-split systems only, with the measured shift; (v) **the test split has been
evaluated three times, under three pre-registrations whose hashes are part of the release**,
so the nominal 95% intervals carry a family-wise cost (§0; Amendment 3 §A3.2 raises the
comparison count to four). Item (v) is not optional: the submission currently contains no
mention of pre-registration, hashing or read counts anywhere, so there is no other place a
reader learns this.

**E10 — the operating point is reported two ways.** **1040–1048** gains one sentence:
$s^{*}$ is selected on the fitting folds (primary, as pre-registered) and also on the
target-disjoint development folds (secondary), both frozen before the test read; if the two
disagree in Tier, the reported result is the worse of the two (Amendment 3 §A3.3). The same
paragraph must not describe the development folds as held out — **1086–1093** uses them for
checkpoint selection.

**E11 — the numbers, per §6.** Whatever Tier the clean primary lands in, the §6 reading
governs: **abstract 49**; IV-C **1224–1241**; **Table IV 1266–1304**; Fig. 2 caption
**1306–1318**; IV-D **1368–1383**; IV-F **1559–1567**; Conclusion **1612–1625**. All are
recomputed from the clean medians rather than carried over. Under Tiers R and A the leaky
figures move into E9's paragraph; under Tier W the headline becomes the per-channel table;
under Tier N the claim is withdrawn as §6 says. The submission does **not** contain a
"2.6 times more uniform" phrase (the 2.6 at **1325** and **1348** is a noise-scale factor,
unrelated), so that item of A6's first draft is withdrawn.

**E12 — the controls and the clash rate go in Table IV.** Per §5.4, B3(c) and Amendment 4,
**Table IV (1266–1304)** gains both model-free controls as rows — spread, $\psi$, realised
torsion ratio, internal RMSF ratio and **full-uncropped-pocket clash rate** — with an
*unmatched* footnote where B3(b) applies, whether or not B3 fires. The existing footnote
machinery at **1299–1303** is the model: MEHF-GS is already footnoted as not reaching a
torsional width ratio of one, which is the precedent an unmatched control inherits. The
clash-rate column is required for every arm (Amendment 4) and is the number A6.3.1 item 4
says must be in the paper.

**E13 — Limitations additions.** After **1596**, two sentences: (i) the frame-0 crop omits
protein atoms the ligand genuinely approaches later in the trajectory (P7), so the clash
term — in both the loss and the projection — fences less than the all-frames crop did; the
fix makes the conditioning honest, not strictly better; (ii) what the model-free controls do
and do not establish, in the language of Amendment 5 §A5.4, naming specifically that the
projection-ON source control is what bounds how much of the channel uniformity is
attributable to the learned velocity rather than to the parameter-free projection, and that
this bound is not yet established.

## A6.5 Release items

`RESULT.md` must list, for the record: the sha256 of this file as amended; the draft-1 hash
`a51668b7…` and the `head -638 | head -c -1` recipe that verifies it; the three test-read
dates and their pre-registrations; `excluded_systems.json` and its 28 entries; the $\tau$
JSON; the frozen $s^{*}$ JSON with both the fitting-fold and the dev values; the
analysis-script hash taken with zero `test_*.json` on disk; and the `LAP_DECISION.txt`
timestamp. Two divergences between the two manuscript versions are part of the release
notes, so that they are never conflated again: the Overleaf master contains an audit bullet
(*"Projection targets and all other conditioning come from the input pose"*) that the
submission does not, and the master contains an amplitude-insensitivity claim about the
spread metric that the submission does not (F4, F5).

---

# AMENDMENT 7 — the noise-scale grid is extended to $[1.0,5.0]$, after the first grid run, 2026-10-24
Written after the fitting- and development-fold grids were run and before `test` is read for
any purpose. **`test` remains untouched**; no test command has been issued under this
document. This is a change to a **pre-registered candidate range made after seeing data**,
and it is the one amendment of this set that must carry its own defence rather than a
reason.

## A7.1 What changed, and the exact ceiling that forced it

§2 froze the operating-point grid at $\{1.0,1.5,\dots,4.0\}$. It is extended to
$\{1.0,1.5,\dots,5.0\}$ — two additional measured points, 4.5 and 5.0. Nothing else
changes: the rule, the metric, the arms, the folds, the seeds, the budget and the inference
command are as frozen.

The first freeze (`outputs/clean_protocol_v1/s_star_frozen.json`, 2026-10-24T05:04:12Z,
kept and superseded, not deleted) returned **UNMATCHED in four of twelve cells**, and the
maximum torsion ratio reached at the top of the frozen grid shows why:

| sample | arm | fold 1 | fold 2 | fold 3 |
|---|---|---|---|---|
| fitting (PRIMARY) | ISO | matched, max 2.258 | matched, max 2.709 | matched, max 2.507 |
| fitting (PRIMARY) | HARM | matched, $s^{*}=3.746$ | matched, $s^{*}=3.339$ | **UNMATCHED, max 0.940** |
| dev (SECONDARY) | ISO | matched | matched | matched |
| dev (SECONDARY) | HARM | **UNMATCHED, max 0.964** | **UNMATCHED, max 0.968** | **UNMATCHED, max 0.877** |

HARM's torsion ratio **topped out below 1 and never bracketed the match** in those four
cells. Under §A3.1's no-extrapolation rule they have no operating point at all, which left
the pre-registered **primary contrast resting on two of three folds and the secondary
(Amendment 3) on none of three.** A secondary that exists in the document and in zero cells
of the data is not a secondary; it is a sentence.

**The alternative to extending the grid was worse in a way that must be on the record.**
Keeping $[1.0,4.0]$ and reporting HARM fold 3 at its nearest grid point means comparing an
arm at torsion ratio 0.940 against an arm at 1.000 — i.e. reintroducing exactly the
amplitude confound that Amendment 5 §A5.3 was written to close, and doing it in the primary
rather than in a control. It would also have put HARM fold 3 outside B1's $[0.85,1.25]$
window on the dev side (0.877) and so blocked its own tier. Extending the measured range is
the only move that keeps the matched-amplitude premise, and the matched-amplitude premise is
the thing this protocol is built on.

## A7.2 Why this is not protocol-shopping — the argument, and the concession

Four reasons, each checkable, and then the part that is a genuine concession.

**1. Measuring more grid points is not extrapolation, and the rule was not touched.**
`scripts/matched_torsion_fit.py` is imported unchanged, sha256
`ca54723b142a141ade9b0fec78eaf27a19a10ca71a280a234647567f97a33bc3`, **identical in both
freeze files**. $s^{*}$ is still the piecewise-linear crossing of torsion ratio $=1$ between
two **adjacent measured** points that bracket it. The prohibition in §A3.1 is on
extrapolating past measurement; extending measurement is the opposite operation, and it is
what read 1's Amendment 2 could not do for MEHF-GS because GS's curve was unmatchable from
*below* the grid, where no amount of extra measurement helps.

**2. The rule is mechanical, so a longer grid cannot be steered.** The matching target is
torsion ratio $=1$, a fixed number, in a channel that is not the reported quantity. There is
no free parameter in which a preference for a low spread could enter: for any grid, the rule
returns the smallest crossing of 1, and extending the grid can only (i) leave an existing
crossing untouched, or (ii) create a crossing where there was none. It cannot move a
crossing to a more flattering place.

**3. The decisive check: every cell that was already matched is bit-identical across the two
freezes.** Verified, all eight: ISO $1.5474482221487433$ / $1.2620987538250512$ /
$1.5455534264901918$ on fit and $1.4076217951622483$ / $1.4941402926913012$ /
$1.4354820841918712$ on dev, HARM $3.746087541521619$ / $3.3387370321094494$ on fit —
identical to the last digit in both files, as are their `spread_at_s_star` values, and each
arm's nine-point torsion curve has the old seven-point curve as an **exact prefix**. So the
extension is strictly additive on the measurements that existed; it changed no number that
had already been produced. This is the form of evidence that distinguishes "we measured
further" from "we re-measured until it worked", and it is why it is recorded rather than
asserted.

**4. It was applied symmetrically, not to the failing cells.** Both new points were run for
**every** arm, **every** fold and **both** the fit and dev samples — twelve cells, including
the eight that did not need them and the ISO cells whose crossings are at $s\approx1.3$–1.6
and for which $s=4.5$ and 5.0 are 3 units of irrelevance. A selective extension of only the
unmatched cells would have been arm- and fold-dependent and would not have been defensible.

**The concession, stated plainly because the hash of draft 1 is worth nothing if it is not.**
The candidate range is nonetheless **wider than the one frozen in draft 1**, and the decision
to widen it was taken with the first grid run in hand. A reader is entitled to see that the
direction of the change was *forced by a ceiling* rather than chosen: the evidence for that
is §A7.1's four UNMATCHED cells with maxima 0.877–0.968, all on one side, all in one arm —
a one-sided failure that no choice of a wider range could have been tuned to exploit, since
the only thing a wider range can do to a ceiling is lift it. Had the extension instead moved
an already-existing crossing, or been applied to one arm, or changed the rule, none of the
above would be a defence and this read would not be worth taking.

## A7.3 The frozen values, all twelve cells matched

`outputs/clean_protocol_v1/s_star_frozen_v2.json`, frozen 2026-10-24T06:40:33Z, **before any
test command**. No UNMATCHED anywhere.

| | arm | fold 1 | fold 2 | fold 3 |
|---|---|---|---|---|
| **PRIMARY** (fitting folds, 96 examples) | ISO | $s^{*}=1.547$, spread 1.917 | $1.262$, 1.576 | $1.546$, 1.863 |
| | HARM | $3.746$, **0.448** | $3.339$, **0.462** | $4.349$, **0.391** |
| **SECONDARY** (dev folds, 160 examples) | ISO | $1.408$, 1.677 | $1.494$, 1.946 | $1.435$, 1.693 |
| | HARM | $4.199$, **0.511** | $4.194$, **0.535** | $4.758$, **0.497** |

§2's grid row is hereby revised to $\{1.0,1.5,\dots,5.0\}$; every other row of §2, as
amended by Amendment 1, stands.

## A7.4 Applying §A3.3 to these numbers

$|s^{*}_{\text{dev}}-s^{*}_{\text{fit}}|$ is **0.139 / 0.232 / 0.111 for ISO** and
**0.453 / 0.855 / 0.409 for HARM**. §A3.3 case 2 requires $\le0.5$ for *every* arm and fold;
**HARM fold 2 at 0.855 breaches it**, so:

> **Case 2 (insensitive) is excluded, and that determination is made here, on train-split
> data, before the test read.** The document is therefore committed to **case 3** — both
> $\Delta$ values and both CIs printed in the same table, with
> $\max|s^{*}_{\text{dev}}-s^{*}_{\text{fit}}|=\mathbf{0.855}$ reported as the
> operating-point fragility measure — **unless the two operating points land in different
> Tiers, in which case case 4's downgrade-only rule applies instead.** Which of the two it
> is cannot be known until the single test session, and the rule for both was fixed in
> Amendment 3 before any of these numbers existed.

**The HARM/ISO asymmetry is a geometric consequence of where each arm operates, not
evidence that HARM is less stable, and that can be computed rather than argued.** $s^{*}$ is
the solution of $r_{\mathrm{tor}}(s)=1$, so a perturbation $\delta r$ in the torsion-width
curve moves the crossing by $\delta s^{*}=\delta r/r'(s^{*})$: the flatter the curve at its
crossing, the larger the shift. From the frozen nine-point curves, the secant slope
$r'$ across the bracketing interval is

| fold | ISO $r'$ | HARM $r'$ | ratio |
|---|---|---|---|
| 1 | $(1.2475-0.9740)/0.5=0.547$ | $(1.0506-0.9509)/0.5=0.199$ | 2.75 |
| 2 | $(1.1694-0.8134)/0.5=0.712$ | $(1.0411-0.9136)/0.5=0.255$ | 2.79 |
| 3 | $(1.2865-0.9713)/0.5=0.630$ | $(1.0260-0.9401)/0.5=0.172$ | 3.66 |

HARM's curve is **2.8–3.7 times flatter** at its crossing. The observed ratio of
$|\Delta s^{*}|$, HARM over ISO, is $0.453/0.139=3.3$, $0.855/0.232=3.7$,
$0.409/0.111=3.7$. **The flatness ratio predicts the disagreement ratio quantitatively**,
which means the two arms' underlying torsion-width curves are disagreeing between fit and
dev by *the same* relative amount and only the conversion to $s^{*}$ differs. This is §4's
mechanism M2 confirmed in its premise — "a larger $\Delta s^{*}$ for the arm whose curve is
flatter near its crossing" — and it settles the premise without settling M2's consequence,
which is about how fast the *spread* moves with scale and is not measurable here (the fit
and dev spreads come from different example sets, so their difference is not a derivative).

One further pre-test observation, from
`outputs/clean_protocol_v1/clash_at_sstar_summary_v2.json`, recorded because it bears on B1:
at the **primary** $s^{*}$, the realised torsion ratio on the dev sample is
**0.916 / 0.844 / 0.933** for HARM and 1.089 / 0.859 / 1.068 for ISO. HARM fold 2 at 0.844
is **below B1's $[0.85,1.25]$ window** on that sample. This is dev, not test, and B1 is
defined on test; it is recorded now as a forewarning that B1 may fire, so that if it does,
it was anticipated rather than discovered.

## A7.5 What the agreement at both operating points does and does not buy

**Does.** All twelve cells matched, so the matched-amplitude premise holds at both operating
points — a stronger position than read 1, where MEHF-GS was unmatchable and had to be
reported as unmatched. The direction and magnitude agree across the two: HARM
**0.391–0.535** against ISO **1.576–1.946**, six cells each, non-overlapping ranges,
separation 1.1–1.5 spread units, with no cell reversing. Whatever the operating point, the
ordering is not in doubt on these samples.

**Does not — and this is the part a reader must not be allowed to skip.** Both samples are
**in-sample or near it**: the fitting folds are the data the arms trained on, and the dev
folds set the checkpoints (§A3.1). The frozen record's own lesson is exactly about this
gap: HARM went **0.216 / 0.270 / 0.460 on fitting folds to 0.770 on test**, a factor
$1.7$–$3.6$, while ISO went $1.71$–$1.94$ to $1.974$, a factor $1.05$. Applying those two
observed degradation factors to the clean numbers above forecasts a clean **test** median of
roughly $0.39\times1.7$ to $0.46\times3.6=\mathbf{0.66\text{–}1.66}$ for HARM and
$1.58\times1.05$ to $1.92\times1.05=\mathbf{1.65\text{–}2.01}$ for ISO, i.e. a paired
difference anywhere in $\mathbf{[-1.35,\,+0.01]}$.

**That interval spans every tier in §6, including N.** So the fit/dev separation of
$0.45$ versus $1.92$ is **not** a preview of the test result, and in particular it must not
be read as the effect being four times larger than published — that reading is precisely the
error the frozen record already caught once, in the arm that already degraded. P2's bands
($\Sigma^{*}\in[0.65,1.10]$, $\text{ISO}\in[1.75,2.20]$, $\Delta\in[-1.45,-0.85]$) remain as
stated and **I am not revising their confidence on in-sample data**; I note only that clean
ISO's fitting-fold spreads ($1.58$–$1.92$) sit slightly below the frozen arm's
($1.71$–$1.94$), which puts the lower edge of P2's ISO band at mild risk. P4 — that HARM is
again the only arm to degrade materially from fitting folds to test — is now the prediction
most directly in play, and nothing above scores it.

## A7.6 $\tau$: P1a is REFUTED, P1b is WRONG by its own bound, and the ceiling's cause is NOT established

**The irony, recorded as instructed, because it is instructive.** $\tau$ is the parameter P1
predicted would be a no-op at 0.95 confidence (revised to 0.55 / 0.85 / 0.50 in Amendment 2),
and it is the named suspect for having pushed HARM's operating point off the end of a
pre-registered grid. A predicted no-op does not usually break a frozen range.

**The enlarged estimate, and P1a's score.**
`outputs/clean_protocol_v1/tau_train_only_large.json`: $N=2{,}330$ systems / 344 targets
(1,998 / 299 for oop), post-exclusion train split, 32 times the earlier sample.

| channel | current | train-only large | rel. change | target-cluster SE | change / SE | vs. small sample |
|---|---|---|---|---|---|---|
| bond (Å) | 0.032637 | 0.0327568 | $+0.367\%$ | $0.265\%$ | 1.4 | $+0.94\%$ |
| angle (deg) | 2.774 | 2.770747 | $-0.117\%$ | $0.576\%$ | 0.2 | $+0.07\%$ |
| torsion (deg) | 14.600 | **14.128950** | $\mathbf{-3.226\%}$ | $2.88\%$ | 1.1 | $-0.23\%$ |
| oop (deg) | 6.480 | 6.465123 | $-0.230\%$ | $0.449\%$ | 0.5 | $+0.77\%$ |

**The $-3\%$ torsion shift was not noise**: at 32 times the sample it reproduced to within
$0.23\%$. Amendment 2 §A2.5 said the enlarged sample was as likely to refute P1a as confirm
it; it refuted it. Redoing §A2.3's decomposition on these values, $\bar\delta=-0.802\%$ and
the residual is $(+1.169,+0.685,-2.425,+0.572)\%$, a **residual spread of 3.59 pp** (draft 1
assumed $0.43$), hence a relative perturbation of $\operatorname{diag}F$ of
$-2\times$residual $=(-2.34,-1.37,+4.85,-1.14)\%$, a **spread of 7.19 pp** against draft 1's
$\le0.86\%$.

**S2 exists, it is paired, and it scores both halves of P1.**
`outputs/clean_protocol_v1/tau_diag/{legacy,train_only_large}/` is the noise-only A/B
Amendment 2 §A2.4 specified: the $\Sigma^{*}$ source with **no flow and no pocket**, same
fold example sets, same `seed 20260918`, same scales, differing only in `tau_set` — 36
paired cells (3 folds $\times$ 2 samples $\times$ 6 scales). Relative change in the source's
median channel width ratios, train-only-large against legacy:

| | bond | angle | torsion | oop | internal RMSF |
|---|---|---|---|---|---|
| mean over 36 cells | $+2.04\%$ | $+2.19\%$ | $+0.14\%$ | $+0.71\%$ | $+0.03\%$ |
| max $|{\cdot}|$ over cells | $3.91\%$ | $3.70\%$ | $2.60\%$ | $4.52\%$ | $0.38\%$ |

- **P1a — REFUTED.** Its threshold was $<1\%$ per channel. Bond and angle move $\approx2\%$
  on average and up to $3.9\%$ and $3.7\%$; oop reaches $4.5\%$. Refuted on the means, not
  only on the extremes.
- **P1b — WRONG by the rule I pre-committed to.** §A2.4 fixed it: CONFIRMED if
  $0.77\,w<0.02$, i.e. $w<2.6\%$, with $w$ = S2's largest per-channel relative width
  difference. $w=4.52\%$ gives the bound $0.035>0.02$, so **WRONG**. The softer reading is
  reported next to it and not substituted for it: propagating the *mean* changes through the
  frozen operating point's channel levels gives
  $\Delta\text{spread}\approx r_{\text{oop}}\varepsilon_{\text{oop}}-r_{\text{tor}}\varepsilon_{\text{tor}}
  =1.89(0.0071)-1.15(0.0014)=\mathbf{0.012}$, below 0.02, so P1b's substance may well
  survive even though its pre-registered test does not. I scored it by the rule.
- **The internal RMSF is invariant to $0.03\%$ (max $0.38\%$)**, which is the common-mode
  invariance of §1 confirmed by measurement rather than by T4 alone: the $\alpha$ rescaling
  does absorb a common change of the four widths, exactly as derived.
- One check for the implementer: S2 as specified held $\bm\xi$ fixed. These runs share the
  seed and the example set, which should make them paired at the draw level, but it is worth
  one assertion that the two $\tau$ sets consume the RNG identically. If they do not, the
  per-cell maxima are noise-inflated and $w$ should be re-read from the means.

**The proposed mechanism for the ceiling is plausible but is NOT supported by the
measurement, and the attribution must stay open.** The stated chain is: $\tau_{\text{tor}}$
falls $3.23\%$ $\to$ $F=\tau^{-2}$ rises $\to$ torsions stiffen in the harmonic energy
$\to$ $\Sigma^{*}$ delivers less torsional variance $\to$ a larger scale is needed to reach
ratio 1. The first three links are right. **The fourth is contradicted at the source level**:
S2 measures the torsion channel moving $+0.14\%$ on average, with the systematic $+2\%$
landing in bond and angle, so the $\alpha$-rescaled source does *not* deliver measurably
less torsional width under the new $\tau$. Two further checks point the same way. (i) Even
taking the worst per-cell torsion change, $-2.5\%$, and the measured torsion exponent
$\approx0.98$ (Amendment 5 §A5.3), the implied shift is
$s^{*}\mapsto s^{*}(1/0.975)^{1/0.98}$, i.e. $+2.6\%$, about $+0.10$ at $s^{*}\approx3.8$.
(ii) The observed shifts against the frozen leaky read ($3.245/3.329/3.787\to3.746/3.339/4.349$)
are $+0.50/+0.01/+0.56$ — five times larger in two folds and essentially zero in the third.
A single $\tau$-driven mechanism cannot produce that pattern.

**What this does not license, in either direction.** S2 bounds the $\tau$ effect on the
**source**. The clean HARM arms were *trained* with the new $\tau$, so the flow learned to
transport from a different source and the trained model's torsional response is not bounded
by the source's. If $\tau$ caused the ceiling, it did so **through training**, and that
channel is unmeasured — the measurement would need the clean-crop, old-$\tau$ training run
that §2's frozen budget excludes and that §A2.4 already declined to add. So: $\tau$ is not
exonerated, it is un-implicated at the source level; the frame-0 crop (a differently trained
model), the post-exclusion example sets, and ordinary retrain variance are the other
candidates; and **the honest record is that the cause of the ceiling is not established.**
The grid extension is justified by the ceiling's existence, which is measured, and does not
depend on its cause, which is not.

## A7.7 Resolved, not open: the silent-fallback audit is clean

`_harmonic_local_noise` contains an isotropic-fallback branch, so part of the HARM arm could
in principle have been trained with isotropic noise, which would have made the central
contrast partly vacuous. `outputs/clean_protocol_v1/harmonic_fallback_audit.json` closes it:
across **all six fold-by-split combinations, 20,502 examples**
($5{,}830+1{,}004+5{,}808+1{,}026+5{,}808+1{,}026$) and 2,048 distinct probed products,
`fraction_examples_isotropic_fallback` is **0.0** in every cell, with every molecule on the
`ok` branch, **under both the legacy and the train-only-large $\tau$ sets**. The audit is
measurement only; the fallback code was not touched and the clean arms were trained as they
are. **The possibility that any part of the HARM arm was silently isotropic is excluded, and
the contrast is clean on that axis.** This is recorded as resolved so it cannot reappear as
an open question after the test read.

Two by-products of that audit, recorded for the record and *not* claimed as findings.
(i) The null-space dimension is 6 for 99.8% of molecules (e.g. 1,646 of 1,649 in fold 1 fit),
with 8 or 9 for two or three per fold. The manuscript states that **4.7%** of ligands carry
additional near-zero modes (`paper/MEHF_TCBB/main.tex:1585–1589`); the two numbers differ by
more than an order of magnitude and are probably measuring different things (an exact
null-space dimension against a threshold on small eigenvalues). It is flagged as a check the
implementer should run before the manuscript repeats the 4.7%, not as a discrepancy
established here.
(ii) At the primary $s^{*}$, the full-pocket clash rate
(`clash_at_sstar_summary_v2.json`, fit/dev only, never test) is **identically 0 for ISO in
all six cells** and $2.1\times10^{-4}$ to $4.4\times10^{-3}$ of frames for HARM, with
1.0–12.5% of examples showing at least one violation — i.e. HARM clashes and ISO does not,
as §4's M3 anticipated from HARM's roughly four-fold wider ensembles. The full-pocket and
crop rates are equal to three digits in most cells and differ slightly only for HARM folds 1
and 3, an early and partial indication that at these amplitudes the frame-0 crop captures
nearly all the contacts that are actually approached (relevant to P7, which is scored on
test). B2's tolerance is still computed as Amendment 4 §A4.3 defines it, on test, from the
leaky arms.
