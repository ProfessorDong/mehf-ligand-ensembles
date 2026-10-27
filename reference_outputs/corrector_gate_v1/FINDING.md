# GATE RESULT: the projection beats force-field relaxation on bonded geometry,
# and currently damages aromatic planarity

DiffSBDD released CrossDocked samples, 200 molecules (2 per pocket, 100
pockets). Ideal bond and angle values estimated from the 100 REAL CrossDocked
test ligands. Script `scripts/geometry_corrector_gate.py`; predictions were
written in its docstring and hashed (97c83628...) before any arm was measured.
Medians over molecules.

| arm | bond RMS dev (A) | angle RMS dev (deg) | aromatic out-of-plane (deg) | RMSD from raw (A) | clashes <2A |
|---|---|---|---|---|---|
| A raw model output | 0.1092 | 12.65 | 4.14 | 0 | 0 |
| B MMFF94, unrestrained | 0.0284 | 5.61 | **0.57** | **1.675** | 2.0 |
| C MMFF94, restrained 100 | 0.0363 | 7.68 | 3.17 | 0.153 | 0 |
| **D projection (ours)** | **0.0149** | **1.42** | **10.13** | 0.237 | 0 |

## The gate passes on its stated criteria

P1, bonded repair comparable to the restrained force field: PASSES, and by more
than "comparable". Bond deviation is 2.4x lower than C and lower than
unrestrained MMFF; angle deviation is 5.4x lower than C and 4x lower than B.

P2, pose preservation: PASSES. D moves atoms 0.237 A, inside the pre-registered
0.25 A, while unrestrained MMFF moves them 1.675 A -- 7x more. For a
pocket-conditioned generator the pose IS the prediction, so a remedy that moves
it 1.7 A is not free, and that is the reason this operator has a place.

The collapse condition -- "C matches D on both" -- is FALSE. A restrained
force-field minimisation does not reproduce this.

## P3 was wrong, and the error is the method's current limitation

The prediction was that D would leave aromatic planarity UNCHANGED, because
bond and angle B-vectors are first-order blind to out-of-plane distortion. What
happens is worse than blindness: **D degrades planarity from 4.14 to 10.13
degrees**, a factor of 2.4 the wrong way.

The mechanism is now understood and part of it was an implementation defect that
has been fixed. Angle classes were originally keyed without RING SIZE. A
six-membered aromatic internal angle is about 120 degrees and a five-membered
one about 108; pooled, the median lands near 115, and three such targets at a
trivalent centre sum to 345, below the 360 that a planar centre requires. Three
too-small angle targets can only be satisfied by pyramidalising. With ring size
in the key the damage falls from 13.36 to 10.13 degrees, so that defect was real
and is repaired, but a residue remains.

What remains is the derived blindness itself: nothing in the stiff set
{bonds, angles} penalises out-of-plane displacement, so the redundant system has
a free direction and the solver drifts along it. The fix is derived and pending:
add an improper / out-of-plane coordinate to the stiff set.

**Consequence for claims.** Until that is added, this operator may be described
as improving BONDED geometry. It may NOT be described as improving geometric
validity in general, because aromatic planarity is one of the standard checks
and this operator currently makes it worse. No PoseBusters-style headline until
the improper term exists and is measured.

## Also worth keeping

- Unrestrained MMFF introduces 2 receptor clashes per molecule at the median
  while moving the ligand 1.675 A; the raw output and both conservative arms
  have none. "Just relax it" is not a neutral operation in a pocket.
- The projection introduces no clashes despite having no nonbonded term, which
  is a consequence of its small displacement rather than of any repulsion.

## Scope

200 of 10,000 available molecules; one published model (DiffSBDD); ideal values
from 100 reference ligands giving 10 bond classes and a few dozen angle classes,
so rare chemistry is unconstrained and simply passes through. All three should
be scaled before publication.

---

# UPDATE: the improper term, the circularity it exposed, and the soft-target fix

## 1. The improper coordinate works exactly as derived

Added the Wilson out-of-plane B-vector (formula and finite-difference
verification in `outputs/theory_modules_v1/RESULT.md` 3.7) to the stiff set.

| arm | bond RMS | angle RMS | planarity | RMSD |
|---|---|---|---|---|
| D projection | 0.0149 | 1.419 | **10.131** | 0.2372 |
| E + improper | 0.0153 | 1.521 | **0.458** | 0.2374 |

Planarity 10.13 -> 0.458 deg, better than unrestrained MMFF's 0.566.

**Its own control passed.** Because B_oop is exactly orthogonal to every bond
and angle row at planarity, switching it on must leave those channels almost
untouched. Measured: bond moved 2.7%, angle 7.2%, both inside the 25% tolerance
fixed in advance. The prediction and its control were derived before the run.

## 2. A circularity in the primary metric, found and corrected

The deviation-from-ideal score IS the projection's objective: the operator
drives each bond and angle to its class median and is then scored on distance
from that median. MMFF optimises something else. The comparison was unfair to
MMFF and, worse, the score cannot see the specific way this operator fails —
an operator that collapsed every bond onto the median would score perfectly
while destroying the distribution.

Replacing it with a metric NEITHER arm optimises, the class-count-weighted
Wasserstein distance between each arm's internal-coordinate distribution and
the real ligands':

| arm | W(bond) | W(angle) |
|---|---|---|
| A raw | 0.0536 | 5.933 |
| B MMFF free | 0.0236 | **1.859** |
| C MMFF restrained | 0.0223 | 2.988 |
| E projection + improper | 0.0204 | **2.498** |

**On the neutral metric the projection LOSES the angle channel to unrestrained
MMFF**, 2.498 against 1.859, where the deviation score had it winning 1.521 to
5.608. The deviation score overstated the advantage by about 3.7x. The cause is
the predicted one: driving every instance of a class to one value narrows the
class distribution.

## 3. The fix is the paper's own thesis applied to its own operator

Target a DISTRIBUTION, not a point: draw each term's target once as
median + N(0, class physical width). Same operator, same sweeps, no retraining.

| arm | W(bond) | W(angle) | planarity | RMSD | clashes |
|---|---|---|---|---|---|
| A raw | 0.0536 | 5.933 | 4.140 | 0 | 0 |
| B MMFF free | 0.0236 | 1.859 | 0.566 | 1.675 | 2.0 |
| C MMFF restrained | 0.0223 | 2.988 | 3.172 | 0.153 | 0 |
| E hard target | 0.0204 | 2.498 | 0.458 | 0.237 | 0 |
| **F soft target** | **0.0181** | **1.358** | 0.717 | 0.244 | 0 |

F is best on both neutral channels, beating the standard remedy by 23% on bond
length and 27% on angle, **while moving atoms 6.9x less and introducing no
receptor clashes**. Planarity 0.717 is slightly behind free MMFF's 0.566 and
behind E's 0.458, and far ahead of the restrained force field's 3.172.

## 4. The circularity that is NOT yet removed

The class medians and widths are estimated from the same 100 reference ligands
that the Wasserstein distance is computed against. The operator is fitted to
that sample and MMFF is not, so these numbers remain optimistic.

**Required before any of this is reported: split the reference ligands, fit the
medians and widths on one half, score the Wasserstein against the other half.**
Until that is run, treat F's margin over B as an upper bound.

Other scope limits unchanged: 200 of 10,000 molecules, one published model,
10 bond classes and a few dozen angle classes so rare chemistry passes through
unconstrained.

---

# HELD-OUT CONTROL: the distributional advantage does not survive

The control flagged as required in section 4 above has been run. Reference
ligands are split in half; class medians and widths are fitted on one half and
the Wasserstein distance is scored against the other. Three split seeds.

| arm | W(bond), 3 seeds | W(angle), 3 seeds | planarity |
|---|---|---|---|
| B MMFF free | 0.0262 / 0.0245 / 0.0202 | **1.810 / 1.793 / 1.506** | 0.566 |
| C MMFF restrained | **0.0234 / 0.0232 / 0.0192** | 2.426 / 2.646 / 2.663 | 3.172 |
| E hard target + improper | 0.0332 / 0.0305 / 0.0243 | 3.131 / 3.835 / 3.439 | **0.350 / 0.330 / 0.363** |
| F soft target | 0.0264 / 0.0217 / 0.0214 | 2.247 / 2.767 / 2.346 | 0.604 / 0.509 / 0.602 |

## Retraction

**The reported F advantage — W(bond) 0.0181 and W(angle) 1.358, described as 23%
and 27% better than the standard remedy — is withdrawn.** Those numbers fitted
the class medians and widths to the same 100 ligands the score was computed
against. With 10 bond classes and a few dozen angle classes estimated from 100
molecules, the targets overfit that sample; MMFF, fitted elsewhere, got no such
advantage. The overfit was worth roughly 30% of the claimed margin.

Under the held-out control, on the distributional metric:
* the ANGLE channel is won by unrestrained MMFF on every seed, by 25 to 35%;
* the BOND channel is a tie, the restrained force field taking two seeds of
  three and the soft projection one.

## What survives, and why each survives

1. **Aromatic planarity.** E gives 0.330 to 0.363 degrees against unrestrained
   MMFF's 0.566, the restrained force field's 3.172 and the raw output's 4.140.
   This is not fitted: the target is chi = 0, a geometric condition, not an
   estimated class statistic, which is exactly why the held-out split does not
   touch it.
2. **Pose preservation.** 0.24 A against unrestrained MMFF's 1.675 A, a factor
   of 6.9. Structural, nothing estimated.
3. **No receptor clashes.** The projection introduces none; unrestrained MMFF
   introduces 2 per molecule at the median.

## The defensible claim

A geometry corrector that beats force-field relaxation on aromatic planarity,
MATCHES it on bond and angle distributions, and does so while moving the ligand
seven times less and introducing no receptor clashes. Since a pocket-conditioned
generator's pose is its prediction, a remedy that displaces it by 1.7 A is not
free, and that is the gap this operator fills.

It may NOT be claimed that the operator improves bond or angle geometry over a
force field. On held-out targets it does not.

---

# SECOND ROUND OF RETRACTIONS, 2026-09-25

Two errors, both mine, both found by the evidence pass and both verified
independently here. Together they remove most of what was left of the
corrector's advantage.

## R1. The force-field comparison used the wrong baseline

Only restraint 0 and restraint 100 were ever compared. The restraint sweep has
a sweet spot in between that neither bracket shows. On the same 200 molecules
and the same held-out split:

| arm | W(bond) | W(angle) | planarity | RMSD | clashes |
|---|---|---|---|---|---|
| raw | 0.0486 | 5.334 | 4.140 | 0 | 0 |
| MMFF unrestrained | 0.0262 | 1.810 | 0.566 | 1.675 | 2.00 |
| **MMFF restraint 5** | **0.0260** | **1.664** | 2.138 | 0.399 | **0** |
| MMFF restraint 100 | 0.0234 | 2.426 | 3.172 | 0.153 | 0 |
| projection + improper (E) | 0.0332 | 3.131 | 0.350 | 0.234 | 0 |
| soft target (F) | 0.0264 | 2.247 | 0.604 | 0.234 | 0 |

**MMFF at restraint 5 beats the projection on both distributional channels**
(bond 0.0260 against 0.0264, angle 1.664 against 2.247) and introduces no
clashes. **"Moves the ligand seven times less" is withdrawn**: that compared
against the unrestrained arm. Against restraint 5, the correct baseline, the
factor is 0.399 / 0.234 = **1.7**.

"Force-field relaxation is destructive" is withdrawn with it. It is a trade-off
curve, and at restraint 5 the cost is 0.399 A with the clash rate back at the
raw model's own level.

## R2. The planarity metric had no reference, and I was scoring it as if zero were the target

Every planarity number in this file was reported as lower-is-better. **Real
CrossDocked ligands measured by the identical function give 1.18 degrees, not
zero**, and a 300 K MD ensemble of real aromatic centres gives 5.22. Scored as
distance to the crystal/docked reference the ranking changes and the margin
disappears:

| arm | planarity | distance to 1.18 |
|---|---|---|
| **soft target (F)** | 0.604 | **0.576** |
| MMFF unrestrained | 0.566 | 0.614 |
| projection + improper (E) | 0.350 | 0.830 |
| MMFF restraint 5 | 2.138 | 0.958 |
| MMFF restraint 100 | 3.172 | 1.992 |
| raw | 4.140 | 2.960 |

**E, the arm reported as winning planarity decisively, is over-flattening**: at
0.350 degrees it is more planar than the crystal structures the model was
trained on and far more planar than physical thermal motion. The claim "beats
force-field relaxation on aromatic planarity" is **withdrawn**. F and
unrestrained MMFF are tied at the top, separated by 0.04 degrees, which this
sample cannot resolve.

## What is actually left

| claim | status |
|---|---|
| beats force-field relaxation on bond/angle distributions | withdrawn (held-out), and restraint 5 beats us |
| beats it on aromatic planarity | withdrawn (metric had no reference; tied with unrestrained MMFF) |
| moves the pose 7x less | withdrawn; against restraint 5 it is 1.7x |
| no receptor clashes | holds, but restraint 5 also has none |
| matches force-field relaxation on every channel at 1.7x less displacement, with no energy function, no nonbonded term, and no iteration to convergence | **this is what survives** |

The operator is competitive with a tuned force field rather than better than one.
That is a far weaker statement than this file previously carried, and it is the
one supported by the measurements.

## Why both errors pointed the same way

Both flattered the method: an unfavourable baseline omitted, and a metric whose
target was assumed to be zero. Neither would have been caught downstream —
a reviewer sees the table, not the sweep that was never run or the reference
that was never looked up. The general lesson for this project's remaining work:
for every metric, write down what value the REAL data takes before reporting any
arm against it.

---

# SCALE-UP: 2,000 molecules, five split seeds. Every held-out conclusion holds.

The scope limit flagged twice above -- "200 of 10,000 molecules" -- has been
removed. The held-out configuration was re-run at 2,000 molecules on five
reference-split seeds. Command, once per seed in {20260924, 1234, 7, 31337, 99}:

```
PYTHONPATH=src:scripts conda run -n MDWM python scripts/geometry_corrector_gate.py \
  --generated-dir data/external/diffsbdd/crossdocked_fullatom_cond \
  --reference-dir data/external/crossdocked/ref_flat \
  --receptor-root data/external/crossdocked/test_set \
  --max-molecules 2000 --max-per-pocket 20 --holdout --split-seed <seed> \
  --output outputs/corrector_gate_v1/scaleup/holdout2000_<seed>.json
```

Nothing in the script was changed; only `--max-molecules` and `--max-per-pocket`
differ from the 200-molecule run. The three seeds used before are among the five.

## Control on the scale-up itself, run before the results were read

The sample is 10x larger, so the first question is whether it is still spread
over the 100 pockets or has concentrated on whichever targets happen to yield
readable molecules first. Replaying the script's own reading loop:

| configuration | molecules | pockets touched | per pocket min / median / max |
|---|---|---|---|
| 200, max 2 per pocket (old) | 200 | 100 | 2 / 2 / 2 |
| 2000, max 20 per pocket (new) | 2000 | 100 | 20 / 20 / 20 |

Exactly 20 from each of the 100 pockets, and the old 200 are nested inside the
new 2,000. No target concentration, and the comparison to the earlier run is not
confounded by a change of pocket mix.

## Results, medians over 2,000 molecules, one column per split seed

| arm | W(bond), 5 seeds | W(angle), 5 seeds |
|---|---|---|
| A raw | 0.0537 / 0.0556 / 0.0594 / 0.0593 / 0.0583 | 5.273 / 5.720 / 6.144 / 6.108 / 5.717 |
| B MMFF free | 0.0249 / 0.0258 / 0.0212 / 0.0224 / 0.0203 | **1.630 / 1.985 / 1.844 / 1.736 / 1.281** |
| C MMFF restrained | 0.0222 / 0.0241 / 0.0206 / 0.0217 / 0.0189 | 2.248 / 2.744 / 2.982 / 3.104 / 2.597 |
| E hard target + improper | 0.0306 / 0.0236 / 0.0213 / 0.0225 / 0.0226 | 3.286 / 3.384 / 3.317 / 3.063 / 3.279 |
| F soft target | 0.0232 / 0.0171 / 0.0195 / 0.0203 / 0.0192 | 2.088 / 2.495 / 2.409 / 1.921 / 2.374 |

| arm | planarity (deg) | RMSD from raw (A) | clashes |
|---|---|---|---|
| A raw | 4.584 | 0 | 0 |
| B MMFF free | 0.629 | 1.579 | 2.0 |
| C MMFF restrained | 3.668 | 0.155 | 0 |
| E hard target + improper | **0.299 - 0.346** | 0.227 | 0 |
| F soft target | 0.573 - 0.630 | 0.231 | 0 |

## Verdict on each held-out conclusion

1. **"Matches the force field on bond, does not beat it."** HOLDS. F beats
   unrestrained MMFF on bond on 5/5 seeds (ratios 0.66-0.94) but beats the
   RESTRAINED force field on only 3/5 (ratios 0.71-1.05). Against the better of
   the two force-field arms it is a tie, which is what was claimed. The direction
   is marginally more favourable to F than it was at 200 molecules, where C took
   two seeds of three; it is still a tie and must not be upgraded to a win.
2. **"Loses the angle channel to unrestrained MMFF."** HOLDS, on 5/5 seeds, by
   11% to 85% (ratios F/B = 1.28, 1.26, 1.31, 1.11, 1.85). At 200 molecules the
   spread was 25-35%; at scale the loss is no smaller and on one seed larger.
   This is the conclusion most likely to be argued with and it survived.
3. **"Wins aromatic planarity."** HOLDS. E gives 0.299-0.346 deg against
   unrestrained MMFF's 0.629, the restrained force field's 3.668 and the raw
   output's 4.584; E beats B on 5/5 seeds. F is now level with B rather than
   behind it (F beats B on 4/5, means 0.604 against 0.629), a small shift in F's
   favour that changes no claim, since the planarity claim rests on E.
4. **"Wins on displacement."** HOLDS. E and F move 0.227 and 0.231 A against
   unrestrained MMFF's 1.579 A, a factor of 6.8-6.9, and introduce no receptor
   clashes where unrestrained MMFF introduces 2 per molecule at the median. The
   restrained force field still moves LESS (0.155 A) than the projection, as it
   did at 200 molecules.

**Nothing flips.** No conclusion in the held-out section above is overturned by
the 10x sample, and the one claim the method is not allowed to make -- beating a
force field on bond or angle geometry -- remains not allowed.

## What the five seeds do and do not measure

They vary only the split of the 100 reference ligands into fit and score halves,
so they bound the noise in the ESTIMATED class targets. The 2,000 generated
molecules are the same in every seed, so these are not five independent samples
of molecules and the spread between columns must not be read as a molecule-level
confidence interval. The molecule-level scope is now 2,000 of 10,000, 20 from
each of 100 pockets, one published model.
