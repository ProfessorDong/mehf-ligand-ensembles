# Pre-registered predictions, paper_evidence_v1 (sections 1 and 2 evidence base)

Written 2026-09-25 BEFORE any number in this directory was computed.
Owner: theory agent. Scope: sections 1 (the problem) and 2 (what everyone does).
No `test` split is read anywhere. `src/mdwm/models/`, `outputs/pareto_v1/`,
`outputs/sigma_star_test_v1/`, `outputs/sigma_star_v1/` are not written.

## a. The matching protocol for the isotropic-noise cost

A1. The per-channel over-dispersion ratio produced by isotropic noise is
    ratio_k = sigma * ||B_k|| / tau_k with a SINGLE sigma per protocol. I
    predict that within each matching protocol the sigma implied by the three
    channels agrees to within 15%, and that the sigma implied from the bond
    channel equals the sigma actually injected to within 10%.
A2. Because ratio_k is linear in sigma and the protocol only chooses sigma, the
    RATIO BETWEEN CHANNELS (bond/angle, angle/dihedral) is protocol-invariant.
    Predicted: bond/angle constant across all rotation-invariant protocols to
    within 10%.
A3. Kabsch-RMSF matching gives a larger bond ratio than pairwise-distance
    matching. Predicted sigma(Kabsch) / sigma(pairwise) in [1.3, 2.0].
A4. A protocol that removes translation but NOT rotation will give a LARGER
    bond ratio than Kabsch (predicted > 13), because the physical arm's rigid
    tumbling inflates its dispersion statistic while contributing zero internal
    distortion. This protocol is ruled out on principle, not on its number.
A5. Over rotation-invariant protocols the bond ratio falls in [6, 14].

## b. Aromatic-only out-of-plane thermal width

B1. Restricting improper centres to sp2 centres (exactly three bonded
    neighbours in the FULL graph, i.e. no hydrogen on the centre) lowers
    tau_oop below 6.48 deg. Predicted 3.5-5.5 deg.
B2. Restricting further to RDKit-perceived AROMATIC centres agrees with B1 to
    within 15%.
B3. The statistic DiffSBDD's 5.09 deg and the crystal 1.18 deg are is the mean
    absolute deviation from planarity of a single conformer, NOT a
    frame-to-frame standard deviation. The like-for-like thermal quantity is
    E|chi| over MD frames. Predicted aromatic-only E|chi| in [2.0, 4.5] deg, so
    DiffSBDD's 5.09 deg sits at 1.1-2.5x the thermal MAD, not 4.3x.
B4. The <10 deg planarity pre-filter on centres is itself a selection; removing
    it raises tau_oop. Predicted it changes aromatic-only tau_oop by < 25%.

## c. The seam: mechanism vs the DiffSBDD refutation

C1. MEHF with `projection_iterations=0` (the audit's arm) has a residual whose
    bond and angle channels are explained by ONE sigma. Predicted |measured
    angle ratio / angle ratio predicted from the bond channel - 1| < 0.20.
    This is the positive control for the mechanism on a trained model.
C2. The same statistic on MEHF with the projection ON (8 iterations) will be
    far from 1: predicted the projection breaks isotropy by more than 2x.
C3. If C1 holds and DiffSBDD fails at 1.74x, the defensible statement is that
    isotropy is the DEFAULT an isotropic objective leaves in place and not a
    law, and that the departures observed are not benign. I predict DiffSBDD's
    departure is in the harmful direction, i.e. the measured angle damage
    EXCEEDS the isotropic prediction rather than falling below it. (Already
    known to be 1.74x; the prediction here is that the same sign holds at a
    larger sample, n >= 1000 molecules.)

## d. Generality of "a geometry loss does nothing"

D1. The angle_term_v1 design cannot detect small effects. Predicted minimum
    detectable effect at 95% confidence, on the angle ratio base of 4.806, is
    a reduction of at least 25%.
D2. Reverse water-filling says the cell-B weight moves the objective's
    residual-variance allocation by 4.8%, so a near-null is the PREDICTED
    outcome for that weight. The experiment therefore cannot support a general
    claim about geometry losses; I predict I will have to withdraw the general
    form.
D3. A second, independent instance exists inside the same corpus: MEHF carries
    `bond_loss_weight = 1.0` and, with the projection removed, its bond channel
    should be indistinguishable from what pure isotropic noise predicts.
    Predicted: within 20% (this is the same statistic as C1).

## e. Is force-field relaxation destructive?

E1. The "destructive" claim is measured on UNRESTRAINED MMFF. The restrained
    arm already in `corrector_gate_v1` (100 kcal/mol/A^2) moves 0.153 A and
    introduces 0 clashes. I predict the general claim "force-field relaxation
    is destructive" must be WITHDRAWN and replaced by a dilemma statement.
E2. The dilemma: the restrained arm is not destructive but does not repair.
    Predicted it closes less than 40% of the raw output's gap to real ligands
    on aromatic planarity, and loses the angle channel to unrestrained MMFF.
E3. A pocket-aware relaxation (ligand relaxed inside a frozen receptor) is the
    strongest published version. Predicted it removes the clash penalty
    (< 0.5 clashes per molecule at the median) but still moves the ligand more
    than 0.8 A at the median, because the displacement is driven by the
    ligand's own internal strain, not by the receptor.
