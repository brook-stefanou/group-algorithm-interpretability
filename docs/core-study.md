# The core study

This registers the core study: the question, the five pre-registered tests
the design can license, the groups, the endpoints, and the run plan.  The design was fixed
on 2026-07-15, and the run plan below — the cell list, the widths, and the
epoch ceiling — was settled on 2026-07-20.  No analysis has been read and no
empirical result exists yet — every number below is a plan.  Selection and
reporting rules are in [the methodology](methodology.md), provenance rules in
[the reproducibility contract](reproducibility.md), and the directions the
study could grow in are in [extensions](extensions.md).

## The question, and why it's open

What algorithm does a small neural network learn when it's trained to
multiply in a finite group?  The circuit-level evidence so far comes from
cyclic groups, S5, and S6, split across three accounts.  Chughtai, Chan and
Nanda (2023) reverse-engineered representation-theoretic (Fourier) circuits.
Stander, Yu, Fan and Biderman (2024) found coset-concentration circuits on S5
and S6 and read them against the representation account.  Wu et al. (arXiv
2410.07476) verified approximate-equivariance explanations.  On that evidence
base the field can't tell "one account is wrong" apart from "the learned
algorithm varies across groups and training regimes."

Theory has moved since.  He et al. (arXiv 2606.02993) proved that every
neuron converges to a single irreducible representation, for an arbitrary
finite group — under a quadratic activation, two separate operand embeddings,
projected gradient flow on a linearised risk, and the complete multiplication
table.  Three silences in that result define my target.

First, the regime.  ReLU, a shared embedding, AdamW, and a partial table with
a held-out split — my setting throughout — is outside every theorem in the
paper, and its authors name the train/test-split (grokking) regime as open.
Second, occupancy.  The theorem says each neuron picks *an* irrep; which
irreps the ensemble occupies is proved only for abelian groups without
self-conjugate irreps, the nonabelian case is named open, and it's exactly
where the coset account predicts concentration on the irreps of an induced
representation while the Fourier account predicts breadth.  Third, the
sector.  The population-level theory excludes self-conjugate representations,
and He et al. chose their one nonabelian experiment (C7⋊C3, order 21)
specifically to avoid the quaternionic sector.

The opening I'm taking is the controlled-contrast gap.  Two non-isomorphic
groups can share a character table.  He et al.'s Theorem 4.3 depends on the
group only through its irreps, so it makes literally the same prediction for
both members of such a pair — and so does any account that reads the model
through the character table alone.  Nobody has trained
character-table-equivalent pairs as a neural falsification instrument, and
the coset account has never been tested against matched pairs.  Closing that
gap is the headline test, C1.

## Evidence grades

Every pre-registered test below carries an evidence grade.  Rung 0 is an association
between separately trained models — a difference between groups.  Rung 1 is a
quantity a probe can decode from a model's activations.  Rung 3 is a causal
effect inside one model: an ablation or a patch moves behaviour in the
predicted direction.  Rung 5 is a fitted functional form for the circuit that
survives an audit of faithfulness, completeness, and minimality.
Between-group contrasts cap at rung 0 by design — a theory that forbids a
difference dies at rung 0, and falsification doesn't need a high rung.  The
mechanism tests (C2, C3, C5) live inside single models and reach higher.

## The five pre-registered tests

### C1 — networks distinguish groups the Fourier account cannot (headline)

Eleven group pairs are, within each pair, identical in character table,
identical in Frobenius–Schur indicators, byte-identical in tensor-rank
bounds, and clean under a coset-template screen at full index: no core-free
subgroup of either member yields a distinguishing induced-representation
template.  So the full Fourier account, the tensor-rank account, and the
coset account all predict no within-pair difference.  The one axis left
varying by construction is power-map structure — element orders and exponent.

I measure the paired within-pair difference in epochs-to-grok (censored; see
the endpoints section) and in the irrep-occupancy vector, at 50 paired seeds
per group.  The pair set is criterion-defined and was closed before any run:
the five pre-registered panel pairs plus every screen-verified clean pair at
order ≤ 81, N = 11 —
`(27,3)/(27,4)`, `(54,10)/(54,11)`, `(64,74)/(64,80)`, `(64,228)/(64,229)`,
`(64,236)/(64,240)`, `(64,241)/(64,242)`, `(81,12)/(81,13)`,
`(125,3)/(125,4)`, `(243,56)/(243,57)`, `(243,65)/(243,66)`,
`(250,10)/(250,11)`.  Four of the order-64 pairs are the first clean
falsifiers known in the even-order world; every previously known 2-group pair
with equal character tables was a Frobenius–Schur flip.  The six pairs at
order ≤ 64 train unconditionally; the five heavier pairs are staged behind
the width-256 probe cells (the staging rule is in the groups section below).
The family definition and the rule that selects it are unchanged by the
staging.

The pair family comes from a re-runnable, hash-pinned screen
(`scripts/derive_falsifiers.py`, whose output is committed at
`results/falsifier_screen_results_full.json`) over the committed
group-properties dataset (`data/group_properties_full.jsonl`, generated by
`scripts/enumerate_groups.g` and `scripts/enumerate_groups.py`), which
reduced 7,274 same-fingerprint candidate pairs — through exact
character-table equality, Frobenius–Schur identity, and the full-index
coset-template criterion — to 370 clean pairs across 450 distinct groups in
orders 21–255.  The rule selecting eleven of them was fixed before any run.

A reproducible within-pair difference on any endpoint is evidence against the
full-Fourier account, attributable to the power-map axis.  Tight intervals
around zero on every trained pair are the Fourier account surviving its
sharpest available behavioural test in this regime, reported as interval
upper bounds — itself a result.  Two of the eleven, `(54,10)/(54,11)` and
`(81,12)/(81,13)`, are the `(27,3)/(27,4)` contrast tensored with a bystander
direct factor; they measure whether the base pair's verdict survives
direct-product embedding, are labelled as embedding probes, and are never
pooled as independent replications.

This is the first controlled contrast at fixed character table.  He et al.'s
theorem makes the same prediction for both members of every pair, and the
occupancy readout is measured against per-group analytic nulls in the
partial-table regime none of the paper's theorems cover.

### C2 — removing the coset route: the D32/QD32/Q32 case study

Groups `(32,18)` D32, `(32,19)` QD32, `(32,20)` Q32.  D32 and Q32 share a
character table and differ in Frobenius–Schur indicators (11/0/0 against
7/0/4), involution count (17 against 1), and coset-route existence: Q32 is
one of exactly three nonabelian groups in orders 21–255 with no faithful
action on fewer than |G| points, so every coset instrument is structurally
undefined on it.  The pair is the extremal removal of the coset route, and
the undefinedness is the design.  QD32 has a different character table from
the other two and serves as a third arm for comparison; it carries no
falsification weight.

The deliverable is a mechanistic account of the D32 circuit — a signed-cyclic
probe, a functional-form fit, and coset probe/ablation/patching, reaching
rung 5 if the fitted form survives audit — with Q32 as the structural
negative control: the signed-cyclic probe must fail on Q32, and a success
there disqualifies the dihedral result.  Occupancy is read from the 50 pooled
seeds, which puts D32 above its decision floor.  A structured failure of
per-neuron sparsity — dense coset-indicator neurons spread over an induced
representation's support — is the most direct coset signature and is a
reportable secondary outcome.  A small fully-connected replication of this
case study covers the architecture axis along which the literature splits.

The target result is a signed-cyclic or coset circuit on D32 with no analogue
on Q32, plus a characterisation of what Q32 learns instead.  If Q32 is
learned as easily and with an equivalent spectral solution, the coset,
extension-coordinate, and power-map accounts are all uninformative on the
most diagnostic group available.  Stander et al. found cosets on S5 and S6,
where coset routes are abundant; here the route is removed entirely, on a
pair the Fourier account can't tell apart, in the quaternionic sector He et
al. exclude and name as open.  The epochs-to-grok contrast between the three
groups is reported descriptively only.

### C3 — carry propagation is a real, causal cost (C128 vs C2⁷)

Groups `(128,1)` C128 and `(128,2328)` C2⁷ at fixed order, with `(127,1)`
C127 as an anchor: the solved modular-addition case, pooled with nothing.
The DFT account predicts comparable difficulty and no digit structure.  The
carry account predicts C2⁷ dramatically easier — pure XOR, linear over F₂ —
with a causal signature: decode the digits, ablate digit directions, fit the
mixed-radix rule.  In C2⁷ ablating bit *i* should cost bit *i* and nothing
else; in C128 ablating the low bit must also cost the high bits.  The
ablation-cost matrix is diagonal in one case and triangular in the other, and
that asymmetry is carry, made causal (rung 3, rising to rung 5 with the
functional-form fit and audit).

Modular addition is the field's solved case (Nanda 2022, Gromov 2023,
Mallinar et al. 2024), but always at one radix.  This contrast isolates carry
as the only varying quantity at fixed order and fixed
representation-theoretic complexity, and tests it causally.  Both groups are
abelian, so the transpose leak (see the endpoints section) is equal across
members: the within-pair contrast survives, and no absolute difficulty number
is reported.

### C4 — satellite tiers: falsification grade and direction tests

The eleven C1 pairs are tier 1 of a three-tier structure over
character-table-equal pairs.  The core samples tiers 2 and 3 at their best
exemplars; the full tiers are extension E1 in
[the extensions roadmap](extensions.md).

Tier 2 keeps Frobenius–Schur identity and relaxes the coset screen:
pure-Fourier still forbids a difference, and a difference's attribution is
shared between the power-map and coset axes.  The exemplar is
`(216,106)/(216,107)`, the pair with the strongest occupancy power in the
panel, where the shape of the occupancy template adjudicates the shared
attribution.

Tier 3 flips the Frobenius–Schur indicators and tests the direction the
strict character-decode account predicts: the quaternionic member pays a
real-dimension cost.  The exemplars are `(64,60)/(64,65)` (involutions 31
against 7) and `(104,4)/(104,6)` (C13:Q8 against D104, involutions 1 against
53).  On the order-104 pair the occupancy statistic sits at a structural
noise floor, so no occupancy estimate is reported there; its endpoints are
epochs-to-grok and the probes, and the signed-cyclic probe must fail on
C13:Q8.

### C5 — a non-split extension forces a learned cocycle

Groups `(48,29)` GL(2,3), a split extension, and `(48,28)` SL(2,3).C2,
non-split, with identical character degrees, class sizes, chief factors,
derived length, exponent, and centre — the only clean split/non-split
contrast in range that avoids being a 2-group artefact.  The
extension-coordinate account says a non-split multiplication rule requires an
irreducible correction term, a 2-cocycle, underivable from the group action.
So it predicts a decodable, ablatable cocycle-carrying structure in the
SL(2,3).C2 model with no analogue in GL(2,3), where a transversal exists on
which the cocycle is identically trivial.

The measurements live inside one model: ablation of the candidate cocycle
subspace against a norm- and dimension-matched random subspace; a
cocycle-value probe held out over quotient-pair cells (many input pairs share
one cell, so holding out raw input pairs would pseudoreplicate); and a
twisted-rule fit with and without the cocycle term, scored on held-out pairs.
Rungs 1–3, and rung 5 if the fitted twisted rule survives audit.  The
character tables of this pair differ, so C5 carries no Fourier-falsification
weight and is counted separately from C1.

The scoped risk, stated up front: the candidate subspace may simply be absent
or unfindable — the hunt is high-variance.  The pre-registered fallback is
the report-everything rule: a null probe or ablation outcome ships as a
measurement.  No prior work looks for explicitly learned extension data — a
2-cocycle probed as a represented object — in network weights.

## The groups

Seeds are paired: the same seed list on both members of every pair, with the
data split seed pinned across the whole campaign.  The campaign is the fixed
cell list in `configs/campaign/core.yaml`, and every count in this section is
counted from that file.  A single smoke cell — D32 at width 128, two seeds —
runs first and verifies the training path and the off-pod shipping end to end
before any core cell starts.

The core phase (width per cell as listed; 50 paired seeds per cell):

| group | name | role | width | why (one line) |
|---|---|---|---|---|
| (27,3) / (27,4) | 3^{1+2}₊ / 3^{1+2}₋ | C1 tier-1 pair, staging probe | 256 | cheapest clean pair; exponent 3 vs 9 at identical character table and FS data |
| (54,10) / (54,11) | — | C1 embedding probe, staging probe | 256 | the (27,3)/(27,4) contrast under a C2 bystander factor |
| (64,74) / (64,80) | — | C1 tier-1 pair | 128 | first clean falsifier family in the even-order world |
| (64,228) / (64,229) | — | C1 tier-1 pair | 128 | even-order clean falsifier, same family |
| (64,236) / (64,240) | — | C1 tier-1 pair | 128 | even-order clean falsifier, same family |
| (64,241) / (64,242) | — | C1 tier-1 pair | 128 | even-order clean falsifier, same family |
| (32,18) / (32,19) / (32,20) | D32 / QD32 / Q32 | C2 case study | 128 and 256 | the most diagnostic triple available; the twist rule is the only mover |
| (128,1) / (128,2328) | C128 / C2⁷ | C3 pair | 128 | radix 128 vs radix 2 at fixed order; carry is the only varying quantity |
| (127,1) | C127 | C3 anchor | 128 | the solved modular-addition case; calibrates every reading |
| (216,106) / (216,107) | — | C4 tier-2 exemplar | 128 | FS-identical falsification-grade pair with the panel's best occupancy power |
| (64,60) / (64,65) | — | C4 tier-3 exemplar | 128 | the star FS flip; involutions 31 vs 7 |
| (104,4) / (104,6) | C13:Q8 / D104 | C4 tier-3 exemplar | 128 | cheap real↔quaternionic direction test; no occupancy estimates |
| (48,29) / (48,28) | GL(2,3) / SL(2,3).C2 | C5 cocycle pair | 128 and 256 | the only clean non-2-group split/non-split contrast |

That is 31 cells over 26 distinct groups: 22 groups at width 128, five of
them — the C2 triple and the C5 pair — duplicated at width 256 as a
width-sensitivity readout, and the four order-27/54 cells at width 256 only.

The five remaining tier-1 pairs — `(81,12)/(81,13)`, `(125,3)/(125,4)`,
`(243,56)/(243,57)`, `(243,65)/(243,66)`, `(250,10)/(250,11)` — are staged,
and the staging is itself pre-registered.  They belong to the same
odd-p-group construction family as the order-27 pair and its order-54 lift,
and they are the most expensive cells in the study; a family that does not
grok returns only both-censored ties, at the largest orders in the design.
The four width-256 cells therefore double as staging probes: the staged pairs
re-enter the cell list only if those probes grok.  The censoring rule and the
ceiling do not change either way.

## Endpoints and reporting

Reporting is estimation-first.  Everything measured is reported as an
estimate with a confidence interval; there's no smallest-effect threshold, no
false-discovery gate, and no significance threshold anywhere in the design.
Descriptive p-values (paired sign-flip permutation) may accompany an interval
as supplements.  A run's record carries its full measurement vector with
provenance — run id, checkpoint, git revision, config hash, data hash — and
no verdict label, per the [methodology](methodology.md).  A narrow interval
near zero is stated as evidence against a large effect; a wide interval is
stated as uninformative; both statements come from the interval itself.
Everything that is run is reported: the pre-registered set filters what is
confirmatory, and a null outcome on any pre-registered test ships as a measurement.

1. Epochs-to-grok: the first epoch beginning a run of >0.99 accuracy on the
   transpose-unleaked held-out subset sustained for 5+ consecutive epochs.
   The censoring rule, pre-registered verbatim: hard 60,000-epoch ceiling
   (`configs/experiment/core.yaml`); a non-grokking seed is recorded as
   censored (>60k); no extensions, no resume.  Non-grokking definition: never
   >0.99 unleaked accuracy for 5+ consecutive epochs before 60k.  Censored
   seeds enter paired summaries as ">60k"; one-side-censored pairs keep their sign,
   both-censored pairs are ties, and the censoring fraction is itself a
   reported per-group measurement.
2. Accuracy means transpose-unleaked held-out accuracy, everywhere.  When a
   held-out pair's transpose sits in the training set and the two elements
   commute, the test label is already visible in training; the leaked share is
   `train_frac × k(G)/|G|` (the group's commuting probability times the train
   fraction), recorded per run as a realised covariate.  No cross-group number
   uses raw test accuracy.
3. The occupancy vector: an energy-weighted histogram over the irreps of `G`
   across the neuron population, measured against the group's own analytic
   null (`π⁰_j = block_rank_j/|G|`) and against the induced-representation
   template library.  Total-variation distances are quoted with the per-group
   noise floor, which scales as `1/√(neurons × seeds)`; pooling the 50 seeds
   lowers the floor about 7×, which is what makes the borderline tier-1
   families decidable.  A null-calibration gate runs every cross-condition
   statistic on random-init models of both members of each pair whose
   character tables differ; a statistic whose null differs across conditions
   does not ship.
4. Per-neuron concentration, as a secondary endpoint.  He et al.'s theorem
   does not apply in this regime, so per-neuron sparsity is a live
   measurement, and its structured failure (dense coset-indicator neurons) is
   the sharpest coset signature available.
5. Probe, ablation, and fit outputs, each reported with its rung, its null,
   and effect sizes in behavioural units — label flips, multiples of
   `1/n_test` — never bare percentages.

The paired-seed design: the same 50 seeds on both members of every pair, the
data split seed pinned across the sweep, unmatched seeds dropped to preserve
the pairing.  Aggregation happens within pair first; the unit of any
cross-group statement is the pair; confidence intervals come from a paired
bootstrap over seeds within a pair.  The tier-1 family is reported pair by
pair — one estimate and one interval per pair, with seeds never pooled across
pairs.  Nine of the eleven are replication across primes and orders; the
remaining two are the embedding probes described under C1.

## Run plan

`scripts/run_campaign.py` runs the cell list in `configs/campaign/core.yaml`
in phase order — the smoke cell, then core, then `extra` — cheapest first
within each phase (ascending group order, ties by ascending index and then
width).  The core phase is the 31 cells in the table: 1,550 runs at 50 paired
seeds per cell, on top of the two-seed smoke cell.

The `extra` phase runs with the campaign and is never pooled into the core
counts: three width-128 cells — `(81,7)` and the `(192,10)/(192,24)`
coset-test pair — supply for extensions E10 and E8.  The config file also
carries two bonus phases, 26 cells of extension supply for E6 and E2 (six and
twenty cells) gated behind `--include-bonus`; this study does not run them,
and they stay in `configs/campaign/core.yaml` as a follow-up worth returning
to.  The full cell list — core, `extra`, and bonus together — is 61 cells and
3,002 runs; every count in this section is counted from
`configs/campaign/core.yaml`.

Why 50 seeds, and pre-registered: within-pair paired tests at n = 50 reach
~81% power at a standardised effect of d = 0.5, whereas n = 25 suffices only
for large effects, and the headline test should be powered for more than the
flattering case.  The occupancy noise floor scales exactly as
`1/√(neurons × seeds)` (Dirichlet-derived, Monte-Carlo-verified), so 50
pooled seeds buy the ~7× floor reduction that turns every borderline tier-1
family decidable with margin.  Censoring at ≤15% non-grok is tolerable, and
one-side-censored ties fall disproportionately on the harder member of a
pair — the direction that flattens a difference result.  C5 keeps n = 50
because a mechanistic test needs seed replication: the reported quantity is
the fraction of seeds in which the cocycle structure is found, a proportion
that wants a tight exact-binomial interval, and at order 48 the runs are
small enough that thinning the seeds saves nothing worth having.

Analysis is separated from training.  Snapshots — evenly spaced, plus dense
powers of two early — are the data; no analysis runs during training, and
every transition criterion and every instrument runs post hoc from snapshots,
so a criterion can change without re-training.  The analysis scripts are
frozen and hashed before the first core run.

## What this study does not assert

- Results are conditional on one architecture: a one-layer transformer at
  width 128, with the named cells duplicated at width 256 and a
  fully-connected replication only for the C2 case study.  No
  cross-architecture universality is asserted.
- Cross-group findings are associations (rung 0) and are stated as such.  No
  intervention inside one model tests an assertion about the difference between
  two models; only the within-model mechanism tests of C2, C3, and C5 rise
  above rung 0.
- A clean-pair difference falsifies without identifying.  It is evidence
  against the full-Fourier account attributable to the power-map *axis*; the
  pairs still differ on element-order histogram, automorphism-group order,
  and subgroup counts, so no single residual invariant is identified as the
  cause.
- Eleven pairs are eleven pairs, and two of them are embedding probes of a
  third.  Class-level statements ("p-groups behave like this") are
  descriptive; the tier-1 family is replication across primes and orders plus
  two embedding probes, and no sampling assertion over a population of groups
  is made.
- No faithfulness bounds are attempted in the core.  Wu et al.'s bar —
  non-vacuous accuracy bounds derived from an explanation — is extension E11.
- No absolute-difficulty assertions are made on abelian groups; the transpose
  leak makes absolute numbers incomparable, and only within-pair contrasts
  are reported (C3).
- Grokking timing is reported separately from the inferred algorithm; timing
  is dynamics, and kernel machines grok too (Mallinar et al. 2024).
