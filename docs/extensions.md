# Extensions

[The core study](core-study.md) is deliberately small.  Each module below is
something it could grow into, and each states the claim it would license,
what it needs, and the measured trigger in the core's own results that would
prompt running it.  Modules inherit the core's design — 50 paired seeds, the
pinned split — unless stated, and they're ordered by promotion priority.  Ids
are stable, so a promoted module keeps its slot as a marker.

## E1 — the remaining character-table-pair tiers

Tier 2 in full: the 15 known pairs with identical character tables and
Frobenius–Schur data whose coset structure differs — pure-Fourier forbids a
difference on every one, with attribution shared between the power-map and
coset axes and adjudicated per pair by occupancy-template shape.  Tier 3 in
full: 47 Frobenius–Schur-flip pairs as direction tests of the strict
character-decode account, including a direction control whose coset degree is
reversed, separating involution-tracking from coset-tracking.  The tier-3
direction tests tolerate n = 25, since direction is the endpoint.  Worth
running when the core's tier 1 shows a difference (the tiers become the
attribution programme) or when it's null (the wider falsification surface
before conceding the account survives).

## E2 — dataset-wide clean-pair supply

The falsifier screen found 370 clean pairs across 450 distinct groups in
orders 21–255; the core's closed selection rule claims 11.  This module is
the remaining 359, 326 of them at order 128 in mutually clean cliques, some
running to nine mutually clean groups — a much denser replication surface
than the 11-pair core.  A cheapest-order-first partial sample scales down
freely.  Worth running when the tier-1 intervals show an effect worth
replicating at larger N, or come out too wide to report with confidence at
N = 11.  A first slice of it, 10 order-128 pairs picked deterministically by
file order, is scheduled to run alongside the core as an opportunistic
extension.

## E3 — ambiguous completion

Train on the 768 of 1,024 entries where the D32 and Q32 tables agree; which
completion the network picks on the held-out quadrant — dihedral or
quaternion — is a fact about the learner with no cross-run variance, and it
sits entirely outside He et al.'s theorems, which assume complete-table
training.  Disclosed confound: the disagreeing quadrant is coset-shaped, so
the readout is which twist, and can't by itself name the algorithm.  The
runs are tiny; the real work is the custom split builder and the completion
classifier.  Worth running when the core's C2 case study finds distinct
D32/Q32 solutions — it then isolates the single twist bit those solutions
differ by, the sharpest instrument in the programme.

## E4 — D/QD/Q replicates up the 2-power ladder

The C2 contrast as a scaling family: dihedral/semidihedral/quaternion triples
at orders 64 and 128, plus a modular fourth arm at each order (same cyclic
backbone, weak central twist).  Upgrades "D32 and Q32 differ" to "the
difference tracks the twist rule across three orders."  Needs no new
instruments.  Worth running when C2 finds any D32/Q32 asymmetry; replication
across orders is the first demand a reviewer will make of that result.

## E5 — reserved

The GL(2,3) / SL(2,3).C2 cocycle experiment is core claim C5, so this slot is
empty; it's kept so module ids stay stable.  Its observational surround — ten
substantive non-prime-power non-split groups — sits in E14's panel remainder.

## E6 — the factorisation bundle

Separates direct-product factorisation from the central-product shared-phase
account: a per-factor readout, a factor-aligned double dissociation, and a
controlled-leak signature, on a verified companion bundle of order-64 pairs
with an order-192 corroborator, plus the Q8∘Q8 / Q8×Q8 pair as the
central-product arm.  Needs two new instruments (factor decomposition with
per-factor readout; factor-aligned ablation and patching).  Worth running if
a novel axis is wanted over depth, or if core occupancy shows factor-shaped
structure anywhere.  Three of the module's pairs are scheduled to run
alongside the core as an opportunistic extension.

## E7 — digit count at fixed everything

Tests whether the number of digits d(G), independent of radix, shows up as
coordinate directions in the embedding: the clean indecomposable pair
`(32,6)/(32,43)` with the odd-prime replicate `(243,22)/(243,60)` as a
discordant-confound design — a same-direction effect across
oppositely-confounded pairs rules the confound out.  Instruments exist from
C3.  Worth running when C3 finds digit or carry structure; run the order-32
pair first and gate the 243 replicate on it.

## E8 — the clean coset test

`(192,10)/(192,24)`: a genuine coset-account test with a 4× coset-index gap
and block count, permutation-degree profile, automorphism-group order, and
socle all matched, adjudicated on template shape.  Disclosed residual: the
power-map axis stays live at headline magnitude, so a positive result
implicates the union of the coset and power-map accounts.  Uses the core's
coset arm plus templates at index 48.  Worth running when the core's
occupancy results show any template-shaped concentration.

## E9 — the class-2 bilinear family and the Arf pair

The class-3 residual left by a fitted bilinear rule, carried on 31 clean
pairs; and `(32,49)/(32,50)`, the second character-table-equivalent fork with
occupancy power, testing quadratic/Arf structure beyond the character table.
Needs a bilinear/quadratic probe-ablate-fit instrument and a
coordinate-bijection derivation that is pure group theory, with no training
runs.  Worth running when the extraspecial tier-1 pairs show structure that
wants a bilinear explanation.

## E10 — wreath/routing and quotient/CRT

Two attribution-gated modules.  Routing: lane-swap patching is the one
instrument that distinguishes data movement from arithmetic; `(32,11)` plus
controls and the odd-prime replicate `(81,7)`; gated on E12.  Quotient/CRT:
quotient-aligned occupancy grouped by kernel class on 67 disjoint
non-nilpotent pairs, carrying the suite's most dangerous normaliser
(deviation-from-own-null mandatory).  Worth running, respectively, when a
core circuit looks permutation-like, and only alongside a full-panel campaign
(E14).

## E11 — faithfulness bounds, width sensitivity, architecture breadth

Four sensitivity axes: a non-vacuous accuracy bound derived from the C2/C3/C5
mechanistic accounts, meeting the bar Wu et al. set; width as a measured
sensitivity axis; architecture replication beyond the C2 case study; and a
partial-table-fraction sweep that makes the memorisation floor quantitative.
Worth running before any external submission — the single-width,
single-architecture caveat is the reviewer objection most worth pre-empting.

## E12 — the socle-calibration subtraction

Prices the confound that shadows every representation-route-availability
claim: groups lacking a faithful irrep systematically carry more minimal
normal subgroups, so this arm measures that effect with the representation
route held off (33 disjoint pairs, 66 groups) and subtracts it.  The core
contains no representation-route-availability claim, so there's nothing yet
to price; the moment any promoted module makes one, this module becomes
mandatory.

## E13 — the "neither route" arm

A parameter-free prediction: occupancy support width at least the minimal
faithful-representation block count, on a pre-registered stratified sample
(n = 30 main + 5 order-128 companion) of the 1,162 groups with neither a
faithful irrep nor a core-free subgroup.  If networks learn these groups,
something other than the representation and coset routes does the work.
n = 25 seeds suffice: the endpoint is a per-group hit/miss count with an
exact-binomial interval.  Worth running once the core's occupancy instruments
are validated end to end; the natural second paper.

## E14 — the full panel

365 groups and 18,250 runs across confirmatory, diagnostic, statistical, and
observational tiers, with class-level typology computed downstream from the
per-run measurement vectors, plus the remaining named experiments and
instrument builds.  The vmapped batch-compaction machinery it needs is already
built.  Worth running when the core lands and the panel is wanted at full
scale.
