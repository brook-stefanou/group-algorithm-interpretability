# Research log

## Convention

This is an append-only record of substantive project decisions, results, and
corrections. A dated bullet may carry an invisible `decision` or `learning` tag,
which marks it for exact copy into decision and learning views that CI renders as
an artifact. Corrections are appended as new entries, and past claims are left as
they were written.

## Jul 13

<!-- learning: matched-pair-panel-pivot -->
- One matched pair is too narrow for the question here.
  - Still useful for building the representation and coset tools.
- Next panel should cover groups with different structures.

<!-- decision: constrained-panel-coverage -->
- Will choose the study panel for structural coverage, not by picking groups from clusters.
  - The main panel is 150 groups from orders 21–255, leaving out 128, with another 50 from order 128.
  - Need to settle the invariant list, constraints, quotas, and weights before choosing group IDs.

## Jul 14

<!-- decision: route-availability-selection -->
- Will choose groups by which algorithmic routes are available and what they cost.
  - Not by an unsupervised diversity score.
  - Routes include Fourier, faithful representation, coset action, direct product,
    extension coordinates, class-2 bilinear correction, power maps, affine,
    dihedral twist, wreath routing, central product, and carry arithmetic.
  - Extends the Jul 13 rejection of clustering. A similarity score cannot say that
    a route is unavailable.

<!-- decision: preregistered-falsification-design -->
- Will pre-register theories, experiments, and predictions before training.
  - A contrast earns a slot only where two theories predict different outcomes.
  - Theories that forbid a difference carry the weight. Any difference kills them.
  - Record "no clear prior" where the mechanism gives no direction. Do not invent one.

<!-- decision: identification-screen-before-selection -->
- Will test each axis for identifiability before spending groups on it.
  - Measure collinearity between the axis and its confounds in the real candidate set.
  - An axis whose confound has fixed sign and no spread is not identified. No sample
    size repairs it.
  - Cut such an axis, or add a calibration arm that prices the confound with the
    axis held off.

<!-- decision: staged-priority-ladder -->
- Correction to the Jul 13 panel sizes. The 150-group main panel and the 50-group
  order-128 extension are withdrawn.
  - Will select in stages by evidential strength. Decisive, then diagnostic, then
    statistical, then observational.
  - Running extra arms is safe. Choosing what to run or report based on results is not.
  - Everything run will be reported.

<!-- learning: independent-convergence-with-literature -->
- The methodology above was settled before the working literature review was read.
  It matches the review's recommended next step.

## Jul 15

<!-- decision: criterion-defined-clean-pair-family -->
- Group selection is settled: 39 groups carrying five pre-registered claims.
  - The headline pairs are criterion-defined, not hand-picked: non-isomorphic
    groups with the same character table and Frobenius–Schur indicators, and no
    coset structure that separates them. A committed script screens all 6,958
    groups for the criterion.
  - The screen finds 370 such pairs; the core runs eleven, and the rest are held
    as extension supply.

- Next: pilot run to fix the model width, then the campaign.
