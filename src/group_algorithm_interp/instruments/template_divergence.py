"""Template-divergence instrument: GCR (sparse-irrep) vs coset, screened.

The character-table-equivalent (CTE) contrast pairs are matched so that the
Fourier, tensor-rank and coset accounts all predict the same thing on both
members of a pair -- that is the whole point of the pair design, and it means
the pairs can never separate the coset account from the representation
account (here called GCR, for the sparse-irrep prediction the representation
account makes once it is refined past "the full Fourier account predicts
breadth"). Separating GCR from coset needs a single group compared against
*its own* templates, not a pair.

This module builds that single-group comparison from three templates over the
group's real isotypic blocks, all in the same units :mod:`occupancy` already
uses (an energy share per block, summing to 1):

* the analytic null, ``pi0_j = block_rank_j / |G|`` (:func:`occupancy.analytic_null`);
* the coset account's ``Ind_H^G 1`` energy template for the minimal-index
  nontrivial core-free subgroup (:mod:`templates`, I-12/I-13);
* the GCR account's sparse-irrep template (:func:`gcr_sparse_template`,
  built here): the representation account's prediction is that the model
  needs only a *minimal separating set* of nontrivial irreps -- the smallest
  set whose direct sum is a faithful representation of ``G`` -- not the whole
  regular representation. No canonical implementation of this exists
  elsewhere in the codebase yet (see the internal TODO's still-open "GCR
  character-readout functional form" item); the operationalisation here is a
  deliberately simple one, documented so it can be superseded without
  surprise: blocks are added greedily in ascending irrep degree (cheapest
  first) until the intersection of their kernels is trivial, and the
  template distributes energy across the selected blocks in proportion to
  their ``block_rank``, exactly as the null and coset templates do. A block's
  kernel -- the elements it sends to the identity matrix -- is read off the
  character alone: for a unitary representation ``rho`` of degree ``d``,
  ``chi(g) == d`` exactly iff ``rho(g)`` is the identity matrix (the trace of
  a unitary matrix has magnitude ``d``, with equality iff every eigenvalue is
  1), so no concrete irrep matrices are needed.

**The degeneracy hazard is binding and must be screened before any of this is
meaningful.** The coset account's target is the isotypic support of
``Ind_H^G 1`` for the minimal-index nontrivial core-free ``H`` -- but nothing
guarantees that support is small. On D32 the minimal-index core-free subgroup
has index 16, and its ``Ind_H^G 1`` support spans 30 of D32's 32 isotypic
dimensions (``templates.py``, ``tests/test_coset.py``): ablating that target
would remove almost the entire representation, so a model's occupancy is
*forced* close to it regardless of which account the model actually
implements, and "closest to coset" would be a foregone, uninformative
conclusion. :func:`degeneracy_screen` runs this check first and reports a
``DEFINED``/``UNDEFINED`` verdict; :func:`compare_to_templates` refuses to
render a coset-vs-GCR verdict when the screen is ``UNDEFINED``, returning the
string :data:`UNDEFINED` for the coset-side readouts exactly as the existing
coset instruments do for their own degenerate cases (``coset.py``,
``templates.py``).

The threshold is :data:`MAX_COSET_SUPPORT_FRACTION` (default ``0.5``): the
coset target's isotypic support must span *less than half* of the group's
dimensions to be considered non-degenerate. This is deliberately a strict cut
rather than a loose one -- of the panel's five locally-exported-subgroup
artifacts, D32 and QD32 sit at 30/32 (0.9375) and GL(2,3) at 26/48 (0.5417):
all comfortably degenerate under this threshold, and none straddle it, so the
exact value is not load-bearing for the panel groups on disk. It is chosen so
that "at least half the group's representation survives ablating the
complement" is the working definition of "ablating the coset target does not
remove almost everything." ``max_support_fraction`` is a named, overridable
parameter throughout, never a bare literal.

Every occupancy vector this module compares against a template must already
be aligned to the group's own isotypic-block order (:func:`occupancy.
population_occupancy`'s output shape, ``[n_blocks]``) -- this module does not
read result JSON files or recompute occupancy itself, only compares vectors
callers already hold in that form.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..groups.group import FiniteGroup
from .occupancy import analytic_null, total_variation, trivial_block_index
from .templates import template_library

_TOL = 1e-6

DEFINED = "DEFINED"
UNDEFINED = "UNDEFINED"

#: The coset target's isotypic support must span a fraction of the group's
#: dimensions strictly below this to be considered non-degenerate (see the
#: module docstring for the D32/QD32/GL(2,3) evidence behind the choice).
MAX_COSET_SUPPORT_FRACTION = 0.5


# ---------------------------------------------------------------------------
# Degeneracy screen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DegeneracyScreen:
    """The coset-target degeneracy verdict for one group, plus the evidence
    behind it. ``verdict`` is :data:`DEFINED` only when a nontrivial core-free
    subgroup exists (I-12) *and* its ``Ind_H^G 1`` isotypic support spans less
    than ``max_support_fraction`` of the group -- otherwise :data:`UNDEFINED`,
    for one of three mutually exclusive reasons distinguishable via the other
    fields: no subgroup data was exported (``artifact_incomplete``), only the
    trivial subgroup is core-free (``coset_defined`` False -- abelian and
    generalised-quaternion groups), or a nontrivial core-free subgroup exists
    but its support is degenerate (``coset_defined`` True, ``verdict``
    UNDEFINED, ``support_fraction`` at or above the threshold -- the D32
    case)."""

    order: int
    n_subgroups_examined: int
    artifact_incomplete: bool
    coset_defined: bool
    min_corefree_index: int
    max_support_fraction: float
    verdict: str
    reason: str
    subgroup_index: int | None
    subgroup_order: int | None
    coset_index: int | None
    coset_template: np.ndarray | None
    support_rank: int | None
    support_fraction: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "n_subgroups_examined": self.n_subgroups_examined,
            "artifact_incomplete": self.artifact_incomplete,
            "coset_defined": self.coset_defined,
            "min_corefree_index": self.min_corefree_index,
            "max_support_fraction": self.max_support_fraction,
            "verdict": self.verdict,
            "reason": self.reason,
            "subgroup_index": self.subgroup_index,
            "subgroup_order": self.subgroup_order,
            "coset_index": self.coset_index,
            "coset_template": (
                self.coset_template.tolist() if self.coset_template is not None else None
            ),
            "support_rank": self.support_rank,
            "support_fraction": self.support_fraction,
        }


def degeneracy_screen(
    group: FiniteGroup,
    *,
    max_support_fraction: float = MAX_COSET_SUPPORT_FRACTION,
) -> DegeneracyScreen:
    """I-screen: the coset-target degeneracy verdict for ``group`` (see the
    module docstring). Reuses :func:`templates.template_library` for the
    core-free enumeration (I-12) and the ``Ind_H^G 1`` energy template
    (I-13) rather than re-deriving either."""
    if not 0.0 < max_support_fraction <= 1.0:
        raise ValueError(f"max_support_fraction must be in (0, 1], got {max_support_fraction}")
    library = template_library(group)
    common: dict[str, Any] = {
        "order": group.order,
        "n_subgroups_examined": library.n_subgroups_examined,
        "artifact_incomplete": library.artifact_incomplete,
        "coset_defined": library.coset_defined,
        "min_corefree_index": library.min_corefree_index,
        "max_support_fraction": max_support_fraction,
    }
    if library.artifact_incomplete:
        return DegeneracyScreen(
            **common,
            verdict=UNDEFINED,
            reason=(
                "artifact incomplete: no subgroup data was exported for this "
                "group, so the coset target -- and therefore the degeneracy "
                "screen -- cannot be evaluated from this artifact"
            ),
            subgroup_index=None,
            subgroup_order=None,
            coset_index=None,
            coset_template=None,
            support_rank=None,
            support_fraction=None,
        )
    if not library.coset_defined:
        return DegeneracyScreen(
            **common,
            verdict=UNDEFINED,
            reason=(
                "no nontrivial core-free subgroup: Ind_1^G 1 is the regular "
                "representation, so the coset account has no distinct "
                "prediction to screen (abelian and generalised-quaternion "
                "groups land here by theorem)"
            ),
            subgroup_index=None,
            subgroup_order=None,
            coset_index=None,
            coset_template=None,
            support_rank=None,
            support_fraction=None,
        )
    min_index = library.min_corefree_index
    at_min = [entry for entry in library.entries if entry.coset_index == min_index]
    entry = min(at_min, key=lambda e: e.subgroup_index)
    support_rank = sum(
        group.isotypic_blocks[j].block_rank
        for j in range(len(entry.template))
        if entry.template[j] > _TOL
    )
    support_fraction = support_rank / group.order
    if support_fraction >= max_support_fraction:
        verdict = UNDEFINED
        reason = (
            f"degenerate: the minimal-index core-free subgroup's Ind_H^G 1 "
            f"support spans {support_rank}/{group.order} isotypic dimensions "
            f"({support_fraction:.4f} >= {max_support_fraction} threshold); "
            "ablating the coset target would remove almost the whole "
            "representation regardless of which account the model "
            "implements, so a coset-vs-GCR comparison here would be vacuous "
            "(the D32 rank-30/32 case this screen exists to catch)"
        )
    else:
        verdict = DEFINED
        reason = (
            f"the minimal-index core-free subgroup's Ind_H^G 1 support spans "
            f"{support_rank}/{group.order} isotypic dimensions "
            f"({support_fraction:.4f} < {max_support_fraction} threshold): "
            "low enough that ablating it leaves most of the representation "
            "intact, so a coset-vs-GCR comparison is meaningfully defined"
        )
    return DegeneracyScreen(
        **common,
        verdict=verdict,
        reason=reason,
        subgroup_index=entry.subgroup_index,
        subgroup_order=entry.subgroup_order,
        coset_index=entry.coset_index,
        coset_template=entry.template,
        support_rank=support_rank,
        support_fraction=support_fraction,
    )


# ---------------------------------------------------------------------------
# GCR sparse-irrep template
# ---------------------------------------------------------------------------


def block_kernel(group: FiniteGroup, block_index: int) -> np.ndarray:
    """The isotypic block's kernel: the element indices every irrep merged
    into the block sends to the identity matrix. For a unitary irrep ``rho``
    of degree ``d``, ``chi(g) == d`` exactly iff ``rho(g)`` is the identity
    (the character is a sum of ``d`` unit-modulus eigenvalues, which can only
    reach magnitude ``d`` -- let alone the real value ``d`` -- when every
    eigenvalue is 1), so the kernel is read off the character alone.
    Complex-conjugate irreps merged into one real block share a kernel
    exactly (``rho`` and its conjugate vanish on the same elements); this is
    asserted, not assumed."""
    block = group.isotypic_blocks[block_index]
    representative = group.irreps[block.irrep_indices[0]]
    kernel_mask = np.abs(representative.character - representative.dimension) < _TOL
    for other_index in block.irrep_indices[1:]:
        other = group.irreps[other_index]
        other_mask = np.abs(other.character - other.dimension) < _TOL
        if not np.array_equal(kernel_mask, other_mask):
            raise ValueError(
                f"block {block_index}'s merged irreps {block.irrep_indices} disagree on "
                "their kernel; this is not a valid real isotypic merge"
            )
    return np.flatnonzero(kernel_mask).astype(np.int64)


def minimal_separating_blocks(group: FiniteGroup) -> tuple[int, ...]:
    """The GCR account's minimal separating set of isotypic blocks: nontrivial
    blocks added greedily in ascending irrep degree (cheapest first, ties
    broken by block index for determinism) until the intersection of their
    kernels is trivial -- i.e. their direct sum is a faithful representation
    of ``G``. Always terminates: the direct sum of every nontrivial irrep is
    faithful (an element fixed by every nontrivial irrep is fixed by the
    regular representation, hence is the identity), so the trivial block is
    excluded (its kernel is the whole group -- it carries no discriminating
    information) but the full remaining set is always enough in the worst
    case. This is a greedy, cheapest-first heuristic, not a guaranteed
    globally-minimum-cardinality search."""
    trivial = trivial_block_index(group)
    candidates = sorted(
        (j for j in range(len(group.isotypic_blocks)) if j != trivial),
        key=lambda j: (group.isotypic_blocks[j].irrep_degree, j),
    )
    intersection = set(range(group.order))
    selected: list[int] = []
    for j in candidates:
        if len(intersection) <= 1:
            break
        intersection &= {int(x) for x in block_kernel(group, j).tolist()}
        selected.append(j)
    if len(intersection) != 1:
        raise ValueError(
            "no faithful subset of nontrivial isotypic blocks was found; the "
            "direct sum of every nontrivial irrep is faithful by the regular-"
            "representation theorem, so this indicates corrupted character data"
        )
    return tuple(selected)


def gcr_sparse_template(group: FiniteGroup) -> np.ndarray:
    """The GCR account's sparse-irrep energy template: zero outside
    :func:`minimal_separating_blocks`, and within it proportional to each
    selected block's ``block_rank``, exactly as the null and coset templates
    distribute energy. Sums to 1."""
    selected = minimal_separating_blocks(group)
    total_rank = sum(group.isotypic_blocks[j].block_rank for j in selected)
    if total_rank <= 0:
        raise ValueError("selected GCR blocks have zero total rank; template is undefined")
    template = np.zeros(len(group.isotypic_blocks), dtype=np.float64)
    for j in selected:
        template[j] = group.isotypic_blocks[j].block_rank / total_rank
    return template


# ---------------------------------------------------------------------------
# Minimal faithful set: cost-based minimality of a block set
#
# The GCR account's claim is that a model needs only a *faithful* set of
# isotypic blocks -- a set whose direct sum distinguishes every element from
# the identity (:func:`block_kernel`'s intersection is trivial). But minimal
# faithful sets are generally NOT UNIQUE: S4 (see the fixture below) has a
# faithful singleton (the degree-3 "three" block alone) and a faithful triple
# ("sign", "two", "three") that does not contain it, and other groups can
# have several faithful sets of the same minimum size. So "is the model's
# used set minimal" cannot be answered by comparing it against one canonical
# minimal set -- it needs a COST that stays well-defined regardless of which
# minimal set happens to be cheapest, and a search that reports how many
# minimal sets tie so a caller can see the non-uniqueness directly rather
# than being handed a single, arbitrarily-chosen "the" minimal set.
#
# Two costs are supported (:func:`minimum_faithful_sets`'s ``cost``
# parameter):
#
# * ``"cardinality"`` (the default): the number of blocks -- "how many
#   irreps does the model need to track".
# * ``"total_block_rank"``: the summed ``block_rank`` (== sum of d^2 over the
#   blocks' irreps) -- the actual dimension-count cost. A single high-degree
#   block can be cardinality-cheap but rank-expensive, so this cost can
#   disagree with cardinality about which set is "smaller": whether a model
#   skipping a high-degree block is "free" is exactly the question the
#   reference table this instrument produces is built to answer.
#
# Both searches share one enumeration strategy: increasing cardinality
# k = 1, 2, 3, ... over combinations of NONTRIVIAL blocks (the trivial block
# carries no discriminating information -- its kernel is the whole group),
# stopping at the first k with any faithful set. Every panel group measured
# lands at a small k in practice (see the reference table), so this stays
# cheap; :data:`MAX_FAITHFUL_SET_SEARCH_CARDINALITY` guards against a
# combinatorial blow-up on a hypothetical group with an unexpectedly large
# minimal faithful set by capping the search and falling back to the full
# nontrivial-block set, which is always faithful by the same
# regular-representation theorem :func:`minimal_separating_blocks` relies on.
# ``cost="total_block_rank"`` additionally widens the search by one level
# (k+1) after the first hit, since a slightly larger set can have a smaller
# total dimension count than every set at the minimum cardinality.
#
# That cardinality cap alone is NOT enough of a guard: ``C(n, k)`` blows up
# from a large *candidate count* ``n`` even at a small, capped ``k``. A group
# with many one-dimensional irreps and no single faithful one -- the
# elementary-abelian (C2)^7 = SmallGroup(128, 2328), 127 nontrivial candidate
# blocks and no faithful singleton -- has ``C(127, 6) ~ 4.8e9``, which hangs
# the exact search for the better part of an hour even though ``k`` never
# exceeds :data:`MAX_FAITHFUL_SET_SEARCH_CARDINALITY`.
#
# A flat cap on the candidate count ``n`` is the obvious guard but is too
# blunt: several real panel groups have a large ``n`` (SmallGroup(127, 1),
# 63 nontrivial blocks; SmallGroup(128, 1), 64) yet the exact search is
# already cheap for them, because they have a faithful block at cardinality
# 1 and the loop stops there -- ``n`` alone does not predict cost, whether a
# small-``k`` faithful set exists does. So the guard instead bounds the
# actual per-level WORK: before enumerating cardinality-``k`` combinations,
# :data:`MAX_FAITHFUL_SET_SEARCH_COMBOS` caps ``C(n, k)``. A group whose
# search would have stopped early (a faithful set found at some small ``k``)
# never reaches an over-budget level regardless of its candidate count, so
# its result is completely unaffected by this guard; only a group with NO
# faithful set below budget -- the (C2)^7 pathology -- falls through to
# :func:`greedy_faithful_cover`, a set-cover-style greedy search over irrep
# KERNELS (not subsets) that is ``O(n^2)`` and always terminates. The greedy
# result is reported with ``certified=False``: it is a valid faithful set and
# an upper bound on the true minimum, never a false minimum, but is not
# guaranteed optimal.
# ---------------------------------------------------------------------------

#: The combinatorial search cap for :func:`minimum_faithful_sets`: increasing
#: cardinality k is searched only up to this bound before giving up and
#: falling back to the full nontrivial-block set. Every panel group measured
#: lands at k <= 3 (see the reference table this instrument produces), so
#: this is generous headroom, not a tight bound -- it exists purely to stop
#: ``C(n, k)`` from blowing up on a group with many conjugacy classes and an
#: unexpectedly large minimal faithful set. Named and overridable via
#: ``max_cardinality``, never a bare literal.
MAX_FAITHFUL_SET_SEARCH_CARDINALITY = 6

#: The per-level combinatorial-search budget for :func:`minimum_faithful_sets`
#: and :func:`recruited_dimension.minimal_faithful_real_dimension`: before
#: enumerating cardinality-``k`` combinations of the ``n`` nontrivial
#: candidates, ``C(n, k)`` is checked against this budget. Exceeding it at
#: any level stops the exact search there (as if the cardinality cap had been
#: reached without a faithful set) and falls back to a greedy kernel-
#: intersection cover, reported uncertified -- see the module-section
#: docstring above for why this is a per-level work budget rather than a flat
#: candidate-count cap. ``C(30, 6) = 593775`` and ``C(40, 6) ~ 3.8e6``, so
#: this comfortably covers every real panel group's candidate count at the
#: default cardinality cap while stopping well short of the ``C(127, 6) ~
#: 4.8e9`` (C2)^7 case. Named and overridable via ``max_combos_per_level``,
#: never a bare literal.
MAX_FAITHFUL_SET_SEARCH_COMBOS = 1_000_000


def is_faithful(group: FiniteGroup, block_indices: Sequence[int]) -> bool:
    """Whether the direct sum of ``block_indices`` is a faithful
    representation of ``group``: the intersection of their kernels
    (:func:`block_kernel`) is exactly the identity. Every kernel already
    contains the identity, so the intersection can only ever shrink to
    exactly ``{identity}`` or stay larger -- never to empty.

    An empty ``block_indices`` is faithful only for the trivial group (order
    1): the intersection of an empty family of subsets of the group is
    conventionally the whole group, which is trivial precisely when the
    group itself is."""
    intersection = set(range(group.order))
    for block_index in block_indices:
        intersection &= {int(x) for x in block_kernel(group, block_index).tolist()}
        if len(intersection) <= 1:
            break
    return len(intersection) == 1


def greedy_faithful_cover(
    group: FiniteGroup,
    candidates: Sequence[int],
    *,
    tie_break_cost: Callable[[int], int] | None = None,
) -> tuple[int, ...]:
    """A greedy set-cover search for a faithful block set: repeatedly select
    the candidate block whose kernel most shrinks the current common-kernel
    intersection (ties broken by ``tie_break_cost`` ascending when given,
    then by block index, for determinism), until the intersection is exactly
    the identity.

    Unlike :func:`minimum_faithful_sets`'s exact search, this never
    enumerates block SUBSETS -- only individual block KERNELS -- so its cost
    is ``O(n^2)`` in the candidate count regardless of how large the minimal
    faithful set turns out to be, and it always terminates in at most
    ``len(candidates)`` steps (the full candidate set is faithful by the
    regular-representation theorem, see :func:`minimal_separating_blocks`).
    It is a fast, always-terminating UPPER BOUND on the minimum faithful set,
    not a certified minimum -- callers that need certification must use the
    exact search instead (only tractable for a small candidate count)."""
    kernels = {j: frozenset(int(x) for x in block_kernel(group, j).tolist()) for j in candidates}
    intersection: frozenset[int] = frozenset(range(group.order))
    remaining = set(candidates)
    selected: list[int] = []
    while len(intersection) > 1 and remaining:

        def _key(j: int, _intersection: frozenset[int] = intersection) -> tuple[int, int, int]:
            reduction = len(_intersection) - len(_intersection & kernels[j])
            tiebreak = tie_break_cost(j) if tie_break_cost is not None else 0
            return (-reduction, tiebreak, j)

        best = min(remaining, key=_key)
        intersection &= kernels[best]
        selected.append(best)
        remaining.discard(best)
    if len(intersection) != 1:
        raise ValueError(
            "greedy faithful-set cover exhausted every candidate without reaching a "
            "trivial kernel intersection; this violates the regular-representation "
            "theorem and indicates corrupted character data"
        )
    return tuple(selected)


def block_set_cost(group: FiniteGroup, block_indices: Sequence[int]) -> dict[str, int]:
    """The cost of a block set on the three bases this module reports:
    ``cardinality`` (block count), ``total_block_rank`` (summed dimension,
    i.e. summed ``d**2`` over the blocks' irreps -- :class:`IsotypicBlock`'s
    own ``block_rank`` convention), and ``max_degree`` (the largest irrep
    degree present, ``0`` for an empty set). Duplicate indices are collapsed
    before costing."""
    unique = sorted({int(b) for b in block_indices})
    blocks = group.isotypic_blocks
    return {
        "cardinality": len(unique),
        "total_block_rank": sum(blocks[j].block_rank for j in unique),
        "max_degree": max((blocks[j].irrep_degree for j in unique), default=0),
    }


_FAITHFUL_SET_COSTS = ("cardinality", "total_block_rank")


def _greedy_faithful_fallback(
    group: FiniteGroup, candidates: Sequence[int], *, cost: str
) -> dict[str, Any]:
    """The uncertified :func:`minimum_faithful_sets` result when the exact
    search is abandoned over budget: a fast, always-terminating greedy cover
    (see :func:`greedy_faithful_cover`), reported as an upper bound."""
    blocks = group.isotypic_blocks
    tie_break_cost = (lambda j: blocks[j].block_rank) if cost == "total_block_rank" else None
    greedy = greedy_faithful_cover(group, candidates, tie_break_cost=tie_break_cost)
    greedy_cost = block_set_cost(group, greedy)
    return {
        "min_cardinality": greedy_cost["cardinality"],
        "n_sets_at_min_cardinality": 1,
        "min_total_block_rank": greedy_cost["total_block_rank"],
        "example_min_set": greedy,
        "cost": cost,
        "fallback_used": False,
        "certified": False,
    }


def minimum_faithful_sets(
    group: FiniteGroup,
    *,
    cost: str = "cardinality",
    max_cardinality: int = MAX_FAITHFUL_SET_SEARCH_CARDINALITY,
    max_combos_per_level: int = MAX_FAITHFUL_SET_SEARCH_COMBOS,
) -> dict[str, Any]:
    """The minimum-cost faithful set(s) among NONTRIVIAL blocks (see the
    module-section docstring above for the two supported ``cost`` options,
    the shared cardinality-increasing search strategy, and why the
    combinatorial guard is a per-level work budget rather than a flat
    candidate-count cap).

    Returns a dict: ``min_cardinality`` (the smallest block count any
    faithful set achieves), ``n_sets_at_min_cardinality`` (how many distinct
    faithful sets of that size exist -- the non-uniqueness count),
    ``min_total_block_rank`` (the smallest ``total_block_rank`` among the
    faithful sets considered: those at ``min_cardinality``, plus one extra
    level ``min_cardinality + 1`` when ``cost="total_block_rank"``),
    ``example_min_set`` (one faithful set achieving both ``min_cardinality``
    and, among ties, the smallest ``total_block_rank`` -- a representative,
    not "the" minimal set, since minimal faithful sets are generally not
    unique), ``cost`` (echoed back), ``fallback_used`` (``True`` when the
    cardinality search hit ``max_cardinality`` without finding a faithful
    set and fell back to the full nontrivial-block set -- the pathological
    case the regular-representation theorem guarantees cannot happen for any
    real group, but which is guarded against rather than assumed away), and
    ``certified`` (``False`` when a search level's ``C(n, k)`` exceeded
    ``max_combos_per_level`` before a faithful set was found, so the exact
    search was abandoned in favour of :func:`greedy_faithful_cover` --
    every field above is then still a valid faithful set and cost, but only
    an UPPER BOUND on the true minimum, never a certified one; ``True``
    whenever the exact search ran to a conclusion, including the
    ``fallback_used`` case, since that fallback set is exactly the full
    candidate set, not a heuristic guess)."""
    if cost not in _FAITHFUL_SET_COSTS:
        raise ValueError(f"cost must be one of {_FAITHFUL_SET_COSTS}, got {cost!r}")
    trivial = trivial_block_index(group)
    candidates = sorted(j for j in range(len(group.isotypic_blocks)) if j != trivial)
    n = len(candidates)
    search_limit = min(n, max_cardinality)

    # Kernels computed once, outside the combinatorial loop: `is_faithful`
    # recomputes `block_kernel` (a character-array scan) on every call, which
    # is fine for a one-off check but would redundantly redo the same O(n)
    # work millions of times over the search's inner loop.
    kernels = {j: frozenset(int(x) for x in block_kernel(group, j).tolist()) for j in candidates}
    full_intersection = frozenset(range(group.order))

    def combo_is_faithful(combo: tuple[int, ...]) -> bool:
        intersection = full_intersection
        for j in combo:
            intersection &= kernels[j]
            if len(intersection) <= 1:
                break
        return len(intersection) == 1

    def faithful_combos(k: int) -> list[tuple[int, ...]]:
        return [
            combo for combo in itertools.combinations(candidates, k) if combo_is_faithful(combo)
        ]

    min_cardinality: int | None = None
    combos_at_min: list[tuple[int, ...]] = []
    budget_exceeded = False
    for k in range(1, search_limit + 1):
        if math.comb(n, k) > max_combos_per_level:
            # This level alone is already too expensive to enumerate (the
            # (C2)^7 case): abandon the exact search rather than hang.
            budget_exceeded = True
            break
        combos = faithful_combos(k)
        if combos:
            min_cardinality = k
            combos_at_min = combos
            break

    if budget_exceeded:
        return _greedy_faithful_fallback(group, candidates, cost=cost)

    fallback_used = False
    if min_cardinality is None:
        # The search cap was reached with no faithful set found: fall back to
        # the full nontrivial-block set, always faithful by the regular-
        # representation theorem (see `minimal_separating_blocks`).
        full = tuple(candidates)
        if not is_faithful(group, full):
            raise ValueError(
                "the full nontrivial-block set is not faithful; this violates "
                "the regular-representation theorem and indicates corrupted "
                "character data"
            )
        min_cardinality = n
        combos_at_min = [full]
        fallback_used = True

    def rank_key(combo: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        return (block_set_cost(group, combo)["total_block_rank"], combo)

    best = min(combos_at_min, key=rank_key)
    min_total_block_rank = block_set_cost(group, best)["total_block_rank"]

    if (
        cost == "total_block_rank"
        and not fallback_used
        and min_cardinality < search_limit
        and math.comb(n, min_cardinality + 1) <= max_combos_per_level
    ):
        # Widen the search by one level: a slightly larger set can have a
        # smaller total dimension count than every set at min_cardinality.
        next_combos = faithful_combos(min_cardinality + 1)
        if next_combos:
            next_best = min(next_combos, key=rank_key)
            next_rank = block_set_cost(group, next_best)["total_block_rank"]
            if next_rank < min_total_block_rank:
                best = next_best
                min_total_block_rank = next_rank

    return {
        "min_cardinality": min_cardinality,
        "n_sets_at_min_cardinality": len(combos_at_min),
        "min_total_block_rank": min_total_block_rank,
        "example_min_set": best,
        "cost": cost,
        "fallback_used": fallback_used,
        "certified": True,
    }


def used_set_minimality(
    group: FiniteGroup,
    used_blocks: Sequence[int],
    *,
    cost: str = "cardinality",
) -> dict[str, Any]:
    """The headline minimal-faithful-set comparison for a MEASURED used block
    set (e.g. from occupancy, I-15 causal ablation, or the GCR matrix-product
    fit): whether it is itself faithful, its cost against
    :func:`minimum_faithful_sets`'s cost-based minimum, and whether it is
    irredundant (removing any single block from it breaks faithfulness).

    A ``used_blocks`` set that is not faithful is reported as such
    (``used_faithful=False``) rather than raising -- the costs are still
    computed and returned, because a not-faithful used set is itself a
    notable finding (the model would be missing information needed to
    distinguish some pair of group elements), not an error condition.

    Returns a dict: ``used_faithful``, ``used_cardinality``,
    ``used_total_block_rank``, ``min_cardinality``, ``min_total_block_rank``
    (the latter two from :func:`minimum_faithful_sets`), ``cardinality_ratio``
    and ``rank_ratio`` (used-over-minimum, ``nan`` in the order-1-group edge
    case where the minimum is zero), ``is_irredundant`` (``True`` iff the used
    set is faithful and removing any one of its blocks breaks that -- ``False``
    whenever the used set is not itself faithful, since redundancy is only
    defined relative to an already-faithful set), and
    ``n_minimal_faithful_sets`` (:func:`minimum_faithful_sets`'s
    ``n_sets_at_min_cardinality``, surfaced here because non-uniqueness of the
    minimal faithful set is exactly what would make a naive single-set
    comparison misleading), and ``minimum_certified``
    (:func:`minimum_faithful_sets`'s ``certified`` -- ``False`` means
    ``min_cardinality``/``min_total_block_rank`` are only an upper bound, not
    a certified minimum, so the ratios above are upper bounds on the true
    ratio too)."""
    used = tuple(sorted({int(b) for b in used_blocks}))
    used_faithful = is_faithful(group, used)
    used_cost = block_set_cost(group, used)
    minimum = minimum_faithful_sets(group, cost=cost)

    is_irredundant = used_faithful and all(
        not is_faithful(group, tuple(b for b in used if b != drop)) for drop in used
    )

    cardinality_ratio = (
        used_cost["cardinality"] / minimum["min_cardinality"]
        if minimum["min_cardinality"] > 0
        else float("nan")
    )
    rank_ratio = (
        used_cost["total_block_rank"] / minimum["min_total_block_rank"]
        if minimum["min_total_block_rank"] > 0
        else float("nan")
    )

    return {
        "used_faithful": used_faithful,
        "used_cardinality": used_cost["cardinality"],
        "used_total_block_rank": used_cost["total_block_rank"],
        "min_cardinality": minimum["min_cardinality"],
        "min_total_block_rank": minimum["min_total_block_rank"],
        "cardinality_ratio": cardinality_ratio,
        "rank_ratio": rank_ratio,
        "is_irredundant": is_irredundant,
        "n_minimal_faithful_sets": minimum["n_sets_at_min_cardinality"],
        "minimum_certified": minimum["certified"],
    }


# ---------------------------------------------------------------------------
# Divergence comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateDivergence:
    """One occupancy vector's total-variation distance to each of the three
    templates, and which it is closest to -- but only when :attr:`screen` is
    :data:`DEFINED`. When it is not, ``tv_to_coset`` and ``closest_template``
    are the string :data:`UNDEFINED`: a coset-vs-GCR verdict is refused
    exactly where the coset target itself is degenerate or absent, even
    though ``tv_to_null`` and ``tv_to_gcr`` remain independently well-defined
    and are always reported."""

    screen: DegeneracyScreen
    null_template: np.ndarray
    gcr_template: np.ndarray
    gcr_selected_blocks: tuple[int, ...]
    tv_to_null: float
    tv_to_gcr: float
    tv_to_coset: float | str
    closest_template: str

    def to_record(self) -> dict[str, Any]:
        return {
            "instrument": "template-divergence",
            "screen": self.screen.to_record(),
            "null_template": self.null_template.tolist(),
            "gcr_template": self.gcr_template.tolist(),
            "gcr_selected_blocks": list(self.gcr_selected_blocks),
            "tv_to_null": self.tv_to_null,
            "tv_to_gcr": self.tv_to_gcr,
            "tv_to_coset": self.tv_to_coset,
            "closest_template": self.closest_template,
        }


def compare_to_templates(
    occupancy: np.ndarray,
    group: FiniteGroup,
    *,
    max_support_fraction: float = MAX_COSET_SUPPORT_FRACTION,
) -> TemplateDivergence:
    """Compare a model's occupancy vector (:func:`occupancy.
    population_occupancy`'s output, aligned to ``group.isotypic_blocks``)
    against the analytic null, the coset ``Ind_H^G 1`` template and the GCR
    sparse-irrep template, reporting the total-variation distance to each and
    which is closest -- gated on :func:`degeneracy_screen`. When the screen is
    not :data:`DEFINED`, the coset-side readouts (``tv_to_coset``,
    ``closest_template``) are the string :data:`UNDEFINED` rather than a
    number: the whole point of this instrument is a coset-vs-GCR
    discrimination, which is not a question a degenerate or absent coset
    target can answer."""
    occ = np.asarray(occupancy, dtype=np.float64)
    n_blocks = len(group.isotypic_blocks)
    if occ.shape != (n_blocks,):
        raise ValueError(f"occupancy must have shape ({n_blocks},), got {occ.shape}")
    screen = degeneracy_screen(group, max_support_fraction=max_support_fraction)
    null_template = analytic_null(group)
    gcr_selected = minimal_separating_blocks(group)
    gcr_template = gcr_sparse_template(group)
    tv_to_null = total_variation(occ, null_template)
    tv_to_gcr = total_variation(occ, gcr_template)
    if screen.verdict != DEFINED:
        return TemplateDivergence(
            screen=screen,
            null_template=null_template,
            gcr_template=gcr_template,
            gcr_selected_blocks=gcr_selected,
            tv_to_null=tv_to_null,
            tv_to_gcr=tv_to_gcr,
            tv_to_coset=UNDEFINED,
            closest_template=UNDEFINED,
        )
    assert screen.coset_template is not None  # DEFINED always carries one
    tv_to_coset = total_variation(occ, screen.coset_template)
    distances = {"null": tv_to_null, "coset": tv_to_coset, "gcr": tv_to_gcr}
    closest = min(distances, key=lambda name: distances[name])
    return TemplateDivergence(
        screen=screen,
        null_template=null_template,
        gcr_template=gcr_template,
        gcr_selected_blocks=gcr_selected,
        tv_to_null=tv_to_null,
        tv_to_gcr=tv_to_gcr,
        tv_to_coset=tv_to_coset,
        closest_template=closest,
    )


# ---------------------------------------------------------------------------
# Data-driven refinement: measured USED/product-carrying sets vs PREDICTED sets
#
# ``compare_to_templates`` above compares occupancy against three *predicted*
# templates -- the analytic null, coset's Ind_H^G 1, and GCR's greedy-cheapest
# faithful-set guess. Three flaws make its coset-vs-GCR non-result untrustworthy
# on its own: (i) the sparse-GCR set is a guess that can pick the wrong faithful
# subset when several exist; (ii) "closest" is an argmin that is meaningless
# when the three distances sit close together; (iii) it is block-total
# resemblance only, never checked against which blocks the model *causally*
# uses.
#
# The functions below build the data-driven side of that comparison from two
# other instruments' outputs, so the sparse-GCR guess and the argmin can both
# be checked against measurement rather than trusted outright:
#
# * :func:`causally_used_blocks` reads I-15 (``coset.isotypic_block_ablation``,
#   aggregated per cell by ``scripts/measure_isotypic_ablation.py``) for the
#   blocks whose ablation costs more than a matched random subspace -- the
#   MEASURED "used" set, upgrading "occupied" to "used";
# * :func:`product_carrying_blocks` reads the GCR matrix-product fit
#   (``gcr_matmul.py``, aggregated by ``scripts/measure_gcr_matmul.py``) for
#   the degree->=2 blocks where the shared-index matrix-product form is
#   actually favoured over the unconstrained bilinear alternative -- a second,
#   independent MEASURED set a caller may use to refine or cross-check the
#   used set before comparison;
# * :func:`data_driven_comparison` compares a MEASURED used set against each
#   account's PREDICTED set (coset: the nonzero-support blocks of Ind_H^G 1;
#   GCR/representation: the minimal faithful set, :func:`minimal_separating_blocks`)
#   by set overlap (Jaccard, precision, recall) AND by the raw TV distance of
#   the occupancy to a template built from each set -- always reporting every
#   raw distance plus a ``separation`` field beside any argmin label, so a
#   caller can see directly when "closest" would be meaningless.
#
# These functions take instrument RECORDS (the dicts/JSON the two upstream
# instruments already emit) as inputs, never re-deriving or re-running either
# instrument, so they can be wired to real result JSON once runs complete.
# ---------------------------------------------------------------------------

#: The per-cell aggregated ``flip_fraction_over_random`` mean (I-15, a given
#: ablation mode) must exceed this for a block to count as causally used --
#: the matched-random-subspace-relative effect that licenses "used" over
#: merely "occupied" (see ``coset.isotypic_block_ablation``'s own docstring).
#: A modest bar rather than a strict one: I-15's matched-random control
#: already screens out "any subspace of this rank does this much damage", so
#: a small positive margin above that control is already informative. Named
#: and overridable, never a bare literal.
MIN_FLIP_OVER_RANDOM = 0.05

#: The per-cell aggregated ``mp_minus_bilinear_heldout`` mean (the GCR
#: matrix-product instrument, ``gcr_matmul.py``) must be at least this for a
#: degree->=2 block to count as product-carrying: a non-negative gap means
#: the unconstrained bilinear alternative does not beat the shared-index
#: matrix-product form out of sample. Zero, not a positive margin, because the
#: bilinear model is a strict superset that can only tie or overfit when the
#: truth is a matrix product (``gcr_matmul.py``'s module docstring); named and
#: overridable.
MIN_MP_MINUS_BILINEAR = 0.0

#: The fraction of measured seeds in which the information criterion must
#: favour the constrained matrix-product model (``mp_favoured_fraction``) for
#: a degree->=2 block to count as product-carrying, when
#: ``require_favoured=True``. A simple majority; named and overridable.
MIN_FAVOURED_FRACTION = 0.5


def causally_used_blocks(
    iso_ablation_record: dict[str, Any],
    *,
    min_flip_over_random: float = MIN_FLIP_OVER_RANDOM,
    mode: str = "zero",
) -> tuple[int, ...]:
    """The isotypic blocks I-15 shows the model actually USES: those whose
    per-cell aggregated ``flip_fraction_over_random`` mean, under ablation
    mode ``mode`` (default ``"zero"``, outright deletion), exceeds
    ``min_flip_over_random`` -- the matched-random-subspace-relative effect
    that licenses "used" over merely "occupied"
    (``coset.isotypic_block_ablation``, I-15).

    ``iso_ablation_record`` is the aggregated per-cell record
    ``scripts/measure_isotypic_ablation.py``'s ``aggregate_cell`` writes: a
    ``"blocks"`` list of ``{"block_index": int, "modes": {mode: {
    "flip_fraction_over_random": {"mean": float, ...}, ...}, ...}}``. A block
    whose aggregated mean is non-finite (``NaN``, e.g. zero measured seeds) is
    never included. Blocks are returned sorted ascending; the trivial block is
    never included in practice because ``isotypic_block_ablation`` does not
    measure it by default, but this function does not special-case it -- it
    only ever reports what the record actually shows crossing the threshold.
    """
    selected: list[int] = []
    for block in iso_ablation_record["blocks"]:
        modes = block["modes"]
        if mode not in modes:
            raise ValueError(
                f"mode {mode!r} not present in block {block['block_index']}'s "
                f"modes ({sorted(modes)})"
            )
        mean = modes[mode]["flip_fraction_over_random"]["mean"]
        if mean is not None and math.isfinite(mean) and mean > min_flip_over_random:
            selected.append(int(block["block_index"]))
    return tuple(sorted(selected))


def product_carrying_blocks(
    gcr_matmul_record: dict[str, Any],
    *,
    require_favoured: bool = True,
    min_favoured_fraction: float = MIN_FAVOURED_FRACTION,
    min_mp_minus_bilinear: float = MIN_MP_MINUS_BILINEAR,
) -> tuple[int, ...]:
    """The degree->=2 isotypic blocks the GCR matrix-product instrument shows
    actually carry the shared-index matrix-product structure: a non-negative
    across-seed mean ``mp_minus_bilinear_heldout`` gap (>= ``
    min_mp_minus_bilinear``) and, when ``require_favoured`` (the default), the
    information criterion favouring the constrained model in at least
    ``min_favoured_fraction`` of measured seeds (``mp_favoured_fraction``).

    ``gcr_matmul_record`` is the aggregated per-cell record
    ``scripts/measure_gcr_matmul.py``'s ``_aggregate_cell`` writes: an
    ``"irreps"`` list of ``{"block_index": int, "mp_minus_bilinear_heldout":
    {"mean": float, ...}, "mp_favoured_fraction": float, ...}``. Every entry in
    that list already has irrep degree ``>= 2`` (``screen_gcr_matmul`` never
    fits lower-degree blocks), so no degree filter is applied here. Blocks are
    returned sorted ascending."""
    selected: list[int] = []
    for irrep in gcr_matmul_record["irreps"]:
        gap = irrep["mp_minus_bilinear_heldout"]["mean"]
        if gap is None or not math.isfinite(gap) or gap < min_mp_minus_bilinear:
            continue
        if require_favoured and irrep["mp_favoured_fraction"] < min_favoured_fraction:
            continue
        selected.append(int(irrep["block_index"]))
    return tuple(sorted(selected))


def _rank_proportional_template(group: FiniteGroup, blocks: Sequence[int]) -> np.ndarray | None:
    """An energy template proportional to ``block_rank`` over exactly
    ``blocks`` and zero elsewhere, summing to 1 -- the same convention the
    null, coset and GCR templates all use. ``None`` (never a zero vector, which
    would silently compare as "all mass elsewhere") when ``blocks`` is empty: a
    template over no blocks has no defined energy distribution."""
    unique = sorted({int(b) for b in blocks})
    if not unique:
        return None
    total_rank = sum(group.isotypic_blocks[j].block_rank for j in unique)
    if total_rank <= 0:
        raise ValueError("selected blocks have zero total rank; template is undefined")
    template = np.zeros(len(group.isotypic_blocks), dtype=np.float64)
    for j in unique:
        template[j] = group.isotypic_blocks[j].block_rank / total_rank
    return template


@dataclass(frozen=True)
class SetOverlap:
    """Set-overlap metrics between a MEASURED block set and one account's
    PREDICTED block set. ``precision`` is the fraction of the predicted set
    the model's measured-used set actually covers (``|used ∩ predicted| /
    |predicted|``); ``recall`` is the fraction of the measured-used set the
    predicted set accounts for (``|used ∩ predicted| / |used|``) -- the
    ordinary retrieval-style convention with "predicted" as "retrieved" and
    "used" as "relevant". Degenerate cases: two empty sets score 1.0 on all
    three (vacuously identical); one empty and one nonempty set scores 0.0 on
    all three."""

    used_size: int
    predicted_size: int
    intersection_size: int
    jaccard: float
    precision: float
    recall: float

    def to_record(self) -> dict[str, Any]:
        return {
            "used_size": self.used_size,
            "predicted_size": self.predicted_size,
            "intersection_size": self.intersection_size,
            "jaccard": self.jaccard,
            "precision": self.precision,
            "recall": self.recall,
        }


def _set_overlap(used: Sequence[int], predicted: Sequence[int]) -> SetOverlap:
    used_set = {int(x) for x in used}
    predicted_set = {int(x) for x in predicted}
    intersection = used_set & predicted_set
    union = used_set | predicted_set
    if not union:  # both empty: vacuously identical
        jaccard = precision = recall = 1.0
    else:
        jaccard = len(intersection) / len(union)
        precision = (len(intersection) / len(predicted_set)) if predicted_set else 0.0
        recall = (len(intersection) / len(used_set)) if used_set else 0.0
    return SetOverlap(
        used_size=len(used_set),
        predicted_size=len(predicted_set),
        intersection_size=len(intersection),
        jaccard=jaccard,
        precision=precision,
        recall=recall,
    )


@dataclass(frozen=True)
class DataDrivenComparison:
    """The data-driven refinement of :func:`compare_to_templates`: a MEASURED
    causally-used block set (I-15, typically :func:`causally_used_blocks`'s
    output, optionally refined against :func:`product_carrying_blocks`)
    compared against each account's PREDICTED block set -- coset's
    nonzero-support blocks of ``Ind_H^G 1``, GCR/representation's minimal
    faithful set (:func:`minimal_separating_blocks`) -- by set overlap
    (:class:`SetOverlap`: Jaccard, precision, recall) AND by the raw TV
    distance of the occupancy to a template built from each set.

    Every raw distance (``tv_to_used``, ``tv_to_coset_predicted``,
    ``tv_to_gcr_predicted``) is always reported, alongside ``separation`` (the
    gap between the two smallest of whichever distances are defined) and
    ``closest_template`` (an argmin label). ``closest_template`` is never
    reported without ``separation`` sitting beside it: an argmin among close
    distances is not evidence, and ``separation`` is what lets a caller see
    that directly rather than trusting the label.

    ``tv_to_used`` and ``coset_overlap``/``tv_to_coset_predicted`` are the
    string :data:`UNDEFINED` when, respectively, ``used_blocks`` is empty (no
    causally-used block was measured) or no coset template is available (no
    explicit override and :func:`degeneracy_screen` is not :data:`DEFINED`
    for this group) -- never a fabricated number."""

    used_blocks: tuple[int, ...]
    coset_predicted_blocks: tuple[int, ...] | str
    gcr_predicted_blocks: tuple[int, ...]
    coset_overlap: SetOverlap | str
    gcr_overlap: SetOverlap
    tv_to_used: float | str
    tv_to_coset_predicted: float | str
    tv_to_gcr_predicted: float
    separation: float | None
    closest_template: str

    def to_record(self) -> dict[str, Any]:
        return {
            "instrument": "template-divergence-data-driven",
            "used_blocks": list(self.used_blocks),
            "coset_predicted_blocks": (
                list(self.coset_predicted_blocks)
                if isinstance(self.coset_predicted_blocks, tuple)
                else self.coset_predicted_blocks
            ),
            "gcr_predicted_blocks": list(self.gcr_predicted_blocks),
            "coset_overlap": (
                self.coset_overlap.to_record()
                if isinstance(self.coset_overlap, SetOverlap)
                else self.coset_overlap
            ),
            "gcr_overlap": self.gcr_overlap.to_record(),
            "tv_to_used": self.tv_to_used,
            "tv_to_coset_predicted": self.tv_to_coset_predicted,
            "tv_to_gcr_predicted": self.tv_to_gcr_predicted,
            "separation": self.separation,
            "closest_template": self.closest_template,
        }


def data_driven_comparison(
    occupancy: np.ndarray,
    group: FiniteGroup,
    used_blocks: Sequence[int],
    *,
    coset_template: np.ndarray | None = None,
    max_support_fraction: float = MAX_COSET_SUPPORT_FRACTION,
) -> DataDrivenComparison:
    """Compare a MEASURED causally-used block set against each account's
    PREDICTED set (see :class:`DataDrivenComparison` for the full field
    contract). ``occupancy`` must already be aligned to
    ``group.isotypic_blocks`` exactly as :func:`compare_to_templates` requires.

    ``used_blocks`` is taken as plain block indices rather than re-deriving
    them, so a caller may pass :func:`causally_used_blocks`'s output directly,
    a set already intersected with :func:`product_carrying_blocks`, or a
    hand-picked set for a specific reviewer question -- this function performs
    no measurement of its own.

    ``coset_template`` defaults to :func:`degeneracy_screen`'s own
    ``coset_template`` (the minimal-index core-free subgroup's ``Ind_H^G 1``
    energy template), gated by ``max_support_fraction`` exactly as there, and
    is :data:`UNDEFINED` under the same conditions. Pass an explicit
    ``coset_template`` array (shape ``(n_blocks,)``) to compare against a
    different subgroup's template instead."""
    occ = np.asarray(occupancy, dtype=np.float64)
    n_blocks = len(group.isotypic_blocks)
    if occ.shape != (n_blocks,):
        raise ValueError(f"occupancy must have shape ({n_blocks},), got {occ.shape}")

    used = tuple(sorted({int(b) for b in used_blocks}))
    used_template = _rank_proportional_template(group, used)

    if coset_template is None:
        screen = degeneracy_screen(group, max_support_fraction=max_support_fraction)
        coset_template_arr = screen.coset_template if screen.verdict == DEFINED else None
    else:
        coset_template_arr = np.asarray(coset_template, dtype=np.float64)
        if coset_template_arr.shape != (n_blocks,):
            raise ValueError(
                f"coset_template must have shape ({n_blocks},), got {coset_template_arr.shape}"
            )

    gcr_predicted = minimal_separating_blocks(group)
    gcr_template = gcr_sparse_template(group)
    gcr_overlap = _set_overlap(used, gcr_predicted)
    tv_to_gcr_predicted = total_variation(occ, gcr_template)

    coset_predicted: tuple[int, ...] | str
    coset_overlap: SetOverlap | str
    tv_to_coset_predicted: float | str
    if coset_template_arr is None:
        coset_predicted = UNDEFINED
        coset_overlap = UNDEFINED
        tv_to_coset_predicted = UNDEFINED
    else:
        coset_predicted = tuple(j for j in range(n_blocks) if coset_template_arr[j] > _TOL)
        coset_overlap = _set_overlap(used, coset_predicted)
        tv_to_coset_predicted = total_variation(occ, coset_template_arr)

    tv_to_used: float | str
    tv_to_used = total_variation(occ, used_template) if used_template is not None else UNDEFINED

    distances: dict[str, float] = {"gcr": tv_to_gcr_predicted}
    if isinstance(tv_to_used, float):
        distances["used"] = tv_to_used
    if isinstance(tv_to_coset_predicted, float):
        distances["coset"] = tv_to_coset_predicted

    closest_template = min(distances, key=lambda name: distances[name])
    separation: float | None = None
    if len(distances) >= 2:
        ordered = sorted(distances.values())
        separation = ordered[1] - ordered[0]

    return DataDrivenComparison(
        used_blocks=used,
        coset_predicted_blocks=coset_predicted,
        gcr_predicted_blocks=gcr_predicted,
        coset_overlap=coset_overlap,
        gcr_overlap=gcr_overlap,
        tv_to_used=tv_to_used,
        tv_to_coset_predicted=tv_to_coset_predicted,
        tv_to_gcr_predicted=tv_to_gcr_predicted,
        separation=separation,
        closest_template=closest_template,
    )


__all__ = [
    "DEFINED",
    "MAX_COSET_SUPPORT_FRACTION",
    "MAX_FAITHFUL_SET_SEARCH_CARDINALITY",
    "MAX_FAITHFUL_SET_SEARCH_COMBOS",
    "MIN_FAVOURED_FRACTION",
    "MIN_FLIP_OVER_RANDOM",
    "MIN_MP_MINUS_BILINEAR",
    "UNDEFINED",
    "DataDrivenComparison",
    "DegeneracyScreen",
    "SetOverlap",
    "TemplateDivergence",
    "block_kernel",
    "block_set_cost",
    "causally_used_blocks",
    "compare_to_templates",
    "data_driven_comparison",
    "degeneracy_screen",
    "gcr_sparse_template",
    "is_faithful",
    "minimal_separating_blocks",
    "minimum_faithful_sets",
    "product_carrying_blocks",
    "used_set_minimality",
]
