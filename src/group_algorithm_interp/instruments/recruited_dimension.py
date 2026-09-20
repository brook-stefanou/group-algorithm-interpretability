"""Recruited-dimension instrument: effective dimension vs three accounts.

The study's thesis is that the network learns a near-minimal *faithful* subset
of the group's irreducible representations, rather than the whole regular
representation. A recent matrix-memory result (Larson, arXiv 2609.12259) argues
that networks recruit exactly the group's **minimal faithful real
representation dimension**. This instrument tests the analogous claim in this
project's MLP-on-one-hot setting and sets it against two competitors:

* the **tensor-rank account** (Shutman et al. 2509.06931): learning difficulty,
  and the dimension a network must carry, scale with the rank of the
  group-algebra multiplication tensor (``groups/tensor_rank.py``);
* **"uses everything"**: the network recruits the full regular representation,
  dimension ``|G|``.

It does so by measuring a principled *effective dimension* of the trained model
(a participation-ratio / stable-rank of a well-justified matrix, all on CPU in
float64) and reporting, per group, how that measured dimension compares -- as a
ratio -- to each of the three theoretical anchors.

The subtle anchor is the first, and it is a **real**-representation quantity.

Minimal faithful *real* representation dimension
------------------------------------------------
Work over the reals. Each complex irreducible representation ``rho`` of complex
degree ``d`` contributes a *real* irreducible whose real dimension depends on
its Frobenius-Schur indicator ``nu(rho) = (1/|G|) sum_g chi(rho)(g^2)``:

* ``nu = +1`` (**real type**): ``rho`` is realisable over the reals; the real
  irreducible has real dimension ``d``.
* ``nu = 0`` (**complex type**): ``rho`` is not self-dual; ``rho`` and its
  conjugate ``rho-bar`` fuse into a single real irreducible of real dimension
  ``2d``.
* ``nu = -1`` (**quaternionic type**): ``rho`` is self-dual but not realisable
  over the reals; the real irreducible has real dimension ``2d``.

The real isotypic blocks this project exports (``groups/data.py``:
:class:`IsotypicBlock`) already merge each complex-conjugate pair into one
block, so the blocks are in one-to-one correspondence with the group's real
irreducibles: a length-2 block is a complex-type pair (real dim ``2d``), a
length-1 block is real (``nu=+1``, real dim ``d``) or quaternionic
(``nu=-1``, real dim ``2d``). The kernel of a real irreducible equals the
kernel of the underlying complex irrep (a conjugate pair shares a kernel), which
is exactly what :func:`template_divergence.block_kernel` reads off the
character, so faithfulness over the reals is the same trivial-kernel-intersection
condition the faithful-set machinery already uses -- only the *cost* changes,
from cardinality / ``d^2`` to the real dimension above.

The **minimal faithful real dimension** is then the smallest total real
dimension of a set of real irreducibles (nontrivial blocks) whose direct sum is
faithful (kernels intersect to the identity alone). The canonical discriminating
case is Q8: its only faithful irrep is the degree-2 quaternionic one, so the
minimal faithful *complex* dimension is 2 but the minimal faithful *real*
dimension is ``2*2 = 4``. A real-type group of the same complex degree (S3, D4:
degree-2 faithful irrep, ``nu=+1``) lands at ``2``. Getting Q8 = 4 and S3 = 2 is
the whole point of doing this over the reals, and is the instrument's headline
correctness test.

Tensor rank is collinear with ``|G|`` at the floor
---------------------------------------------------
``groups/tensor_rank.py`` proves the unconditional lower bound on the
group-algebra tensor rank is exactly ``|G|`` (the block-diagonal flattening
rank), and warns that tensor-rank bounds are a monotone transform of the
character degrees -- never an independent axis. So the tensor-rank *lower bound*
coincides with the "uses everything" anchor for essentially every group; the two
accounts differ only in the tensor rank's *upper* bound (``> |G|`` once the group
has high-degree irreps) and, more to the point, in a *hidden-unit / multiplication
count*, which is the GCR matrix-product instrument's object, not this one. This
instrument therefore reports the tie honestly: an effective dimension near ``|G|``
is "consistent with both the tensor-rank and uses-everything accounts", and the
discriminating power the recruited *dimension* actually delivers is
minimal-faithful (small) versus full-regular (``|G|``). Separating tensor-rank
from uses-everything is deferred to the multiplication-count instruments, and
said so rather than papered over with a spurious argmin.

Effective (recruited) dimension
--------------------------------
The measured dimension is a spectral effective-rank of a matrix, reported in
three flavours so a reader sees robustness (:class:`EffectiveRank`):

* participation ratio of the singular values, ``(sum s_i)^2 / sum s_i^2``;
* participation ratio of the covariance spectrum, ``(sum s_i^2)^2 / sum s_i^4``;
* stable rank, ``sum s_i^2 / max_i s_i^2 = ||A||_F^2 / ||A||_2^2``.

For a matrix with ``k`` equal nonzero singular values all three equal ``k``; they
diverge only as the spectrum becomes unequal, which is exactly the "how many
directions does the model really use" question.

Two matrices are measured, matching the project's rule that the object of study
is activation-space, with the always-available weight-space quantity beside it:

* the **embedding** ``W_E`` restricted to the ``|G|`` group-element rows (the
  ``'='`` token row is dropped -- it is an architectural DC token, not a group
  element);
* the **activation code** over one argument: reusing
  :func:`occupancy.neuron_activations` to get ``A[m, a, b]``, the matrix whose
  ``a``-th row is the flattened ``(m, b)`` response ``A[:, a, :]`` (left
  argument; the ``b``-code is the transpose case). Its effective rank is the
  number of directions the model recruits to encode that argument, capped at
  ``|G|``.

Both are reported **centred and uncentred**. Centring (subtracting the mean over
the group-element index) removes the constant / trivial-block direction, which
at random initialisation carries ~96% of the activation energy (see
``occupancy.py``) and would otherwise dominate the effective rank and mask the
recruited structure -- the same reason the occupancy headline is the
nontrivial-renormalised form. The centred activation-left participation ratio is
the primary recruited-dimension scalar; the rest are reported for transparency
and robustness.

Everything downstream of the forward pass is CPU float64; the ``device``
parameter only moves the forward pass (default CPU), exactly as
:func:`occupancy.neuron_activations` documents.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..groups.group import FiniteGroup
from ..groups.tensor_rank import group_algebra_tensor_rank_bounds
from ..model import GroupModel
from .occupancy import neuron_activations, trivial_block_index
from .template_divergence import (
    MAX_FAITHFUL_SET_SEARCH_CARDINALITY,
    MAX_FAITHFUL_SET_SEARCH_COMBOS,
    block_kernel,
    greedy_faithful_cover,
    is_faithful,
)

_TOL = 1e-6

#: Frobenius-Schur indicator values and their real-representation meaning.
FS_REAL = 1  #: nu = +1: real type, real irreducible dimension d.
FS_COMPLEX = 0  #: nu = 0: complex type, conjugate pair fuses to real dimension 2d.
FS_QUATERNIONIC = -1  #: nu = -1: quaternionic type, real irreducible dimension 2d.


# ---------------------------------------------------------------------------
# Real-representation theory: Frobenius-Schur and minimal faithful real dim
# ---------------------------------------------------------------------------


def frobenius_schur_indicator(group: FiniteGroup, irrep_index: int) -> int:
    """The Frobenius-Schur indicator ``nu(rho) = (1/|G|) sum_g chi(g^2)`` of one
    irrep, computed directly from its character and the squaring power map --
    never from the stored ``frobenius_schur`` array, so it can *verify* that
    array rather than trust it.

    The squaring map ``g -> g^2`` is the Cayley-table diagonal
    ``cayley_table[g, g]``; ``chi`` is :attr:`IrrepData.character` over all
    elements. The sum is exactly ``+1``, ``0`` or ``-1`` for any genuine
    character (Frobenius-Schur), so a non-integer or out-of-range result means
    corrupted character data and is raised, not rounded away silently."""
    irrep = group.irreps[irrep_index]
    character = np.asarray(irrep.character, dtype=np.complex128)
    if character.shape != (group.order,):
        raise ValueError(
            f"irrep {irrep_index} character has shape {character.shape}, "
            f"expected ({group.order},) over all group elements"
        )
    diagonal = np.arange(group.order)
    squares = group.cayley_table[diagonal, diagonal]
    nu = character[squares].sum() / group.order
    if abs(nu.imag) > _TOL:
        raise ValueError(
            f"Frobenius-Schur indicator of irrep {irrep_index} is not real "
            f"({nu}); character data is corrupted"
        )
    rounded = round(nu.real)
    if abs(nu.real - rounded) > _TOL or rounded not in (FS_REAL, FS_COMPLEX, FS_QUATERNIONIC):
        raise ValueError(
            f"Frobenius-Schur indicator of irrep {irrep_index} is {nu.real}, "
            "not one of -1/0/+1; character data is corrupted"
        )
    return int(rounded)


def block_real_dimension(group: FiniteGroup, block_index: int) -> int:
    """The real dimension of the real irreducible represented by one isotypic
    block, from its Frobenius-Schur type and complex degree ``d``
    (:attr:`IsotypicBlock.irrep_degree`):

    * length-1 block, ``nu = +1`` (real): ``d``;
    * length-1 block, ``nu = -1`` (quaternionic): ``2d``;
    * length-2 block (complex-conjugate pair, ``nu = 0`` for both): ``2d``.

    The Frobenius-Schur indicator of each merged irrep is recomputed from its
    character (:func:`frobenius_schur_indicator`) and cross-checked against the
    stored ``group.frobenius_schur`` when present; a mismatch is raised. The
    block structure is validated against the indicator: a length-1 block must be
    real or quaternionic (a complex-type irrep would have been merged with its
    conjugate into a length-2 block), and a length-2 block must be two
    complex-type irreps sharing a degree."""
    block = group.isotypic_blocks[block_index]
    degree = block.irrep_degree
    indices = block.irrep_indices
    indicators = [frobenius_schur_indicator(group, i) for i in indices]

    stored = getattr(group, "frobenius_schur", None)
    if stored is not None:
        for i, computed in zip(indices, indicators, strict=True):
            if int(stored[i]) != computed:
                raise ValueError(
                    f"stored Frobenius-Schur {int(stored[i])} for irrep {i} disagrees "
                    f"with the character-derived value {computed}"
                )

    if len(indices) == 1:
        indicator = indicators[0]
        if indicator == FS_REAL:
            return degree
        if indicator == FS_QUATERNIONIC:
            return 2 * degree
        raise ValueError(
            f"block {block_index} is a single irrep with Frobenius-Schur {indicator} "
            "(complex type); a complex-type irrep must be merged with its conjugate "
            "into a length-2 real block, so this is an invalid real isotypic decomposition"
        )
    if len(indices) == 2:
        if not all(ind == FS_COMPLEX for ind in indicators):
            raise ValueError(
                f"block {block_index} merges two irreps {indices} but their "
                f"Frobenius-Schur indicators {indicators} are not both 0 (complex type); "
                "only a complex-conjugate pair forms a length-2 real block"
            )
        return 2 * degree
    raise ValueError(
        f"block {block_index} merges {len(indices)} irreps {indices}; a real isotypic "
        "block is one irrep (real/quaternionic) or a conjugate pair (complex), never more"
    )


def real_irreducible_dimensions(group: FiniteGroup) -> tuple[int, ...]:
    """The real dimension of every isotypic block, block order preserved
    (:func:`block_real_dimension` per block)."""
    return tuple(block_real_dimension(group, j) for j in range(len(group.isotypic_blocks)))


@dataclass(frozen=True)
class MinimalFaithfulReal:
    """The minimal faithful real representation dimension of a group and the
    evidence behind it.

    ``real_dimension`` is the smallest total real dimension of a set of
    nontrivial real irreducibles (isotypic blocks) whose direct sum is faithful.
    ``example_set`` is one block set achieving it (a representative -- minimal
    faithful sets are generally not unique), ``example_set_cardinality`` its
    block count, and ``example_set_degrees``/``example_set_real_dims`` the complex
    degrees and real dimensions of those blocks. ``n_sets_at_min`` counts how many
    distinct faithful sets attain ``real_dimension`` within the search cap (the
    non-uniqueness surfaced directly). ``fallback_used`` is True only in the
    guarded pathological case where the cardinality search hit ``max_cardinality``
    without a faithful set and fell back to the full nontrivial-block set (always
    faithful by the regular-representation theorem). ``certified`` is ``False``
    when the candidate count exceeded ``exact_max_candidates`` and the exact
    search was skipped in favour of a greedy kernel-intersection cover (see
    :func:`minimal_faithful_real_dimension`) -- ``real_dimension`` and
    ``example_set`` are then still a valid faithful set and its real
    dimension, but only an UPPER BOUND on the true minimum, never a
    certified one."""

    order: int
    real_dimension: int
    example_set: tuple[int, ...]
    example_set_cardinality: int
    example_set_degrees: tuple[int, ...]
    example_set_real_dims: tuple[int, ...]
    n_sets_at_min: int
    all_real_dimensions: tuple[int, ...]
    fallback_used: bool
    certified: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "real_dimension": self.real_dimension,
            "example_set": list(self.example_set),
            "example_set_cardinality": self.example_set_cardinality,
            "example_set_degrees": list(self.example_set_degrees),
            "example_set_real_dims": list(self.example_set_real_dims),
            "n_sets_at_min": self.n_sets_at_min,
            "all_real_dimensions": list(self.all_real_dimensions),
            "fallback_used": self.fallback_used,
            "certified": self.certified,
        }


def minimal_faithful_real_dimension(
    group: FiniteGroup,
    *,
    max_cardinality: int = MAX_FAITHFUL_SET_SEARCH_CARDINALITY,
    max_combos_per_level: int = MAX_FAITHFUL_SET_SEARCH_COMBOS,
) -> MinimalFaithfulReal:
    """The minimal faithful *real* representation dimension of ``group`` (see
    :class:`MinimalFaithfulReal` and the module docstring).

    Search strategy: increasing cardinality ``k = 1, 2, ...`` over combinations
    of nontrivial blocks, tracking the least total real dimension of any faithful
    set found. Because real dimension is not monotone in cardinality (one
    expensive quaternionic block can cost more than several cheap real ones), the
    search does not stop at the first faithful hit: it continues until the sum of
    the ``k`` smallest block real dimensions -- a tight lower bound on any
    cardinality-``k`` set's cost -- strictly exceeds the best cost found, at which
    point no larger set can beat it and the minimum (and the count of sets tying
    at it) is exact within ``max_cardinality``. The cap guards a combinatorial
    blow-up on a hypothetical group with a large minimal faithful set, falling
    back to the full nontrivial-block set (faithful by the
    regular-representation theorem); every panel group's minimal faithful set is
    tiny, so the cap is generous headroom, not a tight bound. Reuses
    :func:`template_divergence.is_faithful` for the trivial-kernel-intersection
    test unchanged -- only the cost is the real dimension.

    The cardinality cap and lower-bound pruning alone are not enough to guard
    a large CANDIDATE count: ``C(n, k)`` explodes from a large ``n`` even at a
    small, capped ``k`` (a group with many one-dimensional irreps and no
    single faithful one -- e.g. the elementary-abelian (C2)^7, 127 nontrivial
    candidates -- has ``C(127, 6) ~ 4.8e9``, and its real-dimension lower
    bound never prunes those levels away since 1-dimensional blocks make
    every level's lower bound small). A flat cap on the candidate count is
    too blunt here for the same reason as :func:`template_divergence.
    minimum_faithful_sets` (see its module-section docstring): several real
    panel groups have a large candidate count but a cheap search because
    pruning kicks in at small ``k``. So the guard instead bounds the
    per-level WORK: before enumerating cardinality-``k`` combinations,
    ``max_combos_per_level`` caps ``C(n, k)``; a group whose search prunes to
    completion before any over-budget level is entirely unaffected.
    Exceeding the budget abandons the exact search (even if a faithful set
    was already found at a smaller ``k`` -- a further level could still beat
    it, and real dimension is not monotone, so a partial result cannot
    safely be certified) in favour of
    :func:`template_divergence.greedy_faithful_cover`, a fast,
    always-terminating greedy kernel-intersection cover; the result is then
    reported with ``certified=False`` (a valid faithful set and upper bound
    on the true minimum, never a false minimum, but not guaranteed
    optimal)."""
    trivial = trivial_block_index(group)
    candidates = [j for j in range(len(group.isotypic_blocks)) if j != trivial]
    all_dims = real_irreducible_dimensions(group)
    real_dims = {j: all_dims[j] for j in candidates}
    n = len(candidates)
    search_limit = min(n, max_cardinality)
    ascending = sorted(real_dims.values())

    # Kernels computed once, outside the combinatorial loop (see
    # `template_divergence.minimum_faithful_sets` for the same optimisation
    # and why: `is_faithful` recomputes `block_kernel` on every call, which
    # would otherwise redo the same O(n) work millions of times over).
    kernels = {j: frozenset(int(x) for x in block_kernel(group, j).tolist()) for j in candidates}
    full_intersection = frozenset(range(group.order))

    def combo_is_faithful(combo: tuple[int, ...]) -> bool:
        intersection = full_intersection
        for j in combo:
            intersection &= kernels[j]
            if len(intersection) <= 1:
                break
        return len(intersection) == 1

    best_cost: int | None = None
    best_set: tuple[int, ...] | None = None
    n_at_best = 0
    budget_exceeded = False
    for k in range(1, search_limit + 1):
        lower_bound = sum(ascending[:k])
        if best_cost is not None and lower_bound > best_cost:
            break  # no cardinality >= k can match or beat best_cost
        if math.comb(n, k) > max_combos_per_level:
            budget_exceeded = True
            break
        for combo in itertools.combinations(candidates, k):
            if not combo_is_faithful(combo):
                continue
            cost = sum(real_dims[j] for j in combo)
            if best_cost is None or cost < best_cost:
                best_cost, best_set, n_at_best = cost, combo, 1
            elif cost == best_cost:
                n_at_best += 1

    if budget_exceeded:
        greedy = greedy_faithful_cover(group, candidates, tie_break_cost=lambda j: real_dims[j])
        greedy_cost = sum(real_dims[j] for j in greedy)
        return MinimalFaithfulReal(
            order=group.order,
            real_dimension=greedy_cost,
            example_set=greedy,
            example_set_cardinality=len(greedy),
            example_set_degrees=tuple(group.isotypic_blocks[j].irrep_degree for j in greedy),
            example_set_real_dims=tuple(real_dims[j] for j in greedy),
            n_sets_at_min=1,
            all_real_dimensions=all_dims,
            fallback_used=False,
            certified=False,
        )

    fallback_used = False
    if best_cost is None:
        full = tuple(candidates)
        if not is_faithful(group, full):
            raise ValueError(
                "the full nontrivial-block set is not faithful; this violates the "
                "regular-representation theorem and indicates corrupted character data"
            )
        best_cost = sum(real_dims[j] for j in candidates)
        best_set, n_at_best, fallback_used = full, 1, True

    assert best_set is not None
    return MinimalFaithfulReal(
        order=group.order,
        real_dimension=best_cost,
        example_set=best_set,
        example_set_cardinality=len(best_set),
        example_set_degrees=tuple(group.isotypic_blocks[j].irrep_degree for j in best_set),
        example_set_real_dims=tuple(real_dims[j] for j in best_set),
        n_sets_at_min=n_at_best,
        all_real_dimensions=all_dims,
        fallback_used=fallback_used,
        certified=True,
    )


# ---------------------------------------------------------------------------
# Theoretical anchors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TheoreticalDimensions:
    """The three theoretical anchors a recruited dimension is compared against.

    ``minimal_faithful_real`` is the minimal faithful real dimension
    (:class:`MinimalFaithfulReal`); ``regular_rep`` is ``|G|`` (full regular
    representation). The tensor-rank bounds come from
    :func:`groups.tensor_rank.group_algebra_tensor_rank_bounds`;
    ``tensor_rank_lower`` is the unconditional floor, which equals ``|G|`` for
    essentially every group (see the module docstring) -- carried explicitly so
    the coincidence with ``regular_rep`` is visible rather than hidden."""

    minimal_faithful_real: MinimalFaithfulReal
    regular_rep: int
    tensor_rank_lower: int
    tensor_rank_upper: int
    tensor_rank_naive_additive: int

    @property
    def tensor_rank_coincides_with_regular(self) -> bool:
        return self.tensor_rank_lower == self.regular_rep

    def to_record(self) -> dict[str, Any]:
        return {
            "minimal_faithful_real": self.minimal_faithful_real.to_record(),
            "minimal_faithful_real_dimension": self.minimal_faithful_real.real_dimension,
            "regular_rep": self.regular_rep,
            "tensor_rank_lower": self.tensor_rank_lower,
            "tensor_rank_upper": self.tensor_rank_upper,
            "tensor_rank_naive_additive": self.tensor_rank_naive_additive,
            "tensor_rank_coincides_with_regular": self.tensor_rank_coincides_with_regular,
        }


def theoretical_dimensions(
    group: FiniteGroup,
    *,
    max_cardinality: int = MAX_FAITHFUL_SET_SEARCH_CARDINALITY,
) -> TheoreticalDimensions:
    """The three theoretical anchors for ``group`` (see
    :class:`TheoreticalDimensions`)."""
    degrees = [irrep.dimension for irrep in group.irreps]
    bounds = group_algebra_tensor_rank_bounds(degrees)
    return TheoreticalDimensions(
        minimal_faithful_real=minimal_faithful_real_dimension(
            group, max_cardinality=max_cardinality
        ),
        regular_rep=group.order,
        tensor_rank_lower=bounds["tensor_rank_lower_bound"],
        tensor_rank_upper=bounds["tensor_rank_upper_bound"],
        tensor_rank_naive_additive=bounds["tensor_rank_naive_additive_lower_estimate"],
    )


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveRank:
    """Three spectral effective-rank measures of one matrix (see the module
    docstring). ``participation_ratio`` is ``(sum s)^2 / sum s^2`` over the
    singular values; ``participation_ratio_sq`` is ``(sum s^2)^2 / sum s^4`` (the
    participation ratio of the covariance spectrum ``s^2``); ``stable_rank`` is
    ``sum s^2 / max s^2``. For a matrix with ``k`` equal nonzero singular values
    all three equal ``k``."""

    participation_ratio: float
    participation_ratio_sq: float
    stable_rank: float
    n_singular_values: int
    max_singular_value: float
    frobenius_energy: float

    def to_record(self) -> dict[str, Any]:
        return {
            "participation_ratio": self.participation_ratio,
            "participation_ratio_sq": self.participation_ratio_sq,
            "stable_rank": self.stable_rank,
            "n_singular_values": self.n_singular_values,
            "max_singular_value": self.max_singular_value,
            "frobenius_energy": self.frobenius_energy,
        }


def effective_rank_from_singular_values(
    singular_values: Sequence[float] | np.ndarray,
) -> EffectiveRank:
    """The three effective-rank measures from a 1-D array of singular values
    (non-negative). A tiny negative value from round-off is clipped to zero; a
    genuinely negative value is raised. An all-zero spectrum (a zero matrix) has
    no defined effective rank and is raised."""
    s = np.asarray(singular_values, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"singular values must be 1-D, got shape {s.shape}")
    if s.size == 0:
        raise ValueError("no singular values; effective rank is undefined")
    if s.min() < -_TOL:
        raise ValueError(f"singular values must be non-negative, got min {s.min()}")
    s = np.clip(s, 0.0, None)
    s2 = s * s
    sum_s = float(s.sum())
    sum_s2 = float(s2.sum())
    sum_s4 = float((s2 * s2).sum())
    s_max = float(s.max())
    if sum_s2 <= 0.0 or s_max <= 0.0:
        raise ValueError("all singular values are zero; effective rank is undefined")
    return EffectiveRank(
        participation_ratio=(sum_s * sum_s) / sum_s2,
        participation_ratio_sq=(sum_s2 * sum_s2) / sum_s4,
        stable_rank=sum_s2 / (s_max * s_max),
        n_singular_values=int(s.size),
        max_singular_value=s_max,
        frobenius_energy=sum_s2,
    )


def effective_rank_of_matrix(matrix: np.ndarray) -> EffectiveRank:
    """Effective rank of a 2-D matrix via its singular values (CPU float64,
    ``compute_uv=False`` so no ``U``/``V`` are formed -- safe for the wide
    activation matrices)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {m.shape}")
    singular_values = np.linalg.svd(m, compute_uv=False)
    return effective_rank_from_singular_values(singular_values)


# ---------------------------------------------------------------------------
# Empirical recruited-dimension measures
# ---------------------------------------------------------------------------


def _group_element_embedding(model: GroupModel, order: int, *, center: bool) -> np.ndarray:
    """``W_E`` restricted to the ``order`` group-element rows (the ``'='`` token
    row ``order`` dropped), CPU float64, optionally mean-centred over the rows."""
    weight = model.W_E.detach().cpu().to(torch.float64).numpy()
    rows = weight[:order]
    if rows.shape[0] != order:
        raise ValueError(
            f"W_E has {weight.shape[0]} rows; expected at least {order} group-element rows"
        )
    if center:
        rows = rows - rows.mean(axis=0, keepdims=True)
    return rows


def embedding_effective_rank(
    model: GroupModel, group: FiniteGroup, *, center: bool = True
) -> EffectiveRank:
    """Recruited dimension read from the embedding ``W_E`` over the group
    elements (weight-space; always available, even for ``use_mlp=false`` models).
    ``center`` subtracts the mean embedding row, removing the constant/DC
    direction (the recommended default)."""
    return effective_rank_of_matrix(_group_element_embedding(model, group.order, center=center))


def _argument_code_matrix(
    activations: np.ndarray, order: int, argument: str, *, center: bool
) -> np.ndarray:
    """The ``|G| x (d_mlp * |G|)`` matrix whose row ``a`` (``argument='left'``) is
    the flattened response ``A[:, a, :]`` -- the model's activation code for the
    left argument over the Cayley grid; ``'right'`` is the ``b``-code. Centred
    (default) subtracts the mean over the group-element index, removing the
    constant/trivial-block direction."""
    a = np.asarray(activations, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != order or a.shape[2] != order:
        raise ValueError(f"activations must have shape [d_mlp, {order}, {order}], got {a.shape}")
    if argument == "left":
        matrix = a.transpose(1, 0, 2).reshape(order, -1)  # row a = flatten over (m, b)
    elif argument == "right":
        matrix = a.transpose(2, 0, 1).reshape(order, -1)  # row b = flatten over (m, a)
    else:
        raise ValueError(f"argument must be 'left' or 'right', got {argument!r}")
    if center:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    return matrix


def activation_effective_rank(
    model: GroupModel,
    group: FiniteGroup,
    *,
    argument: str = "left",
    center: bool = True,
    batch_size: int = 8192,
    device: torch.device = torch.device("cpu"),
) -> EffectiveRank:
    """Recruited dimension read from the MLP activation code for one argument
    over the Cayley grid (activation-space). Reuses
    :func:`occupancy.neuron_activations` for the forward pass (``device`` moves
    only that pass; the analysis is CPU float64), then takes the effective rank
    of the argument-code matrix. Requires an MLP."""
    activations = neuron_activations(model, group.order, batch_size=batch_size, device=device)
    matrix = _argument_code_matrix(activations, group.order, argument, center=center)
    return effective_rank_of_matrix(matrix)


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------


def discriminate(recruited: float, anchors: dict[str, float]) -> dict[str, Any]:
    """Compare a recruited dimension against the theoretical anchors: the ratio
    ``recruited / anchor`` for each, and which anchor it sits closest to on a
    log scale (equal multiplicative distance above and below counts equally).

    Anchors that are ``<= 0`` are skipped (an order-1 group's minimal faithful
    dimension is 0). ``closest`` names the single nearest anchor;
    ``closest_tie`` lists every anchor within ``_TOL`` of that nearest log
    distance, so a coincidence (tensor-rank floor equal to ``|G|``) is reported
    as a tie rather than an arbitrary pick. ``log_distance`` is ``|log ratio|``
    per anchor."""
    ratios = {name: recruited / value for name, value in anchors.items() if value > 0}
    if not ratios:
        raise ValueError("no positive anchors to compare against")
    log_distance = {name: abs(math.log(ratio)) for name, ratio in ratios.items() if ratio > 0}
    nearest = min(log_distance.values())
    tie = sorted(name for name, dist in log_distance.items() if dist - nearest <= _TOL)
    return {
        "recruited": recruited,
        "ratios": ratios,
        "log_distance": log_distance,
        "closest": tie[0],
        "closest_tie": tie,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecruitedDimension:
    """The recruited-dimension instrument's output for one model on one group.

    ``theoretical`` holds the three anchors; the ``*_effective_rank`` fields hold
    the measured effective ranks (centred and uncentred) for the embedding and,
    when the model has an MLP, the left/right activation codes. ``primary_measure``
    names which measure feeds the headline ``discrimination`` (centred
    activation-left when available, else centred embedding), and ``discrimination``
    compares that measure's participation ratio to the anchors
    (:func:`discriminate`). ``activation_available`` is False for ``use_mlp=false``
    models, whose activation measures are ``None``."""

    canonical_id: tuple[int, int]
    theoretical: TheoreticalDimensions
    embedding_centered: EffectiveRank
    embedding_uncentered: EffectiveRank
    activation_available: bool
    activation_left_centered: EffectiveRank | None
    activation_left_uncentered: EffectiveRank | None
    activation_right_centered: EffectiveRank | None
    activation_right_uncentered: EffectiveRank | None
    primary_measure: str
    primary_recruited: float
    discrimination: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        def opt(rank: EffectiveRank | None) -> dict[str, Any] | None:
            return rank.to_record() if rank is not None else None

        return {
            "instrument": "recruited-dimension",
            "canonical_id": list(self.canonical_id),
            "theoretical": self.theoretical.to_record(),
            "embedding_centered": self.embedding_centered.to_record(),
            "embedding_uncentered": self.embedding_uncentered.to_record(),
            "activation_available": self.activation_available,
            "activation_left_centered": opt(self.activation_left_centered),
            "activation_left_uncentered": opt(self.activation_left_uncentered),
            "activation_right_centered": opt(self.activation_right_centered),
            "activation_right_uncentered": opt(self.activation_right_uncentered),
            "primary_measure": self.primary_measure,
            "primary_recruited": self.primary_recruited,
            "discrimination": self.discrimination,
        }


def _has_mlp(model: GroupModel) -> bool:
    """Whether the model exposes MLP neuron activations. ``OneLayerTransformer``
    carries an explicit ``use_mlp`` flag; ``FCModel`` always has a hidden layer."""
    return bool(getattr(model, "use_mlp", True))


def measure_recruited_dimension(
    model: GroupModel,
    group: FiniteGroup,
    *,
    batch_size: int = 8192,
    device: torch.device = torch.device("cpu"),
    max_cardinality: int = MAX_FAITHFUL_SET_SEARCH_CARDINALITY,
) -> RecruitedDimension:
    """Measure the model's recruited/effective dimension and compare it to the
    three theoretical anchors (see :class:`RecruitedDimension` and the module
    docstring). ``device`` moves only the activation forward pass; every
    effective-rank computation is CPU float64.

    The primary recruited scalar is the centred activation-left participation
    ratio when the model has an MLP, else the centred embedding participation
    ratio -- the discrimination is reported against that, with all other measures
    carried for robustness. The tensor-rank anchor used in the discrimination is
    the unconditional lower bound; because it coincides with ``|G|`` for
    essentially every group, a recruited dimension near ``|G|`` reads as a tie
    between the tensor-rank and uses-everything accounts (see the module
    docstring)."""
    theoretical = theoretical_dimensions(group, max_cardinality=max_cardinality)

    embedding_centered = embedding_effective_rank(model, group, center=True)
    embedding_uncentered = embedding_effective_rank(model, group, center=False)

    activation_available = _has_mlp(model)
    activation_left_centered: EffectiveRank | None = None
    activation_left_uncentered: EffectiveRank | None = None
    activation_right_centered: EffectiveRank | None = None
    activation_right_uncentered: EffectiveRank | None = None
    if activation_available:
        activations = neuron_activations(model, group.order, batch_size=batch_size, device=device)
        activation_left_centered = effective_rank_of_matrix(
            _argument_code_matrix(activations, group.order, "left", center=True)
        )
        activation_left_uncentered = effective_rank_of_matrix(
            _argument_code_matrix(activations, group.order, "left", center=False)
        )
        activation_right_centered = effective_rank_of_matrix(
            _argument_code_matrix(activations, group.order, "right", center=True)
        )
        activation_right_uncentered = effective_rank_of_matrix(
            _argument_code_matrix(activations, group.order, "right", center=False)
        )

    if activation_available:
        assert activation_left_centered is not None
        primary_measure = "activation_left_centered"
        primary = activation_left_centered
    else:
        primary_measure = "embedding_centered"
        primary = embedding_centered
    primary_recruited = primary.participation_ratio

    anchors = {
        "minimal_faithful_real": float(theoretical.minimal_faithful_real.real_dimension),
        "tensor_rank_lower": float(theoretical.tensor_rank_lower),
        "regular_rep": float(theoretical.regular_rep),
    }
    discrimination = discriminate(primary_recruited, anchors)

    return RecruitedDimension(
        canonical_id=group.canonical_id,
        theoretical=theoretical,
        embedding_centered=embedding_centered,
        embedding_uncentered=embedding_uncentered,
        activation_available=activation_available,
        activation_left_centered=activation_left_centered,
        activation_left_uncentered=activation_left_uncentered,
        activation_right_centered=activation_right_centered,
        activation_right_uncentered=activation_right_uncentered,
        primary_measure=primary_measure,
        primary_recruited=primary_recruited,
        discrimination=discrimination,
    )


__all__ = [
    "FS_COMPLEX",
    "FS_QUATERNIONIC",
    "FS_REAL",
    "EffectiveRank",
    "MinimalFaithfulReal",
    "RecruitedDimension",
    "TheoreticalDimensions",
    "activation_effective_rank",
    "block_real_dimension",
    "discriminate",
    "effective_rank_from_singular_values",
    "effective_rank_of_matrix",
    "embedding_effective_rank",
    "frobenius_schur_indicator",
    "measure_recruited_dimension",
    "minimal_faithful_real_dimension",
    "real_irreducible_dimensions",
    "theoretical_dimensions",
]
