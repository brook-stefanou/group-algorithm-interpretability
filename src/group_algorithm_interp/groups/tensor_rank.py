"""Group-algebra multiplication-tensor rank bounds, derived from character degrees.

Two 2026 preprints (Shutman et al. 2509.06931; Notsawo et al. 2602.19533) claim
structure-tensor rank governs learning difficulty. For a *group algebra*
``C[G]`` the situation is mathematically special, and this module states it
precisely.

**Wedderburn.** ``C[G] cong bigoplus_i M_{d_i}(C)``, where the ``d_i`` are the
character degrees of ``G`` (one Wedderburn factor per irreducible
representation; ``sum_i d_i^2 = |G|``, the standard Burnside dimension
identity). So the multiplication tensor of the group algebra is a **direct sum
of matrix-multiplication tensors** ``bigoplus_i <d_i, d_i, d_i>`` -- one per
irrep, not one per distinct degree.

**Exact rank is open for d >= 3.** ``R(<d,d,d>)``, the bilinear/multiplicative
complexity of ``d x d`` matrix multiplication, is known EXACTLY only for
``d = 1`` (trivially 1, scalar multiplication) and ``d = 2`` (7, Strassen
1969, "Gaussian elimination is not optimal", Numer. Math. 13). For ``d >= 3``
the exact value is **open** -- it is the matrix-multiplication-exponent
problem -- so no exact rank exists for a group with an irrep of degree >= 3, and
bounds are the best available.

**Upper bounds are unconditionally subadditive.** For a direct sum of tensors
on disjoint underlying coordinate spaces, ``R(T_1 oplus T_2) <= R(T_1) +
R(T_2)`` always: run each summand's optimal algorithm independently on its own
block and take the union. No conjecture is needed. This module's
``tensor_rank_upper_bound`` sums a per-irrep upper bound and is therefore a
valid, unconditional upper bound on the true rank.

**A lower bound that IS additive, unconditionally: flattening rank.** If ``T =
sum_{r=1}^R a_r ox b_r ox c_r`` then any matricization ("flattening") of ``T``
has matrix rank <= R (standard fact, e.g. Landsberg, *Tensors: Geometry and
Applications*, GSM 128, AMS 2012). Because the Wedderburn blocks live on
disjoint coordinate subspaces in every mode, the flattening matrix of the full
tensor is **block-diagonal**, and ordinary matrix rank of a block-diagonal
matrix is EXACTLY (not just boundedly) the sum of the blocks' ranks -- this is
elementary linear algebra, not a tensor-rank conjecture. Every flattening of
``<d,d,d>`` (matrix multiplication is a "1-generic" tensor) has rank exactly
``d^2``. Summing over the Wedderburn blocks: the flattening rank of the whole
group-algebra tensor is exactly ``sum_i d_i^2 = |G|``. So ``R(C[G]) >= |G|``
**unconditionally, exactly, for every finite group** -- this is
``tensor_rank_lower_bound``'s dominant term, and for abelian ``G`` (all
``d_i = 1``) it is tight: ``R(C[G]) = |G|`` exactly, matching the elementary
fact that ``C[G] cong C^{|G|}`` for abelian ``G`` (the tensor is diagonal, and
a diagonal/"unit" tensor of format ``|G| x |G| x |G|`` has rank exactly
``|G|``).

The lower bound also folds in a per-block bound from restriction: since
restricting a bilinear algorithm to one block cannot increase rank,
``R(bigoplus_i T_i) >= R(T_j)`` for every block ``j`` individually (this is
unconditional too -- no additivity needed, just "one summand is a restriction
of the whole"). For ``d >= 3`` the best simple unconditional per-block bound
used here is Blaser's `R(<d,d,d>) >= (5/2) d^2 - 3d` (Markus Blaser, "A 5/2 n^2
Lower Bound for the Multiplicative Complexity of n x n Matrix Multiplication",
FOCS 1999, pp. 45-50; holds over any field, for all n). ``tensor_rank_lower_
bound`` is the max of the flattening bound and this per-block bound, so it is
never weaker than either.

**What is NOT valid: summing per-block lower bounds.** Doing so would require
``R(bigoplus_i T_i) >= sum_i R(T_i)`` -- the lower-bound half of Strassen's
additivity conjecture. Shitov (2019, "Strassen's additivity conjecture is
false", disproving the general conjecture over infinite fields for genuine
direct sums with disjoint coordinate spaces) showed this can FAIL: the whole
can have strictly smaller rank than the sum of its parts' true ranks. This
module therefore reports that quantity separately, as ``tensor_rank_naive_
additive_lower_estimate``, and it is documented as NOT a proven lower bound on
the true rank -- an assumption-dependent number for context only, never to be
compared against as if it were validated.

**Upper-bound construction.** ``R(<d,d,d>)`` for ``d in {1, 2, 3}`` uses the
known best values (1; 7, Strassen 1969; 23, Laderman 1976, "A noncommutative
algorithm for multiplying 3x3 matrices using 23 multiplications", Bull. AMS
82). For other ``d`` this module combines two unconditional facts: (a)
submultiplicativity under the Kronecker/block-recursive construction,
``R(<ab,ab,ab>) <= R(<a,a,a>) * R(<b,b,b>)`` (this is literally how Strassen's
algorithm is applied recursively: view a ``(ab) x (ab)`` product as a ``b x b``
block matrix product where each "scalar" multiplication is itself an ``a x a``
block, so an ``R(a)``-multiplication algorithm nested inside an
``R(b)``-multiplication algorithm gives ``R(a) * R(b)`` total; standard, e.g.
Buergisser/Clausen/Shokrollahi, *Algebraic Complexity Theory*, Springer 1997);
and (b) monotonicity under restriction/padding, ``R(<d,d,d>) <= R(<d',d',d'>)``
for ``d' >= d`` (embed as a submatrix padded with zero rows/columns). This
upper bound is not the tightest published bound for every ``d``; sharper
special-purpose decompositions exist in the literature for some individual
``d`` and are not chased here.

**Critical consequence for this project: tensor rank is NOT an independent
axis.** All of the bounds above are deterministic functions of the multiset of
character degrees alone -- the same input that determines
``fourier_block_cost = sum_i d_i^3`` (see ``scripts/enumerate_groups.py``).
Both quantities are monotonic transforms of the same underlying irrep-degree
multiset, so they are collinear by construction. **Tensor rank bounds must
never enter a distance function or be treated as an independent predictor
column** -- the same error class the project already banned for
``chief_factor_split`` (see ``scripts/enumerate_groups.py``'s column-semantics
header).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Exact known values of R(<d,d,d>), the bilinear rank of d x d matrix
# multiplication. Only d in {1, 2} are exactly known; see module docstring.
# d=1: trivial (scalar multiplication). d=2: Strassen 1969.
_EXACT_RANK: dict[int, int] = {1: 1, 2: 7}

# Best-known small-case upper bounds used as base cases for the recursive
# upper-bound construction. d=3: Laderman 1976.
_UPPER_BASE: dict[int, int] = {1: 1, 2: 7, 3: 23}

# Upper-bound recursion is only ever queried for the character degrees present
# in the panel (max observed degree is 15 for the order-21..255 range this
# project uses), but the table is built generously past that so the module
# does not silently need updating if the panel grows.
_UPPER_TABLE_CEILING = 128


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _build_upper_table(ceiling: int) -> dict[int, int]:
    """DP table of ``R(<d,d,d>) <=`` this value, for ``d`` up to ``ceiling``,
    built from ``_UPPER_BASE`` via submultiplicativity over divisor pairs.
    Entries are only populated where a decomposition exists (composite ``d``
    or a base case); ``irrep_rank_upper_bound`` handles padding for the rest.
    """
    table = dict(_UPPER_BASE)
    for d in range(4, ceiling + 1):
        best: int | None = None
        for a in range(2, d):
            if d % a != 0:
                continue
            b = d // a
            if a in table and b in table:
                candidate = table[a] * table[b]
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            table[d] = best
    return table


_UPPER_TABLE = _build_upper_table(_UPPER_TABLE_CEILING)


def irrep_rank_lower_bound(d: int) -> int:
    """Unconditional lower bound on ``R(<d,d,d>)``.

    ``d = 1``: exact (1). ``d = 2``: exact (7, Strassen 1969). ``d >= 3``:
    Blaser's bound, ``ceil((5/2) d^2 - 3d)`` (FOCS 1999; holds over any
    field). This is a *per-block* bound -- see the module docstring for why it
    must not simply be summed across the irreps of a group.
    """
    if d < 1:
        raise ValueError(f"irrep degree must be >= 1, got {d}")
    if d in _EXACT_RANK:
        return _EXACT_RANK[d]
    return math.ceil(2.5 * d * d - 3 * d)


def irrep_rank_upper_bound(d: int) -> int:
    """Unconditional upper bound on ``R(<d,d,d>)`` -- see module docstring for
    the submultiplicative + padding construction. ``d = 1, 2, 3`` are the
    known best small-case values (1, 7, 23)."""
    if d < 1:
        raise ValueError(f"irrep degree must be >= 1, got {d}")
    if d > _UPPER_TABLE_CEILING:
        limit = _next_power_of_two(d)
        table = _build_upper_table(limit)
    else:
        limit = _UPPER_TABLE_CEILING
        table = _UPPER_TABLE
    candidates = [table[dp] for dp in range(d, limit + 1) if dp in table]
    return min(candidates)


def group_algebra_tensor_rank_bounds(character_degrees: Sequence[int]) -> dict[str, int]:
    """Rank bounds for the multiplication tensor of ``C[G]``, derived from
    ``character_degrees`` alone (Wedderburn: one ``<d,d,d>`` block per irrep).

    Returns a dict with:

    ``tensor_rank_lower_bound``
        Unconditional. ``max(sum_i d_i^2, max_i irrep_rank_lower_bound(d_i))``
        -- the exact block-diagonal flattening-rank bound (equals ``|G|``
        always, since ``sum_i d_i^2 = |G|``) and the restriction bound from
        the single hardest block, whichever is larger. Tight for abelian
        groups (``= |G|`` exactly).

    ``tensor_rank_upper_bound``
        Unconditional. ``sum_i irrep_rank_upper_bound(d_i)`` -- valid by
        subadditivity of rank over direct sums (run each block's algorithm
        independently).

    ``tensor_rank_naive_additive_lower_estimate``
        ``sum_i irrep_rank_lower_bound(d_i)``. **NOT a proven lower bound on
        the true rank** -- it assumes Strassen's additivity conjecture, which
        Shitov (2019) showed is false in general. Reported for context only;
        see module docstring.

    Raises ``ValueError`` if ``character_degrees`` is empty or contains a
    non-positive value.
    """
    degrees = list(character_degrees)
    if not degrees:
        raise ValueError("character_degrees must be non-empty")
    if any(d < 1 for d in degrees):
        raise ValueError(f"all character degrees must be >= 1, got {degrees}")

    flattening_bound = sum(d * d for d in degrees)
    max_block_lower = max(irrep_rank_lower_bound(d) for d in degrees)
    lower_bound = max(flattening_bound, max_block_lower)
    upper_bound = sum(irrep_rank_upper_bound(d) for d in degrees)
    naive_additive_lower_estimate = sum(irrep_rank_lower_bound(d) for d in degrees)

    return {
        "tensor_rank_lower_bound": lower_bound,
        "tensor_rank_upper_bound": upper_bound,
        "tensor_rank_naive_additive_lower_estimate": naive_additive_lower_estimate,
    }


__all__ = [
    "group_algebra_tensor_rank_bounds",
    "irrep_rank_lower_bound",
    "irrep_rank_upper_bound",
]
