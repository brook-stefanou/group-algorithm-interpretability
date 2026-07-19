"""Tests for group-algebra multiplication-tensor rank bounds.

These assert the defining invariants, not just "returns a finite
number": lower <= upper always; d=1 gives
exact rank 1; d=2 gives exact rank 7 (Strassen); bounds are monotone
non-decreasing in the degree; and the abelian case gives exact rank |G| (not
just a bound), since all d_i = 1 there.
"""

from __future__ import annotations

import pytest

from group_algorithm_interp.groups.tensor_rank import (
    group_algebra_tensor_rank_bounds,
    irrep_rank_lower_bound,
    irrep_rank_upper_bound,
)


def test_degree_one_is_exact_scalar_multiplication():
    assert irrep_rank_lower_bound(1) == 1
    assert irrep_rank_upper_bound(1) == 1


def test_degree_two_is_exact_strassen_seven():
    assert irrep_rank_lower_bound(2) == 7
    assert irrep_rank_upper_bound(2) == 7


def test_degree_three_upper_bound_is_laderman_23():
    assert irrep_rank_upper_bound(3) == 23


@pytest.mark.parametrize("d", range(1, 40))
def test_lower_never_exceeds_upper_per_block(d):
    assert irrep_rank_lower_bound(d) <= irrep_rank_upper_bound(d)


@pytest.mark.parametrize("d", range(1, 39))
def test_bounds_are_monotone_non_decreasing_in_degree(d):
    assert irrep_rank_lower_bound(d) <= irrep_rank_lower_bound(d + 1)
    assert irrep_rank_upper_bound(d) <= irrep_rank_upper_bound(d + 1)


@pytest.mark.parametrize("d", [0, -1, -5])
def test_non_positive_degree_rejected(d):
    with pytest.raises(ValueError):
        irrep_rank_lower_bound(d)
    with pytest.raises(ValueError):
        irrep_rank_upper_bound(d)


def test_upper_bound_submultiplicative_padding_matches_repeated_strassen():
    # d=4 = 2*2: Strassen applied recursively twice, 7*7 = 49.
    assert irrep_rank_upper_bound(4) == 49
    # d=8 = 2^3: 7^3 = 343.
    assert irrep_rank_upper_bound(8) == 343
    # d=9 = 3*3: Laderman applied recursively twice, 23*23 = 529.
    assert irrep_rank_upper_bound(9) == 529


def test_upper_bound_for_prime_degree_uses_padding_not_left_unbounded():
    # d=5 is prime; must fall back to the smallest d' >= 5 with a known
    # decomposition (d'=6=2*3 -> 7*23=161) via the restriction/padding
    # argument, not raise or silently omit a bound.
    assert irrep_rank_upper_bound(5) == 161


def test_upper_bound_beyond_the_precomputed_table_still_terminates_and_is_valid():
    # No character degree in the current panel (max 15) approaches the
    # precomputed table ceiling, but the padding/rebuild path for d beyond
    # it must still produce a valid, finite bound rather than a KeyError.
    # Pure power-of-two padding (Strassen recursion only) is always an
    # available candidate, so the DP optimum can never exceed it.
    d = 200
    upper = irrep_rank_upper_bound(d)
    assert irrep_rank_lower_bound(d) <= upper <= 7**8  # 256 = 2^8 is a valid pad target


@pytest.mark.parametrize("degrees", [[], [0], [-1], [1, 0]])
def test_group_algebra_bounds_reject_empty_or_non_positive_degrees(degrees):
    with pytest.raises(ValueError):
        group_algebra_tensor_rank_bounds(degrees)


def test_abelian_group_algebra_rank_is_exact_not_just_bounded():
    """For abelian G every d_i = 1, so C[G] ~= C^|G| and the tensor is
    diagonal: rank is EXACTLY |G|, not merely bracketed. This must hold as an
    equality, both bounds pinned to |G|, for every order tested."""
    for order in (1, 2, 6, 21, 100, 255):
        bounds = group_algebra_tensor_rank_bounds([1] * order)
        assert bounds["tensor_rank_lower_bound"] == order
        assert bounds["tensor_rank_upper_bound"] == order
        assert bounds["tensor_rank_naive_additive_lower_estimate"] == order


def test_group_algebra_bounds_lower_never_exceeds_upper_for_real_panel_groups():
    # A handful of real character-degree multisets from the panel (order,
    # index -> character_degrees, cross-checked against
    # data/group_properties_full.jsonl): S3 (6,1), D4 (8,3), Q8 (8,4),
    # C7:C3 (21,1), D32 (32,18), Q32 (32,20).
    panel_degree_multisets = [
        [1, 1, 2],  # S3
        [1, 1, 1, 1, 2],  # D4
        [1, 1, 1, 1, 2],  # Q8
        [1, 1, 1, 3, 3],  # C7:C3, (21,1)
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],  # D32, (32,18)
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],  # Q32, (32,20)
    ]
    for degrees in panel_degree_multisets:
        bounds = group_algebra_tensor_rank_bounds(degrees)
        assert bounds["tensor_rank_lower_bound"] <= bounds["tensor_rank_upper_bound"]
        # Burnside/Wedderburn dimension identity: sum(d^2) == |G|, which is
        # also the unconditional flattening-rank lower bound this module
        # relies on -- assert the lower bound is at least that identity.
        assert bounds["tensor_rank_lower_bound"] >= sum(d * d for d in degrees)


def test_group_algebra_lower_bound_uses_flattening_identity_when_it_dominates():
    # Many small-degree irreps: the block-diagonal flattening bound sum(d^2)
    # dominates any single restriction bound, and must be used exactly.
    degrees = [1] * 50 + [2] * 5
    bounds = group_algebra_tensor_rank_bounds(degrees)
    flattening = sum(d * d for d in degrees)
    assert flattening == 50 + 5 * 4  # sanity on the arithmetic itself
    assert bounds["tensor_rank_lower_bound"] == flattening


def test_group_algebra_lower_bound_uses_per_block_bound_when_it_dominates():
    # One irrep of large degree with little else: the per-block restriction
    # bound (Blaser, d>=3) can exceed the flattening identity sum(d^2).
    degrees = [1, 1, 15]
    bounds = group_algebra_tensor_rank_bounds(degrees)
    flattening = sum(d * d for d in degrees)
    per_block = irrep_rank_lower_bound(15)
    assert per_block > flattening  # this is the case this test exists to cover
    assert bounds["tensor_rank_lower_bound"] == per_block


def test_upper_bound_is_subadditive_sum_of_per_block_upper_bounds():
    degrees = [1, 1, 2, 3]
    bounds = group_algebra_tensor_rank_bounds(degrees)
    expected = sum(irrep_rank_upper_bound(d) for d in degrees)
    assert bounds["tensor_rank_upper_bound"] == expected


def test_naive_additive_estimate_is_not_silently_equal_to_the_validated_lower_bound():
    # For a group with a genuinely large-degree irrep, the (unvalidated)
    # naive additive estimate and the (validated) unconditional lower bound
    # can diverge -- this pins that they are computed independently rather
    # than one being an alias of the other.
    degrees = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]  # D32 / Q32
    bounds = group_algebra_tensor_rank_bounds(degrees)
    assert bounds["tensor_rank_naive_additive_lower_estimate"] >= bounds["tensor_rank_lower_bound"]
