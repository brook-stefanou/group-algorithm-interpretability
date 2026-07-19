"""Zero-dependency stats helpers."""

import itertools
import math
import random
import statistics

import pytest
from scipy import stats as scipy_stats

from group_algorithm_interp.stats import (
    betainc_reg,
    bootstrap_ci,
    mean_std,
    permutation_test,
    sign_test,
    welch_ttest,
)


def test_mean_std():
    mean, std = mean_std([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    assert std == pytest.approx(statistics.stdev([1.0, 2.0, 3.0, 4.0]))
    assert mean_std([5.0]) == (5.0, 0.0)  # single value -> std 0
    with pytest.raises(ValueError):
        mean_std([])


def test_mean_std_all_identical_values_with_n_greater_than_one():
    # Boundary: zero-width sample that isn't a singleton -- std must be exactly
    # 0.0, not something numerically-near-zero.
    assert mean_std([3.0, 3.0, 3.0, 3.0]) == (3.0, 0.0)


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    lo, hi = bootstrap_ci(vals, n_resamples=2000, seed=0)
    assert lo < statistics.fmean(vals) < hi
    assert (lo, hi) == bootstrap_ci(vals, n_resamples=2000, seed=0)  # seeded => deterministic
    # a constant sample has a degenerate CI
    assert bootstrap_ci([5.0, 5.0, 5.0], n_resamples=200) == (5.0, 5.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])


def test_bootstrap_ci_does_not_touch_global_random_state():
    # bootstrap_ci's docstring promises a private seeded RNG that leaves
    # global state untouched.
    state_before = random.getstate()
    bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_resamples=500, seed=0)
    assert random.getstate() == state_before


def test_bootstrap_ci_minimum_size_two():
    # n=2 is the smallest input bootstrap_ci accepts -- every resample is one
    # of {a, b, mean(a,b)}, so the CI is necessarily a subset of that range.
    lo, hi = bootstrap_ci([1.0, 3.0], n_resamples=500, seed=0)
    assert 1.0 <= lo <= hi <= 3.0


def test_bootstrap_ci_empty_confidence_collapses_to_a_point():
    # confidence=0.0 -> alpha=0.5 -> lo and hi indices coincide (both at the
    # median resample), so the interval collapses to a single point rather
    # than raising or inverting (lo > hi).
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0], confidence=0.0, n_resamples=200, seed=0)
    assert lo == hi


def test_sign_test():
    all_pos = sign_test([0.1, 0.2, 0.3])
    assert all_pos["n"] == 3 and all_pos["n_positive"] == 3
    assert all_pos["p_two_sided"] == pytest.approx(0.25)  # 2 * C(3,0) * 0.5^3
    mixed = sign_test([1.0, -1.0, 1.0, 1.0])
    assert mixed["n"] == 4 and mixed["n_positive"] == 3
    assert mixed["p_two_sided"] == pytest.approx(0.625)  # 2 * (C(4,0)+C(4,1)) * 0.5^4
    ties = sign_test([0.0, 0.0], zero_tol=0.0)
    assert ties["n"] == 0 and ties["p_two_sided"] == 1.0


def test_sign_test_empty_input():
    r = sign_test([])
    assert r == {"n": 0.0, "n_positive": 0.0, "p_two_sided": 1.0}


def test_sign_test_all_ties_with_nonzero_tolerance():
    # every |diff| is within zero_tol -> all dropped as ties, not misclassified
    # as positive/negative.
    r = sign_test([0.01, -0.02, 0.03], zero_tol=0.05)
    assert r == {"n": 0.0, "n_positive": 0.0, "p_two_sided": 1.0}


def test_sign_test_single_nonzero_difference():
    # n=1 is the minimum nondegenerate input: one sign, necessarily maximal
    # p-value.
    r = sign_test([0.5])
    assert r["n"] == 1.0 and r["n_positive"] == 1.0
    assert r["p_two_sided"] == pytest.approx(1.0)


def test_identical_groups_give_zero_t_and_p_one():
    r = welch_ttest([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert r["t"] == 0.0
    assert r["p_two_sided"] == pytest.approx(1.0)


def test_clearly_separated_groups_are_significant():
    r = welch_ttest([1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
    assert r["t"] < 0  # first group's mean is lower
    assert r["p_two_sided"] < 0.01
    assert 0.0 <= r["p_two_sided"] <= 1.0
    sp = scipy_stats.ttest_ind([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], equal_var=False)
    assert r["t"] == pytest.approx(sp.statistic)
    assert r["dof"] == pytest.approx(sp.df)
    assert r["p_two_sided"] == pytest.approx(sp.pvalue)


def test_requires_two_observations_per_group():
    with pytest.raises(ValueError):
        welch_ttest([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        welch_ttest([1.0, 2.0], [3.0])


# -- Boundary cases for the zero-within-group-variance bug -----------------
#
# A plain Welch t-test on saturated accuracies (every seed hits exactly 0.0 or
# 1.0) has zero within-group variance in one or both groups. The previous
# implementation silently returned t=0.0, p=1.0 ("no difference") whenever
# *both* groups were zero-variance, even when the groups were perfectly
# separated -- exactly the shape a real grokking-style result produces. It
# also reported the pooled dof (na+nb-2) rather than a Welch-Satterthwaite
# value in that branch.


def test_both_zero_variance_different_means_raises():
    # Perfect separation: all of group a hit 1, all of group b hit 2. scipy
    # reports t=-inf, p=0.0 here -- technically consistent but the wrong tool,
    # so this raises instead of returning a number that invites misreporting.
    with pytest.raises(ValueError, match="degenerate"):
        welch_ttest([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    sp = scipy_stats.ttest_ind([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], equal_var=False)
    assert math.isinf(sp.statistic) and sp.pvalue == 0.0  # documents the scipy contrast


def test_both_zero_variance_realistic_saturation_case_raises():
    # The project's realistic case: all 5 seeds of one group generalize
    # (test accuracy 1.0), none of the other group's seeds do (0.0).
    with pytest.raises(ValueError, match="degenerate"):
        welch_ttest([1.0] * 5, [0.0] * 5)


def test_both_zero_variance_identical_means_is_genuine_no_difference():
    # Both groups are the same nonzero-length constant -- a legitimate
    # "no difference" input, not an instrument failure. dof should equal the
    # pooled na+nb-2 when na == nb (the scale-invariant Welch-Satterthwaite
    # limit collapses to the pooled value in that case).
    r = welch_ttest([7.0, 7.0, 7.0], [7.0, 7.0, 7.0])
    assert r["t"] == 0.0
    assert r["p_two_sided"] == 1.0
    assert r["dof"] == pytest.approx(4.0)  # na + nb - 2 for na == nb == 3


def test_both_zero_variance_identical_means_unequal_sizes():
    # Same degenerate-identical case but with na != nb, where the dof is *not*
    # the pooled na+nb-2 -- exercises the general Welch-Satterthwaite limit
    # formula rather than the coincidental equal-n special case above.
    r = welch_ttest([2.0] * 5, [2.0] * 3)
    assert r["t"] == 0.0
    assert r["p_two_sided"] == 1.0
    expected_dof = (1.0 / 5 + 1.0 / 3) ** 2 / ((1.0 / 25) / 4 + (1.0 / 9) / 2)
    assert r["dof"] == pytest.approx(expected_dof)
    assert r["dof"] != pytest.approx(6.0)  # na + nb - 2 would be wrong here


def test_one_group_zero_variance_other_not():
    # Only one group is degenerate -- se2 is still strictly positive (driven
    # entirely by the non-degenerate group), so this is not a special case at
    # all and must match scipy exactly.
    a = [1.0, 1.0, 1.0, 1.0]
    b = [0.0, 1.0, 2.0, 3.0]
    r = welch_ttest(a, b)
    sp = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert r["t"] == pytest.approx(sp.statistic)
    assert r["dof"] == pytest.approx(sp.df)
    assert r["p_two_sided"] == pytest.approx(sp.pvalue)


def test_minimum_two_observations_with_zero_variance():
    # n=2 is the smallest sample that has a defined (here: zero) variance.
    with pytest.raises(ValueError, match="degenerate"):
        welch_ttest([1.0, 1.0], [2.0, 2.0])
    r = welch_ttest([1.0, 1.0], [1.0, 1.0])
    assert r["t"] == 0.0
    assert r["p_two_sided"] == 1.0


def test_near_degenerate_tiny_variance_does_not_raise():
    # Variance is nonzero but tiny -- must take the normal Welch path (huge
    # |t|, tiny dof-appropriate p), not the zero-variance branch, and must
    # match scipy.
    a = [1.0, 1.0, 1.0 + 1e-9]
    b = [2.0, 2.0, 2.0 - 1e-9]
    r = welch_ttest(a, b)
    sp = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert r["t"] == pytest.approx(sp.statistic)
    assert r["dof"] == pytest.approx(sp.df)
    assert r["p_two_sided"] == pytest.approx(sp.pvalue, abs=1e-8)
    assert r["p_two_sided"] < 1e-6


def test_betainc_reg_boundaries():
    assert betainc_reg(2.0, 3.0, 0.0) == 0.0
    assert betainc_reg(2.0, 3.0, 1.0) == 1.0
    mid = betainc_reg(2.0, 2.0, 0.5)
    assert mid == pytest.approx(0.5, abs=1e-9)  # symmetric case


def test_betainc_reg_matches_scipy_off_boundary():
    assert betainc_reg(3.0, 5.0, 0.3) == pytest.approx(
        scipy_stats.beta.cdf(0.3, 3.0, 5.0), abs=1e-10
    )


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------


def _brute_force_permutation_p(a, b):
    """Independent (non-shared-code) exact enumeration, for cross-checking
    ``permutation_test``'s own exact branch."""
    pooled = list(a) + list(b)
    n = len(pooled)
    na = len(a)
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    hits = 0
    total = 0
    for combo in itertools.combinations(range(n), na):
        in_a = set(combo)
        group_a = [pooled[i] for i in combo]
        group_b = [pooled[i] for i in range(n) if i not in in_a]
        total += 1
        if abs(statistics.fmean(group_a) - statistics.fmean(group_b)) >= observed:
            hits += 1
    return hits / total


def test_permutation_test_matches_brute_force_enumeration():
    a = [1.0, 2.0, 5.0]
    b = [3.0, 8.0, 9.0, 10.0]
    result = permutation_test(a, b)
    assert result["exact"] == 1.0
    assert result["p_two_sided"] == pytest.approx(_brute_force_permutation_p(a, b))


def test_permutation_test_p_value_is_in_unit_interval():
    result = permutation_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert 0.0 <= result["p_two_sided"] <= 1.0


def test_permutation_test_identical_inputs_give_p_approx_one():
    # Every relabeling of a constant pool has the same (zero) statistic, so the
    # observed labeling is exactly as extreme as all of them -- p is exactly 1,
    # not just close to it.
    result = permutation_test([5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0])
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_permutation_test_perfect_separation_gives_minimum_p_not_an_error():
    # Non-overlapping ranges: only the observed labeling and its label-swapped
    # complement reach the maximum |mean diff| among all C(6,3)=20 relabelings,
    # so the minimum achievable exact p-value here is 2/20 -- not zero, and not
    # a raised exception (unlike welch_ttest on the analogous saturated case).
    a = [1.0, 2.0, 3.0]
    b = [100.0, 101.0, 102.0]
    result = permutation_test(a, b)
    assert result["exact"] == 1.0
    assert result["n_permutations"] == pytest.approx(20.0)
    assert result["p_two_sided"] == pytest.approx(2.0 / 20.0)
    assert result["p_two_sided"] == pytest.approx(_brute_force_permutation_p(a, b))


def test_permutation_test_saturated_accuracy_vectors_do_not_raise():
    # The motivating case: welch_ttest raises here (zero within-group variance,
    # different means). permutation_test must handle it and report a small,
    # non-zero p rather than the smallest-possible p being unreachable.
    generalizing = [1.0] * 5
    not_generalizing = [0.0] * 5
    with pytest.raises(ValueError, match="degenerate"):
        welch_ttest(generalizing, not_generalizing)
    result = permutation_test(generalizing, not_generalizing)
    assert result["p_two_sided"] < 0.05


def test_permutation_test_exact_to_sampling_boundary():
    # C(10, 5) = 252: pin the exact/sampling switch at a threshold we control,
    # rather than relying on the (much larger) default.
    a = list(range(5))
    b = list(range(5, 10))
    exact = permutation_test(a, b, exact_threshold=252)
    assert exact["exact"] == 1.0
    assert exact["n_permutations"] == pytest.approx(252.0)

    sampled = permutation_test(a, b, exact_threshold=251, n_resamples=500, seed=0)
    assert sampled["exact"] == 0.0
    assert sampled["n_permutations"] == pytest.approx(500.0)

    # Same seed -> identical result (private RNG, not wall-clock/global state).
    sampled_again = permutation_test(a, b, exact_threshold=251, n_resamples=500, seed=0)
    assert sampled == sampled_again


def test_permutation_test_does_not_touch_global_random_state():
    state_before = random.getstate()
    # Force the sampling branch so the private RNG path actually runs.
    permutation_test(list(range(5)), list(range(5, 10)), exact_threshold=1, n_resamples=200, seed=0)
    assert random.getstate() == state_before


def test_permutation_test_requires_nonempty_groups():
    with pytest.raises(ValueError):
        permutation_test([], [1.0])
    with pytest.raises(ValueError):
        permutation_test([1.0], [])
