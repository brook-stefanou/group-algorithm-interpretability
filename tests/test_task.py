"""Tests for group_algorithm_interp.task — GroupTask, TrainTestSplit, build_group_task, train_test_split."""

import numpy as np
import pytest

from group_algorithm_interp.groups.data import load_group
from group_algorithm_interp.task import (
    GroupTask,
    TrainTestSplit,
    build_group_task,
    commuting_probability,
    train_test_split,
    transpose_leak_fraction,
    transpose_leaked_mask,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def group():
    """Load the C8 group artifact (order 8, index 1)."""
    return load_group(8, 1)


@pytest.fixture(scope="module")
def task(group):
    """Pre-built GroupTask for the C8 group."""
    return build_group_task(group)


def _task_of_size(n: int, group_order: int = 2) -> GroupTask:
    """A GroupTask with `n` examples; contents are irrelevant to the split."""
    return GroupTask(
        group_order=group_order,
        inputs=np.arange(n * 2).reshape(n, 2),
        targets=np.arange(n),
    )


def test_task_and_split_are_immutable(task):
    """Frozen dataclasses: a run's data cannot be mutated out from under it.
    Construction, repr, and field comparison are @dataclass's own job and
    aren't retested here."""
    with pytest.raises(Exception):
        task.group_order = 16  # type: ignore[misc]

    split = train_test_split(task, train_frac=0.5, seed=42)
    assert isinstance(split, TrainTestSplit)
    with pytest.raises(Exception):
        split.train_inputs = np.array([[9, 9]])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_group_task
# ---------------------------------------------------------------------------


class TestBuildGroupTask:
    def test_group_order(self, task, group):
        assert task.group_order == 8
        assert task.group_order == group.order

    def test_inputs_shape(self, task, group):
        n = group.order
        assert task.inputs.shape == (n * n, 2)

    def test_targets_shape(self, task, group):
        n = group.order
        assert task.targets.shape == (n * n,)

    def test_targets_match_cayley_table_flat(self, task, group):
        expected = group.cayley_table.reshape(-1).copy()
        np.testing.assert_array_equal(task.targets, expected)

    def test_inputs_cover_all_ordered_pairs(self, task, group):
        n = group.order
        expected_pairs = {(a, b) for a in range(n) for b in range(n)}
        actual_pairs = {tuple(row) for row in task.inputs}
        assert actual_pairs == expected_pairs
        assert len(actual_pairs) == task.inputs.shape[0]  # no duplicate rows

    def test_input_target_alignment(self, task, group):
        table = group.cayley_table
        for k in range(task.inputs.shape[0]):
            a, b = int(task.inputs[k, 0]), int(task.inputs[k, 1])
            assert table[a, b] == task.targets[k]

    def test_targets_in_valid_range(self, task, group):
        n = group.order
        assert task.targets.min() >= 0
        assert task.targets.max() < n

    @pytest.mark.parametrize(
        "order,index",
        [
            (8, 1),
            (21, 2),
            (32, 1),
        ],
    )
    def test_different_groups(self, order, index):
        g = load_group(order, index)
        t = build_group_task(g)
        n = g.order
        assert t.group_order == n
        assert t.inputs.shape == (n * n, 2)
        assert t.targets.shape == (n * n,)
        np.testing.assert_array_equal(t.targets, g.cayley_table.reshape(-1))


# ---------------------------------------------------------------------------
# train_test_split — happy path
# ---------------------------------------------------------------------------


class TestTrainTestSplitHappyPath:
    def test_basic_sizes(self, task):
        split = train_test_split(task, train_frac=0.8, seed=42)
        n_total = task.inputs.shape[0]
        n_train_expected = round(0.8 * n_total)
        n_test_expected = n_total - n_train_expected

        assert split.train_inputs.shape[0] == n_train_expected
        assert split.train_targets.shape[0] == n_train_expected
        assert split.test_inputs.shape[0] == n_test_expected
        assert split.test_targets.shape[0] == n_test_expected

    def test_deterministic(self, task):
        s1 = train_test_split(task, train_frac=0.8, seed=42)
        s2 = train_test_split(task, train_frac=0.8, seed=42)
        np.testing.assert_array_equal(s1.train_inputs, s2.train_inputs)
        np.testing.assert_array_equal(s1.train_targets, s2.train_targets)
        np.testing.assert_array_equal(s1.test_inputs, s2.test_inputs)
        np.testing.assert_array_equal(s1.test_targets, s2.test_targets)

    def test_different_seeds_produce_different_permutation(self, task):
        """Different seeds should, with overwhelming probability, shuffle differently."""
        s1 = train_test_split(task, train_frac=0.8, seed=42)
        s2 = train_test_split(task, train_frac=0.8, seed=99)
        combined1 = np.concatenate([s1.train_inputs, s1.test_inputs], axis=0)
        combined2 = np.concatenate([s2.train_inputs, s2.test_inputs], axis=0)
        assert not np.array_equal(combined1, combined2)

    def test_disjoint_train_test(self, task):
        split = train_test_split(task, train_frac=0.7, seed=42)
        train_set = {tuple(row) for row in split.train_inputs}
        test_set = {tuple(row) for row in split.test_inputs}
        assert train_set.isdisjoint(test_set)

    def test_union_covers_all_examples(self, task):
        split = train_test_split(task, train_frac=0.7, seed=42)
        all_pairs = {tuple(row) for row in task.inputs}
        union = {tuple(row) for row in split.train_inputs} | {
            tuple(row) for row in split.test_inputs
        }
        assert union == all_pairs

    def test_preserves_input_target_mapping(self, task):
        lookup = {tuple(task.inputs[k]): task.targets[k] for k in range(task.inputs.shape[0])}

        split = train_test_split(task, train_frac=0.7, seed=42)
        for i in range(split.train_inputs.shape[0]):
            key = tuple(split.train_inputs[i])
            assert split.train_targets[i] == lookup[key]

        for i in range(split.test_inputs.shape[0]):
            key = tuple(split.test_inputs[i])
            assert split.test_targets[i] == lookup[key]

    @pytest.mark.parametrize("train_frac", [0.01, 0.1, 0.5, 0.9, 0.99])
    def test_various_fracs(self, task, train_frac):
        """A sweep of valid train_frac values; both sides stay non-empty for |G|=8."""
        split = train_test_split(task, train_frac=train_frac, seed=42)
        n_total = task.inputs.shape[0]
        n_train = round(train_frac * n_total)
        assert split.train_inputs.shape[0] == n_train
        assert split.test_inputs.shape[0] == n_total - n_train
        assert split.train_inputs.shape[0] > 0 and split.test_inputs.shape[0] > 0

    def test_rounding_half(self, task):
        """With 64 examples (order-8 group) and train_frac=0.5, n_train == 32."""
        assert task.inputs.shape[0] == 64
        split = train_test_split(task, train_frac=0.5, seed=42)
        assert split.train_inputs.shape[0] == 32
        assert split.test_inputs.shape[0] == 32

    def test_rounding_third(self, task):
        """round(1/3 * 64) = 21."""
        split = train_test_split(task, train_frac=1 / 3, seed=42)
        assert split.train_inputs.shape[0] == 21
        assert split.test_inputs.shape[0] == 43

    def test_rounding_odd_n(self):
        """With 9 examples and train_frac=0.5, round(4.5)=4 (banker's rounding)."""
        split = train_test_split(_task_of_size(9, group_order=3), train_frac=0.5, seed=42)
        assert split.train_inputs.shape[0] == 4
        assert split.test_inputs.shape[0] == 5

    def test_shapes_and_dtypes_are_preserved(self, task):
        split = train_test_split(task, train_frac=0.6, seed=42)
        assert split.train_targets.shape[0] == split.train_inputs.shape[0]
        assert split.test_targets.shape[0] == split.test_inputs.shape[0]
        assert split.train_inputs.dtype == task.inputs.dtype
        assert split.test_inputs.dtype == task.inputs.dtype
        assert split.train_inputs.shape[0] + split.test_inputs.shape[0] == task.inputs.shape[0]

    def test_split_independent_of_group_order_field(self):
        inputs = np.arange(36 * 2).reshape(36, 2)
        targets = np.arange(36)
        t_a = GroupTask(group_order=6, inputs=inputs, targets=targets)
        t_b = GroupTask(group_order=999, inputs=inputs, targets=targets)

        s_a = train_test_split(t_a, train_frac=0.5, seed=42)
        s_b = train_test_split(t_b, train_frac=0.5, seed=42)
        np.testing.assert_array_equal(s_a.train_inputs, s_b.train_inputs)
        np.testing.assert_array_equal(s_a.test_inputs, s_b.test_inputs)


# ---------------------------------------------------------------------------
# train_test_split — rejected splits
# ---------------------------------------------------------------------------


class TestTrainTestSplitValueErrors:
    """train_frac outside (0, 1) is rejected."""

    @pytest.mark.parametrize("train_frac", [0.0, 1.0, -0.1, 1.5, -100.0, float("nan")])
    def test_frac_outside_the_open_interval_raises(self, task, train_frac):
        # NaN too: every NaN comparison is False, so it fails the interval check.
        with pytest.raises(ValueError, match="train_frac must be in the open interval"):
            train_test_split(task, train_frac=train_frac, seed=42)

    def test_error_message_includes_bad_value(self, task):
        with pytest.raises(ValueError, match="got 1.5"):
            train_test_split(task, train_frac=1.5, seed=42)


class TestTrainTestSplitEmptySide:
    """A split that empties either side is silent poison: cross-entropy over an
    empty tensor is nan, a nan never reaches the generalisation bar, and the run
    trains its whole budget and finalises as `completed` with nan metrics. The
    splitter must refuse instead."""

    def test_empty_test_split_raises(self):
        # C2's 4 pairs with train_frac=0.9: round(3.6) == 4 train, 0 test.
        with pytest.raises(ValueError, match="4 train / 0 test"):
            train_test_split(_task_of_size(4, group_order=2), train_frac=0.9, seed=42)

    def test_empty_test_split_raises_for_a_larger_group(self):
        # C8's 64 pairs with train_frac=0.995: round(63.68) == 64 train, 0 test.
        with pytest.raises(ValueError, match="64 train / 0 test"):
            train_test_split(_task_of_size(64, group_order=8), train_frac=0.995, seed=42)

    def test_empty_train_split_raises(self):
        with pytest.raises(ValueError, match="0 train / 64 test"):
            train_test_split(_task_of_size(64, group_order=8), train_frac=1e-6, seed=42)

    def test_error_message_names_the_group_and_the_fraction(self):
        with pytest.raises(ValueError) as excinfo:
            train_test_split(_task_of_size(4, group_order=2), train_frac=0.9, seed=42)
        message = str(excinfo.value)
        assert "train_frac=0.9" in message
        assert "order-2 group" in message
        assert "nan" in message  # states why an empty side breaks training

    def test_the_last_admissible_fraction_still_works(self):
        """The guard rejects only the empty case, not everything near the edge."""
        split = train_test_split(_task_of_size(4, group_order=2), train_frac=0.7, seed=42)
        assert split.train_inputs.shape[0] == 3
        assert split.test_inputs.shape[0] == 1


# ---------------------------------------------------------------------------
# Transpose-leak covariate: commuting_probability
# ---------------------------------------------------------------------------


class TestCommutingProbability:
    """Pr(G) = k(G)/|G|, the number of conjugacy classes over the order. By
    Burnside this is exactly the fraction of ordered pairs that commute."""

    def test_abelian_group_has_probability_one(self):
        # Every element is its own conjugacy class, so k == |G| and Pr == 1.
        assert commuting_probability(load_group(8, 1)) == pytest.approx(1.0)  # C8
        assert commuting_probability(load_group(4, 1)) == pytest.approx(1.0)  # C4

    def test_s3_has_three_classes_over_six(self):
        # S3: {e}, the three reflections, the two rotations -> k = 3, |G| = 6.
        g = load_group(6, 1)
        assert len(g.conjugacy_classes) == 3
        assert commuting_probability(g) == pytest.approx(0.5)

    def test_matches_the_realised_commuting_fraction(self):
        # The analytic Pr(G) must equal the fraction of ordered pairs (a, b)
        # with a*b == b*a computed directly off the Cayley table.
        for order, index in [(6, 1), (8, 3), (8, 4), (8, 1)]:
            g = load_group(order, index)
            table = g.cayley_table
            commute = table == table.T
            assert commuting_probability(g) == pytest.approx(commute.mean())


# ---------------------------------------------------------------------------
# Transpose-leak covariate: transpose_leaked_mask / transpose_leak_fraction
# ---------------------------------------------------------------------------


def _split_from_pairs(train_pairs, test_pairs, table) -> TrainTestSplit:
    """Build a TrainTestSplit directly from explicit (a, b) pair lists so a test
    can pin the exact realised split, bypassing the random splitter."""
    train = np.array(train_pairs, dtype=np.int64).reshape(-1, 2)
    test = np.array(test_pairs, dtype=np.int64).reshape(-1, 2)
    return TrainTestSplit(
        train_inputs=train,
        train_targets=table[train[:, 0], train[:, 1]],
        test_inputs=test,
        test_targets=table[test[:, 0], test[:, 1]],
    )


# C2 as addition mod 2: everything commutes.
_C2_TABLE = np.array([[0, 1], [1, 0]], dtype=np.int64)


class TestTransposeLeakedMask:
    def test_abelian_leak_is_exactly_transpose_in_train(self):
        """In an abelian group every pair commutes, so a test pair is leaked iff
        its transpose is in train -- the commuting condition is always met."""
        # test={(1,0),(0,0),(1,1)}, train={(0,1)}.
        split = _split_from_pairs([(0, 1)], [(1, 0), (0, 0), (1, 1)], _C2_TABLE)
        mask = transpose_leaked_mask(split, _C2_TABLE)
        # (1,0): transpose (0,1) IS in train, commutes -> leaked.
        # (0,0): transpose (0,0) not in train -> not leaked.
        # (1,1): transpose (1,1) not in train -> not leaked.
        np.testing.assert_array_equal(mask, [True, False, False])
        assert transpose_leak_fraction(split, _C2_TABLE) == pytest.approx(1 / 3)

    def test_diagonal_pair_is_never_leaked(self):
        """A pair (a, a) is its own transpose; train and test are disjoint, so its
        transpose can never be in train and it is never leaked."""
        split = _split_from_pairs([(0, 1), (1, 0)], [(0, 0), (1, 1)], _C2_TABLE)
        mask = transpose_leaked_mask(split, _C2_TABLE)
        np.testing.assert_array_equal(mask, [False, False])

    def test_noncommuting_pair_never_leaks_even_with_transpose_in_train(self):
        """The commuting condition is load-bearing: a test pair whose transpose
        is in train is still unleaked when the pair does not commute, because
        a*b then differs from the memorised b*a."""
        g = load_group(6, 1)  # S3, nonabelian
        table = g.cayley_table
        # A noncommuting pair (a, b): a*b != b*a.
        noncommuting = next(
            (a, b) for a in range(6) for b in range(6) if table[a, b] != table[b, a]
        )
        a, b = noncommuting
        # Put the pair in test and its transpose in train.
        split = _split_from_pairs([(b, a)], [(a, b)], table)
        mask = transpose_leaked_mask(split, table)
        assert mask.tolist() == [False]  # transpose in train, but does not commute

        # A commuting off-diagonal pair with its transpose in train IS leaked.
        commuting = next(
            (a, b) for a in range(6) for b in range(6) if a != b and table[a, b] == table[b, a]
        )
        c, d = commuting
        split2 = _split_from_pairs([(d, c)], [(c, d)], table)
        assert transpose_leaked_mask(split2, table).tolist() == [True]

    def test_empty_unleaked_subset_degenerate_case(self):
        """A split where every test pair is leaked: the transpose-unleaked subset
        is empty, the leak fraction is 1.0, and no downstream code may divide by
        the (zero) unleaked count."""
        # test={(0,1)} whose transpose (1,0) is in train; the only off-diagonal
        # test pair is leaked, so unleaked is empty.
        split = _split_from_pairs([(1, 0), (0, 0), (1, 1)], [(0, 1)], _C2_TABLE)
        mask = transpose_leaked_mask(split, _C2_TABLE)
        assert mask.all()  # every test pair leaked
        assert (~mask).sum() == 0  # unleaked subset empty
        assert transpose_leak_fraction(split, _C2_TABLE) == pytest.approx(1.0)

    def test_mask_length_matches_test_set_and_fraction_in_unit_interval(self):
        g = load_group(8, 3)  # D8, nonabelian
        task = build_group_task(g)
        split = train_test_split(task, train_frac=0.8, seed=7)
        mask = transpose_leaked_mask(split, g.cayley_table)
        assert mask.shape == (split.test_inputs.shape[0],)
        frac = transpose_leak_fraction(split, g.cayley_table)
        assert 0.0 <= frac <= 1.0
        # The realised leak fraction cannot exceed the group's commuting ceiling
        # by more than sampling noise allows on the leaked side: every leaked pair
        # commutes, so leaked pairs are a subset of the commuting test pairs.
        table = g.cayley_table
        commutes = (
            table[split.test_inputs[:, 0], split.test_inputs[:, 1]]
            == table[split.test_inputs[:, 1], split.test_inputs[:, 0]]
        )
        assert np.all(mask <= commutes)  # leaked => commutes
