"""The group-multiplication learning task: predict a * b from the pair (a, b).

Inputs are pairs of element indices into the group; the target is the index of
their product (read straight off the Cayley table). The train/test split
fraction is the knob that drives generalisation -- a small train fraction forces the
model to generalise rather than memorise.
"""

from dataclasses import dataclass

import numpy as np

from group_algorithm_interp.config import MIN_TEST_EXAMPLES, MIN_TRAIN_EXAMPLES, split_sizes
from group_algorithm_interp.groups.group import FiniteGroup


@dataclass(frozen=True)
class GroupTask:
    """All |G|^2 multiplication examples for a group."""

    group_order: int
    inputs: np.ndarray  # shape (|G|^2, 2): element-index pairs (a, b)
    targets: np.ndarray  # shape (|G|^2,): index of a * b


@dataclass(frozen=True)
class TrainTestSplit:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray


def build_group_task(group: FiniteGroup) -> GroupTask:
    """Enumerate every ordered pair (a, b) and its product a * b."""
    n = group.order
    grid = np.arange(n)
    # Row k = (k // n, k % n), so it lines up with cayley_table.reshape(-1).
    inputs = np.stack([np.repeat(grid, n), np.tile(grid, n)], axis=1)
    targets = group.cayley_table.reshape(-1)
    return GroupTask(group_order=n, inputs=inputs, targets=targets)


def train_test_split(task: GroupTask, train_frac: float, seed: int) -> TrainTestSplit:
    """Randomly split the task into train/test, seeded for reproducibility.

    Inputs and targets are indexed by the same permutation, so every example
    keeps its correct label.

    A train_frac in the open interval (0, 1) is *not* enough to guarantee two
    non-empty sides: ``round(train_frac * n)`` can still be ``0`` or ``n`` for a
    small group (C2 has only 4 pairs, so train_frac=0.9 rounds to 4 train / 0
    test). An empty side is silent poison -- cross-entropy over an empty tensor
    is nan, so the run trains its full budget and finalises as ``completed`` with
    nan metrics -- so it is rejected here, naming the group and the fraction.
    ``DataConfig`` rejects the same condition at validation time; this guard also
    covers direct callers that never went through a config.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in the open interval (0, 1), got {train_frac}")

    n = task.inputs.shape[0]
    n_train, n_test = split_sizes(n, train_frac)
    if n_train < MIN_TRAIN_EXAMPLES or n_test < MIN_TEST_EXAMPLES:
        raise ValueError(
            f"train_frac={train_frac} splits the {n} examples of the "
            f"order-{task.group_order} group into {n_train} train / {n_test} "
            f"test, but at least {MIN_TRAIN_EXAMPLES} train and "
            f"{MIN_TEST_EXAMPLES} test example(s) are required: an empty side "
            "makes the loss and accuracy nan without failing the run."
        )

    permutation = np.random.default_rng(seed).permutation(n)
    train_idx, test_idx = permutation[:n_train], permutation[n_train:]

    return TrainTestSplit(
        train_inputs=task.inputs[train_idx],
        train_targets=task.targets[train_idx],
        test_inputs=task.inputs[test_idx],
        test_targets=task.targets[test_idx],
    )


def commuting_probability(group: FiniteGroup) -> float:
    """The group's commuting probability ``Pr(G) = k(G)/|G|``, where ``k(G)`` is
    the number of conjugacy classes.

    By Burnside's counting of commuting pairs, ``k(G)/|G|`` is exactly the
    fraction of ordered pairs ``(a, b)`` with ``a*b == b*a``. It is the
    theoretical ceiling the realised transpose-leak fraction approaches as
    ``train_frac -> 1``: abelian groups have ``Pr(G) == 1`` (every transpose is a
    shortcut), and it falls with noncommutativity. Recorded per run as a
    group-level covariate alongside the realised ``transpose_leak_fraction``.
    """
    return len(group.conjugacy_classes) / group.order


def transpose_leaked_mask(split: TrainTestSplit, cayley_table: np.ndarray) -> np.ndarray:
    """Boolean mask over the test pairs: ``True`` where a test pair is *transpose-
    leaked* on this realised split.

    A test pair ``(a, b)`` is transpose-leaked when its transpose ``(b, a)`` is in
    the *training* set AND the pair commutes (``a*b == b*a``). Both conditions are
    required: when they hold, the label ``a*b`` equals the memorised training
    label ``b*a``, so the model can read the test answer straight off a training
    example -- the pair is reachable by the commuting-transpose shortcut and does
    not test generalisation. A noncommuting pair whose transpose is in train is
    *not* leaked (``a*b != b*a``), and a diagonal pair ``(a, a)`` is never leaked
    (its transpose is itself, and train/test are disjoint).

    Computed on the ACTUAL realised split, never a theoretical expectation. The
    returned mask is aligned row-for-row with ``split.test_inputs`` (and hence
    with the test tokens/targets built from it in the same order).
    """
    n = int(cayley_table.shape[0])
    # Encode a pair (a, b) as the integer a*n + b -- a bijection onto [0, n^2),
    # so pair membership becomes cheap integer-set membership.
    train_a = split.train_inputs[:, 0].astype(np.int64)
    train_b = split.train_inputs[:, 1].astype(np.int64)
    train_codes = train_a * n + train_b
    test_a = split.test_inputs[:, 0].astype(np.int64)
    test_b = split.test_inputs[:, 1].astype(np.int64)
    transpose_codes = test_b * n + test_a  # the code of the transpose (b, a)
    transpose_in_train = np.isin(transpose_codes, train_codes)
    commutes = cayley_table[test_a, test_b] == cayley_table[test_b, test_a]
    return np.asarray(transpose_in_train & commutes, dtype=bool)


def transpose_leak_fraction(split: TrainTestSplit, cayley_table: np.ndarray) -> float:
    """Fraction of test pairs that are transpose-leaked on the realised split (see
    :func:`transpose_leaked_mask`). In ``[0, 1]``; the test side is non-empty by
    construction (the splitter refuses an empty side)."""
    return float(transpose_leaked_mask(split, cayley_table).mean())
