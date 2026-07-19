"""Cross-check the committed Sage/GAP artifacts against the hand-built fixtures.

``tests/data/group_artifacts/*.npz`` is produced by the real exporter
(``scripts/export_group.py``, passagemath/libgap).  ``tests/support_artifacts.py``
builds the same four groups from scratch in pure Python.  The two are wholly
independent implementations of the same mathematics, so where they disagree, one
of them is wrong -- that is the value of comparing them.

What is compared is the *invariant* content, never the representation-theoretic
bookkeeping that is pure convention:

  compared      |G|, irrep degree multiset, sum(d^2) == |G|, Frobenius-Schur
                indicator multiset, isotypic block_rank multiset, and the
                character table up to a permutation of rows (irreps) and
                columns (classes)

  NOT compared  element labelling/ordering, the order of Irr(G), the choice of
                basis for each irrep matrix, the order of conjugacy classes

GAP's element ordering and irrep basis are its own; the fixtures' are ours.  Any
comparison that assumed they agreed would be testing a convention, not the group.

D4 and Q8 are deliberately both here.  They are NON-isomorphic groups with the
SAME character table, so a comparison built only on character data cannot tell
them apart -- and would happily pass if the exporter emitted one where the other
was asked for.  ``test_d4_and_q8_share_a_character_table_but_are_different_groups``
pins that down from data the character table does not determine: the element-order
profile, and the Frobenius-Schur indicator of the degree-2 irrep (+1 for D4, which
is real; -1 for Q8, which is quaternionic).
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from support_artifacts import write_test_artifacts

from group_algorithm_interp.groups.data import GroupData, load_group

GOLDEN_DIR = Path(__file__).parent / "data" / "group_artifacts"

# (order, index, name) -- exactly the groups exported by the real exporter.
GOLDEN: list[tuple[int, int, str]] = [
    (6, 1, "S3"),
    (8, 3, "D4"),
    (8, 4, "Q8"),
    (8, 1, "C8"),
]
IDS = [name for _, _, name in GOLDEN]
PARAMS = [(order, index) for order, index, _ in GOLDEN]


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A private hand-built corpus, independent of the session-wide one."""
    directory = tmp_path_factory.mktemp("hand_built")
    write_test_artifacts(directory)
    return directory


def _sage(order: int, index: int) -> GroupData:
    return load_group(order, index, GOLDEN_DIR)


def _element_order_profile(table: np.ndarray) -> dict[int, int]:
    """Multiset of element orders, derived from the Cayley table alone.

    This is an isomorphism invariant that the character table does NOT
    determine -- which is exactly why it separates D4 from Q8.
    """
    n = len(table)
    identity = next(e for e in range(n) if np.array_equal(table[e], np.arange(n)))
    profile: dict[int, int] = {}
    for g in range(n):
        current, k = g, 1
        while current != identity:
            current = int(table[current, g])
            k += 1
        profile[k] = profile.get(k, 0) + 1
    return profile


def _character_tables_match(left: np.ndarray, right: np.ndarray, atol: float = 1e-8) -> bool:
    """True iff some row+column permutation carries ``left`` onto ``right``.

    Brute force is fine here: these groups have at most 5 conjugacy classes, so
    the search is at most 5! * 5! = 14,400 candidate permutations.
    """
    if left.shape != right.shape:
        return False
    rows, cols = left.shape
    for column_permutation in permutations(range(cols)):
        candidate = left[:, list(column_permutation)]
        for row_permutation in permutations(range(rows)):
            if np.allclose(candidate[list(row_permutation)], right, atol=atol):
                return True
    return False


@pytest.mark.parametrize(("order", "index"), PARAMS, ids=IDS)
def test_sage_and_hand_built_agree_on_the_group(order: int, index: int, fixture_dir: Path) -> None:
    sage = _sage(order, index)
    hand = load_group(order, index, fixture_dir)

    assert sage.order == hand.order == order
    # The Cayley tables use different element labellings, so they are NOT
    # expected to be equal entrywise. The order profile is labelling-free.
    assert _element_order_profile(sage.cayley_table) == _element_order_profile(hand.cayley_table)
    assert len(sage.conjugacy_classes) == len(hand.conjugacy_classes)
    assert sorted(len(c) for c in sage.conjugacy_classes) == sorted(
        len(c) for c in hand.conjugacy_classes
    )


@pytest.mark.parametrize(("order", "index"), PARAMS, ids=IDS)
def test_sage_and_hand_built_agree_on_irrep_degrees(
    order: int, index: int, fixture_dir: Path
) -> None:
    sage = sorted(irrep.dimension for irrep in _sage(order, index).irreps)
    hand = sorted(irrep.dimension for irrep in load_group(order, index, fixture_dir).irreps)
    assert sage == hand
    # ...and the degrees are right in absolute terms, not merely consistent.
    assert sum(d**2 for d in sage) == order


@pytest.mark.parametrize(("order", "index"), PARAMS, ids=IDS)
def test_sage_and_hand_built_agree_on_the_character_table(
    order: int, index: int, fixture_dir: Path
) -> None:
    """Equal up to a permutation of irreps (rows) and classes (columns)."""
    sage = _sage(order, index).character_table
    hand = load_group(order, index, fixture_dir).character_table
    assert _character_tables_match(sage, hand), (
        f"SmallGroup({order},{index}): Sage-exported and hand-built character "
        f"tables are not equal under any row/column permutation.\n"
        f"sage=\n{np.round(sage, 4)}\nhand=\n{np.round(hand, 4)}"
    )


@pytest.mark.parametrize(("order", "index"), PARAMS, ids=IDS)
def test_sage_and_hand_built_agree_on_frobenius_schur(
    order: int, index: int, fixture_dir: Path
) -> None:
    sage = _sage(order, index)
    hand = load_group(order, index, fixture_dir)
    # Row order differs between the two, so compare (degree, indicator) pairs:
    # the multiset of indicators alone would not notice an indicator attached to
    # the wrong degree.
    sage_pairs = sorted(
        (irrep.dimension, int(sage.frobenius_schur[irrep.table_index])) for irrep in sage.irreps
    )
    hand_pairs = sorted(
        (irrep.dimension, int(hand.frobenius_schur[irrep.table_index])) for irrep in hand.irreps
    )
    assert sage_pairs == hand_pairs
    assert all(indicator in (-1, 0, 1) for _, indicator in sage_pairs)


@pytest.mark.parametrize(("order", "index"), PARAMS, ids=IDS)
def test_sage_and_hand_built_agree_on_isotypic_blocks(
    order: int, index: int, fixture_dir: Path
) -> None:
    """irrep_degree and block_rank -- equal only when d == 1, easy to
    conflate otherwise -- checked on both corpora."""
    sage = _sage(order, index)
    hand = load_group(order, index, fixture_dir)
    sage_blocks = sorted((b.irrep_degree, b.block_rank) for b in sage.isotypic_blocks)
    hand_blocks = sorted((b.irrep_degree, b.block_rank) for b in hand.isotypic_blocks)
    assert sage_blocks == hand_blocks
    # Absolute anchor, not just agreement: ranks partition the regular rep.
    assert sum(rank for _, rank in sage_blocks) == order
    for degree, rank in sage_blocks:
        # d^2 for a lone real irrep, 2*d^2 for a merged conjugate pair.
        assert rank in (degree**2, 2 * degree**2)


def test_d4_and_q8_share_a_character_table_but_are_different_groups(
    fixture_dir: Path,
) -> None:
    """The trap this whole comparison could otherwise fall into.

    D4 = SmallGroup(8,3) and Q8 = SmallGroup(8,4) are non-isomorphic yet have
    identical character tables.  So character data alone CANNOT verify that the
    exporter emitted the group it was asked for.  Assert both halves.
    """
    d4, q8 = _sage(8, 3), _sage(8, 4)

    # 1. Same character table -- so character-only comparisons are blind here.
    assert _character_tables_match(d4.character_table, q8.character_table)
    assert sorted(i.dimension for i in d4.irreps) == sorted(i.dimension for i in q8.irreps)

    # 2. Different groups. D4 has 2 elements of order 4 and 5 of order 2;
    #    Q8 has 6 of order 4 and exactly 1 of order 2 (its unique involution).
    assert _element_order_profile(d4.cayley_table) == {1: 1, 2: 5, 4: 2}
    assert _element_order_profile(q8.cayley_table) == {1: 1, 2: 1, 4: 6}

    # 3. Frobenius-Schur separates them where the character table does not: the
    #    degree-2 irrep is real (+1) for D4 and quaternionic (-1) for Q8.
    def two_dim_indicator(group: GroupData) -> int:
        (irrep,) = [i for i in group.irreps if i.dimension == 2]
        return int(group.frobenius_schur[irrep.table_index])

    assert two_dim_indicator(d4) == 1
    assert two_dim_indicator(q8) == -1

    # 4. And the hand-built fixtures agree on that separation, independently.
    hand_d4 = load_group(8, 3, fixture_dir)
    hand_q8 = load_group(8, 4, fixture_dir)
    assert two_dim_indicator(hand_d4) == 1
    assert two_dim_indicator(hand_q8) == -1


def test_golden_artifacts_carry_real_sage_provenance() -> None:
    """They must be genuine exporter output, not a fixture copied into place."""
    for order, index, _ in GOLDEN:
        group = _sage(order, index)
        assert group.provenance["backend"] == "SageMath/libgap"
        assert group.provenance["sage_version"]
        assert group.provenance["gap_version"]
