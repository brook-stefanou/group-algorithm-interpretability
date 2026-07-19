"""Verify that every fixture Cayley table in ``support_artifacts.py`` is
actually a group.

This module never calls ``load_group`` and does not exercise ``data.py``'s
``_validate_group_axioms`` or ``_validate_isotypic_blocks``: it checks the
fixture builders directly. See ``tests/test_artifact_validation.py`` for the
tests that exercise those validators, via hand-built malformed ``.npz``
artifacts.

``support_artifacts.py`` hand-builds cyclic/dihedral/quaternion multiplication
tables in pure Python, and the rest of the suite trusts those tables as
mathematical oracles for hundreds of tests. This module checks each fixture
table directly against the group axioms, independently of any src-side
validation logic, so a bug shared between a builder and a validator cannot
hide from both.

Every fixture is small (order <= 32), so all checks -- including the O(n^3)
associativity check -- run exhaustively; no sampling is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from support_artifacts import _cyclic, _dihedral, _quaternion

# (case id, Cayley table, expected order, expected abelian)
_CASES: list[tuple[str, np.ndarray, int, bool]] = [
    *(
        (f"cyclic-{order}-{index}", _cyclic(n)[0], order, True)
        for (order, index), n in {
            (2, 1): 2,
            (4, 1): 4,
            (6, 2): 6,
            (7, 1): 7,
            (8, 1): 8,
            (21, 2): 21,
            (32, 1): 32,
        }.items()
    ),
    ("dihedral-6-1", _dihedral(3)[0], 6, False),
    ("dihedral-8-3", _dihedral(4)[0], 8, False),
    ("quaternion-8-4", _quaternion()[0], 8, False),
]

_IDS = [case[0] for case in _CASES]


@pytest.fixture(params=_CASES, ids=_IDS)
def fixture_group(request: pytest.FixtureRequest) -> tuple[str, np.ndarray, int, bool]:
    return request.param


def test_closure(fixture_group: tuple[str, np.ndarray, int, bool]) -> None:
    """Every table entry is a valid element index in [0, n)."""
    _, table, order, _ = fixture_group
    assert table.shape == (order, order)
    assert table.min() >= 0
    assert table.max() < order


def test_latin_square_rows(fixture_group: tuple[str, np.ndarray, int, bool]) -> None:
    """Every row is a permutation of all n elements."""
    _, table, order, _ = fixture_group
    expected = np.arange(order)
    row_sorted = np.sort(table, axis=1)
    bad_rows = np.flatnonzero(np.any(row_sorted != expected, axis=1))
    assert bad_rows.size == 0, f"row {int(bad_rows[0])} is not a permutation of 0..{order - 1}"


def test_latin_square_columns(fixture_group: tuple[str, np.ndarray, int, bool]) -> None:
    """Every column is a permutation of all n elements."""
    _, table, order, _ = fixture_group
    expected = np.arange(order)
    col_sorted = np.sort(table, axis=0)
    bad_cols = np.flatnonzero(np.any(col_sorted != expected[:, None], axis=0))
    assert bad_cols.size == 0, f"column {int(bad_cols[0])} is not a permutation of 0..{order - 1}"


def _find_identity(table: np.ndarray) -> int:
    n = table.shape[0]
    idx = np.arange(n)
    candidates = np.flatnonzero(
        np.all(table == idx[None, :], axis=1) & np.all(table == idx[:, None], axis=0)
    )
    assert candidates.size > 0, "no two-sided identity element exists"
    return int(candidates[0])


def test_identity_exists_and_is_unique(
    fixture_group: tuple[str, np.ndarray, int, bool],
) -> None:
    """There is exactly one e with e*g == g*e == g for all g."""
    _, table, order, _ = fixture_group
    n = table.shape[0]
    idx = np.arange(n)
    candidates = np.flatnonzero(
        np.all(table == idx[None, :], axis=1) & np.all(table == idx[:, None], axis=0)
    )
    assert candidates.size == 1, f"expected exactly one identity, found {candidates.tolist()}"


def test_inverses_exist(fixture_group: tuple[str, np.ndarray, int, bool]) -> None:
    """Every g has some h with g*h == h*g == e."""
    _, table, order, _ = fixture_group
    e = _find_identity(table)
    has_inverse = np.any((table == e) & (table.T == e), axis=1)
    missing = np.flatnonzero(~has_inverse)
    assert missing.size == 0, f"element {int(missing[0])} has no two-sided inverse"


def test_associativity_exhaustive(fixture_group: tuple[str, np.ndarray, int, bool]) -> None:
    """(a*b)*c == a*(b*c) for every triple, checked exhaustively.

    All fixtures here have order <= 32 (32**3 = 32768 triples), so a full,
    vectorized O(n^3) check is trivially fast (well under a second) -- no
    sampling is used.
    """
    _, table, order, _ = fixture_group
    idx = np.arange(order)
    left = table[table[:, :, None], idx[None, None, :]]
    right = table[idx[:, None, None], table[None, :, :]]
    mismatches = np.argwhere(left != right)
    assert mismatches.size == 0, (
        f"associativity fails at a={mismatches[0][0]}, b={mismatches[0][1]}, "
        f"c={mismatches[0][2]}: "
        f"(a*b)*c={int(left[tuple(mismatches[0])])} != "
        f"a*(b*c)={int(right[tuple(mismatches[0])])}"
    )


def test_known_order(fixture_group: tuple[str, np.ndarray, int, bool]) -> None:
    """Each fixture's table has the order the builder claims to produce."""
    _, table, order, _ = fixture_group
    assert table.shape[0] == order


def test_known_commutativity(fixture_group: tuple[str, np.ndarray, int, bool]) -> None:
    """Cyclic fixtures are abelian; dihedral/quaternion fixtures are not.

    A fixture that silently degenerated into the wrong group (e.g. a
    dihedral builder that accidentally produced a cyclic table of the same
    order) would still pass every other axiom check here but fail this one.
    """
    _, table, order, expected_abelian = fixture_group
    is_abelian = bool(np.array_equal(table, table.T))
    if expected_abelian:
        assert is_abelian, "expected an abelian (cyclic) fixture but table is not symmetric"
    else:
        mismatches = np.argwhere(table != table.T)
        assert mismatches.size > 0, "expected a non-abelian fixture but table is symmetric"
        a, b = (int(v) for v in mismatches[0])
        assert int(table[a, b]) != int(table[b, a])
