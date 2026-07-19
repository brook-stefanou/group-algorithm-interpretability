"""Tests for the group-name -> SmallGroup(order, index) resolver.

``catalog.py`` is the single place a config string like ``"D8"`` becomes a
concrete ``(order, index)`` SmallGroup id.  A silent transcription error here
(e.g. "D4" secretly resolving to the quaternion group) would be invisible
downstream and would invalidate every mechanistic claim built on top of it.
Every assertion below is checked against an *independent* fact about the
named group -- not against a restatement of ``catalog.py``'s own source:

  * the order implied by the group's name (e.g. "C5" must have order 5,
    "S3" must have order 3! = 6, "D8"/"Q8" must have order 8, using the
    algebraist's ``D_n`` = order-``n`` dihedral convention that this catalog
    itself documents by mapping "D8" to an order-8 SmallGroup);
  * abelian-ness implied by the name ("C_n" is always abelian; "S3", "D8",
    "Q8" are always non-abelian), read off the *fixture Cayley table* in
    ``tests/support_artifacts.py``, not off catalog.py;
  * the classic D4-vs-Q8 fingerprint: both are non-abelian of order 8, but
    Q8 has a *unique* non-identity involution ("-1") while D4/D8 has several
    (the four reflections plus the 180-degree rotation) -- this is exactly
    the invariant that a "D4 mislabelled as Q8" transcription bug would
    violate;
  * cross-consistency with the independently-maintained shorthand table in
    ``group_algorithm_interp.config`` (``_GROUP_NAME_TO_ID``), which a
    catalog.py edit could silently drift from.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from group_algorithm_interp.config import _GROUP_NAME_TO_ID
from group_algorithm_interp.groups.catalog import _NAMED, build_from_id, resolve_group
from group_algorithm_interp.groups.data import GroupData

# Names for which tests/support_artifacts.py::write_test_artifacts actually
# writes a fixture .npz (see that module's ``cyclic`` dict plus its explicit
# dihedral(3)/dihedral(4)/quaternion calls).  "C3" and "C5" are claimed by
# _NAMED but have no fixture artifact anywhere in the test corpus, so they
# cannot be structurally verified here -- see test_c3_c5_have_no_fixture_artifact.
_NAMES_WITH_FIXTURES = {"C2", "C4", "C6", "C7", "C8", "S3", "D8", "Q8"}
_NAMES_WITHOUT_FIXTURES = set(_NAMED) - _NAMES_WITH_FIXTURES


def _expected_order(name: str) -> int:
    """Order implied purely by group-theoretic convention for ``name``.

    Independent of anything in catalog.py: derived from the mathematical
    meaning of the name, not from the module under test.
    """
    if name.startswith("C"):
        return int(name[1:])  # C_n is the cyclic group of order n.
    if name.startswith("D"):
        # Algebraist's D_n convention (order n), which is the convention this
        # catalog uses since it maps "D8" -> an order-8 SmallGroup(8, 3).
        return int(name[1:])
    if name == "S3":
        return 6  # |S_3| = 3! = 6.
    if name == "Q8":
        return 8  # The quaternion group has exactly 8 elements: +-1,+-i,+-j,+-k.
    raise AssertionError(f"no independent order rule known for {name!r}; add one")


def _identity_index(table: np.ndarray) -> int:
    n = len(table)
    for i in range(n):
        if np.array_equal(table[i], np.arange(n)):
            return i
    raise AssertionError("Cayley table has no left-identity row; malformed fixture")


def _element_order(table: np.ndarray, identity: int, x: int) -> int:
    n = len(table)
    power = x
    k = 1
    while power != identity:
        power = int(table[power, x])
        k += 1
        if k > n:
            raise AssertionError("element order exceeds |G|; malformed Cayley table")
    return k


def _is_abelian(table: np.ndarray) -> bool:
    return bool(np.array_equal(table, table.T))


# --------------------------------------------------------------------------
# 1. Every named shorthand resolves to a group of the order its name implies.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_NAMED))
def test_named_order_matches_name_semantics(name: str) -> None:
    """The order component of _NAMED[name] must equal the order implied by name.

    Catches the class of bug where a name's (order, index) tuple has the
    wrong *order* digit -- e.g. "C5": (4, 1) -- independent of whether a
    fixture artifact exists to load.
    """
    expected_order = _expected_order(name)
    order_in_table, _index = _NAMED[name]
    assert order_in_table == expected_order, (
        f"{name!r} is mapped to order {order_in_table}, but its name implies order {expected_order}"
    )


@pytest.mark.parametrize("name", sorted(_NAMES_WITH_FIXTURES))
def test_named_group_resolves_and_has_expected_order(name: str) -> None:
    group = resolve_group(name)
    assert isinstance(group, GroupData)
    assert group.order == _expected_order(name)
    # canonical_id must round-trip exactly through the catalog's own tuple.
    assert group.canonical_id == _NAMED[name]


def test_c3_c5_have_no_fixture_artifact() -> None:
    """C3/C5 are claimed by _NAMED but the test corpus has no artifact for them.

    This is a real gap, not a bug in catalog.py: it documents that "C3" and
    "C5" cannot be structurally verified by this test suite (no Cayley table
    is available anywhere in the repo's offline fixtures), only their order
    digit (checked above) and their presence in both independent shorthand
    tables (checked below).
    """
    for name in sorted(_NAMES_WITHOUT_FIXTURES):
        with pytest.raises(FileNotFoundError, match=r"sage -python scripts/export_group\.py"):
            resolve_group(name)


# --------------------------------------------------------------------------
# 2. Abelian-ness implied by the name, read off the real fixture Cayley table.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(n for n in _NAMES_WITH_FIXTURES if n.startswith("C")))
def test_cyclic_names_resolve_to_abelian_groups(name: str) -> None:
    group = resolve_group(name)
    assert _is_abelian(group.cayley_table), f"{name!r} must be abelian (it is a cyclic group)"


@pytest.mark.parametrize("name", ["S3", "D8", "Q8"])
def test_nonabelian_names_resolve_to_nonabelian_groups(name: str) -> None:
    group = resolve_group(name)
    assert not _is_abelian(group.cayley_table), f"{name!r} must be non-abelian"


@pytest.mark.parametrize("name", sorted(n for n in _NAMES_WITH_FIXTURES if n.startswith("C")))
def test_cyclic_names_have_a_generator_of_full_order(name: str) -> None:
    """A cyclic group of order n must contain an element of order n --
    an abelian-but-non-cyclic group of the same order would not, so this is
    what actually pins down "cyclic" as opposed to merely "abelian"."""
    group = resolve_group(name)
    table = group.cayley_table
    identity = _identity_index(table)
    orders = {_element_order(table, identity, x) for x in range(group.order)}
    assert group.order in orders, f"{name!r} has no element of order {group.order}"


# --------------------------------------------------------------------------
# 3. The classic D4-vs-Q8 transcription bug: both are non-abelian of order 8,
#    but they are distinguished by how many non-identity involutions they have.
# --------------------------------------------------------------------------


def test_d8_and_q8_are_distinct_nonabelian_order_eight_groups() -> None:
    d8 = resolve_group("D8")
    q8 = resolve_group("Q8")
    assert d8.order == 8
    assert q8.order == 8
    assert not _is_abelian(d8.cayley_table)
    assert not _is_abelian(q8.cayley_table)
    # Same order/character-table-fingerprint family, but must be different
    # concrete SmallGroup ids (this is the fact a D4<->Q8 mixup would break).
    assert d8.canonical_id != q8.canonical_id


def test_q8_has_a_unique_involution() -> None:
    """Q8 has exactly one non-identity element x with x*x = identity (namely
    -1); every other non-identity element (+-i, +-j, +-k) has order 4. If
    "Q8" ever silently resolved to D4/D8, this would fail: D4/D8 has five
    non-identity involutions (four reflections plus the 180-degree rotation).
    """
    group = resolve_group("Q8")
    table = group.cayley_table
    identity = _identity_index(table)
    involutions = [
        x for x in range(group.order) if x != identity and _element_order(table, identity, x) == 2
    ]
    assert len(involutions) == 1, (
        f"Q8 must have exactly one non-identity involution, found {len(involutions)}: "
        f"{involutions}. This is the signature that distinguishes Q8 from D4/D8."
    )


def test_d8_has_multiple_involutions() -> None:
    """D4/D8 (dihedral of order 8) has more than one non-identity involution:
    reflections and the 180-degree rotation all qualify, only the two
    order-4 rotations do not. Mirrors the Q8 uniqueness check above; together
    they pin down which fixture is which independent of what catalog.py or
    the fixture author called them.
    """
    group = resolve_group("D8")
    table = group.cayley_table
    identity = _identity_index(table)
    involutions = [
        x for x in range(group.order) if x != identity and _element_order(table, identity, x) == 2
    ]
    assert len(involutions) > 1, (
        f"D4/D8 must have more than one non-identity involution, found "
        f"{len(involutions)}: {involutions}"
    )


def test_s3_is_the_unique_nonabelian_group_of_order_six() -> None:
    """S3 (order 6, non-abelian) must be structurally distinct from C6
    (order 6, abelian). Both name order-6 groups, and catalog.py only tells
    them apart via the index; this checks the fixtures actually differ the
    way the names promise.
    """
    s3 = resolve_group("S3")
    c6 = resolve_group("C6")
    assert s3.order == 6
    assert c6.order == 6
    assert not _is_abelian(s3.cayley_table)
    assert _is_abelian(c6.cayley_table)
    assert s3.canonical_id != c6.canonical_id


# --------------------------------------------------------------------------
# 4. Cross-consistency with the independently-maintained shorthand table in
#    group_algorithm_interp.config -- catches silent drift between the two.
# --------------------------------------------------------------------------


def test_catalog_and_config_shorthand_tables_agree() -> None:
    assert _NAMED == _GROUP_NAME_TO_ID, (
        "groups/catalog.py's _NAMED and config.py's _GROUP_NAME_TO_ID have "
        "diverged; a group name would resolve to a different SmallGroup id "
        "depending on which module resolves it."
    )


# --------------------------------------------------------------------------
# 5. GAP SmallGroup indices are 1-based; guard against an accidental 0-based
#    off-by-one anywhere in the shorthand table.
# --------------------------------------------------------------------------


def test_all_named_indices_are_one_based() -> None:
    for name, (order, index) in _NAMED.items():
        assert order >= 1, f"{name!r} has non-positive order {order}"
        assert index >= 1, f"{name!r} has index {index} < 1; GAP SmallGroup indices are 1-based"


# --------------------------------------------------------------------------
# 6. build_from_id: string dispatch, "order,index" parsing, tuple passthrough.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_NAMES_WITH_FIXTURES))
def test_build_from_id_named_string_matches_resolve_group(name: str) -> None:
    via_build = build_from_id(name)
    via_resolve = resolve_group(name)
    assert via_build.canonical_id == via_resolve.canonical_id
    assert np.array_equal(via_build.cayley_table, via_resolve.cayley_table)


def test_build_from_id_tuple_form() -> None:
    group = build_from_id((8, 3))  # D8's canonical id.
    assert group.canonical_id == (8, 3)
    named = resolve_group("D8")
    assert np.array_equal(group.cayley_table, named.cayley_table)


def test_build_from_id_comma_string_form_matches_named_and_tuple() -> None:
    from_comma = build_from_id("8,3")
    from_tuple = build_from_id((8, 3))
    from_name = resolve_group("D8")
    assert from_comma.canonical_id == from_tuple.canonical_id == from_name.canonical_id
    assert np.array_equal(from_comma.cayley_table, from_tuple.cayley_table)


@pytest.mark.parametrize(
    "malformed",
    [
        "bogus",  # no comma at all
        "1,2,3",  # too many comma-separated fields
        "6",  # not enough fields
        "a,b",  # non-integer fields
        "6,",  # empty second field
        ",1",  # empty first field
        "d8",  # lowercase is not a recognised shorthand and is not "order,index"
    ],
)
def test_build_from_id_rejects_malformed_strings_with_actionable_message(
    malformed: str,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        build_from_id(malformed)
    message = str(excinfo.value)
    assert malformed in message
    assert "named shorthand" in message
    assert "order,index" in message


def test_build_from_id_rejects_out_of_range_index_loudly() -> None:
    # (8, 999) parses fine as an id but no such SmallGroup artifact exists;
    # this must fail loudly (FileNotFoundError with a fix-it command), never
    # silently resolve to a different, wrong group.
    with pytest.raises(FileNotFoundError, match=r"sage -python scripts/export_group\.py"):
        build_from_id("8,999")
    with pytest.raises(FileNotFoundError, match=r"sage -python scripts/export_group\.py"):
        build_from_id((8, 999))


# --------------------------------------------------------------------------
# 7. resolve_group: str/_NAMED, str/"order,index", tuple passthrough, and the
#    object-with-.order-and-.index dispatch branch.
# --------------------------------------------------------------------------


def test_resolve_group_plain_tuple_matches_named_lookup() -> None:
    via_tuple = resolve_group((8, 4))  # Q8's canonical id, no .order/.index attrs on a plain tuple.
    via_name = resolve_group("Q8")
    assert via_tuple.canonical_id == via_name.canonical_id
    assert np.array_equal(via_tuple.cayley_table, via_name.cayley_table)


def test_resolve_group_object_with_order_and_index_attributes() -> None:
    spec = SimpleNamespace(order=6, index=1)  # S3's canonical id, via attribute dispatch.
    group = resolve_group(spec)
    assert group.canonical_id == (6, 1)
    named = resolve_group("S3")
    assert np.array_equal(group.cayley_table, named.cayley_table)


def test_resolve_group_object_attributes_are_coerced_to_int() -> None:
    # hasattr(..., "order")/"index" dispatch calls int() on the attribute
    # values, so string-typed attributes on a config-like object must work.
    spec = SimpleNamespace(order="8", index="3")
    group = resolve_group(spec)
    assert group.canonical_id == (8, 3)


def test_resolve_group_object_missing_index_attribute_fails_loudly() -> None:
    # Only "order" is present, so the object-dispatch branch is skipped and
    # the object falls through to build_from_id, which cannot unpack a
    # SimpleNamespace as *args -- this must raise, never silently resolve.
    spec = SimpleNamespace(order=8)
    with pytest.raises(TypeError):
        resolve_group(spec)


def test_resolve_group_unknown_plain_string_raises_actionable_error() -> None:
    with pytest.raises(ValueError, match="Unrecognised group identifier"):
        resolve_group("Z5")


def test_resolve_group_comma_string_not_in_named_table() -> None:
    # "S3" -> (6, 1) is a named shorthand, but its raw "order,index" spelling
    # is not itself a key in _NAMED; it must still resolve identically via
    # the comma-parsing branch.
    assert "6,1" not in _NAMED
    via_comma = resolve_group("6,1")
    via_name = resolve_group("S3")
    assert via_comma.canonical_id == via_name.canonical_id


def test_resolve_group_out_of_range_index_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError, match=r"sage -python scripts/export_group\.py"):
        resolve_group("8,999")
