"""The exporter's subgroup/coset section is gated behind a flag (default OFF).

The subgroup/coset section of ``scripts/export_group.py`` emits one ``.npz``
member per subgroup and per left coset.  On elementary-abelian 2-groups this
explodes -- ``C2^7`` = ``SmallGroup(128, 2328)`` has 29,212 subgroups and
387,987 left cosets, so a single archive would hold ~417,000 members -- and the
core study includes ``C2^7``.  The section is therefore OFF by default and
enabled only for the specific groups whose coset templates an instrument needs.

These tests need no Sage: they exercise the gate function ``_subgroup_section``
directly (a fake ``libgap`` stands in for the ON branch, and the OFF branch must
not dereference ``libgap`` at all) and confirm that a gated-OFF artifact --
``subgroups: 0`` / ``coset_counts: []`` with no subgroup/coset members -- still
loads cleanly through the Sage-free loading path, yielding empty subgroup and
coset tuples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_artifact_validation import _c4_metadata_and_arrays, _write  # noqa: E402

from group_algorithm_interp.groups.data import load_group  # noqa: E402
from scripts.export_group import _subgroup_section  # noqa: E402


class _Exploding:
    """Any attribute access raises: proves the OFF branch never enumerates."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"subgroup enumeration must not run when include_subgroups=False "
            f"(tried to access libgap.{name})"
        )


class _FakeLibgap:
    """Minimal stand-in for the handful of ``libgap`` calls the ON branch makes.

    Subgroups are identified by integer handles; ``Elements`` and
    ``LeftCosets``/``AsList`` resolve those handles to element-label lists.
    """

    def __init__(self, subgroups: list[list[str]], cosets: list[list[list[str]]]) -> None:
        self._subgroups = subgroups
        self._cosets = cosets

    def AllSubgroups(self, group: object) -> list[int]:  # noqa: N802
        return list(range(len(self._subgroups)))

    def Elements(self, subgroup: int) -> list[str]:  # noqa: N802
        return self._subgroups[subgroup]

    def LeftCosets(self, group: object, subgroup: int) -> list[tuple[int, int]]:  # noqa: N802
        return [(subgroup, j) for j in range(len(self._cosets[subgroup]))]

    def AsList(self, coset: tuple[int, int]) -> list[str]:  # noqa: N802
        subgroup, j = coset
        return self._cosets[subgroup][j]


def test_flag_off_returns_empty_section_without_enumerating() -> None:
    """OFF (the default): no counts, no members, and ``libgap`` is never touched."""
    count, coset_counts, payload = _subgroup_section(
        _Exploding(), _Exploding(), {}, include_subgroups=False
    )
    assert count == 0
    assert coset_counts == []
    assert payload == {}


def test_flag_on_emits_subgroup_and_coset_members() -> None:
    """ON: every subgroup and coset becomes a named array member."""
    # Two subgroups of a notional order-4 group: the whole group and a C2.
    token_of = {"e": 0, "a": 1, "b": 2, "c": 3}
    subgroups = [["e", "a", "b", "c"], ["e", "b"]]
    cosets = [
        [["e", "a", "b", "c"]],  # the whole group is its own single coset
        [["e", "b"], ["a", "c"]],  # C2 has two left cosets
    ]
    fake = _FakeLibgap(subgroups, cosets)

    count, coset_counts, payload = _subgroup_section(
        fake, object(), token_of, include_subgroups=True
    )

    assert count == 2
    assert coset_counts == [1, 2]
    assert set(payload) == {"subgroup_0", "subgroup_1", "coset_0_0", "coset_1_0", "coset_1_1"}
    # Element labels are resolved to sorted token arrays.
    np.testing.assert_array_equal(payload["subgroup_1"], np.array([0, 2]))
    np.testing.assert_array_equal(payload["coset_1_1"], np.array([1, 3]))


def test_gated_off_artifact_loads_with_empty_subgroups(tmp_path: Path) -> None:
    """A gated-OFF artifact (subgroups: 0, coset_counts: []) round-trips: the
    loader reads ``metadata`` counts, so an artifact exported without the
    subgroup section must load cleanly and expose empty
    ``subgroups``/``left_cosets`` -- nothing under ``src/`` consumes them
    today.
    """
    metadata, arrays = _c4_metadata_and_arrays()
    metadata = dict(metadata)
    metadata["subgroups"] = 0
    metadata["coset_counts"] = []
    arrays = {k: v for k, v in arrays.items() if not k.startswith(("subgroup_", "coset_"))}

    _write(tmp_path, 4, 1, metadata, arrays)
    group = load_group(4, 1, directory=tmp_path)

    assert group.subgroups == ()
    assert group.left_cosets == ()
    # The rest of the artifact is unaffected.
    assert group.order == 4
    assert len(group.irreps) == 4
    with pytest.raises(ValueError, match="not part of this Sage/GAP artifact"):
        group.cosets_for_subgroup(np.array([0, 2]))
