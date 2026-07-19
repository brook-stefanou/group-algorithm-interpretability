"""Malformed-artifact tests that exercise ``data.py``'s own validators.

``test_group_axioms.py`` checks that the *fixture builders* in
``support_artifacts.py`` produce genuine groups; it never calls
``load_group`` and so never runs ``_validate_group_axioms`` or
``_validate_isotypic_blocks``. This module is the one that does: every test
here hand-builds a malformed ``.npz`` artifact (valid container format,
invalid group-theoretic or format content) and asserts that ``load_group``
rejects it for the specific reason intended.

Each corrupted artifact is constructed to trip exactly one validator branch:
earlier-running checks (closure before Latin-square, Latin-square before
identity, and so on -- see the order documented on
``_validate_group_axioms``) are kept valid so the intended branch is the one
that actually fires.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from support_artifacts import _classes, _cosets, _cyclic, _projector, _subgroups

from group_algorithm_interp.groups.data import ARTIFACT_FORMAT, load_group


def _c4_metadata_and_arrays(order: int = 4, index: int = 1) -> tuple[dict, dict[str, np.ndarray]]:
    """A full, valid SmallGroup(4,1) (cyclic C4) artifact's ingredients.

    Mirrors ``_write_group``'s construction but returns the metadata/arrays
    dicts instead of writing them, so callers can corrupt exactly one field
    before serializing. Structurally COMPLETE -- every field ``load_group``
    reads is present and valid -- so a single-field corruption test can
    attribute the resulting rejection to that corruption alone, and a
    mutation test that neutralizes a validator sees a clean "DID NOT RAISE"
    instead of an unrelated missing-field crash.
    """
    table, matrices, characters, fs, fields = _cyclic(order)
    classes = _classes(table)
    subgroups = _subgroups(table)
    class_values = np.array([[char[cls[0]] for cls in classes] for char in characters])

    blocks: list[tuple[np.ndarray, int, int, tuple[int, ...]]] = []
    used: set[int] = set()
    for i, char in enumerate(characters):
        if i in used:
            continue
        conjugate = next(
            (j for j, other in enumerate(characters) if np.allclose(other, char.conjugate())), i
        )
        projector = _projector(table, char, matrices[i].shape[1])
        indices = (i,)
        if conjugate != i:
            projector = projector + _projector(
                table, characters[conjugate], matrices[conjugate].shape[1]
            )
            indices = (i, conjugate)
            used.add(conjugate)
        used.add(i)
        real = np.real_if_close(projector, tol=1000).real
        degree = int(matrices[i].shape[1])
        rank = sum(int(matrices[j].shape[1]) ** 2 for j in indices)
        blocks.append((real, degree, rank, indices))

    metadata = {
        "format": ARTIFACT_FORMAT,
        "small_group": [order, index],
        "order": order,
        "description": "artifact-validation test fixture",
        "element_labels": [str(i) for i in range(order)],
        "classes": len(classes),
        "subgroups": len(subgroups),
        "coset_counts": [len(_cosets(table, s)) for s in subgroups],
        "irreps": [
            {"dimension": matrix.shape[1], "field": field, "basis": "test-fixture"}
            for matrix, field in zip(matrices, fields, strict=True)
        ],
        "blocks": [
            {"irrep_degree": degree, "block_rank": rank, "irrep_indices": list(indices)}
            for _, degree, rank, indices in blocks
        ],
        "provenance": {"backend": "test fixture", "purpose": "artifact validation tests"},
    }
    arrays: dict[str, np.ndarray] = {
        "cayley_table": table,
        "character_real": class_values.real,
        "character_imag": class_values.imag,
        "frobenius_schur": np.array(fs, dtype=np.int64),
    }
    for i, cls in enumerate(classes):
        arrays[f"class_{i}"] = cls
    for i, matrix in enumerate(matrices):
        arrays[f"irrep_{i}_real"] = matrix.real
        arrays[f"irrep_{i}_imag"] = matrix.imag
        arrays[f"irrep_{i}_character_real"] = characters[i].real
        arrays[f"irrep_{i}_character_imag"] = characters[i].imag
    for i, (projector, _, _, _) in enumerate(blocks):
        arrays[f"block_{i}_projector"] = projector
    for i, subgroup in enumerate(subgroups):
        arrays[f"subgroup_{i}"] = subgroup
        for j, coset in enumerate(_cosets(table, subgroup)):
            arrays[f"coset_{i}_{j}"] = coset
    return metadata, arrays


def _write(
    tmp_path: Path,
    order: int,
    index: int,
    metadata: dict,
    arrays: dict[str, np.ndarray],
) -> Path:
    """Serialize ``metadata``/``arrays`` at the path ``load_group(order, index, ...)``
    will look for. A plain, non-atomic ``np.savez`` is fine here: these are
    throwaway malformed fixtures written once and read once in-process, not
    artifacts a concurrent reader could observe mid-write.
    """
    payload = dict(arrays)
    payload["metadata"] = np.array(json.dumps(metadata))
    path = tmp_path / f"smallgroup_{order}_{index}.npz"
    np.savez(path, **payload)
    return path


def _axiom_artifact(tmp_path: Path, table: np.ndarray, order: int, index: int) -> None:
    """Write a structurally COMPLETE artifact wrapping a possibly-non-group
    ``table`` of order ``order``.

    Every test targeting ``_validate_group_axioms`` hands a ``table`` that
    deliberately fails the axioms, so real characters, irreps, and isotypic
    blocks can't be derived from it -- everything below ``cayley_table`` here
    is structural filler, not real representation theory. Its job is to be
    *readable* by ``load_group`` and to *pass* ``_validate_isotypic_blocks``,
    so that when ``_validate_group_axioms`` is neutralized, execution reaches
    a clean "DID NOT RAISE" instead of tripping on an unrelated missing field
    or the other validator.

    The filler models ``order`` independent degree-1 irreps, each its own
    isotypic block with projector ``E_ii``: ``trace(E_ii) == 1 ==
    block_rank``, ``sum(d_i**2) == 1 == block_rank``, degrees agree with
    ``irrep_degree`` -- every per-block check in ``_validate_isotypic_blocks``
    passes, and the block ranks sum to ``order``. Subgroup/coset filler is
    the trivial one-subgroup-is-the-whole-group case; neither validator
    inspects it.
    """
    n = order
    metadata = {
        "format": ARTIFACT_FORMAT,
        "small_group": [order, index],
        "order": order,
        "description": "artifact-validation test fixture (deliberately non-group table)",
        "element_labels": [str(i) for i in range(n)],
        "classes": n,
        "subgroups": 1,
        "coset_counts": [1],
        "irreps": [{"dimension": 1, "field": "real", "basis": "filler"} for _ in range(n)],
        "blocks": [{"irrep_degree": 1, "block_rank": 1, "irrep_indices": [i]} for i in range(n)],
        "provenance": {"backend": "test fixture", "purpose": "artifact validation tests"},
    }
    arrays: dict[str, np.ndarray] = {
        "cayley_table": table,
        "character_real": np.ones((n, n)),
        "character_imag": np.zeros((n, n)),
        "frobenius_schur": np.ones(n, dtype=np.int64),
    }
    for i in range(n):
        arrays[f"class_{i}"] = np.array([i], dtype=np.int64)
        arrays[f"irrep_{i}_real"] = np.ones((n, 1, 1))
        arrays[f"irrep_{i}_imag"] = np.zeros((n, 1, 1))
        arrays[f"irrep_{i}_character_real"] = np.ones(n)
        arrays[f"irrep_{i}_character_imag"] = np.zeros(n)
        projector = np.zeros((n, n))
        projector[i, i] = 1.0
        arrays[f"block_{i}_projector"] = projector
    arrays["subgroup_0"] = np.arange(n, dtype=np.int64)
    arrays["coset_0_0"] = np.arange(n, dtype=np.int64)
    _write(tmp_path, order, index, metadata, arrays)


# ---------------------------------------------------------------------------
# Baseline control
# ---------------------------------------------------------------------------


def test_baseline_c4_artifact_loads_successfully(tmp_path: Path) -> None:
    """Control: the uncorrupted C4 artifact from ``_c4_metadata_and_arrays``
    loads successfully end to end.

    Every other test in this module corrupts exactly one field of this
    fixture and asserts the resulting rejection; without this control, a
    passing rejection test could equally be explained by a latent defect
    already in the fixture rather than the corruption under test.
    """
    metadata, arrays = _c4_metadata_and_arrays()
    _write(tmp_path, 4, 1, metadata, arrays)
    group = load_group(4, 1, directory=tmp_path)
    assert group.canonical_id == (4, 1)
    assert group.order == 4
    assert len(group.conjugacy_classes) == 4
    assert len(group.irreps) == 4
    # C4's two complex conjugate degree-1 irreps (k=1, k=3) merge into one
    # real isotypic block, so 4 irreps yield only 3 blocks (k=0, k=2 self-
    # conjugate; k=1/k=3 merged) -- see ``_write_group``'s block-merging loop.
    assert len(group.isotypic_blocks) == 3
    assert sum(block.block_rank for block in group.isotypic_blocks) == 4
    assert len(group.subgroups) == len(group.left_cosets)
    assert len(group.subgroups) == 3  # divisors of 4: subgroups of order 1, 2, 4


# ---------------------------------------------------------------------------
# _validate_group_axioms
# ---------------------------------------------------------------------------


def test_associativity_violation_rejected(tmp_path: Path) -> None:
    """A Latin square with a two-sided identity and inverses for every
    element, but which fails associativity, is rejected.

    This is a genuine order-5 loop (found by exhaustive search over reduced
    Latin squares of order 5), not merely a corrupted group table: order <=4
    loops with full inverses turn out to always be associative, so a
    non-associative example only appears at order 5.
    """
    table = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 3, 4, 2],
            [2, 4, 0, 1, 3],
            [3, 2, 4, 0, 1],
            [4, 3, 1, 2, 0],
        ],
        dtype=np.int64,
    )
    _axiom_artifact(tmp_path, table, order=5, index=1)
    with pytest.raises(ValueError, match="associativity fails at"):
        load_group(5, 1, directory=tmp_path)


def test_latin_square_row_violation_rejected(tmp_path: Path) -> None:
    """A row containing a duplicate value (not a permutation of 0..n-1) is
    rejected, naming that row."""
    table, *_ = _cyclic(4)
    table = table.copy()
    table[1, 3] = table[1, 0]  # row 1 now has a duplicate, missing a value
    _axiom_artifact(tmp_path, table, order=4, index=1)
    with pytest.raises(ValueError, match="row 1 is not a permutation"):
        load_group(4, 1, directory=tmp_path)


def test_latin_square_column_violation_rejected(tmp_path: Path) -> None:
    """A column that is not a permutation, while every row still is (so the
    row check -- which runs first -- does not fire instead)."""
    table, *_ = _cyclic(4)
    table = table.copy()
    # Swap two entries within row 0: row 0 stays a permutation of 0..3, but
    # columns 1 and 2 each end up with a duplicate.
    table[0, 1], table[0, 2] = table[0, 2], table[0, 1]
    _axiom_artifact(tmp_path, table, order=4, index=1)
    with pytest.raises(ValueError, match="column 1 is not a permutation"):
        load_group(4, 1, directory=tmp_path)


def test_missing_identity_rejected(tmp_path: Path) -> None:
    """A genuine Latin square (every row and column a permutation) that has
    no two-sided identity element."""
    table = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [3, 2, 0, 1],
            [2, 3, 1, 0],
        ],
        dtype=np.int64,
    )
    _axiom_artifact(tmp_path, table, order=4, index=1)
    with pytest.raises(ValueError, match="no two-sided identity element"):
        load_group(4, 1, directory=tmp_path)


def test_missing_inverse_rejected(tmp_path: Path) -> None:
    """A Latin square with a two-sided identity, but where one element has
    no two-sided inverse (an order-5 loop, found by exhaustive search)."""
    table = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 3, 4, 2],
            [2, 3, 4, 0, 1],
            [3, 4, 1, 2, 0],
            [4, 2, 0, 1, 3],
        ],
        dtype=np.int64,
    )
    _axiom_artifact(tmp_path, table, order=5, index=1)
    with pytest.raises(ValueError, match="element 2 has no two-sided inverse"):
        load_group(5, 1, directory=tmp_path)


def test_closure_violation_rejected(tmp_path: Path) -> None:
    """An out-of-range Cayley-table entry (>= n) is rejected as a closure
    violation, before any Latin-square/identity/inverse/associativity check
    runs."""
    table, *_ = _cyclic(4)
    table = table.copy()
    table[0, 0] = 99
    _axiom_artifact(tmp_path, table, order=4, index=1)
    with pytest.raises(ValueError, match="closure violation"):
        load_group(4, 1, directory=tmp_path)


# ---------------------------------------------------------------------------
# _validate_isotypic_blocks
# ---------------------------------------------------------------------------


def test_block_rank_disagrees_with_projector_trace_rejected(tmp_path: Path) -> None:
    """``block_rank`` matches ``sum(d_i**2)`` (so the earlier check passes),
    but the block's own projector's trace does not match ``block_rank``."""
    metadata, arrays = _c4_metadata_and_arrays()
    arrays = dict(arrays)
    arrays["block_0_projector"] = arrays["block_0_projector"] * 2
    _write(tmp_path, 4, 1, metadata, arrays)
    with pytest.raises(ValueError, match="but trace"):
        load_group(4, 1, directory=tmp_path)


def test_block_rank_disagrees_with_sum_of_squares_rejected(tmp_path: Path) -> None:
    """``block_rank`` does not equal ``sum(d_i**2)`` over the block's
    irreps; this check runs before the trace check, so it fires regardless
    of the (untouched, still-consistent) projector."""
    metadata, arrays = _c4_metadata_and_arrays()
    metadata = copy.deepcopy(metadata)
    metadata["blocks"][0]["block_rank"] = 99
    _write(tmp_path, 4, 1, metadata, arrays)
    with pytest.raises(ValueError, match="sum of d\\^2"):
        load_group(4, 1, directory=tmp_path)


def test_block_ranks_do_not_sum_to_group_order_rejected(tmp_path: Path) -> None:
    """Every individual block is internally consistent, but the blocks'
    ranks don't sum to ``|G|`` -- achieved by appending a verbatim duplicate
    of block 0 (same projector, degree, rank, indices), so each per-block
    check still passes but the total now exceeds the group order.
    """
    metadata, arrays = _c4_metadata_and_arrays()
    metadata = copy.deepcopy(metadata)
    arrays = dict(arrays)
    metadata["blocks"].append(copy.deepcopy(metadata["blocks"][0]))
    new_index = len(metadata["blocks"]) - 1
    arrays[f"block_{new_index}_projector"] = arrays["block_0_projector"].copy()
    _write(tmp_path, 4, 1, metadata, arrays)
    with pytest.raises(ValueError, match="block ranks sum to"):
        load_group(4, 1, directory=tmp_path)


def test_block_irrep_degree_disagrees_with_irrep_dimension_rejected(tmp_path: Path) -> None:
    """``block.irrep_degree`` disagrees with the actual dimension of the
    irrep(s) it claims to cover; this check runs first in
    ``_validate_isotypic_blocks``, before the rank/trace checks."""
    metadata, arrays = _c4_metadata_and_arrays()
    metadata = copy.deepcopy(metadata)
    metadata["blocks"][0]["irrep_degree"] = 2  # actual irrep 0 has dimension 1
    _write(tmp_path, 4, 1, metadata, arrays)
    with pytest.raises(ValueError, match="disagrees with the degrees"):
        load_group(4, 1, directory=tmp_path)


# ---------------------------------------------------------------------------
# Format/identity checks in load_group itself
# ---------------------------------------------------------------------------


def test_format_one_artifact_rejected(tmp_path: Path) -> None:
    """A format-1 artifact is rejected outright: format 1's per-block
    ``dimension`` field was wrong for every irrep of degree >= 2, which is
    why ``ARTIFACT_FORMAT`` was bumped to 2."""
    metadata, arrays = _c4_metadata_and_arrays()
    metadata = copy.deepcopy(metadata)
    metadata["format"] = 1
    _write(tmp_path, 4, 1, metadata, arrays)
    with pytest.raises(ValueError, match="unsupported group artifact format"):
        load_group(4, 1, directory=tmp_path)


def test_small_group_mismatch_rejected(tmp_path: Path) -> None:
    metadata, arrays = _c4_metadata_and_arrays(order=4, index=1)
    # File lives at smallgroup_4_2.npz (what load_group(4, 2, ...) looks
    # for), but its metadata still claims SmallGroup(4, 1).
    _write(tmp_path, 4, 2, metadata, arrays)
    with pytest.raises(ValueError, match="does not match SmallGroup"):
        load_group(4, 2, directory=tmp_path)
