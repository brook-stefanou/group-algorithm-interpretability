"""Sage-free ground-truth verification of the exported representation theory.

Every assertion here is self-verifying: a mathematical identity that pins down
the correct answer without trusting a memorised character table. Conjugacy
classes, inverses and the identity are re-derived from the Cayley table alone,
so the artifact's own class/character/projector arrays are always checked
against structure computed independently of them.

The only hard-coded anchors are the irrep degree multisets of a handful of tiny
groups (all 1s for cyclic; {1,1,2} for S3; {1,1,1,1,2} for D4 and Q8) -- these
are trivially checkable by hand, and the sum-of-squares and orthogonality
relations below confirm them independently anyway.

Every test below runs against TWO artifact corpora, and needs no SageMath to do
so (both are already on disk at test time):

``fixture``
    The pure-Python corpus written at session start by ``support_artifacts``
    (see ``conftest``).  Covers all eight groups.

``sage``
    The committed golden corpus in ``tests/data/group_artifacts``, produced by
    the REAL exporter (``scripts/export_group.py``) under passagemath/libgap and
    checked into the repository.  Covers S3, D4, Q8 and C8.

The second corpus is the point: ``representations/*`` is pure accessors over an
``.npz``, so the fixture corpus alone verifies the artifact contract and the
loader while leaving ``scripts/export_group.py`` executed by nothing -- the
seam the historical isotypic-dimension bug lived in. The golden artifacts are
committed DATA, so testing against them costs the offline gate nothing and
requires no Sage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from group_algorithm_interp.groups.data import GroupData, load_group
from group_algorithm_interp.representations import (
    compute_character_table,
    extract_irreps,
    frobenius_schur_indicators,
    real_isotypic_blocks,
)

# Golden artifacts produced by scripts/export_group.py and committed to the repo.
GOLDEN_DIR = Path(__file__).parent / "data" / "group_artifacts"

# (order, index, human name, irrep degree multiset)
GROUPS: list[tuple[int, int, str, list[int]]] = [
    (2, 1, "C2", [1, 1]),
    (4, 1, "C4", [1, 1, 1, 1]),
    (6, 2, "C6", [1] * 6),
    (7, 1, "C7", [1] * 7),
    (8, 1, "C8", [1] * 8),
    (6, 1, "S3", [1, 1, 2]),
    (8, 3, "D4", [1, 1, 1, 1, 2]),
    (8, 4, "Q8", [1, 1, 1, 1, 2]),
]

# The subset for which a real Sage/GAP export is committed under tests/data.
GOLDEN_GROUPS: frozenset[tuple[int, int]] = frozenset({(6, 1), (8, 3), (8, 4), (8, 1)})

# A case is one (group, corpus) pair.  ``directory=None`` resolves to the
# generated fixture corpus via GROUP_ARTIFACTS_DIR (set in conftest).
Case = tuple[int, int, list[int], Path | None]


def _cases(only: set[tuple[int, int]] | None = None) -> tuple[list[Case], list[str]]:
    cases: list[Case] = []
    ids: list[str] = []
    for order, index, name, degrees in GROUPS:
        if only is not None and (order, index) not in only:
            continue
        cases.append((order, index, degrees, None))
        ids.append(f"{name}-fixture")
        if (order, index) in GOLDEN_GROUPS:
            cases.append((order, index, degrees, GOLDEN_DIR))
            ids.append(f"{name}-sage")
    return cases, ids


CASES, IDS = _cases()
PARAMS = CASES

ATOL = 1e-9


def test_golden_sage_corpus_is_present() -> None:
    """A missing golden corpus must fail, never silently skip -- otherwise every
    ``*-sage`` case below vanishes from the run and the exporter goes untested
    again."""
    missing = [
        f"smallgroup_{order}_{index}.npz"
        for order, index in sorted(GOLDEN_GROUPS)
        if not (GOLDEN_DIR / f"smallgroup_{order}_{index}.npz").is_file()
    ]
    assert not missing, (
        f"committed Sage/GAP artifacts missing from {GOLDEN_DIR}: {missing}. "
        "Regenerate with: uv sync --extra sage && "
        "uv run python scripts/export_group.py --order O --index I "
        "--output tests/data/group_artifacts/smallgroup_O_I.npz"
    )


@pytest.fixture(params=PARAMS, ids=IDS)
def group(request: pytest.FixtureRequest) -> GroupData:
    order, index, _, directory = request.param
    return load_group(order, index, directory)


# --------------------------------------------------------------------------
# Structure re-derived from the Cayley table alone (never from the artifact's
# own class / character / projector arrays).
# --------------------------------------------------------------------------


def identity_of(table: np.ndarray) -> int:
    n = len(table)
    for e in range(n):
        if np.array_equal(table[e], np.arange(n)) and np.array_equal(table[:, e], np.arange(n)):
            return e
    raise AssertionError("Cayley table has no two-sided identity")


def inverses_of(table: np.ndarray) -> np.ndarray:
    e = identity_of(table)
    return np.array([int(np.flatnonzero(table[g] == e)[0]) for g in range(len(table))])


def conjugacy_classes_of(table: np.ndarray) -> list[list[int]]:
    """Classes {x g x^-1 : x in G}, derived only from the multiplication table."""
    inv = inverses_of(table)
    n = len(table)
    remaining = set(range(n))
    classes: list[list[int]] = []
    while remaining:
        g = min(remaining)
        cls = sorted({int(table[table[x, g], inv[x]]) for x in range(n)})
        classes.append(cls)
        remaining -= set(cls)
    return classes


def regular_representation(table: np.ndarray) -> np.ndarray:
    """Left regular rep: R[g][x, y] == 1 iff x == g * y."""
    n = len(table)
    reg = np.zeros((n, n, n), dtype=np.float64)
    for g in range(n):
        reg[g, table[g], np.arange(n)] = 1.0
    return reg


# --------------------------------------------------------------------------
# The Cayley table itself must be a group (validates the ground-truth source).
# --------------------------------------------------------------------------


def test_cayley_table_is_a_group(group: GroupData) -> None:
    table = group.cayley_table
    n = group.order
    assert table.shape == (n, n)
    # Closure and a Latin square (each row/column a permutation).
    for i in range(n):
        assert sorted(table[i].tolist()) == list(range(n))
        assert sorted(table[:, i].tolist()) == list(range(n))
    # Associativity: (a*b)*c == a*(b*c) for all triples.
    left = table[table[:, :, None], np.arange(n)[None, None, :]]
    right = table[np.arange(n)[:, None, None], table[None, :, :]]
    assert np.array_equal(left, right), "Cayley table is not associative"
    e = identity_of(table)
    inv = inverses_of(table)
    assert np.array_equal(table[np.arange(n), inv], np.full(n, e))
    assert np.array_equal(table[inv, np.arange(n)], np.full(n, e))


# --------------------------------------------------------------------------
# Irrep matrices.
# --------------------------------------------------------------------------


def test_irreps_are_homomorphisms(group: GroupData) -> None:
    """rho(a) @ rho(b) == rho(a*b) for EVERY pair, read from the Cayley table."""
    table = group.cayley_table
    for irrep in extract_irreps(group):
        rho = irrep.matrices
        assert rho.shape == (group.order, irrep.dimension, irrep.dimension)
        products = np.einsum("aij,bjk->abik", rho, rho)
        images = rho[table]
        worst = float(np.abs(products - images).max())
        assert worst < ATOL, (
            f"irrep {irrep.table_index} (deg {irrep.dimension}) is not a homomorphism; "
            f"max |rho(a)rho(b) - rho(ab)| = {worst:.3e}"
        )


def test_irreps_are_unitary(group: GroupData) -> None:
    for irrep in extract_irreps(group):
        rho = irrep.matrices
        eye = np.broadcast_to(np.eye(irrep.dimension), rho.shape)
        gram = rho @ np.conjugate(np.transpose(rho, (0, 2, 1)))
        worst = float(np.abs(gram - eye).max())
        assert worst < ATOL, (
            f"irrep {irrep.table_index} is not unitary; max |rho rho^H - I| = {worst:.3e}"
        )


def test_identity_maps_to_identity_matrix(group: GroupData) -> None:
    e = identity_of(group.cayley_table)
    for irrep in extract_irreps(group):
        assert np.allclose(irrep.matrices[e], np.eye(irrep.dimension), atol=ATOL)
        # chi(e) is the degree, by definition.
        assert abs(complex(irrep.character[e]) - irrep.dimension) < ATOL


def test_character_is_the_trace_of_the_irrep(group: GroupData) -> None:
    for irrep in extract_irreps(group):
        traces = np.trace(irrep.matrices, axis1=1, axis2=2)
        worst = float(np.abs(traces - irrep.character).max())
        assert worst < ATOL, (
            f"irrep {irrep.table_index}: character != trace(rho); max diff {worst:.3e}"
        )


def test_character_of_inverse_is_the_conjugate(group: GroupData) -> None:
    inv = inverses_of(group.cayley_table)
    for irrep in extract_irreps(group):
        chi = irrep.character
        assert np.allclose(chi[inv], np.conjugate(chi), atol=ATOL)


# --------------------------------------------------------------------------
# Conjugacy classes and class functions.
# --------------------------------------------------------------------------


def test_exported_conjugacy_classes_match_the_cayley_table(group: GroupData) -> None:
    derived = conjugacy_classes_of(group.cayley_table)
    exported = sorted((sorted(int(x) for x in cls) for cls in group.conjugacy_classes))
    assert sorted(derived) == exported
    assert sum(len(cls) for cls in derived) == group.order


def test_characters_are_class_functions(group: GroupData) -> None:
    """Each character is constant on every independently derived conjugacy class."""
    for irrep in extract_irreps(group):
        for cls in conjugacy_classes_of(group.cayley_table):
            values = irrep.character[cls]
            spread = float(np.abs(values - values[0]).max())
            assert spread < ATOL, (
                f"irrep {irrep.table_index} is not constant on class {cls}; spread {spread:.3e}"
            )


def test_number_of_irreps_equals_number_of_conjugacy_classes(group: GroupData) -> None:
    table, classes = compute_character_table(group)
    derived = conjugacy_classes_of(group.cayley_table)
    assert len(extract_irreps(group)) == len(derived)
    assert table.shape == (len(derived), len(derived))
    assert len(classes) == len(derived)


def test_stored_class_character_table_agrees_with_per_element_characters(
    group: GroupData,
) -> None:
    table, classes = compute_character_table(group)
    for row, irrep in enumerate(extract_irreps(group)):
        for col, cls in enumerate(classes):
            assert abs(complex(table[row, col]) - complex(irrep.character[cls[0]])) < ATOL


# --------------------------------------------------------------------------
# Orthogonality relations and the degree formula.
# --------------------------------------------------------------------------


def test_sum_of_squares_of_degrees_equals_group_order(group: GroupData) -> None:
    total = sum(irrep.dimension**2 for irrep in extract_irreps(group))
    assert total == group.order, f"sum of d_i^2 = {total}, expected |G| = {group.order}"


def test_first_orthogonality_relation(group: GroupData) -> None:
    """(1/|G|) * sum_g chi_i(g) conj(chi_j(g)) == delta_ij, weighted by class sizes."""
    table, classes = compute_character_table(group)
    sizes = np.array([len(cls) for cls in classes], dtype=np.float64)
    gram = (table * sizes) @ np.conjugate(table).T / group.order
    eye = np.eye(table.shape[0])
    worst = float(np.abs(gram - eye).max())
    assert worst < 1e-8, (
        f"rows of the character table are not orthonormal; "
        f"max |<chi_i, chi_j> - delta_ij| = {worst:.3e}\n{np.round(gram.real, 4)}"
    )


def test_second_orthogonality_relation(group: GroupData) -> None:
    """sum_i chi_i(c) conj(chi_i(c')) == delta_{c,c'} * |G| / |c| (column relation)."""
    table, classes = compute_character_table(group)
    sizes = np.array([len(cls) for cls in classes], dtype=np.float64)
    columns = np.conjugate(table).T @ table
    expected = np.diag(group.order / sizes)
    worst = float(np.abs(columns - expected).max())
    assert worst < 1e-8, (
        f"columns of the character table are not orthogonal; max diff {worst:.3e}\n"
        f"{np.round(columns.real, 4)}"
    )


def test_frobenius_schur_indicator_matches_its_definition(group: GroupData) -> None:
    """FS(chi) = (1/|G|) sum_g chi(g^2), and lies in {-1, 0, 1}."""
    table = group.cayley_table
    squares = table[np.arange(group.order), np.arange(group.order)]
    exported = frobenius_schur_indicators(group)
    for irrep in extract_irreps(group):
        derived = complex(irrep.character[squares].sum()) / group.order
        assert abs(derived.imag) < 1e-8
        assert abs(derived.real - round(derived.real)) < 1e-8
        assert round(derived.real) in (-1, 0, 1)
        assert int(exported[irrep.table_index]) == round(derived.real), (
            f"irrep {irrep.table_index}: exported FS={int(exported[irrep.table_index])}, "
            f"(1/|G|) sum chi(g^2) = {derived.real:.6f}"
        )


@pytest.mark.parametrize(("order", "index", "expected", "directory"), PARAMS, ids=IDS)
def test_irrep_degree_multiset(
    order: int, index: int, expected: list[int], directory: Path | None
) -> None:
    """Hard anchor: the degree multiset of each small group (hand-checkable)."""
    loaded = load_group(order, index, directory)
    assert sorted(irrep.dimension for irrep in loaded.irreps) == sorted(expected)


# --------------------------------------------------------------------------
# Isotypic projectors on the regular representation.
# --------------------------------------------------------------------------


def test_projectors_are_idempotent_and_symmetric(group: GroupData) -> None:
    for i, block in enumerate(real_isotypic_blocks(group)):
        p = block.projector
        assert p.shape == (group.order, group.order)
        idem = float(np.abs(p @ p - p).max())
        assert idem < 1e-8, f"block {i}: P@P != P; max diff {idem:.3e}"
        sym = float(np.abs(p - p.T).max())
        assert sym < 1e-8, f"block {i}: P is not symmetric; max diff {sym:.3e}"


def test_projectors_are_mutually_orthogonal_and_resolve_the_identity(
    group: GroupData,
) -> None:
    blocks = real_isotypic_blocks(group)
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks):
            if i == j:
                continue
            worst = float(np.abs(a.projector @ b.projector).max())
            assert worst < 1e-8, f"blocks {i},{j} are not orthogonal; max |P_i P_j| = {worst:.3e}"
    total = sum((block.projector for block in blocks), np.zeros((group.order, group.order)))
    worst = float(np.abs(total - np.eye(group.order)).max())
    assert worst < 1e-8, f"isotypic projectors do not sum to I; max diff {worst:.3e}"


def test_projectors_commute_with_the_regular_representation(group: GroupData) -> None:
    """A genuine isotypic projector is G-equivariant: P R(g) == R(g) P for all g."""
    reg = regular_representation(group.cayley_table)
    for i, block in enumerate(real_isotypic_blocks(group)):
        p = block.projector
        worst = float(np.abs(p @ reg - reg @ p).max())
        assert worst < 1e-8, (
            f"block {i} does not commute with the regular representation; max diff {worst:.3e}"
        )


def test_regular_representation_multiplicity(group: GroupData) -> None:
    """Each irrep occurs in the regular rep with multiplicity equal to its degree,
    so the isotypic block of a degree-d irrep has rank d^2."""
    irreps = extract_irreps(group)
    for i, block in enumerate(real_isotypic_blocks(group)):
        expected = sum(irreps[j].dimension ** 2 for j in block.irrep_indices)
        trace = float(np.trace(block.projector).real)
        rank = int(np.linalg.matrix_rank(block.projector, tol=1e-8))
        assert abs(trace - expected) < 1e-8, (
            f"block {i} (irreps {block.irrep_indices}): trace(P) = {trace:.6f}, "
            f"expected sum of d^2 = {expected}"
        )
        assert rank == expected, (
            f"block {i} (irreps {block.irrep_indices}): rank(P) = {rank}, expected {expected}"
        )
    total = sum(
        sum(irreps[j].dimension ** 2 for j in block.irrep_indices)
        for block in real_isotypic_blocks(group)
    )
    assert total == group.order


def test_every_irrep_belongs_to_exactly_one_block(group: GroupData) -> None:
    blocks = real_isotypic_blocks(group)
    seen = [j for block in blocks for j in block.irrep_indices]
    assert sorted(seen) == list(range(len(extract_irreps(group))))


# --------------------------------------------------------------------------
# The two stored block integers, pinned against convention-free ground truth.
#
# ``irrep_degree`` and ``block_rank`` coincide for every degree-1 irrep, so an
# assertion made only against an abelian group can't tell them apart -- the
# exporter stored len(irrep_indices) * d, which is neither quantity once d >= 2.
# Every check below runs over abelian AND non-abelian fixtures, with expected
# values recomputed from the group itself: chi(e) for the degree, trace/rank of
# the projector for the block rank.
# --------------------------------------------------------------------------


def test_stored_irrep_degree_is_chi_at_the_identity(group: GroupData) -> None:
    """irrep_degree == chi(e) for every irrep merged into the block."""
    irreps = extract_irreps(group)
    e = identity_of(group.cayley_table)
    for i, block in enumerate(real_isotypic_blocks(group)):
        for j in block.irrep_indices:
            chi_e = complex(irreps[j].character[e])
            assert abs(chi_e.imag) < ATOL
            assert abs(chi_e.real - block.irrep_degree) < ATOL, (
                f"block {i}: irrep_degree = {block.irrep_degree}, but irrep {j} has "
                f"chi(e) = {chi_e.real:.6f}"
            )
        # A conjugate pair shares a degree, so a block has exactly one.
        degrees = {irreps[j].dimension for j in block.irrep_indices}
        assert degrees == {block.irrep_degree}


def test_stored_block_rank_is_the_rank_of_its_projector(group: GroupData) -> None:
    """block_rank == trace(P) == rank(P) == sum of d_i^2, and the ranks sum to |G|."""
    irreps = extract_irreps(group)
    blocks = real_isotypic_blocks(group)
    for i, block in enumerate(blocks):
        trace = float(np.trace(block.projector).real)
        rank = int(np.linalg.matrix_rank(block.projector, tol=1e-8))
        sum_of_squares = sum(irreps[j].dimension ** 2 for j in block.irrep_indices)
        assert block.block_rank == round(trace), (
            f"block {i} (irreps {block.irrep_indices}): block_rank = {block.block_rank}, "
            f"trace(P) = {trace:.6f}"
        )
        assert block.block_rank == rank, (
            f"block {i} (irreps {block.irrep_indices}): block_rank = {block.block_rank}, "
            f"rank(P) = {rank}"
        )
        assert block.block_rank == sum_of_squares
        # d^2 for a single real irrep; 2*d^2 for a merged complex-conjugate pair.
        assert block.block_rank == len(block.irrep_indices) * block.irrep_degree**2
    total = sum(block.block_rank for block in blocks)
    assert total == group.order, f"block ranks sum to {total}, expected |G| = {group.order}"


_TWO_DIM_CASES, _TWO_DIM_IDS = _cases(only={(6, 1), (8, 3), (8, 4)})


@pytest.mark.parametrize(
    ("order", "index", "_degrees", "directory"), _TWO_DIM_CASES, ids=_TWO_DIM_IDS
)
def test_degree_two_block_has_rank_four(
    order: int, index: int, _degrees: list[int], directory: Path | None
) -> None:
    """The sole degree-2 block of S3/D4/Q8 spans d^2 = 4 dimensions. The
    irrep is self-conjugate, so it is never merged, yet the block still has
    rank 4, not 2 -- the old exporter recorded ``len(irrep_indices) * d == 2``.
    The ``-sage`` corpus is what puts ``scripts/export_group.py`` itself under
    this assertion."""
    loaded = load_group(order, index, directory)
    two_dim = [block for block in loaded.isotypic_blocks if block.irrep_degree == 2]
    assert len(two_dim) == 1
    block = two_dim[0]
    assert len(block.irrep_indices) == 1
    assert block.block_rank == 4
    assert round(float(np.trace(block.projector))) == 4
    assert int(np.linalg.matrix_rank(block.projector, tol=1e-8)) == 4
    assert sum(item.block_rank for item in loaded.isotypic_blocks) == loaded.order


_MERGED_CASES, _MERGED_IDS = _cases(only={(4, 1), (8, 1)})


@pytest.mark.parametrize(
    ("order", "index", "_degrees", "directory"), _MERGED_CASES, ids=_MERGED_IDS
)
def test_merged_conjugate_pairs_have_rank_two_d_squared(
    order: int, index: int, _degrees: list[int], directory: Path | None
) -> None:
    """A cyclic group of order >= 3 has complex characters that merge in pairs:
    each merged block holds two degree-1 irreps, rank 2 * 1^2 == 2, while its
    irrep_degree stays 1 -- the one case where the old exporter formula
    happened to give the right answer."""
    loaded = load_group(order, index, directory)
    merged = [block for block in loaded.isotypic_blocks if len(block.irrep_indices) == 2]
    assert merged, f"SmallGroup({order},{index}) should have merged conjugate pairs"
    for block in merged:
        assert block.irrep_degree == 1
        assert block.block_rank == 2 * block.irrep_degree**2
        assert block.block_rank == round(float(np.trace(block.projector)))
    unmerged = [block for block in loaded.isotypic_blocks if len(block.irrep_indices) == 1]
    for block in unmerged:
        assert block.block_rank == block.irrep_degree**2
    assert sum(block.block_rank for block in loaded.isotypic_blocks) == loaded.order


def test_random_init_energy_baseline_is_block_rank_over_order(group: GroupData) -> None:
    """An isotropic random vector puts a block_rank/|G| share of its squared
    norm in each isotypic block."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((4096, group.order))
    total = float((vectors**2).sum())
    for block in real_isotypic_blocks(group):
        share = float((vectors @ block.projector.T * vectors).sum()) / total
        expected = block.block_rank / group.order
        assert abs(share - expected) < 0.02, (
            f"block {block.irrep_indices}: energy share {share:.4f}, "
            f"expected block_rank/|G| = {expected:.4f}"
        )
