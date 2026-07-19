"""Serialisable, model-facing exports of Sage/GAP finite-group computations.

This module intentionally contains no group-theory algorithms.  ``GroupData``
is an adapter over an artifact written by ``scripts/export_group.py`` under
``sage -python``.  The only operations here are array indexing and artifact
validation, keeping Sage/GAP objects out of PyTorch processes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Format 2 replaced the ambiguous per-block ``dimension`` field (which meant
# different things in the exporter and the test fixtures) with the explicitly
# named ``irrep_degree`` and ``block_rank``.  Format-1 artifacts are rejected
# rather than reinterpreted, because their block dimensions were wrong for
# every irrep of degree >= 2.
ARTIFACT_FORMAT = 2


@dataclass(frozen=True)
class IrrepData:
    """One GAP-provided concrete irrep, converted at the exporter boundary."""

    matrices: np.ndarray  # [order, degree, degree], complex128
    character: np.ndarray  # [order], complex128
    dimension: int
    table_index: int
    field: str
    basis: str


@dataclass(frozen=True)
class IsotypicBlock:
    """One real isotypic component of the regular representation of G.

    Two different integers describe a block and must never be conflated (they
    agree only when the irrep degree is 1, which is why the difference hid for
    so long behind abelian test cases):

    ``irrep_degree``
        The degree ``d`` of the irrep(s) in this block, i.e. ``chi(e)``.  Both
        members of a merged complex-conjugate pair have the same degree, so a
        block has exactly one degree.  This is the number to use when a
        consumer wants to say "the 2-dimensional irrep".

    ``block_rank``
        The dimension of the isotypic block itself, i.e. the rank (equivalently
        the trace) of ``projector``.  An irrep of degree ``d`` occurs in the
        regular representation with multiplicity ``d``, so it spans ``d**2``
        dimensions; a block formed by merging a complex-conjugate pair spans
        ``2 * d**2``.  In general ``block_rank == sum_i d_i**2`` over the
        irreps merged into the block, and the block ranks sum to ``|G|``.  This
        is the number an energy/variance baseline needs: the random-init
        expectation of a block's energy share is ``block_rank / |G|``.
    """

    projector: np.ndarray
    irrep_degree: int
    block_rank: int
    irrep_indices: tuple[int, ...]


@dataclass(frozen=True)
class GroupData:
    """The complete, immutable model adapter for one canonical SmallGroup.

    All algebraic structure is exported by Sage/GAP.  ``subgroups`` and
    ``left_cosets`` are element-index arrays generated there too; this class
    never discovers group structure from the Cayley table.
    """

    order: int
    index: int
    description: str
    element_labels: tuple[str, ...]
    cayley_table: np.ndarray
    conjugacy_classes: tuple[np.ndarray, ...]
    character_table: np.ndarray
    frobenius_schur: np.ndarray
    irreps: tuple[IrrepData, ...]
    isotypic_blocks: tuple[IsotypicBlock, ...]
    subgroups: tuple[np.ndarray, ...]
    left_cosets: tuple[tuple[np.ndarray, ...], ...]
    provenance: dict[str, Any]

    @property
    def canonical_id(self) -> tuple[int, int]:
        return (self.order, self.index)

    @property
    def canonical_name(self) -> str:
        return f"SmallGroup({self.order},{self.index})"

    @property
    def elements(self) -> tuple[int, ...]:
        """Contiguous model tokens in the exported, stable element order."""
        return tuple(range(self.order))

    def label(self, token: int) -> str:
        return self.element_labels[token]

    def token(self, label: str) -> int:
        return self.element_labels.index(label)

    def cosets_for_subgroup(self, subgroup: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return Sage/GAP-exported left cosets for an exported subgroup."""
        for exported, cosets in zip(self.subgroups, self.left_cosets, strict=True):
            if np.array_equal(exported, subgroup):
                return cosets
        raise ValueError("subgroup is not part of this Sage/GAP artifact")


def artifact_dir() -> Path:
    configured = os.environ.get("GROUP_ARTIFACTS_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "data" / "group_artifacts"


def artifact_path(order: int, index: int, directory: Path | None = None) -> Path:
    return (directory or artifact_dir()) / f"smallgroup_{order}_{index}.npz"


def _complex_array(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return np.asarray(real, dtype=np.float64) + 1j * np.asarray(imag, dtype=np.float64)


def _validate_group_axioms(table: np.ndarray, path: Path) -> None:
    """Validate that ``table`` genuinely defines a finite group, not just an
    array of the right shape and index range.

    Checks, in order (each raises a specific ``ValueError`` naming the
    failing axiom and element/triple):

    1. Closure: every entry is a valid element index in ``[0, n)``.
    2. Latin square: every row and every column is a permutation of all n
       elements (necessary for a group; catches most corruption cheaply).
    3. Identity: some ``e`` with ``e*g == g*e == g`` for all ``g``.
    4. Inverses: every ``g`` has some ``h`` with ``g*h == h*g == e``.
    5. Associativity: ``(a*b)*c == a*(b*c)`` for every triple, exhaustively.

    All SmallGroup artifacts used by this project have order <= 255 (the
    panel spans orders 21-255; see docs/methodology.md). The associativity
    check is the only O(n^3) step. Benchmarked with numpy on a vectorised
    cyclic table of order 255 (the worst case in this project): ~0.10s wall
    time and ~265 MB of transient array memory (two (255,255,255) int64
    arrays) -- well under a second, so it always runs in full. There is no
    opt-out: loading a group artifact is not a hot path, and silently
    skipping the one check that actually verifies the group law would defeat
    the purpose of this function. If artifacts of order well beyond 255 are
    ever introduced, re-benchmark before assuming this still holds.
    """
    n = table.shape[0]
    if table.shape != (n, n) or table.min() < 0 or table.max() >= n:
        raise ValueError(
            f"invalid Cayley table in {path}: entries must be indices in [0, {n}) "
            "(closure violation)"
        )

    idx = np.arange(n)

    row_sorted = np.sort(table, axis=1)
    bad_rows = np.flatnonzero(np.any(row_sorted != idx, axis=1))
    if bad_rows.size:
        raise ValueError(
            f"invalid Cayley table in {path}: row {int(bad_rows[0])} is not a permutation of "
            f"0..{n - 1} (Latin-square violation)"
        )
    col_sorted = np.sort(table, axis=0)
    bad_cols = np.flatnonzero(np.any(col_sorted != idx[:, None], axis=0))
    if bad_cols.size:
        raise ValueError(
            f"invalid Cayley table in {path}: column {int(bad_cols[0])} is not a permutation of "
            f"0..{n - 1} (Latin-square violation)"
        )

    identity_candidates = np.flatnonzero(
        np.all(table == idx[None, :], axis=1) & np.all(table == idx[:, None], axis=0)
    )
    if identity_candidates.size == 0:
        raise ValueError(f"invalid Cayley table in {path}: no two-sided identity element")
    e = int(identity_candidates[0])

    has_inverse = np.any((table == e) & (table.T == e), axis=1)
    missing = np.flatnonzero(~has_inverse)
    if missing.size:
        raise ValueError(
            f"invalid Cayley table in {path}: element {int(missing[0])} has no two-sided inverse"
        )

    left = table[table[:, :, None], idx[None, None, :]]
    right = table[idx[:, None, None], table[None, :, :]]
    mismatches = np.argwhere(left != right)
    if mismatches.size:
        a, b, c = (int(v) for v in mismatches[0])
        raise ValueError(
            f"invalid Cayley table in {path}: associativity fails at "
            f"({a}*{b})*{c}={int(left[a, b, c])} != {a}*({b}*{c})={int(right[a, b, c])}"
        )


def _validate_isotypic_blocks(
    blocks: tuple[IsotypicBlock, ...],
    irreps: tuple[IrrepData, ...],
    order: int,
    path: Path,
) -> None:
    """Check the two block integers against the artifact's own projectors.

    ``block_rank`` must equal ``trace(projector)`` (projectors are idempotent
    and symmetric, so the trace is the rank) and must equal ``sum_i d_i**2``
    over the irreps merged into the block; the ranks must sum to ``|G|``.  This
    is the invariant that a format-1 artifact violated for every irrep of
    degree >= 2.
    """
    for i, block in enumerate(blocks):
        degrees = [irreps[j].dimension for j in block.irrep_indices]
        if any(degree != block.irrep_degree for degree in degrees):
            raise ValueError(
                f"invalid isotypic block {i} in {path}: irrep_degree={block.irrep_degree} "
                f"disagrees with the degrees {degrees} of its irreps {block.irrep_indices}"
            )
        expected = sum(degree**2 for degree in degrees)
        if block.block_rank != expected:
            raise ValueError(
                f"invalid isotypic block {i} in {path}: block_rank={block.block_rank}, "
                f"but sum of d^2 over irreps {block.irrep_indices} is {expected}"
            )
        trace = float(np.trace(block.projector))
        if abs(trace - block.block_rank) > 1e-6:
            raise ValueError(
                f"invalid isotypic block {i} in {path}: block_rank={block.block_rank} "
                f"but trace(projector)={trace:.6f}"
            )
    total = sum(block.block_rank for block in blocks)
    if total != order:
        raise ValueError(
            f"invalid isotypic blocks in {path}: block ranks sum to {total}, expected |G|={order}"
        )


def load_group(order: int, index: int, directory: Path | None = None) -> GroupData:
    """Load a Sage/GAP artifact or explain exactly how to create it."""
    path = artifact_path(order, index, directory)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing Sage/GAP group artifact {path}. Create it with: "
            f"sage -python scripts/export_group.py --order {order} --index {index}. "
            "The project does not fall back to bespoke finite-group code."
        )
    with np.load(path, allow_pickle=False) as raw:
        metadata = json.loads(str(raw["metadata"].item()))
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"unsupported group artifact format in {path}")
        if tuple(metadata["small_group"]) != (order, index):
            raise ValueError(f"artifact {path} does not match SmallGroup({order},{index})")
        n = int(metadata["order"])
        table = np.asarray(raw["cayley_table"], dtype=np.int64)
        _validate_group_axioms(table, path)
        classes = tuple(
            np.asarray(raw[f"class_{i}"], dtype=np.int64) for i in range(metadata["classes"])
        )
        characters = _complex_array(raw["character_real"], raw["character_imag"])
        fs = np.asarray(raw["frobenius_schur"], dtype=np.int64)
        irreps = tuple(
            IrrepData(
                matrices=_complex_array(raw[f"irrep_{i}_real"], raw[f"irrep_{i}_imag"]),
                character=_complex_array(
                    raw[f"irrep_{i}_character_real"], raw[f"irrep_{i}_character_imag"]
                ),
                dimension=int(metadata["irreps"][i]["dimension"]),
                table_index=i,
                field=str(metadata["irreps"][i]["field"]),
                basis=str(metadata["irreps"][i]["basis"]),
            )
            for i in range(len(metadata["irreps"]))
        )
        blocks = tuple(
            IsotypicBlock(
                projector=np.asarray(raw[f"block_{i}_projector"], dtype=np.float64),
                irrep_degree=int(block["irrep_degree"]),
                block_rank=int(block["block_rank"]),
                irrep_indices=tuple(block["irrep_indices"]),
            )
            for i, block in enumerate(metadata["blocks"])
        )
        _validate_isotypic_blocks(blocks, irreps, n, path)
        subgroups = tuple(
            np.asarray(raw[f"subgroup_{i}"], dtype=np.int64) for i in range(metadata["subgroups"])
        )
        cosets = tuple(
            tuple(np.asarray(raw[f"coset_{i}_{j}"], dtype=np.int64) for j in range(count))
            for i, count in enumerate(metadata["coset_counts"])
        )
    return GroupData(
        order=n,
        index=index,
        description=str(metadata["description"]),
        element_labels=tuple(metadata["element_labels"]),
        cayley_table=table,
        conjugacy_classes=classes,
        character_table=characters,
        frobenius_schur=fs,
        irreps=irreps,
        isotypic_blocks=blocks,
        subgroups=subgroups,
        left_cosets=cosets,
        provenance=dict(metadata["provenance"]),
    )
