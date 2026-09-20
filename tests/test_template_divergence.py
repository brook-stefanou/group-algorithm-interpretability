"""Tests for the template-divergence instrument (GCR sparse-irrep vs coset,
screened for degeneracy).

The panel's only locally-exported-subgroup real artifacts are D32, QD32, Q32
(SmallGroup(32,18/19/20)) and SL(2,3).C2, GL(2,3) (SmallGroup(48,28/29)) --
every one of those with a nontrivial core-free subgroup lands degenerate
under this instrument's threshold (D32/QD32 at 30/32, GL(2,3) at 26/48), and
Q32 has none at all. There is no real exported artifact locally with a
low-rank core-free subgroup to serve as the DEFINED positive case, so one is
constructed by hand: S4 (SmallGroup(24,12)) with H = Stab(3) =~ S3, the
textbook core-free point-stabiliser (index 4, Ind_H^G 1 support 10/24). The
Cayley table, conjugacy classes (by cycle type) and subgroup are all computed
directly from the 24 permutations of {0,1,2,3}, not hand-typed; only the
character table is the standard, universally-tabulated one for S4 (matrix
entries are unused by this module and left as placeholders).
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from group_algorithm_interp.groups.data import GroupData, IrrepData, IsotypicBlock, load_group
from group_algorithm_interp.instruments.occupancy import analytic_null
from group_algorithm_interp.instruments.template_divergence import (
    DEFINED,
    MAX_COSET_SUPPORT_FRACTION,
    MIN_FAVOURED_FRACTION,
    MIN_FLIP_OVER_RANDOM,
    MIN_MP_MINUS_BILINEAR,
    UNDEFINED,
    SetOverlap,
    block_kernel,
    causally_used_blocks,
    compare_to_templates,
    data_driven_comparison,
    degeneracy_screen,
    gcr_sparse_template,
    is_faithful,
    minimal_separating_blocks,
    minimum_faithful_sets,
    product_carrying_blocks,
)
from group_algorithm_interp.instruments.templates import induction_template

_REAL_ARTIFACTS = (
    __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "group_artifacts"
)


def _require_real_artifact(order: int, index: int):
    path = _REAL_ARTIFACTS / f"smallgroup_{order}_{index}.npz"
    if not path.is_file():
        pytest.skip(f"real artifact {path} not present locally")
    return load_group(order, index, directory=_REAL_ARTIFACTS)


# ---------------------------------------------------------------------------
# A hand-constructed S4: the DEFINED positive case (see module docstring)
# ---------------------------------------------------------------------------


def _build_s4() -> GroupData:
    perms = list(itertools.permutations(range(4)))
    n = len(perms)
    index_of = {p: i for i, p in enumerate(perms)}

    def compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(p[q[x]] for x in range(4))

    table = np.array(
        [[index_of[compose(perms[a], perms[b])] for b in range(n)] for a in range(n)],
        dtype=np.int64,
    )

    def cycle_type(p: tuple[int, ...]) -> tuple[int, ...]:
        seen = [False] * 4
        lengths = []
        for start in range(4):
            if seen[start]:
                continue
            length = 0
            y = start
            while not seen[y]:
                seen[y] = True
                y = p[y]
                length += 1
            lengths.append(length)
        return tuple(sorted(lengths, reverse=True))

    types = [cycle_type(p) for p in perms]

    # The standard S4 character table, by cycle type (identity; transposition;
    # double-transposition; 3-cycle; 4-cycle). Textbook values (e.g. the
    # character table of S4): orthogonality is asserted below rather than
    # trusted.
    char_by_type: dict[tuple[int, ...], dict[str, complex]] = {
        (1, 1, 1, 1): {"triv": 1, "sign": 1, "two": 2, "three": 3, "three_p": 3},
        (2, 1, 1): {"triv": 1, "sign": -1, "two": 0, "three": 1, "three_p": -1},
        (2, 2): {"triv": 1, "sign": 1, "two": 2, "three": -1, "three_p": -1},
        (3, 1): {"triv": 1, "sign": 1, "two": -1, "three": 0, "three_p": 0},
        (4,): {"triv": 1, "sign": -1, "two": 0, "three": -1, "three_p": 1},
    }
    class_sizes = {t: types.count(t) for t in char_by_type}
    assert class_sizes == {(1, 1, 1, 1): 1, (2, 1, 1): 6, (2, 2): 3, (3, 1): 8, (4,): 6}

    def char_array(key: str) -> np.ndarray:
        return np.array([complex(char_by_type[t][key]) for t in types], dtype=np.complex128)

    names = ["triv", "sign", "two", "three", "three_p"]
    degrees = {"triv": 1, "sign": 1, "two": 2, "three": 3, "three_p": 3}
    characters = {name: char_array(name) for name in names}

    # Orthogonality of the assumed character table, checked against the
    # actual (computed) class sizes rather than assumed.
    class_reps = list(char_by_type.keys())
    sizes = np.array([class_sizes[t] for t in class_reps])
    for a in names:
        for b in names:
            va = np.array([char_by_type[t][a] for t in class_reps])
            vb = np.array([char_by_type[t][b] for t in class_reps])
            inner = float(np.real(np.sum(sizes * va * np.conj(vb)))) / 24
            expected = 1.0 if a == b else 0.0
            assert abs(inner - expected) < 1e-9, (a, b, inner)

    irreps = tuple(
        IrrepData(
            matrices=np.zeros((n, degrees[name], degrees[name]), dtype=np.complex128),
            character=characters[name],
            dimension=degrees[name],
            table_index=i,
            field="Q",
            basis="hand-constructed (unused placeholder matrices)",
        )
        for i, name in enumerate(names)
    )

    def projector_placeholder() -> np.ndarray:
        # Not used by template_divergence.py (which reads only block_rank,
        # irrep_degree and irrep_indices); kept as an explicit zero array
        # rather than a fabricated real projector.
        return np.zeros((n, n), dtype=np.float64)

    blocks = tuple(
        IsotypicBlock(
            projector=projector_placeholder(),
            irrep_degree=degrees[name],
            block_rank=degrees[name] ** 2,
            irrep_indices=(i,),
        )
        for i, name in enumerate(names)
    )
    assert sum(b.block_rank for b in blocks) == n

    identity_perm = tuple(range(4))
    identity_index = index_of[identity_perm]
    # H = Stab(3), the point stabiliser of {0,1,2,3} fixing 3: the textbook
    # core-free subgroup of S4 (index 4, minimal faithful permutation degree).
    subgroup_h = np.array(sorted(i for i, p in enumerate(perms) if p[3] == 3), dtype=np.int64)
    assert subgroup_h.size == 6
    trivial_subgroup = np.array([identity_index], dtype=np.int64)
    full_group = np.arange(n, dtype=np.int64)
    subgroups = (trivial_subgroup, subgroup_h, full_group)

    def cosets_for(members: np.ndarray) -> tuple[np.ndarray, ...]:
        unseen = set(range(n))
        result = []
        while unseen:
            rep = min(unseen)
            coset = np.array(sorted({int(table[rep, h]) for h in members.tolist()}), dtype=np.int64)
            result.append(coset)
            unseen -= set(coset.tolist())
        return tuple(result)

    left_cosets = tuple(cosets_for(s) for s in subgroups)

    classes_by_type: dict[tuple[int, ...], np.ndarray] = {}
    for t in char_by_type:
        classes_by_type[t] = np.array(
            sorted(i for i, p in enumerate(perms) if cycle_type(p) == t), dtype=np.int64
        )
    conjugacy_classes = tuple(classes_by_type[t] for t in char_by_type)
    character_table = np.array(
        [[char_by_type[t][name] for t in char_by_type] for name in names], dtype=np.complex128
    )
    frobenius_schur = np.array([1, 1, 1, 1, 1], dtype=np.int64)

    return GroupData(
        order=n,
        index=12,  # S4 is GAP's SmallGroup(24,12); used here for a real label only
        description="S4 (hand-constructed for template_divergence tests)",
        element_labels=tuple(str(p) for p in perms),
        cayley_table=table,
        conjugacy_classes=conjugacy_classes,
        character_table=character_table,
        frobenius_schur=frobenius_schur,
        irreps=irreps,
        isotypic_blocks=blocks,
        subgroups=subgroups,
        left_cosets=left_cosets,
        provenance={"backend": "hand-constructed", "purpose": "template_divergence tests"},
    )


@pytest.fixture(scope="module")
def s4() -> GroupData:
    return _build_s4()


# ---------------------------------------------------------------------------
# Degeneracy screen
# ---------------------------------------------------------------------------


def test_d32_coset_target_is_degenerate():
    group = _require_real_artifact(32, 18)
    screen = degeneracy_screen(group)
    assert screen.coset_defined is True
    assert screen.min_corefree_index == 16
    assert screen.support_rank == 30
    assert screen.support_fraction == pytest.approx(30 / 32)
    assert screen.verdict == UNDEFINED
    assert "degenerate" in screen.reason.lower()


def test_qd32_coset_target_is_degenerate():
    group = _require_real_artifact(32, 19)
    screen = degeneracy_screen(group)
    assert screen.coset_defined is True
    assert screen.support_fraction == pytest.approx(30 / 32)
    assert screen.verdict == UNDEFINED


def test_q32_generalised_quaternion_has_no_corefree_subgroup():
    group = _require_real_artifact(32, 20)
    screen = degeneracy_screen(group)
    assert screen.coset_defined is False
    assert screen.min_corefree_index == group.order
    assert screen.verdict == UNDEFINED
    assert "no nontrivial core-free subgroup" in screen.reason


def test_abelian_c32_is_undefined_via_artifact_incomplete():
    group = _require_real_artifact(32, 1)
    screen = degeneracy_screen(group)
    # C32 is exported with no subgroup section (n_subgroups_examined == 0):
    # abelian groups have only the trivial core-free subgroup by theorem, but
    # this artifact cannot itself distinguish that from an unexamined group,
    # so it must come back UNDEFINED via the artifact-incomplete branch.
    assert screen.n_subgroups_examined == 0
    assert screen.artifact_incomplete is True
    assert screen.verdict == UNDEFINED
    assert "artifact incomplete" in screen.reason


def test_gl23_coset_target_is_also_degenerate():
    # A second, structurally different real example (not dihedral-family):
    # GL(2,3), min-index core-free H has index 8 but support 26/48 = 0.5417,
    # still at or above the 0.5 threshold.
    group = _require_real_artifact(48, 29)
    screen = degeneracy_screen(group)
    assert screen.coset_defined is True
    assert screen.support_fraction == pytest.approx(26 / 48)
    assert screen.verdict == UNDEFINED


def test_s4_stab3_is_the_defined_positive_case(s4: GroupData):
    screen = degeneracy_screen(s4)
    assert screen.coset_defined is True
    assert screen.min_corefree_index == 4
    assert screen.subgroup_order == 6
    assert screen.coset_index == 4
    assert screen.support_rank == 10  # trivial block (1) + std 3-dim block (9)
    assert screen.support_fraction == pytest.approx(10 / 24)
    assert screen.verdict == DEFINED
    assert screen.support_fraction is not None
    assert screen.support_fraction < MAX_COSET_SUPPORT_FRACTION


def test_screen_threshold_is_a_named_overridable_parameter():
    group = _require_real_artifact(32, 18)
    lenient = degeneracy_screen(group, max_support_fraction=0.99)
    assert lenient.verdict == DEFINED
    strict = degeneracy_screen(group, max_support_fraction=0.5)
    assert strict.verdict == UNDEFINED


def test_screen_rejects_an_out_of_range_threshold():
    group = _require_real_artifact(32, 18)
    with pytest.raises(ValueError, match="max_support_fraction"):
        degeneracy_screen(group, max_support_fraction=0.0)
    with pytest.raises(ValueError, match="max_support_fraction"):
        degeneracy_screen(group, max_support_fraction=1.5)


# ---------------------------------------------------------------------------
# GCR sparse-irrep template
# ---------------------------------------------------------------------------


def test_gcr_template_on_s4_selects_the_faithful_low_degree_blocks(s4: GroupData):
    selected = minimal_separating_blocks(s4)
    names = ["triv", "sign", "two", "three", "three_p"]
    selected_names = [names[j] for j in selected]
    # Greedy ascending-degree: sign (kernel A4) then two (kernel V4) then the
    # faithful standard 3-dim irrep (kernel {e}) finishes it off; three_p is
    # never needed.
    assert selected_names == ["sign", "two", "three"]
    template = gcr_sparse_template(s4)
    assert template.shape == (5,)
    assert template[0] == 0.0  # trivial
    assert template[4] == 0.0  # three_p, unused
    assert np.isclose(template.sum(), 1.0)
    # Energy proportional to block_rank: sign=1, two=4, three=9, total=14.
    assert np.allclose(template, [0.0, 1 / 14, 4 / 14, 9 / 14, 0.0])


def test_block_kernel_matches_known_s4_kernels(s4: GroupData):
    perms = list(itertools.permutations(range(4)))
    identity_index = perms.index(tuple(range(4)))
    # The sign irrep's kernel is A4 (12 even permutations).
    sign_block = 1
    assert block_kernel(s4, sign_block).size == 12
    # The standard 3-dim irrep is faithful: kernel is just the identity.
    three_block = 3
    kernel = block_kernel(s4, three_block)
    assert kernel.tolist() == [identity_index]


def test_minimal_separating_blocks_excludes_trivial(s4: GroupData):
    selected = minimal_separating_blocks(s4)
    assert 0 not in selected  # index 0 is the trivial block


# ---------------------------------------------------------------------------
# Minimum faithful sets: combinatorial-blowup guard
#
# minimum_faithful_sets enumerates cardinality-k subsets of the nontrivial
# blocks looking for a faithful one. Elementary-abelian (C2)^k is the
# pathological case: every irrep is one-dimensional, no single irrep is
# faithful (each kernel is an index-2 subgroup), and no faithful subset
# exists below cardinality k (a spanning set of k independent characters is
# needed to cut the common kernel down to the identity) -- so, uncapped, the
# search would enumerate every subset up to MAX_FAITHFUL_SET_SEARCH_
# CARDINALITY without success. (C2)^7 = SmallGroup(128, 2328) has 127
# nontrivial candidate blocks; C(127, 6) ~ 4.8e9 hung the exact search for
# the better part of an hour before the guard below was added. Built by hand
# here (elements are length-k bit tuples, Cayley table is XOR, characters
# are chi_v(x) = (-1)**popcount(v & x)) so the test needs no external GAP
# artifact for a group this large.
# ---------------------------------------------------------------------------


def _build_elementary_abelian_2group(k: int) -> GroupData:
    n = 1 << k
    elements = list(range(n))
    table = np.array([[a ^ b for b in elements] for a in elements], dtype=np.int64)

    def popcount(x: int) -> int:
        return bin(x).count("1")

    irreps = []
    blocks = []
    for v in range(n):
        character = np.array(
            [1.0 if popcount(v & x) % 2 == 0 else -1.0 for x in elements], dtype=np.complex128
        )
        irreps.append(
            IrrepData(
                matrices=np.zeros((n, 1, 1), dtype=np.complex128),
                character=character,
                dimension=1,
                table_index=v,
                field="Q",
                basis="hand-constructed (C2)^k character (unused placeholder matrix)",
            )
        )
        blocks.append(
            IsotypicBlock(
                projector=np.zeros((n, n), dtype=np.float64),
                irrep_degree=1,
                block_rank=1,
                irrep_indices=(v,),
            )
        )

    conjugacy_classes = tuple(np.array([x], dtype=np.int64) for x in elements)
    character_table = np.array([irrep.character for irrep in irreps], dtype=np.complex128)
    frobenius_schur = np.ones(n, dtype=np.int64)
    trivial_subgroup = np.array([0], dtype=np.int64)
    full_group = np.array(elements, dtype=np.int64)

    return GroupData(
        order=n,
        index=-1,
        description=f"(C2)^{k} (hand-constructed, many one-dim irreps, no single faithful one)",
        element_labels=tuple(str(x) for x in elements),
        cayley_table=table,
        conjugacy_classes=conjugacy_classes,
        character_table=character_table,
        frobenius_schur=frobenius_schur,
        irreps=tuple(irreps),
        isotypic_blocks=tuple(blocks),
        subgroups=(trivial_subgroup, full_group),
        left_cosets=((trivial_subgroup,), (full_group,)),
        provenance={"backend": "hand-constructed", "purpose": "combinatorial-guard test"},
    )


@pytest.fixture(scope="module")
def elementary_abelian_c2_7() -> GroupData:
    return _build_elementary_abelian_2group(7)


def test_minimum_faithful_sets_many_one_dim_irreps_does_not_hang(elementary_abelian_c2_7):
    """The guard's whole point: this used to run for the better part of an
    hour (confirmed on SmallGroup(128, 2328), the real-world instance of
    this pathology). It must now return promptly, a genuinely faithful set,
    and flag the result as uncertified (an upper bound, not a proven
    minimum) since the exact search was abandoned over budget."""
    group = elementary_abelian_c2_7
    assert len(group.isotypic_blocks) == 128  # 1 trivial + 127 nontrivial

    start = time.perf_counter()
    result = minimum_faithful_sets(group)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, (
        f"minimum_faithful_sets took {elapsed:.2f}s; the guard should keep it fast"
    )
    assert result["certified"] is False
    assert is_faithful(group, result["example_min_set"])
    # The true minimum for (C2)^7 is exactly 7 (a basis of the dual space);
    # the greedy cover should land there or close to it, never above n.
    assert 1 <= result["min_cardinality"] <= 127


def test_minimum_faithful_sets_small_groups_unchanged_by_the_guard(s4: GroupData):
    """The guard must change behaviour only for the large-candidate-count,
    no-small-faithful-set case: S4 (5 candidates, well within budget at
    every level) gets the exact, certified answer exactly as before."""
    result = minimum_faithful_sets(s4)
    assert result["certified"] is True
    assert result["min_cardinality"] == 1
    assert is_faithful(s4, result["example_min_set"])


# ---------------------------------------------------------------------------
# Divergence comparison
# ---------------------------------------------------------------------------


def test_compare_classifies_null_occupancy_as_null(s4: GroupData):
    from group_algorithm_interp.instruments.occupancy import analytic_null

    occ = analytic_null(s4)
    result = compare_to_templates(occ, s4)
    assert result.screen.verdict == DEFINED
    assert result.closest_template == "null"
    assert result.tv_to_null == pytest.approx(0.0, abs=1e-9)
    assert result.tv_to_coset != UNDEFINED
    assert result.tv_to_coset > result.tv_to_null


def test_compare_classifies_coset_occupancy_as_coset(s4: GroupData):
    subgroup_h = s4.subgroups[1]
    occ = induction_template(s4, subgroup_h)
    result = compare_to_templates(occ, s4)
    assert result.screen.verdict == DEFINED
    assert result.closest_template == "coset"
    assert isinstance(result.tv_to_coset, float)
    assert result.tv_to_coset == pytest.approx(0.0, abs=1e-9)
    assert result.tv_to_null > 0.0
    assert result.tv_to_gcr > 0.0


def test_compare_classifies_gcr_occupancy_as_gcr(s4: GroupData):
    occ = gcr_sparse_template(s4)
    result = compare_to_templates(occ, s4)
    assert result.screen.verdict == DEFINED
    assert result.closest_template == "gcr"
    assert result.tv_to_gcr == pytest.approx(0.0, abs=1e-9)
    assert result.tv_to_null > 0.0
    assert isinstance(result.tv_to_coset, float)
    assert result.tv_to_coset > 0.0


def test_compare_refuses_a_verdict_when_the_screen_is_undefined():
    group = _require_real_artifact(32, 18)
    from group_algorithm_interp.instruments.occupancy import analytic_null

    occ = analytic_null(group)
    result = compare_to_templates(occ, group)
    assert result.screen.verdict == UNDEFINED
    assert result.tv_to_coset == UNDEFINED
    assert result.closest_template == UNDEFINED
    # tv_to_null and tv_to_gcr remain independently well-defined numbers.
    assert isinstance(result.tv_to_null, float)
    assert isinstance(result.tv_to_gcr, float)


def test_compare_refuses_a_verdict_on_q32_with_no_corefree_subgroup():
    group = _require_real_artifact(32, 20)
    from group_algorithm_interp.instruments.occupancy import analytic_null

    occ = analytic_null(group)
    result = compare_to_templates(occ, group)
    assert result.screen.coset_defined is False
    assert result.closest_template == UNDEFINED
    assert result.tv_to_coset == UNDEFINED


def test_compare_rejects_a_mismatched_occupancy_shape(s4: GroupData):
    with pytest.raises(ValueError, match="shape"):
        compare_to_templates(np.zeros(3), s4)


def test_screen_and_divergence_to_record_are_json_shaped(s4: GroupData):
    screen = degeneracy_screen(s4)
    record = screen.to_record()
    assert record["verdict"] == DEFINED
    assert isinstance(record["coset_template"], list)

    from group_algorithm_interp.instruments.occupancy import analytic_null

    result = compare_to_templates(analytic_null(s4), s4)
    divergence_record = result.to_record()
    assert divergence_record["instrument"] == "template-divergence"
    assert divergence_record["closest_template"] == "null"


# ---------------------------------------------------------------------------
# Data-driven refinement: causally_used_blocks / product_carrying_blocks /
# data_driven_comparison
# ---------------------------------------------------------------------------


def _iso_ablation_record(means_by_block: dict[int, float], *, mode: str = "zero") -> dict:
    """A minimal synthetic I-15 aggregated-cell record: just the ``blocks``
    list shape :func:`causally_used_blocks` reads (``block_index`` and the one
    mode's ``flip_fraction_over_random.mean``); the rest of the real schema
    (``clean_accuracy``, ``runs``, ...) is irrelevant to that function."""
    return {
        "blocks": [
            {"block_index": idx, "modes": {mode: {"flip_fraction_over_random": {"mean": mean}}}}
            for idx, mean in means_by_block.items()
        ]
    }


def test_causally_used_blocks_extracts_intended_set_at_default_threshold():
    record = _iso_ablation_record({1: 0.02, 2: 0.10, 3: 0.30, 4: float("nan")})
    assert causally_used_blocks(record) == (2, 3)
    assert MIN_FLIP_OVER_RANDOM == 0.05  # the documented default


def test_causally_used_blocks_threshold_is_named_and_overridable():
    record = _iso_ablation_record({1: 0.02, 2: 0.10, 3: 0.30})
    assert causally_used_blocks(record, min_flip_over_random=0.15) == (3,)
    assert causally_used_blocks(record, min_flip_over_random=0.0) == (1, 2, 3)


def test_causally_used_blocks_reads_the_requested_mode():
    record = {
        "blocks": [
            {
                "block_index": 1,
                "modes": {
                    "zero": {"flip_fraction_over_random": {"mean": 0.30}},
                    "mean": {"flip_fraction_over_random": {"mean": 0.01}},
                },
            }
        ]
    }
    assert causally_used_blocks(record, mode="zero") == (1,)
    assert causally_used_blocks(record, mode="mean") == ()


def test_causally_used_blocks_rejects_an_unknown_mode():
    record = _iso_ablation_record({1: 0.30})
    with pytest.raises(ValueError, match="mode"):
        causally_used_blocks(record, mode="not-a-mode")


def _gcr_matmul_record(entries: list[dict]) -> dict:
    """A minimal synthetic GCR-matmul aggregated-cell record: just the
    ``irreps`` list shape :func:`product_carrying_blocks` reads."""
    return {"irreps": entries}


def test_product_carrying_blocks_extracts_intended_set_at_defaults():
    record = _gcr_matmul_record(
        [
            {
                "block_index": 2,
                "mp_minus_bilinear_heldout": {"mean": 0.01},
                "mp_favoured_fraction": 0.8,
            },
            {
                "block_index": 3,
                "mp_minus_bilinear_heldout": {"mean": -0.02},
                "mp_favoured_fraction": 0.9,
            },
            {
                "block_index": 4,
                "mp_minus_bilinear_heldout": {"mean": 0.05},
                "mp_favoured_fraction": 0.3,
            },
        ]
    )
    # block 2: positive gap, favoured often enough -> included.
    # block 3: negative gap -> excluded regardless of favoured_fraction.
    # block 4: positive gap but favoured too rarely -> excluded (require_favoured).
    assert product_carrying_blocks(record) == (2,)
    assert MIN_MP_MINUS_BILINEAR == 0.0
    assert MIN_FAVOURED_FRACTION == 0.5


def test_product_carrying_blocks_require_favoured_toggle():
    record = _gcr_matmul_record(
        [
            {
                "block_index": 2,
                "mp_minus_bilinear_heldout": {"mean": 0.01},
                "mp_favoured_fraction": 0.8,
            },
            {
                "block_index": 3,
                "mp_minus_bilinear_heldout": {"mean": -0.02},
                "mp_favoured_fraction": 0.9,
            },
            {
                "block_index": 4,
                "mp_minus_bilinear_heldout": {"mean": 0.05},
                "mp_favoured_fraction": 0.3,
            },
        ]
    )
    # With the favoured requirement dropped, only the sign of the gap matters.
    assert product_carrying_blocks(record, require_favoured=False) == (2, 4)


def test_product_carrying_blocks_non_finite_gap_excluded():
    record = _gcr_matmul_record(
        [
            {
                "block_index": 1,
                "mp_minus_bilinear_heldout": {"mean": float("nan")},
                "mp_favoured_fraction": 1.0,
            }
        ]
    )
    assert product_carrying_blocks(record) == ()


# ---------------------------------------------------------------------------
# data_driven_comparison
# ---------------------------------------------------------------------------
#
# On the s4 fixture: GCR's predicted (faithful) set is {sign, two, three} =
# block indices (1, 2, 3) (see test_gcr_template_on_s4_selects_the_faithful_
# low_degree_blocks above); coset's predicted set (H = Stab(3), index 4) is
# {trivial, three} = block indices (0, 3), by Frobenius reciprocity computed
# by hand: m_triv=1, m_three=1, all other multiplicities 0 (sum m_i d_i =
# 1 + 3 = 4 = [G:H], as induction_template asserts internally).


def test_data_driven_comparison_set_overlap_and_distances(s4: GroupData):
    used = (1, 2, 3)  # exactly the GCR faithful set
    occ = analytic_null(s4)
    result = data_driven_comparison(occ, s4, used)

    assert result.used_blocks == (1, 2, 3)
    assert result.gcr_predicted_blocks == (1, 2, 3)
    assert result.coset_predicted_blocks == (0, 3)

    assert isinstance(result.gcr_overlap, SetOverlap)
    assert result.gcr_overlap.jaccard == pytest.approx(1.0)
    assert result.gcr_overlap.precision == pytest.approx(1.0)
    assert result.gcr_overlap.recall == pytest.approx(1.0)

    assert isinstance(result.coset_overlap, SetOverlap)
    # used={1,2,3}, coset-predicted={0,3}: intersection {3}, union {0,1,2,3}.
    assert result.coset_overlap.intersection_size == 1
    assert result.coset_overlap.jaccard == pytest.approx(1 / 4)
    assert result.coset_overlap.precision == pytest.approx(1 / 2)  # |inter|/|predicted|
    assert result.coset_overlap.recall == pytest.approx(1 / 3)  # |inter|/|used|

    # tv_to_used uses a template built from the used set, which here is the
    # same block set (and hence the same rank-proportional template) as the
    # GCR predicted template.
    assert isinstance(result.tv_to_used, float)
    assert isinstance(result.tv_to_gcr_predicted, float)
    assert result.tv_to_used == pytest.approx(result.tv_to_gcr_predicted, abs=1e-9)
    assert isinstance(result.tv_to_coset_predicted, float)
    assert result.tv_to_coset_predicted == pytest.approx(0.583333, abs=1e-5)
    assert result.tv_to_gcr_predicted == pytest.approx(0.416667, abs=1e-5)

    # closest_template is never reported without separation beside it.
    assert result.separation is not None
    assert result.closest_template in ("used", "gcr", "coset")


def test_data_driven_comparison_flags_low_separation_when_near_equidistant(s4: GroupData):
    # Occupancy placed exactly at the midpoint of the coset and GCR predicted
    # templates is, by construction, equidistant (in TV) from both -- the
    # case an argmin alone would misreport as a confident verdict.
    screen = degeneracy_screen(s4)
    coset_template = screen.coset_template
    assert coset_template is not None
    gcr_template = gcr_sparse_template(s4)
    midpoint = 0.5 * coset_template + 0.5 * gcr_template
    assert midpoint.sum() == pytest.approx(1.0)

    result = data_driven_comparison(midpoint, s4, used_blocks=())

    assert result.used_blocks == ()
    assert result.tv_to_used == UNDEFINED  # no causally-used block was given
    assert isinstance(result.tv_to_coset_predicted, float)
    assert isinstance(result.tv_to_gcr_predicted, float)
    assert result.separation is not None
    assert result.separation == pytest.approx(0.0, abs=1e-9)
    # The label is reported, but separation sitting beside it is what shows
    # it should not be trusted here.
    assert result.closest_template in ("coset", "gcr")


def test_data_driven_comparison_empty_used_blocks_is_undefined_not_zero(s4: GroupData):
    occ = analytic_null(s4)
    result = data_driven_comparison(occ, s4, used_blocks=())
    assert result.used_blocks == ()
    assert result.tv_to_used == UNDEFINED
    # An empty used set against nonempty predicted sets scores 0 on every
    # overlap metric, never 1 (that is reserved for two empty sets).
    assert result.gcr_overlap.jaccard == 0.0
    assert result.gcr_overlap.precision == 0.0
    assert result.gcr_overlap.recall == 0.0


def test_data_driven_comparison_explicit_coset_template_bypasses_the_screen(s4: GroupData):
    occ = gcr_sparse_template(s4)
    override = analytic_null(s4)  # every block has positive mass
    result = data_driven_comparison(occ, s4, used_blocks=(1, 2, 3), coset_template=override)
    assert result.coset_predicted_blocks == (0, 1, 2, 3, 4)
    assert isinstance(result.tv_to_coset_predicted, float)


def test_data_driven_comparison_rejects_a_mismatched_occupancy_shape(s4: GroupData):
    with pytest.raises(ValueError, match="shape"):
        data_driven_comparison(np.zeros(3), s4, used_blocks=(1,))


def test_data_driven_comparison_rejects_a_mismatched_coset_template_shape(s4: GroupData):
    with pytest.raises(ValueError, match="shape"):
        data_driven_comparison(analytic_null(s4), s4, used_blocks=(1,), coset_template=np.zeros(2))


def test_data_driven_comparison_coset_side_is_undefined_when_the_screen_is():
    group = _require_real_artifact(32, 18)  # D32: coset target degenerate (30/32)
    occ = analytic_null(group)
    result = data_driven_comparison(occ, group, used_blocks=())
    assert result.coset_predicted_blocks == UNDEFINED
    assert result.coset_overlap == UNDEFINED
    assert result.tv_to_coset_predicted == UNDEFINED
    # The GCR side and the used-set machinery remain independently defined.
    assert isinstance(result.gcr_overlap, SetOverlap)
    assert isinstance(result.tv_to_gcr_predicted, float)
    assert result.tv_to_used == UNDEFINED  # used_blocks was empty here too
    # Only one distance is defined (gcr), so no separation can be computed.
    assert result.separation is None
    assert result.closest_template == "gcr"


def test_data_driven_comparison_to_record_is_json_shaped(s4: GroupData):
    result = data_driven_comparison(analytic_null(s4), s4, used_blocks=(1, 2, 3))
    record = result.to_record()
    assert record["instrument"] == "template-divergence-data-driven"
    assert record["used_blocks"] == [1, 2, 3]
    assert record["gcr_predicted_blocks"] == [1, 2, 3]
    assert record["coset_predicted_blocks"] == [0, 3]
    assert isinstance(record["coset_overlap"], dict)
    assert isinstance(record["gcr_overlap"], dict)
    assert record["gcr_overlap"]["jaccard"] == pytest.approx(1.0)
    assert isinstance(record["tv_to_coset_predicted"], float)
    assert isinstance(record["separation"], float)


def test_data_driven_comparison_to_record_is_json_shaped_when_undefined():
    group = _require_real_artifact(32, 20)  # Q32: no core-free subgroup at all
    result = data_driven_comparison(analytic_null(group), group, used_blocks=())
    record = result.to_record()
    assert record["coset_predicted_blocks"] == UNDEFINED
    assert record["coset_overlap"] == UNDEFINED
    assert record["tv_to_coset_predicted"] == UNDEFINED
    assert record["tv_to_used"] == UNDEFINED
