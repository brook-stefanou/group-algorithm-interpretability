"""Tests for the recruited-dimension instrument.

The load-bearing correctness claim is the minimal faithful *real* representation
dimension, so it is checked on groups whose answer is hand-verifiable and whose
Frobenius-Schur type is exactly what makes the real-vs-complex distinction bite:

* Q8 (SmallGroup(8,4)): the only faithful irrep is the degree-2 *quaternionic*
  one (nu = -1), so the minimal faithful real dimension is 2*2 = 4, not 2 -- the
  headline FS test case.
* S3 (SmallGroup(6,1)) and D4 (SmallGroup(8,3)): a degree-2 *real* faithful irrep
  (nu = +1), minimal faithful real dimension 2.
* C4 (SmallGroup(4,1)): the faithful degree-1 irrep is *complex* type (nu = 0),
  so its real form is 2-dimensional -- complex degree 1 but real dimension 2.
* C2 (SmallGroup(2,1)): the faithful degree-1 irrep is real, real dimension 1.

The effective-rank measures are checked against a constructed matrix with a known
number of equal singular values (all three measures must return exactly that
rank) and against an unequal spectrum (they must fall strictly between 1 and the
count of nonzero values).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from group_algorithm_interp.groups.data import GroupData, IrrepData, IsotypicBlock, load_group
from group_algorithm_interp.instruments.recruited_dimension import (
    FS_COMPLEX,
    FS_QUATERNIONIC,
    FS_REAL,
    activation_effective_rank,
    block_real_dimension,
    discriminate,
    effective_rank_from_singular_values,
    effective_rank_of_matrix,
    embedding_effective_rank,
    frobenius_schur_indicator,
    measure_recruited_dimension,
    minimal_faithful_real_dimension,
    real_irreducible_dimensions,
    theoretical_dimensions,
)
from group_algorithm_interp.instruments.template_divergence import is_faithful
from group_algorithm_interp.model import OneLayerTransformer
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

# SmallGroup ids of the fixtures the conftest corpus provides.
Q8 = (8, 4)
S3 = (6, 1)
D4 = (8, 3)
C4 = (4, 1)
C2 = (2, 1)
C7 = (7, 1)


def _group(order: int, index: int):
    return load_group(order, index)


def _build_elementary_abelian_2group(k: int) -> GroupData:
    """(C2)^k, hand-constructed: elements are length-k bit tuples, the
    Cayley table is XOR, and the 2**k one-dimensional real characters are
    chi_v(x) = (-1)**popcount(v & x). Every nontrivial character's kernel is
    an index-2 subgroup -- never trivial alone -- so no faithful set exists
    below cardinality k, the combinatorial-blowup pathology
    minimal_faithful_real_dimension's guard exists for (see its docstring
    and SmallGroup(128, 2328), the real-world instance)."""
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


# ---------------------------------------------------------------------------
# Frobenius-Schur indicator, computed from the character + power map
# ---------------------------------------------------------------------------


def test_frobenius_schur_matches_stored_array_for_every_fixture_irrep():
    """The character-derived indicator (mean of chi(g^2)) must agree with the
    stored ``frobenius_schur`` array for every irrep of every fixture group --
    the cross-check ``block_real_dimension`` relies on."""
    for order, index in [Q8, S3, D4, C4, C2, C7, (6, 2), (21, 2), (32, 1)]:
        group = _group(order, index)
        for i in range(len(group.irreps)):
            assert frobenius_schur_indicator(group, i) == int(group.frobenius_schur[i])


def test_frobenius_schur_types_of_q8_s3_c4():
    """Q8's degree-2 irrep is quaternionic (-1), S3's is real (+1), C4's
    faithful degree-1 irrep is complex (0)."""
    q8 = _group(*Q8)
    degree_two = [i for i in range(len(q8.irreps)) if q8.irreps[i].dimension == 2]
    assert len(degree_two) == 1
    assert frobenius_schur_indicator(q8, degree_two[0]) == FS_QUATERNIONIC

    s3 = _group(*S3)
    degree_two_s3 = [i for i in range(len(s3.irreps)) if s3.irreps[i].dimension == 2]
    assert frobenius_schur_indicator(s3, degree_two_s3[0]) == FS_REAL

    c4 = _group(*C4)
    # C4 characters: k=0 trivial (real), k=2 (real), k=1 and k=3 complex conjugates.
    assert sorted(frobenius_schur_indicator(c4, i) for i in range(len(c4.irreps))) == [
        FS_COMPLEX,
        FS_COMPLEX,
        FS_REAL,
        FS_REAL,
    ]


# ---------------------------------------------------------------------------
# Real dimension of a block, and the minimal faithful real dimension
# ---------------------------------------------------------------------------


def test_block_real_dimension_quaternionic_doubles():
    """Q8's degree-2 quaternionic block has real dimension 4 (2d), while its
    degree-1 real blocks have real dimension 1 (d)."""
    q8 = _group(*Q8)
    dims = real_irreducible_dimensions(q8)
    degrees = [b.irrep_degree for b in q8.isotypic_blocks]
    assert sorted(dims) == [1, 1, 1, 1, 4]
    # The single degree-2 block is the one that doubled.
    (two_block,) = [j for j, d in enumerate(degrees) if d == 2]
    assert block_real_dimension(q8, two_block) == 4


def test_block_real_dimension_real_type_does_not_double():
    """S3's and D4's degree-2 blocks are real type: real dimension equals the
    complex degree, 2."""
    for order, index in (S3, D4):
        group = _group(order, index)
        for j, block in enumerate(group.isotypic_blocks):
            if block.irrep_degree == 2:
                assert block_real_dimension(group, j) == 2


def test_block_real_dimension_complex_pair_doubles():
    """C4's complex-conjugate pair merges into one block of real dimension 2
    despite each irrep having complex degree 1."""
    c4 = _group(*C4)
    pair_blocks = [j for j, b in enumerate(c4.isotypic_blocks) if len(b.irrep_indices) == 2]
    assert pair_blocks, "C4 must have a merged complex-conjugate block"
    for j in pair_blocks:
        assert c4.isotypic_blocks[j].irrep_degree == 1
        assert block_real_dimension(c4, j) == 2


def test_minimal_faithful_real_dimension_q8_is_four_not_two():
    """The instrument's headline claim: Q8's minimal faithful real dimension is
    4 (its only faithful irrep is quaternionic), where the naive complex answer
    would be 2. The witnessing set is the single degree-2 block."""
    result = minimal_faithful_real_dimension(_group(*Q8))
    assert result.real_dimension == 4
    assert result.example_set_cardinality == 1
    assert result.example_set_degrees == (2,)
    assert result.example_set_real_dims == (4,)


def test_minimal_faithful_real_dimension_real_type_groups_are_two():
    """S3 and D4 reach a faithful representation with a single real degree-2
    irrep, real dimension 2 -- half of Q8's, from the FS type alone."""
    for order, index in (S3, D4):
        result = minimal_faithful_real_dimension(_group(order, index))
        assert result.real_dimension == 2
        assert result.example_set_degrees == (2,)


def test_minimal_faithful_real_dimension_cyclic_cases():
    """C4 (complex faithful char, real dim 2), C2 (real faithful char, real dim
    1), and C7 (three faithful conjugate pairs, so three distinct minimal sets
    of real dimension 2 tie)."""
    assert minimal_faithful_real_dimension(_group(*C4)).real_dimension == 2
    assert minimal_faithful_real_dimension(_group(*C2)).real_dimension == 1
    c7 = minimal_faithful_real_dimension(_group(*C7))
    assert c7.real_dimension == 2
    assert c7.n_sets_at_min == 3


def test_minimal_faithful_set_is_actually_faithful():
    """The reported witnessing set must genuinely be faithful (kernels intersect
    to the identity alone) for every checked group."""
    for order, index in [Q8, S3, D4, C4, C2, C7]:
        group = _group(order, index)
        result = minimal_faithful_real_dimension(group)
        assert is_faithful(group, result.example_set)


def test_minimal_faithful_real_dimension_many_one_dim_irreps_does_not_hang():
    """The combinatorial-blowup guard applies here too: elementary-abelian
    (C2)^7 has 127 one-dimensional nontrivial irreps and no single faithful
    one (each kernel is an index-2 subgroup), so the exact cardinality
    search would otherwise enumerate C(127, 6) ~ 4.8e9 combinations without
    ever succeeding -- the same pathology confirmed to hang for the better
    part of an hour on SmallGroup(128, 2328). Must now return promptly, a
    genuinely faithful set, and be flagged uncertified."""
    group = _build_elementary_abelian_2group(7)
    start = time.perf_counter()
    result = minimal_faithful_real_dimension(group)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"minimal_faithful_real_dimension took {elapsed:.2f}s"
    assert result.certified is False
    assert is_faithful(group, result.example_set)
    assert 1 <= result.example_set_cardinality <= 127


def test_minimal_faithful_real_dimension_small_groups_unchanged_by_the_guard():
    """The guard must change behaviour only for the large-candidate-count,
    no-small-faithful-set case: every fixture group here has a handful of
    candidates and gets the exact, certified answer exactly as before."""
    for order, index in [Q8, S3, D4, C4, C2, C7]:
        result = minimal_faithful_real_dimension(_group(order, index))
        assert result.certified is True


# ---------------------------------------------------------------------------
# Theoretical anchors
# ---------------------------------------------------------------------------


def test_theoretical_anchors_distinct_on_s3():
    """S3 separates all three anchors: minimal faithful real 2, tensor-rank
    lower bound 7 (Strassen's exact rank for the degree-2 block exceeds |G|), and
    |G| = 6 -- so tensor rank does NOT coincide with the regular rep here."""
    t = theoretical_dimensions(_group(*S3))
    assert t.minimal_faithful_real.real_dimension == 2
    assert t.regular_rep == 6
    assert t.tensor_rank_lower == 7
    assert t.tensor_rank_upper == 9
    assert not t.tensor_rank_coincides_with_regular


def test_theoretical_tensor_rank_coincides_with_regular_for_abelian():
    """For an abelian group all irreps are degree 1, so the tensor-rank lower
    bound equals |G| exactly -- the coincidence the discrimination must report
    as a tie rather than hide."""
    t = theoretical_dimensions(_group(*C4))
    assert t.tensor_rank_lower == t.regular_rep == 4
    assert t.tensor_rank_coincides_with_regular


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------


def test_effective_rank_equal_singular_values_returns_the_rank():
    """A matrix with exactly k equal nonzero singular values has all three
    effective-rank measures equal to k."""
    set_seed(0, deterministic=False)
    k = 4
    ambient = 10
    q, _ = np.linalg.qr(np.random.randn(ambient, ambient))
    matrix = q[:, :k] * 3.0  # k orthonormal columns scaled: k equal singular values
    rank = effective_rank_of_matrix(matrix)
    assert rank.participation_ratio == pytest.approx(k)
    assert rank.participation_ratio_sq == pytest.approx(k)
    assert rank.stable_rank == pytest.approx(k)
    assert rank.frobenius_energy == pytest.approx(k * 9.0)


def test_effective_rank_unequal_spectrum_between_one_and_count():
    """With two unequal nonzero singular values every measure sits strictly
    between 1 and 2 (partial participation of the second direction)."""
    rank = effective_rank_from_singular_values([4.0, 1.0])
    for value in (rank.participation_ratio, rank.participation_ratio_sq, rank.stable_rank):
        assert 1.0 < value < 2.0
    # A heavier tail (more equal) participates more than a lighter one.
    assert (
        effective_rank_from_singular_values([2.0, 2.0]).participation_ratio
        > effective_rank_from_singular_values([4.0, 1.0]).participation_ratio
    )


def test_effective_rank_rejects_zero_and_negative():
    with pytest.raises(ValueError, match="zero"):
        effective_rank_from_singular_values([0.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        effective_rank_from_singular_values([1.0, -0.5])


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------


def test_discriminate_picks_nearest_anchor_on_log_scale():
    """A recruited dimension near the minimal-faithful anchor is labelled
    closest to it, with the reported ratio matching."""
    out = discriminate(4.2, {"minimal_faithful_real": 4.0, "regular_rep": 8.0})
    assert out["closest"] == "minimal_faithful_real"
    assert out["ratios"]["minimal_faithful_real"] == pytest.approx(4.2 / 4.0)


def test_discriminate_reports_coincident_anchors_as_a_tie():
    """When two anchors are equal, both appear in the tie list rather than the
    label silently picking one."""
    out = discriminate(8.0, {"tensor_rank_lower": 8.0, "regular_rep": 8.0})
    assert set(out["closest_tie"]) == {"regular_rep", "tensor_rank_lower"}


# ---------------------------------------------------------------------------
# Empirical measures on real models
# ---------------------------------------------------------------------------


def _model(order: int, index: int, *, use_mlp: bool = True, seed: int = 0):
    from group_algorithm_interp.config import ProjectConfig

    group = _group(order, index)
    config = ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}},
        model={"arch": "transformer", "d_model": 32, "d_mlp": 64, "n_heads": 4, "use_mlp": use_mlp},
        logging={"mode": "disabled"},
    )
    set_seed(seed, deterministic=False)
    return build_model(config, group), group


def test_embedding_effective_rank_bounded_by_dimensions():
    """The embedding effective rank cannot exceed min(|G|, d_model), and a
    genuine (non-constant) embedding is more than one-dimensional."""
    model, group = _model(*S3)
    rank = embedding_effective_rank(model, group, center=True)
    assert 1.0 < rank.participation_ratio <= min(group.order, model.d_model) + 1e-6


def test_activation_effective_rank_runs_and_is_bounded_by_order():
    """The activation-code effective rank is defined and capped at |G|."""
    model, group = _model(*S3)
    rank = activation_effective_rank(model, group, argument="left", center=True)
    assert 1.0 <= rank.participation_ratio <= group.order + 1e-6


def test_measure_recruited_dimension_assembles_record():
    """The top-level measurement assembles all measures, uses the centred
    activation-left participation ratio as the primary scalar, and discriminates
    it against the three anchors."""
    model, group = _model(*Q8)
    result = measure_recruited_dimension(model, group)
    record = result.to_record()
    assert record["instrument"] == "recruited-dimension"
    assert result.activation_available
    assert result.primary_measure == "activation_left_centered"
    assert result.theoretical.minimal_faithful_real.real_dimension == 4
    assert set(result.discrimination["ratios"]) == {
        "minimal_faithful_real",
        "tensor_rank_lower",
        "regular_rep",
    }
    assert result.discrimination["closest"] in result.discrimination["ratios"]
    # The primary recruited scalar is a finite effective dimension in [1, |G|].
    assert 1.0 <= result.primary_recruited <= group.order + 1e-6


def test_no_mlp_model_falls_back_to_embedding_measure():
    """A ``use_mlp=false`` transformer has no neuron activations, so the
    activation measures are absent and the primary scalar comes from the
    embedding."""
    group = _group(*S3)
    set_seed(0, deterministic=False)
    model = OneLayerTransformer(
        d_vocab_in=group.order + 1,
        d_vocab_out=group.order,
        n_ctx=3,
        d_model=32,
        n_heads=4,
        use_mlp=False,
        d_mlp=64,
        activation="relu",
    )
    result = measure_recruited_dimension(model, group)
    assert not result.activation_available
    assert result.activation_left_centered is None
    assert result.primary_measure == "embedding_centered"


def test_measure_recruited_dimension_device_cpu_matches_default():
    """The explicit ``--device cpu`` path returns the same result as the default
    (the analysis is CPU float64 regardless of where the forward pass ran)."""
    model, group = _model(*S3)
    default = measure_recruited_dimension(model, group)
    explicit = measure_recruited_dimension(model, group, device=torch.device("cpu"))
    assert default.primary_recruited == pytest.approx(explicit.primary_recruited)
