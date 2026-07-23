"""The C5 cocycle instrument (instruments/cocycle.py): I-21 precompute and the
I-22/I-22b probe/ablation/fit.

The crux is the split-vs-non-split discriminator: the ``f == 1`` (untwisted)
fit reproduces the whole Cayley table iff the extension splits. The fixture
contrast is the GL(2,3) [split] vs SL(2,3).C2 [non-split] contrast in miniature:

* split: S3 = C3 : C2 over its C3, and D8 = C4 : C2 over its C4 -- a complement
  exists, so the untwisted rule is exact;
* non-split: Q8 over <i> (order-4 cyclic normal subgroup) -- Q8's unique
  involution -1 lies inside every order-4 subgroup, so no C2 complement exists,
  exactly as the binary octahedral group SL(2,3).C2 is the non-split extension
  of SL(2,3) by C2.
"""

from __future__ import annotations

import numpy as np
import pytest

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments import cocycle as C
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

# Named group, its normal subgroup order N, and whether G splits over that N.
SPLIT_CASES = [("S3", 3, True), ("D8", 4, True), ("Q8", 4, False)]


def _first_normal(group, order: int) -> np.ndarray:
    for subgroup in group.subgroups:
        members = np.asarray(subgroup, dtype=np.int64)
        if members.size == order and C.is_normal(group.cayley_table, members):
            return members
    raise AssertionError(f"no normal subgroup of order {order} in {group.canonical_name}")


def _model(name: str, order: int, index: int, *, seed: int = 0, arch: str = "transformer"):
    group = resolve_group(name)
    config = ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}},
        model={"arch": arch, "d_model": 32, "d_mlp": 64, "n_heads": 4},
        logging={"mode": "disabled"},
    )
    set_seed(seed, deterministic=False)
    return build_model(config, group), group


# ---------------------------------------------------------------------------
# Cayley-table primitives
# ---------------------------------------------------------------------------


def test_is_subgroup_and_is_normal():
    q8 = resolve_group("Q8")
    table = q8.cayley_table
    centre = np.array([0, 1])  # {1, -1}
    assert C.is_subgroup(table, centre)
    assert C.is_normal(table, centre)
    not_closed = np.array([0, 2])  # {1, i} -- i*i = -1 is absent
    assert not C.is_subgroup(table, not_closed)
    assert not C.is_normal(table, not_closed)
    assert not C.is_subgroup(table, np.array([], dtype=np.int64))


def test_subgroup_closure_generates_the_whole_subgroup():
    q8 = resolve_group("Q8")
    closure = C.subgroup_closure(q8.cayley_table, {2})  # <i>
    assert closure == frozenset({0, 1, 2, 3})


def test_d8_reflection_subgroup_is_not_normal():
    d8 = resolve_group("D8")
    reflection = np.array([0, 4])  # {e, s}
    assert C.is_subgroup(d8.cayley_table, reflection)
    assert not C.is_normal(d8.cayley_table, reflection)


# ---------------------------------------------------------------------------
# I-21: extension precompute and its self-verifying identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,n_order,_", SPLIT_CASES)
def test_build_extension_asserts_identities(name, n_order, _):
    """The cocycle identity, the normalisation, and the coordinate bijection all
    hold (build_extension raises if any fails); the twisted rule is exact."""
    group = resolve_group(name)
    normal = _first_normal(group, n_order)
    extension = C.build_extension(group, normal)  # asserts the cocycle identity internally
    assert extension.quotient_order == group.order // n_order
    assert extension.normal_order == n_order
    # Coordinate bijection G <-> N x Q (asserted inside, re-checked here).
    n_of, q_of = C.coordinates(extension, group.cayley_table)
    assert len({(int(n_of[g]), int(q_of[g])) for g in range(group.order)}) == group.order
    # The twisted rule reproduces the whole table exactly.
    assert C.coordinate_product_accuracy(
        extension, group.cayley_table, include_cocycle=True
    ) == pytest.approx(1.0)


def test_build_extension_rejects_a_non_normal_subgroup():
    d8 = resolve_group("D8")
    with pytest.raises(ValueError, match="not normal"):
        C.build_extension(d8, np.array([0, 4]))


def test_build_extension_rejects_a_bad_transversal():
    s3 = resolve_group("S3")
    normal = _first_normal(s3, 3)
    with pytest.raises(ValueError, match="one representative per coset"):
        C.build_extension(s3, normal, np.array([0, 1, 2]))
    # s(0) must be the identity.
    with pytest.raises(ValueError, match="s\\(0\\) = e"):
        C.build_extension(s3, normal, np.array([3, 0]))


def test_canonical_transversal_covers_each_coset_once():
    q8 = resolve_group("Q8")
    normal = _first_normal(q8, 4)
    extension = C.build_extension(q8, normal)
    labels = extension.coset_label[extension.transversal]
    assert sorted(labels.tolist()) == list(range(extension.quotient_order))
    assert int(extension.transversal[0]) == 0  # s(0) = e


# ---------------------------------------------------------------------------
# THE decisive discriminator: f == 1 fit SUCCEEDS on split, FAILS on non-split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,n_order,splits", SPLIT_CASES)
def test_f_equiv_one_fit_discriminates_split_from_nonsplit(name, n_order, splits):
    """The GL(2,3) vs SL(2,3).C2 discriminator. On a split extension the
    untwisted (``f == 1``) rule is exact (fit SUCCEEDS); on a non-split one it
    is not, for every transversal (fit FAILS). The twisted rule is always
    exact."""
    group = resolve_group(name)
    normal = _first_normal(group, n_order)
    fit = C.f_equiv_one_fit(group, normal)
    assert fit.splits is splits
    assert fit.twisted_accuracy == pytest.approx(1.0)
    if splits:
        assert fit.untwisted_accuracy == pytest.approx(1.0)  # fit SUCCEEDS
        assert fit.used_trivialising_transversal is True
    else:
        assert fit.untwisted_accuracy < 1.0  # fit FAILS
        assert fit.used_trivialising_transversal is False


@pytest.mark.parametrize("name,n_order,splits", SPLIT_CASES)
def test_split_status_and_complement_witness(name, n_order, splits):
    group = resolve_group(name)
    normal = _first_normal(group, n_order)
    result = C.split_status(group, normal)
    assert result.splits is splits
    if splits:
        complement = np.array(result.complement, dtype=np.int64)
        # A genuine complement: a subgroup of order [G:N] meeting N only in e.
        assert C.is_subgroup(group.cayley_table, complement)
        assert complement.size == group.order // n_order
        assert set(complement.tolist()) & set(normal.tolist()) == {0}
    else:
        assert result.complement is None


def test_trivialising_transversal_present_iff_split():
    s3 = resolve_group("S3")
    assert C.trivialising_transversal(s3, _first_normal(s3, 3)) is not None
    q8 = resolve_group("Q8")
    assert C.trivialising_transversal(q8, _first_normal(q8, 4)) is None


def test_nonsplit_cocycle_is_nontrivial_for_every_transversal():
    """Q8 over <i>: the cocycle is nontrivial for *every* transversal, not just
    the canonical one -- the definition of a non-split extension."""
    q8 = resolve_group("Q8")
    normal = _first_normal(q8, 4)  # coset 0; coset 1 = {4, 5, 6, 7}
    for rep in (4, 5, 6, 7):
        extension = C.build_extension(q8, normal, np.array([0, rep]))
        assert not C.cocycle_is_trivial(extension)
        assert (
            C.coordinate_product_accuracy(extension, q8.cayley_table, include_cocycle=False) < 1.0
        )


def test_split_cocycle_is_trivial_on_the_complement_transversal():
    s3 = resolve_group("S3")
    normal = _first_normal(s3, 3)
    transversal = C.trivialising_transversal(s3, normal)
    extension = C.build_extension(s3, normal, transversal)
    assert C.cocycle_is_trivial(extension)


def test_split_status_rejects_non_normal_subgroup():
    d8 = resolve_group("D8")
    with pytest.raises(ValueError, match="not normal"):
        C.split_status(d8, np.array([0, 4]))


# ---------------------------------------------------------------------------
# I-22: cocycle-value probe (held out over (q1, q2) cells)
# ---------------------------------------------------------------------------


def test_cell_labels_are_shared_across_pairs():
    q8 = resolve_group("Q8")
    normal = _first_normal(q8, 4)
    extension = C.build_extension(q8, normal)
    cell, f_label = C.cell_labels(extension, q8.order)
    assert cell.shape == (q8.order * q8.order,)
    # Many (a, b) pairs share one (q1, q2) cell -- the pseudoreplication unit.
    assert np.unique(cell).size == extension.quotient_order**2
    assert np.unique(cell).size < cell.size


def test_probe_is_undefined_on_a_split_member():
    """The split member's control is a fit residual, not a probe score: with the
    complement transversal f is constant, so the decode target is degenerate."""
    s3 = resolve_group("S3")
    normal = _first_normal(s3, 3)
    transversal = C.trivialising_transversal(s3, normal)
    extension = C.build_extension(s3, normal, transversal)
    model, group = _model("S3", 6, 1)
    features = C.pair_features(model, group.order)
    result = C.probe_cocycle(features, extension, group.order)
    assert result.defined is False
    assert result.to_record()["status"] == "UNDEFINED"


def test_probe_is_defined_on_a_nonsplit_member():
    q8 = resolve_group("Q8")
    normal = _first_normal(q8, 4)
    extension = C.build_extension(q8, normal)
    model, group = _model("Q8", 8, 4)
    features = C.pair_features(model, group.order, site="mlp_post")
    result = C.probe_cocycle(features, extension, group.order, site="mlp_post")
    assert result.defined is True
    record = result.to_record()
    assert record["status"] == "measured"
    assert 0.0 <= record["balanced_accuracy"] <= 1.0
    assert 0.0 <= record["accuracy"] <= 1.0
    assert record["n_classes"] >= 2


def test_pair_features_shape_and_validation():
    model, group = _model("Q8", 8, 4)
    features = C.pair_features(model, group.order)
    assert features.shape[0] == group.order * group.order
    with pytest.raises(ValueError, match="site"):
        C.pair_features(model, group.order, site="embed")


# ---------------------------------------------------------------------------
# I-22b: cocycle-direction ablation vs a matched random subspace
# ---------------------------------------------------------------------------


def test_ablation_is_undefined_on_a_split_member():
    s3 = resolve_group("S3")
    normal = _first_normal(s3, 3)
    transversal = C.trivialising_transversal(s3, normal)
    extension = C.build_extension(s3, normal, transversal)
    model, group = _model("S3", 6, 1)
    result = C.cocycle_ablation(model, group, extension)
    assert result.defined is False
    assert result.to_record()["status"] == "UNDEFINED"


def test_ablation_reports_both_deltas_on_a_nonsplit_member():
    q8 = resolve_group("Q8")
    normal = _first_normal(q8, 4)
    extension = C.build_extension(q8, normal)
    model, group = _model("Q8", 8, 4)
    result = C.cocycle_ablation(model, group, extension, seed=1)
    assert result.defined is True
    record = result.to_record()
    assert record["subspace_dim"] >= 1
    assert isinstance(record["delta_accuracy"], float)
    assert isinstance(record["random_delta_accuracy"], float)
    assert 0.0 <= record["baseline_accuracy"] <= 1.0


def test_extension_to_record_is_serialisable():
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    record = extension.to_record()
    assert record["quotient_order"] == 2
    assert record["cocycle_trivial"] is False
    assert len(record["cocycle"]) == 2
