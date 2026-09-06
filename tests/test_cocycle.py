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

import json
import sys
from pathlib import Path

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


def test_build_extension_rejects_a_permuted_transversal():
    """A permutation of a valid transversal (right representatives, wrong slots)
    covers every coset yet mislabels the quotient; it must be rejected up front
    with a membership error, not downstream as 'cocycle value fell outside N'.
    Negative / out-of-range indices are rejected before they can index-wrap."""
    c6 = resolve_group("C6")
    normal = _first_normal(c6, 2)  # |Q| = 3, canonical reps in cosets 0, 1, 2
    with pytest.raises(ValueError, match="does not lie in coset"):
        C.build_extension(c6, normal, np.array([0, 2, 1]))
    with pytest.raises(ValueError, match="element indices"):
        C.build_extension(c6, normal, np.array([0, -1, 2]))


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


def test_probe_is_structurally_degenerate_on_a_q2_member():
    """The critical fix. For Q = C2 (here Q8 over <i>, |Q| = 2) the normalised
    cocycle is nontrivial in exactly one cell (1, 1), so leave-one-cell-out holds
    that cell out with its class absent from training and the probe is capped at
    chance whatever the model encodes. The record must be marked structurally
    degenerate, never a plain measured chance-level number -- even features that
    literally one-hot encode f_label (a perfect decoder input) must trip the guard
    rather than ship 3/4."""
    q8 = resolve_group("Q8")
    normal = _first_normal(q8, 4)
    extension = C.build_extension(q8, normal)
    assert extension.quotient_order == 2
    _, f_label = C.cell_labels(extension, q8.order)
    onehot = np.eye(int(f_label.max()) + 1)[f_label]  # features that literally encode f
    result = C.probe_cocycle(onehot, extension, q8.order)
    assert result.fold_degenerate is True
    assert result.accuracy is None  # not shipped as a measured chance number
    record = result.to_record()
    assert record["status"] == "STRUCTURALLY_DEGENERATE"
    assert record["degenerate_cells"]  # the sole nontrivial cell (1, 1)
    assert record["degenerate_classes"]
    assert "chance" not in record  # no misleading measured accuracy


def test_probe_positive_control_recovers_planted_f_on_a_q3_member():
    """The positive control that would have caught the |Q| = 2 bug. On a |Q| = 3
    member (C6 over its C2, canonical transversal: nontrivial cocycle spread over
    several cells) no fold is degenerate, so features that one-hot encode f_label
    must be decoded above chance. This is the shape the probe is valid for."""
    c6 = resolve_group("C6")
    normal = _first_normal(c6, 2)
    extension = C.build_extension(c6, normal)  # canonical transversal -> f nontrivial
    assert extension.quotient_order == 3
    _, f_label = C.cell_labels(extension, c6.order)
    onehot = np.eye(int(f_label.max()) + 1)[f_label]
    result = C.probe_cocycle(onehot, extension, c6.order)
    assert result.defined is True
    assert result.fold_degenerate is False
    record = result.to_record()
    assert record["status"] == "measured"
    assert result.chance is not None
    assert result.accuracy > result.chance  # planted f recovers above chance
    assert record["lam"] == pytest.approx(1.0)


def test_probe_rejects_non_finite_features():
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    features = np.zeros((q8.order * q8.order, 4), dtype=np.float64)
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        C.probe_cocycle(features, extension, q8.order)


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


def test_ablation_rejects_zero_subspace_dim():
    """Explicit subspace_dim=0 is a caller error, not a stand-in for 'unset':
    ``0 or (k - 1)`` used to silently swallow it. Reject it explicitly."""
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    model, group = _model("Q8", 8, 4)
    with pytest.raises(ValueError, match="subspace_dim"):
        C.cocycle_ablation(model, group, extension, subspace_dim=0)


def test_ablation_flags_single_cell_f_partition_on_q2():
    """At |Q| = 2 the f partition is the (1, 1)-cell indicator, so the ablation is
    confounded with quotient-pair sensitivity; the record surfaces the confound."""
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    model, group = _model("Q8", 8, 4)
    record = C.cocycle_ablation(model, group, extension).to_record()
    assert record["f_partition_is_single_cell_indicator"] is True
    assert record["f_partition_single_cell"] == 3  # cell (1, 1) = 1 * 2 + 1


def test_records_carry_seed_lam_and_normal_membership():
    """Provenance the audit found missing: the normal subgroup's membership on the
    extension record, the random-control seed on the ablation record, and lam on
    the probe record."""
    q8 = resolve_group("Q8")
    normal = _first_normal(q8, 4)
    extension = C.build_extension(q8, normal)
    assert extension.to_record()["normal"] == sorted(int(x) for x in normal.tolist())

    model, group = _model("Q8", 8, 4)
    ablation_record = C.cocycle_ablation(model, group, extension, seed=7).to_record()
    assert ablation_record["seed"] == 7

    # lam appears on a genuinely measured probe record (a non-degenerate |Q| = 3).
    c6 = resolve_group("C6")
    c6_extension = C.build_extension(c6, _first_normal(c6, 2))
    _, f_label = C.cell_labels(c6_extension, c6.order)
    onehot = np.eye(int(f_label.max()) + 1)[f_label]
    probe_record = C.probe_cocycle(onehot, c6_extension, c6.order, lam=2.5).to_record()
    assert probe_record["lam"] == pytest.approx(2.5)


def test_extension_to_record_is_serialisable():
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    record = extension.to_record()
    assert record["quotient_order"] == 2
    assert record["cocycle_trivial"] is False
    assert len(record["cocycle"]) == 2
    assert record["normal"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# I-22c: twisted-rule FVE fit (held out over (q1, q2) cells)
# ---------------------------------------------------------------------------


def test_leave_one_cell_out_never_splits_a_cell():
    """The FVE fold structure is the anti-pseudoreplication guard: each fold
    holds out one whole (q1, q2) cell, so no (a, b) row of the held-out cell ever
    appears in that fold's training set, and every row is held out exactly once
    (leave-one-cell-out coverage)."""
    q8 = resolve_group("Q8")
    normal = _first_normal(q8, 4)
    extension = C.build_extension(q8, normal)
    cell, _ = C.cell_labels(extension, q8.order)
    held_once = np.zeros(cell.size, dtype=bool)
    n_folds = 0
    for train_mask, test_mask in C._leave_one_cell_out(cell):
        n_folds += 1
        # The held-out rows are exactly one cell ...
        test_cells = set(cell[test_mask].tolist())
        assert len(test_cells) == 1
        # ... and that whole cell is absent from the training rows: no (a, b)
        # pair of the held-out cell leaks into training.
        (held,) = test_cells
        assert not np.any(cell[train_mask] == held)
        # Train and test partition the rows.
        assert np.array_equal(train_mask, ~test_mask)
        held_once |= test_mask
    assert n_folds == np.unique(cell).size == extension.quotient_order**2
    assert held_once.all()  # every row held out in exactly one fold


def test_cell_held_out_fve_positive_control():
    """A known twisted signal: logits concentrated on the true answer. The
    twisted design (which predicts the true answer) earns near-perfect FVE, while
    the untwisted design (which predicts a different answer on every row) earns a
    far lower FVE -- the fit tracks the twist, not an artefact."""
    n_classes = 4
    cell = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    wrong = np.array([1, 0, 3, 2, 1, 0, 3, 2], dtype=np.int64)  # differs on every row
    logits = np.zeros((true.size, n_classes), dtype=np.float64)
    logits[np.arange(true.size), true] = 10.0
    twisted = C._rule_indicator_design(true, n_classes)
    untwisted = C._rule_indicator_design(wrong, n_classes)
    twisted_fve = C._cell_held_out_fve(logits, twisted, cell)
    untwisted_fve = C._cell_held_out_fve(logits, untwisted, cell)
    assert twisted_fve > 0.99
    assert untwisted_fve < 0.5
    assert twisted_fve - untwisted_fve > 0.5


def test_twisted_rule_fve_gain_is_zero_on_split_positive_signal_on_nonsplit():
    """The GL(2,3) [split] vs SL(2,3).C2 [non-split] contrast in miniature. On the
    split member with the trivialising transversal f == e, so the twisted and
    untwisted rules coincide and the gain is exactly 0. On the non-split member
    the cocycle is nontrivial, so the two rules differ and the fit is a genuine
    measurement (finite, held out over cells)."""
    # Split S3: f == e under the complement transversal -> gain is 0.
    s3 = resolve_group("S3")
    normal = _first_normal(s3, 3)
    transversal = C.trivialising_transversal(s3, normal)
    extension = C.build_extension(s3, normal, transversal)
    model, group = _model("S3", 6, 1)
    result = C.twisted_rule_fve(model, group, extension)
    assert result.twist_is_trivial is True
    assert result.twist_fve_gain == pytest.approx(0.0, abs=1e-9)
    assert result.twisted_fve == pytest.approx(result.untwisted_fve, abs=1e-9)
    assert result.n_cells == extension.quotient_order**2
    assert result.to_record()["status"] == "measured"

    # Non-split Q8: the cocycle is nontrivial, so the two rules genuinely differ.
    q8 = resolve_group("Q8")
    q8_normal = _first_normal(q8, 4)
    q8_extension = C.build_extension(q8, q8_normal)
    q8_model, q8_group = _model("Q8", 8, 4)
    q8_result = C.twisted_rule_fve(q8_model, q8_group, q8_extension)
    assert q8_result.twist_is_trivial is False
    assert q8_result.n_features == 1
    assert q8_result.n_cells == q8_extension.quotient_order**2
    for value in (q8_result.twisted_fve, q8_result.untwisted_fve, q8_result.twist_fve_gain):
        assert np.isfinite(value)
    assert q8_result.twisted_fve <= 1.0 + 1e-9
    assert q8_result.untwisted_fve <= 1.0 + 1e-9


def test_twisted_fve_flags_the_q2_mechanism_confound():
    """At |Q| = 2 the twisted and untwisted designs differ only on the held-out
    cell, so a positive gain measures accuracy on that cell, not the cocycle
    mechanism. The record stays measured (the FVE is a real number) but sets
    fold_degenerate so the mechanistic reading is not overclaimed."""
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    model, group = _model("Q8", 8, 4)
    record = C.twisted_rule_fve(model, group, extension).to_record()
    assert record["status"] == "measured"
    assert record["fold_degenerate"] is True
    assert record["degenerate_cells"]


def test_cell_held_out_fve_rejects_non_finite_logits():
    n_classes = 4
    cell = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    logits = np.zeros((true.size, n_classes), dtype=np.float64)
    logits[0, 0] = np.nan
    design = C._rule_indicator_design(true, n_classes)
    with pytest.raises(ValueError, match="non-finite"):
        C._cell_held_out_fve(logits, design, cell)


def test_twisted_rule_fve_runs_on_the_fc_architecture():
    """The FVE target ``resid_final @ W_U`` is the logit path of both
    architectures; the FC model exercises the second one."""
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    model, group = _model("Q8", 8, 4, arch="fc")
    result = C.twisted_rule_fve(model, group, extension)
    assert result.site == "resid_final"
    assert result.twist_is_trivial is False
    assert np.isfinite(result.twist_fve_gain)


# ---------------------------------------------------------------------------
# I-22d: intervention-based predicted-error test (the |Q| = 2 replacement)
# ---------------------------------------------------------------------------


def _synthetic_logits(extension, table, *, twisted: bool, A=3.0, B=2.0, noise=0.05, seed=1):
    """Planted read-position logits ``[|G|^2, |G|]`` for a synthetic model whose
    unembedding is the identity (logits == the read representation).

    ``twisted=True`` plants a genuine twisted mechanism: a strong spike on the true
    answer AND a secondary spike on the row-specific untwisted answer (the
    represented intermediate the cocycle correction sits on top of), so knocking
    out the winner reveals the untwisted product. ``twisted=False`` is the confound
    control -- a pure per-row lookup spike on the true answer with no untwisted
    substructure, exactly the cell-membership model I-22b/I-22c cannot rule out.
    Both are argmax-correct on every row (a grokked model); the discriminator is
    where the *induced* errors land, not accuracy."""
    order = table.shape[0]
    true = table.reshape(-1)
    untwisted = C._predicted_products(extension, table, include_cocycle=False)
    rng = np.random.default_rng(seed)
    logits = np.zeros((order * order, order), dtype=np.float64)
    logits[np.arange(order * order), true] += A
    if twisted:
        active = np.flatnonzero(untwisted != true)
        logits[active, untwisted[active]] += B
    logits += noise * rng.standard_normal(logits.shape)
    return logits


def test_i22d_positive_control_twisted_model_shows_twist_specific_effect():
    """POSITIVE CONTROL. A synthetic model that genuinely computes via twisted
    coordinates (untwisted product represented, then corrected) must report a
    clear twist-specific effect: knocking out the winner drops the induced errors
    onto the row-specific untwisted target, far above both matched nulls."""
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    assert extension.quotient_order == 2  # the registered |Q| = 2 shape
    logits = _synthetic_logits(extension, q8.cayley_table, twisted=True)
    result = C.predicted_error_intervention(logits, q8, extension, seed=0, n_shuffle=200)
    assert result.status == "measured"
    assert result.untwist_hit_rate is not None and result.untwist_hit_rate > 0.9
    assert result.effect_vs_shuffle is not None and result.effect_vs_shuffle > 0.5
    # |N| = 4 >= 3, so the counterfactual-cocycle control exists and is beaten too.
    assert result.counterfactual_null_mean is not None
    assert result.effect_vs_counterfactual is not None and result.effect_vs_counterfactual > 0.5


def test_i22d_confound_control_lookup_model_shows_no_twist_effect():
    """CONFOUND CONTROL -- the case the decode designs (I-22/I-22b/I-22c) fail. A
    synthetic model that encodes cell membership and answers correctly but does NOT
    route through the twisted rule (a pure lookup spike) must report no
    twist-specific effect: the untwisted-target hit rate stays inside the
    shuffled-correspondence null band, so the effect is ~0 -- not the confounded
    positive an f-decode would give here."""
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    logits = _synthetic_logits(extension, q8.cayley_table, twisted=False)
    result = C.predicted_error_intervention(logits, q8, extension, seed=0, n_shuffle=200)
    assert result.status == "measured"
    # The hit rate does not exceed the null's 97.5% quantile: no twist signal.
    assert result.untwist_hit_rate <= result.shuffle_null_ci[1] + 1e-9
    assert result.effect_vs_shuffle < 0.2
    # Decisively separated from the twisted model's effect.
    twisted = C.predicted_error_intervention(
        _synthetic_logits(extension, q8.cayley_table, twisted=True), q8, extension, seed=0
    )
    assert twisted.effect_vs_shuffle - result.effect_vs_shuffle > 0.4


def test_i22d_split_member_is_twist_inactive():
    """SPLIT-MEMBER CONTROL. On a split member coordinatised with the trivialising
    transversal ``f == e``, so there is no twist-active cell: the effect is
    structurally absent and the record reports ``twist_inactive`` (a UNDEFINED-
    shaped verdict), never a number. This is the built-in negative-control arm."""
    s3 = resolve_group("S3")
    normal = _first_normal(s3, 3)
    extension = C.build_extension(s3, normal, C.trivialising_transversal(s3, normal))
    logits = _synthetic_logits(extension, s3.cayley_table, twisted=False)
    result = C.predicted_error_intervention(logits, s3, extension, seed=0)
    assert result.status == "twist_inactive"
    record = result.to_record()
    assert record["status"] == "twist_inactive"
    assert record["reason"]
    assert "untwist_hit_rate" not in record  # no measured number on the split control


def test_i22d_positive_control_generalises_to_q3():
    """Not hard-coded to |Q| = 2: a |Q| = 3 member (C6 over its C2, canonical
    transversal -- cocycle spread over several cells, |N| = 2) still recovers the
    planted twist via the universal shuffled-correspondence null, where the
    counterfactual control is undefined (|N| < 3)."""
    c6 = resolve_group("C6")
    extension = C.build_extension(c6, _first_normal(c6, 2))  # canonical -> non-trivialising
    assert extension.quotient_order == 3
    logits = _synthetic_logits(extension, c6.cayley_table, twisted=True)
    result = C.predicted_error_intervention(logits, c6, extension, seed=0, n_shuffle=200)
    assert result.status == "measured"
    assert result.n_active_cells >= 1
    assert result.untwist_hit_rate > 0.9
    assert result.effect_vs_shuffle > 0.4
    assert result.counterfactual_null_mean is None  # |N| = 2: no counterfactual value
    assert result.effect_vs_counterfactual is None


def test_i22d_is_deterministic_for_a_fixed_seed():
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    logits = _synthetic_logits(extension, q8.cayley_table, twisted=True)
    a = C.predicted_error_intervention(logits, q8, extension, seed=3, n_shuffle=128)
    b = C.predicted_error_intervention(logits, q8, extension, seed=3, n_shuffle=128)
    assert a.to_record() == b.to_record()


def test_i22d_no_correct_active_rows_when_model_is_wrong():
    """A censored/ungrokked model wrong on every twist-active row has no correct
    prediction to knock out: the record says so honestly, not a spurious number."""
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    logits = np.zeros((q8.order * q8.order, q8.order), dtype=np.float64)
    logits[:, 5] = 10.0  # always predicts class 5; cell (1,1) answers live in N = {0..3}
    result = C.predicted_error_intervention(logits, q8, extension, seed=0)
    assert result.status == "no_correct_active_rows"
    assert result.to_record()["reason"]


def test_i22d_rejects_wrong_shape_and_non_finite_logits():
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    with pytest.raises(ValueError, match="logits must be"):
        C.predicted_error_intervention(np.zeros((10, 3)), q8, extension)
    bad = np.zeros((q8.order * q8.order, q8.order), dtype=np.float64)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        C.predicted_error_intervention(bad, q8, extension)


def test_i22d_measured_record_schema():
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    logits = _synthetic_logits(extension, q8.cayley_table, twisted=True)
    record = C.predicted_error_intervention(logits, q8, extension, seed=0, n_shuffle=64).to_record()
    assert record["instrument"] == "cocycle_predicted_error"
    assert record["status"] == "measured"
    for key in (
        "untwist_hit_rate",
        "untwist_hit_ci",
        "shuffle_null",
        "effect_vs_shuffle",
        "counterfactual_null",
        "effect_vs_counterfactual",
        "n_active_cells",
        "n_correct_active_rows",
        "active_cells",
        "seed",
        "n_shuffle",
    ):
        assert key in record
    assert set(record["shuffle_null"]) == {"mean", "ci_95"}
    import json

    json.dumps(record)  # JSON-serialisable


def test_cocycle_predicted_error_runs_on_a_trained_model_path():
    """The model-facing wrapper reads ``resid_final @ W_U`` and runs end to end on
    a real (tiny, untrained-ish) model; the status is an honest one of the
    defined outcomes, never a crash."""
    q8 = resolve_group("Q8")
    extension = C.build_extension(q8, _first_normal(q8, 4))
    model, group = _model("Q8", 8, 4)
    result = C.cocycle_predicted_error(model, group, extension, seed=0, n_shuffle=64)
    assert result.status in {"measured", "no_correct_active_rows"}
    assert result.to_record()["instrument"] == "cocycle_predicted_error"


def test_select_normal_subgroup_is_deterministic_and_records_membership():
    q8 = resolve_group("Q8")
    normal = C.select_normal_subgroup(q8, 4)
    assert normal is not None
    assert C.is_normal(q8.cayley_table, normal)
    assert normal.tolist() == sorted(normal.tolist())
    assert C.select_normal_subgroup(q8, 5) is None  # no order-5 subgroup of an order-8 group


# ---------------------------------------------------------------------------
# End-to-end: scripts/measure_cocycle.py on a small trained fixture run
# ---------------------------------------------------------------------------

_COCYCLE_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_cocycle.py"


def _load_cocycle_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("measure_cocycle", _COCYCLE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_cocycle"] = module
    spec.loader.exec_module(module)
    return module


def _train_run(runs_root: Path, name: str) -> Path:
    from group_algorithm_interp.config import ExperimentConfig, LoggingConfig, ProjectConfig
    from group_algorithm_interp.experiment import GroupGeneralizationExperiment

    config = ProjectConfig(
        device="cpu",
        seed=0,
        data={"group": name, "train_frac": 0.5, "split_seed": 0},
        model={"d_model": 16, "d_mlp": 32, "n_heads": 1},
        optim={"epochs": 8, "log_every": 1, "print_every": 1},
        snapshot={
            "enabled": True,
            "interval": 4,
            "log_dense_until": 4,
            "event_based": False,
            "final_window_epochs": 3,
        },
        logging=LoggingConfig(mode="disabled"),
        experiment=ExperimentConfig(name="cocycle-measure"),
    )
    experiment = GroupGeneralizationExperiment(config, runs_root=runs_root)
    experiment.execute()
    return experiment.run_dir


@pytest.fixture(scope="module")
def nonsplit_and_split_runs(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("cocycle-runs")
    return _train_run(root, "Q8"), _train_run(root, "D8")


def test_measure_cocycle_run_wires_the_full_arm_on_the_nonsplit_member(nonsplit_and_split_runs):
    """End to end: the driver's per-run record carries I-21, the I-22/I-22b/I-22c
    guarded decode/ablation/fit, and the I-22d predicted-error test, with the full
    provenance block -- on the non-split (Q8) member (|Q| = 2), where I-22 is
    structurally degenerate but I-22d is defined."""
    q8_run, _ = nonsplit_and_split_runs
    record = C.measure_cocycle_run(q8_run, threshold=0.0, n_shuffle=64)
    assert record["status"] == "measured"
    assert record["group"]["name"] == "SmallGroup(8,4)"
    assert record["normal_order"] == 4
    assert record["normal_membership"]
    assert record["f_equiv_one_fit"]["splits"] is False
    # I-22 is structurally degenerate at |Q| = 2; I-22d is the live instrument.
    assert record["i22_cocycle_probe"]["status"] == "STRUCTURALLY_DEGENERATE"
    assert record["i22d_predicted_error"]["status"] in {"measured", "no_correct_active_rows"}
    prov = record["provenance"]
    assert prov["analysis_git_commit"] is not None
    assert "analysis_git_dirty" in prov
    assert len(prov["checkpoint_sha256"]) == 64
    assert prov["group_artifact"].endswith("smallgroup_8_4.npz")
    assert len(prov["group_artifact_sha256"]) == 64
    assert prov["instrument_code_sha256"]["cocycle.py"]


def test_measure_cocycle_run_reports_split_control_on_the_split_member(nonsplit_and_split_runs):
    """On the split member (D8) the trivialising transversal makes ``f == e``, so
    I-22 is UNDEFINED and I-22d reports ``twist_inactive`` -- the negative-control
    arm, carried as data not a number."""
    _, d8_run = nonsplit_and_split_runs
    record = C.measure_cocycle_run(d8_run, threshold=0.0, n_shuffle=64)
    assert record["status"] == "measured"
    assert record["f_equiv_one_fit"]["splits"] is True
    assert record["i22_cocycle_probe"]["status"] == "UNDEFINED"
    assert record["i22d_predicted_error"]["status"] == "twist_inactive"


def test_measure_cocycle_cli_writes_records_and_exit_codes(nonsplit_and_split_runs, tmp_path):
    script = _load_cocycle_script()
    q8_run, d8_run = nonsplit_and_split_runs
    out = tmp_path / "cocycle.json"
    code = script.main(
        [
            "measure",
            str(q8_run),
            str(d8_run),
            "--threshold",
            "0.0",
            "--n-shuffle",
            "32",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert len(payload["runs"]) == 2
    for run_dir in (q8_run, d8_run):
        per_run = json.loads((run_dir / "analysis" / "cocycle.json").read_text())
        assert per_run["status"] == "measured"
    # A run with no stable checkpoint is skipped and the CLI exits non-zero.
    skip_code = script.main(["measure", str(q8_run), "--threshold", "0.99"])
    assert skip_code == 1
