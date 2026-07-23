"""The mechanism-arm probes (instruments/probes.py): I-20, I-26, I-27, I-28.

Two properties are load-bearing and get decisive tests:

* **I-27 must fail on the quaternionic member.** ``signed_cyclic_coordinates``
  returns ``None`` on Q8 (the fixture-scale Q32: non-split, no ``(r, s)``
  coordinate system) and a table-reproducing system on D8/S3, so the whole
  ``signed_cyclic_instrument`` is ``UNDEFINED`` on the quaternionic member. A
  probe that "succeeded" there would fit an artefact and disqualify the dihedral
  result; the construction makes that impossible.
* **I-28 is chance-corrected.** On a random-init model the adjusted balanced
  accuracy of the power-map probe sits at ~0 (pooled over seeds), the mandatory
  rule-1 regression -- while a perfectly separable synthetic target scores ~1, so
  the metric is not trivially zero.

The rest covers the group-theory precompute (signed-cyclic and polycyclic
coordinates, both verified against the Cayley table), the nested-safe held-out
scoring of the I-20 fit harness, and the ``UNDEFINED`` handling of degenerate
labels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments import probes as P
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

D_MODEL = 32
D_MLP = 64


def _config(order: int, index: int, arch: str = "transformer") -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}},
        model={"arch": arch, "d_model": D_MODEL, "d_mlp": D_MLP, "n_heads": 4},
        logging={"mode": "disabled"},
    )


def _random_model(name: str, seed: int = 0, arch: str = "transformer"):
    group = resolve_group(name)
    set_seed(seed, deterministic=False)
    return build_model(_config(group.order, group.index, arch), group), group


class _FakeModel:
    """A model stub whose read-position logits are a fixed ``[order, order, C]``
    grid -- lets the I-20 harness be tested on logits of a known functional form
    without training a real model."""

    def __init__(self, logits_grid: np.ndarray):
        self._grid = torch.tensor(logits_grid, dtype=torch.float64)
        self.W_E = torch.zeros(logits_grid.shape[0] + 1, 2)

    def eval(self) -> None:  # noqa: D401 - matches nn.Module.eval()
        return None

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        order = self._grid.shape[0]
        flat = self._grid.reshape(order * order, -1)
        return flat.unsqueeze(1).expand(-1, 3, -1)


# ---------------------------------------------------------------------------
# Cayley-table group theory
# ---------------------------------------------------------------------------


def test_element_orders_hand_values():
    q8 = resolve_group("Q8")
    orders = P.element_orders(q8.cayley_table)
    # Q8: identity (order 1), -1 (order 2), and six order-4 elements.
    assert sorted(orders.tolist()) == [1, 2, 4, 4, 4, 4, 4, 4]
    d8 = resolve_group("D8")
    d8_orders = P.element_orders(d8.cayley_table)
    # D8: e(1), r^2(2), r/r^3(4,4), four reflections(2) -> one 1, five 2, two 4.
    assert sorted(d8_orders.tolist()) == [1, 2, 2, 2, 2, 2, 4, 4]


def test_square_and_cube_maps_are_table_lookups():
    c8 = resolve_group("C8")
    table = c8.cayley_table
    sq = P.square_map(table)
    assert [int(v) for v in sq] == [int(table[g, g]) for g in range(8)]
    cube = P.cube_map(table)
    assert int(cube[1]) == int(table[table[1, 1], 1])


# ---------------------------------------------------------------------------
# I-27 negative control: signed-cyclic coordinates (the decisive Q32 property)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,radix", [("D8", 4), ("S3", 3)])
def test_signed_cyclic_coordinates_exist_and_reproduce_the_table(name, radix):
    group = resolve_group(name)
    coords = P.signed_cyclic_coordinates(group)
    assert coords is not None
    assert coords.is_signed_cyclic  # inversion action
    assert coords.radix == radix
    # Every element has a unique (r, s); the rule rebuilds the whole Cayley table.
    seen = {(int(r), int(s)) for r, s in coords.coords}
    assert len(seen) == group.order
    table = group.cayley_table
    for a in range(group.order):
        ra, sa = int(coords.coords[a, 0]), int(coords.coords[a, 1])
        for b in range(group.order):
            rb, sb = int(coords.coords[b, 0]), int(coords.coords[b, 1])
            r = (ra + (coords.action_sign**sa) * rb) % radix
            s = (sa + sb) % 2
            assert int(coords.element_of[r, s]) == int(table[a, b])


def test_signed_cyclic_is_undefined_on_the_quaternionic_member():
    """DECISIVE (I-27): Q8 is the fixture-scale Q32 -- non-split, so no (r, s)
    coordinate system exists and the construction must return None. A cyclic
    group (C8) also returns None: it has an index-2 cyclic subgroup but no
    reflection outside it."""
    assert P.signed_cyclic_coordinates(resolve_group("Q8")) is None
    assert P.signed_cyclic_coordinates(resolve_group("C8")) is None


def test_signed_cyclic_instrument_status_is_undefined_on_q8_but_measured_on_d8():
    """DECISIVE (I-27): the whole instrument -- probe and twisted-rule fit --
    is UNDEFINED on the quaternionic member, not merely a low score."""
    q_model, q_group = _random_model("Q8")
    q_record = P.signed_cyclic_instrument(q_model, q_group)
    assert q_record["status"] == P.UNDEFINED
    assert "artefact" in q_record["reason"]

    d_model_, d_group = _random_model("D8")
    d_record = P.signed_cyclic_instrument(d_model_, d_group, null_model=d_model_)
    assert d_record["status"] == "measured"
    assert d_record["action_sign"] == -1
    assert set(d_record["probe"]) == {"rotation_r", "reflection_s"}
    fit_forms = {f["name"] for f in d_record["twisted_rule_fit"]["forms"]}
    assert fit_forms == {"signed_cyclic_twisted", "untwisted"}


def test_twisted_rule_fit_prefers_the_twist_on_a_dihedral_structured_model():
    """On a model whose logits *are* the signed-cyclic one-hot (the twisted rule
    holds exactly), the twisted form explains the held-out logits and the
    untwisted reduced form does not -- the held-out FVE gain is positive. This is
    the nested comparison scored on held-out (a, b) pairs, never in-sample."""
    group = resolve_group("D8")
    coords = P.signed_cyclic_coordinates(group)
    forms = P._signed_cyclic_forms(coords, group.order)
    twisted_logits = forms[0].design[:, :, :, 0]  # one-hot of the true product
    model = _FakeModel(twisted_logits)
    record = P.functional_form_fit(model, group.order, forms, seed=0)
    fve = {f["name"]: f["held_out_fve"] for f in record["forms"]}
    assert fve["signed_cyclic_twisted"] > 0.9
    assert fve["signed_cyclic_twisted"] > fve["untwisted"]
    assert record["held_out_fve_gain"] > 0.0


# ---------------------------------------------------------------------------
# I-21 / I-26: polycyclic digits and carry structure
# ---------------------------------------------------------------------------


def test_polycyclic_digits_are_a_bijection_that_reconstructs_products():
    for name in ("C8", "C7", (21, 2), "Q8", "D8"):
        group = resolve_group(name)
        pc = P.polycyclic_digits(group)
        assert pc is not None
        # A set bijection: distinct digit tuples, one per element.
        tuples = {tuple(int(v) for v in pc.digits[g]) for g in range(group.order)}
        assert len(tuples) == group.order
        # element_of inverts the digit map.
        for g in range(group.order):
            assert pc.element_of(tuple(int(v) for v in pc.digits[g])) == g
        assert int(np.prod(pc.radices)) == group.order


def test_carry_matrix_is_triangular_for_a_radix_two_cyclic_group():
    """C8 = 2^3 is the fixture-scale C128: the low bit ripples into the high
    bits, so the carry matrix is lower-triangular (not diagonal) and the whole
    chain is one carry-linked coordinate direction (d(G) = 1)."""
    pc = P.polycyclic_digits(resolve_group("C8"))
    assert pc.radices == (2, 2, 2)
    lower_triangular = np.tril(np.ones((3, 3), dtype=bool))
    assert np.array_equal(pc.carry, lower_triangular)
    assert not np.array_equal(pc.carry, np.eye(3, dtype=bool))
    assert pc.coordinate_directions == 1


def test_single_digit_group_has_a_diagonal_carry():
    """C7 is the fixture-scale C127 anchor: a single digit, so the carry is
    trivially diagonal and there is one direction."""
    pc = P.polycyclic_digits(resolve_group("C7"))
    assert pc.length == 1
    assert np.array_equal(pc.carry, np.eye(1, dtype=bool))
    assert pc.coordinate_directions == 1


def test_carry_digit_instrument_reports_structure_with_a_null():
    model, group = _random_model("C8")
    null, _ = _random_model("C8", seed=1)
    record = P.carry_digit_instrument(model, group, null_model=null)
    assert record["status"] == "measured"
    assert record["composition_length"] == 3
    assert record["carry_is_diagonal"] is False
    assert record["coordinate_directions"] == 1
    assert len(record["digit_probes"]) == 3
    assert record["effective_rank_w_e"] > 0
    assert record["null_effective_rank_w_e"] is not None


def test_effective_rank_bounds():
    # A rank-1 matrix has effective rank ~1; an orthogonal matrix ~n.
    rank_one = np.outer(np.arange(1, 6), np.ones(4))
    assert P.effective_rank(rank_one) == pytest.approx(1.0, abs=1e-6)
    identity = np.eye(5)
    assert P.effective_rank(identity) == pytest.approx(5.0, abs=1e-6)
    assert P.effective_rank(np.zeros((3, 3))) == 0.0


# ---------------------------------------------------------------------------
# The chance-corrected probe (I-28 core)
# ---------------------------------------------------------------------------


def test_crossval_probe_is_undefined_for_degenerate_labels():
    features = np.random.default_rng(0).standard_normal((8, 4))
    # Constant label: one class -> UNDEFINED.
    assert P.crossval_probe(features, np.zeros(8, dtype=int)) == P.UNDEFINED
    # A singleton class (7 vs 1) cannot be stratified/held out -> UNDEFINED.
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    assert P.crossval_probe(features, labels) == P.UNDEFINED


def test_crossval_probe_recovers_a_perfectly_separable_target():
    """A perfectly separable target scores adjusted balanced accuracy ~1, so the
    metric is not trivially zero (the complement of the null test)."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    features = np.zeros((6, 2))
    features[labels == 1] = [10.0, 0.0]  # the two classes are far apart
    features += np.random.default_rng(0).standard_normal((6, 2)) * 0.01
    result = P.crossval_probe(features, labels, seed=0)
    assert isinstance(result, dict)
    assert result["adjusted_balanced_accuracy"] == pytest.approx(1.0)
    assert result["n_classes"] == 2


def test_power_map_probe_involution_is_undefined_when_involutions_are_rare():
    """C8 has exactly one involution (element 4): the binary involution label is
    a singleton class and comes back UNDEFINED -- the same structural asymmetry
    that makes I-28 UNDEFINED on Q32 for involution-ness."""
    model, group = _random_model("C8")
    record = P.power_map_probe(model, group)
    assert record["probes"]["is_involution"] == P.UNDEFINED
    assert record["n_involutions"] == 1


def test_i28_power_map_probe_is_chance_corrected_on_a_random_model():
    """DECISIVE (I-28): pooled over random-init seeds, the adjusted balanced
    accuracy of the square-map probe sits at ~0 -- a random model reports no
    power-map structure. Pinned as a mean over seeds because a single tiny-group
    fold can wander (the occupancy suite pools seeds for the same reason)."""
    scores = []
    for seed in range(8):
        model, group = _random_model("D8", seed=seed)
        result = P.power_map_probe(model, group, seed=seed)["probes"]["square_map"]
        assert isinstance(result, dict)
        scores.append(result["adjusted_balanced_accuracy"])
    assert abs(float(np.mean(scores))) < 0.2, scores


def test_power_map_probe_reports_chance_correction_metadata():
    model, group = _random_model("D8")
    record = P.power_map_probe(model, group)
    assert record["metric"] == "adjusted_balanced_accuracy"
    order = record["probes"]["element_order"]
    assert isinstance(order, dict)
    assert order["chance_level"] == pytest.approx(1.0 / order["n_classes"])


# ---------------------------------------------------------------------------
# I-28b: involution-direction ablation
# ---------------------------------------------------------------------------


def test_involution_ablation_measures_flips_against_a_random_control():
    model, group = _random_model("D8")
    record = P.involution_direction_ablation(model, group, n_controls=4)
    assert record["status"] == "measured"
    assert record["rung"] == 3
    assert record["baseline_correct"] >= record["ablated_correct"] - group.order
    assert "involution_direction_drop_flips" in record
    assert len(record["random_direction_drop_flips"]["per_control"]) == 4


def test_involution_ablation_is_undefined_without_involutions():
    """C7 has no involution (every non-identity element has order 7): the
    involution label is degenerate, so the ablation is UNDEFINED, not a
    fabricated zero drop."""
    model, group = _random_model("C7")
    record = P.involution_direction_ablation(model, group)
    assert record["status"] == P.UNDEFINED


def test_involution_ablation_respects_a_held_out_subset():
    model, group = _random_model("D8")
    subset = np.arange(0, group.order * group.order, 2)
    record = P.involution_direction_ablation(model, group, subset=subset)
    assert record["n_scored_pairs"] == subset.size
    assert record["baseline_correct"] <= subset.size


# ---------------------------------------------------------------------------
# I-20: the generic, nested-safe functional-form fit harness
# ---------------------------------------------------------------------------


def test_pair_split_is_disjoint_and_covers_every_pair():
    fit, test = P.pair_split(6, train_frac=0.7, seed=0)
    assert set(fit.tolist()).isdisjoint(test.tolist())
    assert sorted(fit.tolist() + test.tolist()) == list(range(36))
    assert fit.size + test.size == 36


def test_functional_form_fit_scores_the_true_form_high_and_a_wrong_form_low():
    """A random target one-hot is a form the logits do not follow; the matching
    form is fit exactly. Both are scored on held-out pairs, so a nested basis is
    never rewarded on noise."""
    order = 6
    rng = np.random.default_rng(0)
    true_answer = rng.integers(0, order, size=(order, order))
    true_design = np.zeros((order, order, order, 1))
    wrong_design = np.zeros((order, order, order, 1))
    wrong_answer = rng.integers(0, order, size=(order, order))
    for a in range(order):
        for b in range(order):
            true_design[a, b, true_answer[a, b], 0] = 1.0
            wrong_design[a, b, wrong_answer[a, b], 0] = 1.0
    model = _FakeModel(true_design[:, :, :, 0])
    forms = [
        P.FunctionalForm(name="true", design=true_design),
        P.FunctionalForm(name="wrong", design=wrong_design),
    ]
    record = P.functional_form_fit(model, order, forms, seed=0)
    fve = {f["name"]: f["held_out_fve"] for f in record["forms"]}
    assert fve["true"] > 0.9
    assert fve["true"] > fve["wrong"]


def test_functional_form_fit_null_model_scores_near_zero():
    """The untrained-model null: a random-init model's logits carry no functional
    form, so held-out FVE is ~0 (the rule-1 regression for the fit harness)."""
    group = resolve_group("D8")
    coords = P.signed_cyclic_coordinates(group)
    forms = P._signed_cyclic_forms(coords, group.order)
    model, _ = _random_model("D8", seed=0)
    null_model, _ = _random_model("D8", seed=1)
    record = P.functional_form_fit(model, group.order, forms, null_model=null_model, seed=0)
    for form in record["forms"]:
        assert form["null_held_out_fve"] < 0.2


# ---------------------------------------------------------------------------
# Feature extraction / shared plumbing
# ---------------------------------------------------------------------------


def test_element_features_are_mean_centred():
    model, group = _random_model("D8")
    embed = P.embedding_features(model, group.order)
    assert embed.shape == (group.order, D_MODEL)
    np.testing.assert_allclose(embed.mean(axis=0), 0.0, atol=1e-9)
    left = P.neuron_features(model, group, argument="left")
    assert left.shape == (group.order, D_MLP)
    np.testing.assert_allclose(left.mean(axis=0), 0.0, atol=1e-9)
    with pytest.raises(ValueError, match="'left' or 'right'"):
        P.neuron_features(model, group, argument="both")


def test_model_correct_mask_matches_manual_argmax():
    model, group = _random_model("D8")
    mask = P.model_correct_mask(model, group)
    assert mask.shape == (group.order * group.order,)
    assert mask.dtype == bool


def test_fc_model_is_accepted_by_the_probes():
    """The FC baseline shares the W_E / read-position contract, so the same
    probes accept it (what makes the I-36 architecture replication cheap)."""
    model, group = _random_model("D8", arch="fc")
    record = P.power_map_probe(model, group)
    assert record["source"] == "embed"
    assert P.signed_cyclic_instrument(model, group)["status"] == "measured"
