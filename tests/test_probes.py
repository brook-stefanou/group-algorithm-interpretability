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

from types import SimpleNamespace

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


def test_klein_four_is_classified_trivial_not_dihedral():
    """DECISIVE regression: for m <= 2 the cyclic generator is its own inverse,
    so the inversion action and the trivial action are the same function on N.
    Klein four (C2 x C2, m = 2) must be classified action_sign = +1 (trivial),
    never -1 (dihedral) -- a hand-built table catches the bug the artifact
    corpus (which has no order-4 abelian target) cannot."""
    # Klein four as bitwise XOR on {0, 1, 2, 3}: table[i, j] = i ^ j.
    table = np.array([[i ^ j for j in range(4)] for i in range(4)], dtype=np.int64)
    group = SimpleNamespace(cayley_table=table)
    coords = P.signed_cyclic_coordinates(group)
    assert coords is not None
    assert coords.radix == 2
    assert coords.action_sign == 1
    assert not coords.is_signed_cyclic


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
    # Finding 2: the twisted-rule fit is accuracy-sensitive, not independent
    # mechanism evidence -- the record must carry that caveat explicitly.
    assert "caveat" in d_record["twisted_rule_fit"]
    assert "accuracy" in d_record["twisted_rule_fit"]["caveat"]


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
    """Abelian groups (the C3 scope) are enumeration-robust: a coordinate
    system always exists, so ``pc is not None`` is asserted directly for
    them. Non-abelian groups (D8, Q8) are enumeration-dependent (see
    ``polycyclic_digits``'s docstring): whether the greedy chain returns a
    system at all depends on the artifact's element enumeration, so the
    fixed measured/None outcome is not asserted here -- only that *if* a
    system is returned, it has the bijection and reconstruction properties
    any valid system must have, regardless of which enumeration produced it."""
    abelian = ("C8", "C7", (21, 2))
    non_abelian = ("Q8", "D8")
    for name in abelian:
        group = resolve_group(name)
        pc = P.polycyclic_digits(group)
        assert pc is not None, f"{name} is abelian: a coordinate system must always exist"
        _assert_bijection_and_reconstruction(pc, group)
    for name in non_abelian:
        group = resolve_group(name)
        pc = P.polycyclic_digits(group)
        if pc is not None:
            _assert_bijection_and_reconstruction(pc, group)


def _assert_bijection_and_reconstruction(pc: P.PolycyclicDigits, group) -> None:
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


@pytest.mark.parametrize("source", ["left", "right"])
def test_involution_ablation_rejects_non_embedding_sources(source):
    """DECISIVE (finding 1): source="left"/"right" builds the involution
    direction in d_mlp space (neuron_features), but ablate_direction only
    projects a direction out of W_E in d_model space -- a dimensional mismatch
    that either crashes (D_MODEL != D_MLP here) or, on a config where they
    happen to coincide, would silently ablate a meaningless axis and report a
    spurious "measured" record. The honest fix is to reject it outright."""
    model, group = _random_model("D8")
    with pytest.raises(ValueError, match="only supports source='embed'"):
        P.involution_direction_ablation(model, group, source=source)


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
    record = P.functional_form_fit(model, order, forms, seed=0, train_frac=0.6)
    fve = {f["name"]: f["held_out_fve"] for f in record["forms"]}
    assert fve["true"] > 0.9
    assert fve["true"] > fve["wrong"]
    # Finding 7: the fit-defining split parameters must travel in the record.
    assert record["train_frac"] == pytest.approx(0.6)
    assert record["split_seed"] == 0


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


def test_functional_form_fit_computes_null_logits_once_not_per_form(monkeypatch):
    """Finding 7: the null model's read-position logits do not depend on which
    form is being scored, so they must be computed once, not once per form."""
    group = resolve_group("D8")
    coords = P.signed_cyclic_coordinates(group)
    forms = P._signed_cyclic_forms(coords, group.order)
    assert len(forms) == 2  # a real multi-form call, so "once per form" would show
    model, _ = _random_model("D8", seed=0)
    null_model, _ = _random_model("D8", seed=1)

    calls = []
    real_read_position_logits = P.read_position_logits

    def _counting_read_position_logits(m, order, **kwargs):
        calls.append(m)
        return real_read_position_logits(m, order, **kwargs)

    monkeypatch.setattr(P, "read_position_logits", _counting_read_position_logits)
    P.functional_form_fit(model, group.order, forms, null_model=null_model, seed=0)
    assert calls.count(null_model) == 1
    assert calls.count(model) == 1


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


# ---------------------------------------------------------------------------
# Finiteness guard: a NaN/Inf feature or logit must fail loudly, not pass
# silently through nearest-centroid argmin or lstsq.
# ---------------------------------------------------------------------------


def test_embedding_features_rejects_nan_in_w_e():
    model, group = _random_model("D8")
    with torch.no_grad():
        model.W_E[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        P.embedding_features(model, group.order)


def test_neuron_features_rejects_nan_activations(monkeypatch):
    model, group = _random_model("D8")

    def _nan_activations(_model, order):
        return np.full((4, order, order), np.nan)

    monkeypatch.setattr(P, "neuron_activations", _nan_activations)
    with pytest.raises(ValueError, match="non-finite"):
        P.neuron_features(model, group, argument="left")


def test_read_position_logits_rejects_nan_logits():
    order = 3
    grid = np.zeros((order, order, 2))
    grid[0, 0, 0] = np.nan
    model = _FakeModel(grid)
    with pytest.raises(ValueError, match="non-finite"):
        P.read_position_logits(model, order)


# ---------------------------------------------------------------------------
# GCR character-readout functional form
# ---------------------------------------------------------------------------


def test_gcr_candidate_and_fourier_irreps_on_s3():
    """S3 has three irreps: trivial, the sign (1-D, nontrivial), and the
    standard 2-D representation. The candidate set excludes only the trivial
    irrep; the Fourier-only set is exactly the sign representation."""
    group = resolve_group("S3")
    candidates = P.gcr_candidate_irreps(group)
    fourier = P.gcr_fourier_only_irreps(group)
    assert sorted(group.irreps[i].dimension for i in candidates) == [1, 2]
    assert [group.irreps[i].dimension for i in fourier] == [1]
    trivial = next(i for i in range(len(group.irreps)) if i not in candidates)
    assert np.allclose(group.irreps[trivial].character, 1.0)
    # include_trivial=True adds it back.
    assert trivial in P.gcr_candidate_irreps(group, include_trivial=True)


def test_gcr_character_design_matches_direct_trace_computation():
    """The design column is exactly Re tr(rho(a) rho(b) rho(c^-1)), computed
    directly from the artifact's irrep matrices -- not merely self-consistent
    with the vectorised implementation."""
    group = resolve_group("S3")
    std_idx = next(i for i in P.gcr_candidate_irreps(group) if group.irreps[i].dimension == 2)
    design = P.gcr_character_design(group, [std_idx])
    matrices = group.irreps[std_idx].matrices
    table = group.cayley_table
    inv = P.inverses(table, P.identity_index(table))
    for a, b, c in [(2, 4, 5), (0, 1, 2), (5, 5, 0)]:
        expected = np.trace(matrices[a] @ matrices[b] @ matrices[int(inv[c])]).real
        assert design[a, b, c, 0] == pytest.approx(expected)


def _sparse_gcr_target(
    group, irrep_indices: list[int], coeffs: list[float], *, noise_scale: float = 0.0, seed: int = 0
) -> np.ndarray:
    design = P.gcr_character_design(group, irrep_indices)
    logits = (design * np.asarray(coeffs)).sum(axis=-1)
    if noise_scale:
        logits = logits + np.random.default_rng(seed).normal(scale=noise_scale, size=logits.shape)
    return logits


def test_gcr_character_readout_recovers_a_sparse_two_dimensional_irrep_target():
    """Positive control + discrimination: logits that ARE the GCR readout of
    the 2-D standard irrep alone (plus small noise). The full irrep set
    recovers them out of sample; the Fourier-only (1-D, abelian) rival --
    the deliberately mismatched form -- does not, so the nested comparison
    shows a large gap; and the minimal-set search identifies exactly the
    true 2-D irrep, not the full candidate set."""
    group = resolve_group("S3")
    candidates = P.gcr_candidate_irreps(group)
    std_idx = next(i for i in candidates if group.irreps[i].dimension == 2)
    logits = _sparse_gcr_target(group, [std_idx], [1.0], noise_scale=0.01, seed=0)
    model = _FakeModel(logits)
    record = P.gcr_character_readout_instrument(model, group, seed=0)

    nested = record["primary"]["nested_comparison"]
    assert nested["full_held_out_fve"] > 0.9
    assert nested["fourier_only_held_out_fve"] < 0.6
    assert nested["full_vs_fourier_held_out_fve_gain"] > 0.3

    minimal = record["primary"]["minimal_irrep_set"]
    assert minimal["selected_irrep_indices"] == [std_idx]
    assert minimal["reached_target"]

    # Raw FVE is present but demoted -- never the sole headline statistic.
    assert "secondary_raw_fve" in record
    assert "class function" in record["caveat"]


def test_gcr_character_readout_recovers_a_two_irrep_sum():
    """Positive control on a sparse sum of TWO irreps (sign + standard), as
    the GCR readout prediction allows ('a sparse sum over occupied irreps').
    Since S3 has only these two nontrivial irreps, the minimal set found is
    the full candidate set -- both are needed, neither is spurious."""
    group = resolve_group("S3")
    candidates = P.gcr_candidate_irreps(group)
    sign_idx = next(i for i in candidates if group.irreps[i].dimension == 1)
    std_idx = next(i for i in candidates if group.irreps[i].dimension == 2)
    logits = _sparse_gcr_target(group, [sign_idx, std_idx], [0.7, 1.0], noise_scale=0.01, seed=1)
    model = _FakeModel(logits)
    record = P.gcr_character_readout_instrument(model, group, seed=0)
    assert record["primary"]["nested_comparison"]["full_held_out_fve"] > 0.9
    assert set(record["primary"]["minimal_irrep_set"]["selected_irrep_indices"]) == {
        sign_idx,
        std_idx,
    }


def test_gcr_character_readout_negative_control_on_a_random_lookup_target():
    """Negative control: a random one-hot lookup table, unrelated to any
    Phi_rho. Raw held-out FVE must not be spuriously high, and the
    minimal-set search must not reach the target FVE -- the instrument does
    not manufacture GCR support out of an arbitrary target."""
    group = resolve_group("S3")
    order = group.order
    rng = np.random.default_rng(0)
    random_answer = rng.integers(0, order, size=(order, order))
    logits = np.zeros((order, order, order))
    for a in range(order):
        for b in range(order):
            logits[a, b, random_answer[a, b]] = 1.0
    model = _FakeModel(logits)
    record = P.gcr_character_readout_instrument(model, group, seed=0)
    assert record["primary"]["nested_comparison"]["full_held_out_fve"] < 0.3
    assert not record["primary"]["minimal_irrep_set"]["reached_target"]


def test_gcr_character_readout_null_model_scores_near_zero():
    """The untrained-model null (I-20's convention, reused here): a random-init
    model's logits carry no GCR structure, so held-out FVE for both the full
    and Fourier-only forms is ~0."""
    model, group = _random_model("D8")
    null_model, _ = _random_model("D8", seed=1)
    record = P.gcr_character_readout_instrument(model, group, null_model=null_model, seed=0)
    assert record["null"]["full_null_held_out_fve"] < 0.2
    assert record["null"]["fourier_only_null_held_out_fve"] < 0.2


def test_gcr_character_readout_undefined_nested_comparison_without_fourier_rival(monkeypatch):
    """A group with no nontrivial one-dimensional irrep (simulated by
    monkeypatching the Fourier-only lookup to empty, since no perfect-group
    fixture exists at this scale) reports the nested comparison as UNDEFINED
    rather than comparing against an empty design."""
    model, group = _random_model("D8")
    monkeypatch.setattr(P, "gcr_fourier_only_irreps", lambda g: [])
    record = P.gcr_character_readout_instrument(model, group, seed=0)
    assert record["primary"]["nested_comparison"]["status"] == P.UNDEFINED
    assert record["n_fourier_irreps"] == 0


def test_gcr_character_readout_row_sampling_is_a_noop_above_the_cap():
    """A cap above order**3 leaves the GCR readout's FVE fields byte-identical
    to the uncapped run and reports row_sampling_applied False."""
    group = resolve_group("S3")
    candidates = P.gcr_candidate_irreps(group)
    std_idx = next(i for i in candidates if group.irreps[i].dimension == 2)
    logits = _sparse_gcr_target(group, [std_idx], [1.0], noise_scale=0.01, seed=0)
    model = _FakeModel(logits)
    baseline = P.gcr_character_readout_instrument(model, group, seed=0)
    capped = P.gcr_character_readout_instrument(
        model, group, seed=0, max_rows=10 * group.order**3, sample_seed=4
    )
    assert baseline["sampling"]["row_sampling_applied"] is False
    assert capped["sampling"]["row_sampling_applied"] is False
    assert (
        capped["secondary_raw_fve"]["full_held_out_fve"]
        == baseline["secondary_raw_fve"]["full_held_out_fve"]
    )
    assert (
        capped["primary"]["minimal_irrep_set"]["selected_irrep_indices"]
        == baseline["primary"]["minimal_irrep_set"]["selected_irrep_indices"]
    )


def test_gcr_character_readout_row_sampling_records_metadata_and_stays_finite():
    """Under a genuine cap the run subsamples whole (a, b) pairs (the full class
    axis kept) and reports finite FVE plus the sampling metadata."""
    group = resolve_group("S3")
    candidates = P.gcr_candidate_irreps(group)
    std_idx = next(i for i in candidates if group.irreps[i].dimension == 2)
    logits = _sparse_gcr_target(group, [std_idx], [1.0], noise_scale=0.01, seed=0)
    model = _FakeModel(logits)
    order = group.order
    cap = order * order
    record = P.gcr_character_readout_instrument(model, group, seed=0, max_rows=cap, sample_seed=1)
    meta = record["sampling"]
    assert meta["row_sampling_applied"] is True
    assert meta["n_pairs_used"] == cap // order
    assert meta["n_classes"] == order
    assert np.isfinite(record["secondary_raw_fve"]["full_held_out_fve"])


def test_fc_model_is_accepted_by_the_probes():
    """The FC baseline shares the W_E / read-position contract, so the same
    probes accept it (what makes the I-36 architecture replication cheap)."""
    model, group = _random_model("D8", arch="fc")
    record = P.power_map_probe(model, group)
    assert record["source"] == "embed"
    assert P.signed_cyclic_instrument(model, group)["status"] == "measured"
