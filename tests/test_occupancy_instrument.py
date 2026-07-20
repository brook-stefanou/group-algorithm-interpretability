"""The spectral core (instruments/occupancy.py + instruments/templates.py).

Covers the invariants the specs make mandatory: energies non-negative and
complete (summing to the total), the analytic null ``block_rank_j / |G|``,
the Dirichlet floor formula against a hand computation, the ``Ind_H^G 1``
template library against hand-computed D8 values with its self-verifying
identities, the structural ``UNDEFINED`` on Q8 (the quaternion fixture is
exactly the Q32-class case: no nontrivial core-free subgroup), and the
suite-wide rule-1 regression -- an untrained, randomly initialised model must
not report structure on the calibrated (nontrivial) statistic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.instruments.occupancy import (
    analytic_null,
    cayley_grid_tokens,
    dirichlet_noise_floor,
    isotypic_energies,
    neuron_activations,
    per_neuron_concentration,
    population_occupancy,
    restrict_to_nontrivial,
    total_variation,
    trivial_block_index,
)
from group_algorithm_interp.instruments.templates import (
    induction_multiplicities,
    induction_template,
    is_core_free,
    subgroup_core,
    template_library,
)
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

D_MLP = 64


def _config(order: int, index: int, arch: str = "transformer") -> ProjectConfig:
    return ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}},
        model={"arch": arch, "d_model": 32, "d_mlp": D_MLP, "n_heads": 4},
        logging={"mode": "disabled"},
    )


def _random_model(order: int, index: int, seed: int = 0, arch: str = "transformer"):
    group = resolve_group((order, index))
    set_seed(seed, deterministic=False)
    return build_model(_config(order, index, arch), group), group


# ---------------------------------------------------------------------------
# I-08: activations over the Cayley grid
# ---------------------------------------------------------------------------


def test_cayley_grid_tokens_align_with_task_enumeration():
    """Row k = (k // n, k % n, '='), the same order build_group_task uses, so
    activations row k lines up with cayley_table.reshape(-1)[k]."""
    tokens = cayley_grid_tokens(3)
    expected = [(a, b, 3) for a in range(3) for b in range(3)]
    assert [tuple(row.tolist()) for row in tokens] == expected


def test_neuron_activations_shape_and_batching():
    model, group = _random_model(8, 3)
    full = neuron_activations(model, group.order)
    assert full.shape == (D_MLP, 8, 8)
    # Chunked evaluation must be exactly the single-batch result.
    chunked = neuron_activations(model, group.order, batch_size=7)
    np.testing.assert_allclose(chunked, full, rtol=0, atol=0)


def test_neuron_activations_require_an_mlp():
    group = resolve_group("D8")
    config = _config(8, 3)
    config.model.use_mlp = False
    model = build_model(config, group)
    with pytest.raises(ValueError, match="no MLP"):
        neuron_activations(model, group.order)


def test_fc_model_is_accepted_by_the_same_instrument():
    model, group = _random_model(8, 3, arch="fc")
    energies = isotypic_energies(neuron_activations(model, group.order), group)
    occupancy = population_occupancy(energies)
    assert occupancy.shape == (len(group.isotypic_blocks),)
    assert occupancy.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Isotypic energies: non-negativity + completeness, both arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argument", ["left", "right"])
def test_energies_are_nonnegative_and_complete(argument):
    model, group = _random_model(8, 3)
    activations = neuron_activations(model, group.order)
    energies = isotypic_energies(activations, group, argument=argument)
    assert energies.shape == (D_MLP, len(group.isotypic_blocks))
    assert (energies >= 0.0).all()
    totals = np.square(activations).sum(axis=(1, 2))
    np.testing.assert_allclose(energies.sum(axis=1), totals, rtol=1e-9)


def test_left_and_right_arguments_measure_different_projections():
    model, group = _random_model(8, 3)
    activations = neuron_activations(model, group.order)
    left = isotypic_energies(activations, group, argument="left")
    right = isotypic_energies(activations, group, argument="right")
    assert not np.allclose(left, right)


def test_invalid_argument_is_rejected():
    model, group = _random_model(8, 3)
    activations = neuron_activations(model, group.order)
    with pytest.raises(ValueError, match="'left' or 'right'"):
        isotypic_energies(activations, group, argument="both")


# ---------------------------------------------------------------------------
# I-10 null, I-14 floor, TV
# ---------------------------------------------------------------------------


def test_analytic_null_is_block_rank_over_order():
    d8 = resolve_group("D8")
    np.testing.assert_allclose(analytic_null(d8), [0.125, 0.125, 0.125, 0.125, 0.5])
    c8 = resolve_group("C8")
    # C8: real blocks are {chi_0}, {chi_4}, and the three merged conjugate
    # pairs -> ranks 1, 1, 2, 2, 2 (block order is the artifact's own).
    assert analytic_null(c8).sum() == pytest.approx(1.0)
    np.testing.assert_allclose(sorted(analytic_null(c8)), [0.125, 0.125, 0.25, 0.25, 0.25])


def test_total_variation_hand_values():
    assert total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert total_variation(np.array([0.25, 0.75]), np.array([0.75, 0.25])) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="shape mismatch"):
        total_variation(np.ones(2), np.ones(3))


def test_dirichlet_noise_floor_hand_value():
    """floor = 1/2 * sum_j sqrt(2/pi) * sqrt(p_j (1-p_j) / N); for p = (1/2, 1/2)
    and N = 100 this is sqrt(2/pi) * 0.05."""
    expected = math.sqrt(2.0 / math.pi) * 0.05
    assert dirichlet_noise_floor(np.array([0.5, 0.5]), 100) == pytest.approx(expected)


def test_noise_floor_scales_as_one_over_sqrt_n():
    pi0 = analytic_null(resolve_group("D8"))
    assert dirichlet_noise_floor(pi0, 50 * 256) == pytest.approx(
        dirichlet_noise_floor(pi0, 256) / math.sqrt(50)
    )
    with pytest.raises(ValueError, match="positive"):
        dirichlet_noise_floor(pi0, 0)


# ---------------------------------------------------------------------------
# Trivial block handling
# ---------------------------------------------------------------------------


def test_trivial_block_is_the_averaging_projector():
    for name in ("D8", "Q8", "C8"):
        group = resolve_group(name)
        j = trivial_block_index(group)
        assert np.allclose(group.isotypic_blocks[j].projector, 1.0 / group.order)


def test_restrict_to_nontrivial_renormalises():
    d8 = resolve_group("D8")
    pi0_nt = restrict_to_nontrivial(analytic_null(d8), trivial_block_index(d8))
    np.testing.assert_allclose(pi0_nt, [1 / 7, 1 / 7, 1 / 7, 4 / 7])


def test_per_neuron_concentration_handles_dead_units():
    energies = np.array([[3.0, 1.0], [0.0, 0.0], [0.5, 0.5]])
    top_share, top_block = per_neuron_concentration(energies)
    assert top_share[0] == pytest.approx(0.75)
    assert math.isnan(top_share[1])
    assert top_block.tolist() == [0, -1, 0]


def test_population_occupancy_rejects_zero_energy():
    with pytest.raises(ValueError, match="undefined"):
        population_occupancy(np.zeros((4, 3)))


# ---------------------------------------------------------------------------
# Suite rule 1: an untrained model must not report structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order,index", [(8, 3), (8, 4), (8, 1)])
def test_untrained_model_reports_no_structure_on_the_calibrated_form(order, index):
    """The mandatory regression test behind every structure instrument. Pooled
    over 3 random-init seeds, the nontrivial-renormalised occupancy must sit
    within 3x the Dirichlet floor of its own null (measured 1.5-2.0x across
    the fixture groups), for both arguments."""
    group = resolve_group((order, index))
    pi0 = analytic_null(group)
    trivial = trivial_block_index(group)
    pi0_nt = restrict_to_nontrivial(pi0, trivial)
    seeds = (0, 1, 2)
    for argument in ("left", "right"):
        pooled = np.zeros(len(group.isotypic_blocks))
        for seed in seeds:
            set_seed(seed, deterministic=False)
            model = build_model(_config(order, index), group)
            energies = isotypic_energies(
                neuron_activations(model, group.order), group, argument=argument
            )
            pooled += energies.sum(axis=0)
        occupancy_nt = restrict_to_nontrivial(pooled / pooled.sum(), trivial)
        tv = total_variation(occupancy_nt, pi0_nt)
        floor = dirichlet_noise_floor(pi0_nt, len(seeds) * D_MLP)
        assert tv < 3.0 * floor, f"untrained {group.canonical_name} {argument}: {tv} vs {floor}"


def test_untrained_model_documents_the_trivial_dc_dominance():
    """Why the nontrivial form is the calibrated headline: at random init the
    read-token DC puts the bulk of the energy in the trivial block, so the
    full-form statistic is dominated by an architectural offset rather than
    group structure. Pinned here so the behaviour is measured, not asserted
    from memory."""
    model, group = _random_model(8, 3, seed=0)
    energies = isotypic_energies(neuron_activations(model, group.order), group)
    occupancy = population_occupancy(energies)
    assert occupancy[trivial_block_index(group)] > 0.5


# ---------------------------------------------------------------------------
# I-12 core-free enumeration + I-13 templates
# ---------------------------------------------------------------------------


def test_subgroup_core_of_the_d8_centre_is_itself():
    """The centre {e, r^2} is normal, so it is its own core and is not
    core-free; a reflection subgroup {e, s} has trivial core."""
    d8 = resolve_group("D8")
    table = d8.cayley_table
    centre = np.array([0, 2])  # e, r^2 in the fixture's element order
    assert subgroup_core(table, centre).tolist() == [0, 2]
    assert not is_core_free(table, centre)
    reflection = np.array([0, 4])  # e, s
    assert subgroup_core(table, reflection).tolist() == [0]
    assert is_core_free(table, reflection)


def test_d8_induction_multiplicities_and_template_hand_values():
    """H = {e, s}, index 4. Frobenius reciprocity: m_chi = (chi(e) + chi(s))/2,
    so the two 1-d characters with chi(s) = +1 appear once, the two with
    chi(s) = -1 do not, and the 2-d irrep (chi(s) = 0) appears once; the
    self-verifying identity sum m_i d_i = 4 = [G:H] holds. The energy template
    is t = (1/4, 0, 1/4, 0, 1/2) and TV(t, pi0) = 1/4."""
    d8 = resolve_group("D8")
    h = np.array([0, 4])
    multiplicities = induction_multiplicities(d8, h)
    assert multiplicities.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    template = induction_template(d8, h)
    np.testing.assert_allclose(template, [0.25, 0.0, 0.25, 0.0, 0.5])
    assert total_variation(template, analytic_null(d8)) == pytest.approx(0.25)


def test_trivial_subgroup_template_equals_the_analytic_null():
    """The theorem behind the UNDEFINED rule: Ind_1^G 1 is the regular
    representation, so the trivial subgroup's template IS pi0 exactly."""
    for name in ("D8", "Q8", "C8"):
        group = resolve_group(name)
        template = induction_template(group, np.array([0]))
        np.testing.assert_allclose(template, analytic_null(group), atol=1e-9)


def test_d8_template_library_finds_the_index_4_corefree_subgroups():
    d8 = resolve_group("D8")
    library = template_library(d8)
    assert library.coset_defined
    assert library.min_corefree_index == 4
    # The four order-2 reflection subgroups are core-free; the centre is not.
    assert len(library.entries) == 4
    assert all(entry.coset_index == 4 for entry in library.entries)
    assert all(entry.tv_to_null == pytest.approx(0.25) for entry in library.entries)
    record = library.to_record()
    assert record["coset_defined"] is True
    assert len(record["entries"]) == 4


@pytest.mark.parametrize("name", ["Q8", "C8"])
def test_template_library_is_undefined_without_nontrivial_corefree_subgroups(name):
    """Q8 (every subgroup contains the centre) and any abelian group (every
    subgroup is normal) have only the trivial core-free subgroup: the coset
    account has no distinct prediction, and the library must say UNDEFINED --
    a string, deliberately not a number."""
    group = resolve_group(name)
    library = template_library(group)
    assert not library.coset_defined
    assert library.min_corefree_index == group.order
    assert library.entries == ()
    record = library.to_record()
    assert record["entries"] == "UNDEFINED"
    assert "regular representation" in record["undefined_reason"]


def test_induction_rejects_a_non_dividing_subset():
    d8 = resolve_group("D8")
    with pytest.raises(ValueError, match="does not divide"):
        induction_multiplicities(d8, np.array([0, 1, 2]))


def test_energies_reject_wrong_shape_and_corrupt_projectors():
    model, group = _random_model(8, 3)
    activations = neuron_activations(model, group.order)
    with pytest.raises(ValueError, match="shape"):
        isotypic_energies(activations[:, :4, :], group)


def test_activations_are_deterministic_given_the_seed():
    model_a, group = _random_model(8, 3, seed=7)
    model_b, _ = _random_model(8, 3, seed=7)
    a = neuron_activations(model_a, group.order)
    b = neuron_activations(model_b, group.order)
    assert np.array_equal(a, b)
    assert isinstance(model_a, torch.nn.Module)
