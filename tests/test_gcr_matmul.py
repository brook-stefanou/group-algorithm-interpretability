"""The GCR matrix-product instrument (instruments/gcr_matmul.py).

The tests are the instrument's warrant: they show it *credits* activations that
are genuinely a shared-index matrix product and *refuses* the two ways a
non-GCR account fakes a high fit -- a generic bilinear form with independent
indices (structure the matrix product cannot reach) and a function of ``a b``
that ties a failing bilinear model (structure neither reaches). They also pin
the basis-invariance property on which the whole identification rests, and the
structural ``UNDEFINED`` for a degree-1-dominated group.

Synthetic activation tensors are used for the discriminating controls, so the
ground truth is known exactly -- and they stand for the *post-nonlinearity*
activation the instrument now fits, since the matrix product ``rho(a) rho(b)``
is realised by the MLP nonlinearity and is absent from the linear ``mlp_pre``.
One lighter integration check drives a real (untrained) model through the
post-activation extraction path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from group_algorithm_interp.config import ProjectConfig
from group_algorithm_interp.groups.catalog import resolve_group
from group_algorithm_interp.groups.data import GroupData, load_group
from group_algorithm_interp.instruments.gcr_matmul import (
    UNDEFINED,
    bilinear_features,
    fit_matmul_gcr,
    matrix_product_features,
    measure_gcr_matmul,
    post_neuron_activations,
    screen_gcr_matmul,
)
from group_algorithm_interp.instruments.occupancy import neuron_activations
from group_algorithm_interp.seed import set_seed
from group_algorithm_interp.training.trainer import build_model

_REAL_ARTIFACTS = Path(__file__).resolve().parents[1] / "data" / "group_artifacts"


# ---------------------------------------------------------------------------
# Helpers: pick a degree->=2 block and build ground-truth activation tensors
# ---------------------------------------------------------------------------


def _first_high_degree_block(group: GroupData) -> tuple[int, np.ndarray]:
    for block_index, block in enumerate(group.isotypic_blocks):
        if block.irrep_degree >= 2:
            return block_index, group.irreps[block.irrep_indices[0]].matrices
    raise AssertionError(f"{group.canonical_name} has no degree->=2 irrep")


def _matrix_product_activations(
    group: GroupData, matrices: np.ndarray, n_neurons: int, noise: float, seed: int
) -> np.ndarray:
    """Activations that ARE a shared-index matrix product of ``rho``.

    ``A_m[a, b] = Re(sum_{ij} c^m_{ij} rho(a b)_{ij})`` -- the exact GCR form,
    plus small Gaussian noise. The positive control.
    """
    rng = np.random.default_rng(seed)
    order, degree, _ = matrices.shape
    left = np.repeat(np.arange(order), order)
    right = np.tile(np.arange(order), order)
    products = group.cayley_table[left, right]
    activations = np.empty((n_neurons, order, order), dtype=np.float64)
    for m in range(n_neurons):
        coef = rng.standard_normal((degree, degree)) + 1j * rng.standard_normal((degree, degree))
        values = np.real(np.sum(coef[None] * matrices[products], axis=(1, 2)))
        activations[m] = values.reshape(order, order)
    activations += noise * rng.standard_normal(activations.shape)
    return activations


def _bilinear_activations(
    group: GroupData, matrices: np.ndarray, n_neurons: int, noise: float, seed: int
) -> np.ndarray:
    """Activations that are a generic bilinear form with INDEPENDENT indices.

    ``A_m[a, b] = Re(vec(rho(a))^T T^m vec(rho(b)))`` with a full random ``T^m``
    -- structure dominated by the off-diagonal ``k != l`` couplings the
    matrix-product form cannot represent. The negative control that isolates the
    shared-index signature.
    """
    rng = np.random.default_rng(seed)
    order, degree, _ = matrices.shape
    left = np.repeat(np.arange(order), order)
    right = np.tile(np.arange(order), order)
    flat = matrices.reshape(order, degree * degree)
    activations = np.empty((n_neurons, order, order), dtype=np.float64)
    for m in range(n_neurons):
        tensor = rng.standard_normal((degree**2, degree**2)) + 1j * rng.standard_normal(
            (degree**2, degree**2)
        )
        values = np.real(np.einsum("ci,ij,cj->c", flat[left], tensor, flat[right]))
        activations[m] = values.reshape(order, order)
    activations += noise * rng.standard_normal(activations.shape)
    return activations


def _function_of_ab_activations(
    group: GroupData, n_neurons: int, noise: float, seed: int
) -> np.ndarray:
    """Activations that are an arbitrary per-product lookup ``g(a b)``.

    A random value per product -- a function of ``a b`` the restricted
    matrix-product form cannot reproduce (it can only reach the projection onto
    one irrep's entry span). The second negative control: the matrix-product
    model may tie a *failing* bilinear model here, which is not GCR.
    """
    rng = np.random.default_rng(seed)
    order = group.order
    left = np.repeat(np.arange(order), order)
    right = np.tile(np.arange(order), order)
    products = group.cayley_table[left, right]
    per_product = rng.standard_normal((n_neurons, order))
    activations = np.empty((n_neurons, order, order), dtype=np.float64)
    for m in range(n_neurons):
        activations[m] = per_product[m][products].reshape(order, order)
    activations += noise * rng.standard_normal(activations.shape)
    return activations


# ---------------------------------------------------------------------------
# UNDEFINED screen (degree-1-dominated group)
# ---------------------------------------------------------------------------


def test_degree_one_group_is_undefined():
    """C8 is abelian: every rho(g) is a scalar, so rho(a) rho(b) is ordinary
    multiplication and the shared-index contraction is vacuous. The screen must
    return the UNDEFINED verdict -- a string, deliberately not a number."""
    group = resolve_group("C8")
    result = screen_gcr_matmul(np.zeros((4, group.order, group.order)), group)
    assert not result.defined
    assert result.fits == ()
    record = result.to_record()
    assert record["status"] == UNDEFINED
    assert "vacuous" in record["reason"]
    assert record["max_irrep_degree"] == 1


def test_fit_rejects_a_degree_one_irrep():
    """The core fit refuses a scalar irrep outright rather than fitting a
    degenerate model -- callers must screen first."""
    group = resolve_group("C8")
    scalar = group.irreps[1].matrices  # degree 1
    with pytest.raises(ValueError, match="cannot distinguish"):
        fit_matmul_gcr(np.zeros((2, group.order, group.order)), scalar, group.cayley_table)


# ---------------------------------------------------------------------------
# Positive control: an exact matrix product is credited
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["D8", "Q8", "S3"])
def test_positive_control_credits_matrix_product(name):
    """Activations that ARE a shared-index matrix product: the matrix-product
    form must capture essentially all the achievable variance and credit GCR."""
    group = resolve_group(name)
    _, matrices = _first_high_degree_block(group)
    activations = _matrix_product_activations(group, matrices, n_neurons=16, noise=0.05, seed=0)
    fit = fit_matmul_gcr(activations, matrices, group.cayley_table, n_splits=4, seed=0)
    assert fit.mp_fve_heldout > 0.95
    assert fit.mp_fraction_of_ab > 0.95
    assert fit.credits_gcr


@pytest.mark.parametrize("name", ["D8", "Q8", "S3"])
def test_positive_control_favours_the_constraint_over_bilinear(name):
    """The decisive part of the positive control: the matrix-product model must
    be favoured over the strictly more general bilinear model. It uses far fewer
    parameters, ties or beats it on held-out cells, and wins the information
    criterion -- the shared-index contraction costs nothing, so it is real."""
    group = resolve_group(name)
    _, matrices = _first_high_degree_block(group)
    activations = _matrix_product_activations(group, matrices, n_neurons=16, noise=0.05, seed=1)
    fit = fit_matmul_gcr(activations, matrices, group.cayley_table, n_splits=4, seed=0)
    assert fit.n_params_mp < fit.n_params_bilinear
    assert fit.mp_minus_bilinear_heldout >= -0.02
    assert fit.mp_favoured
    assert fit.bic_mp < fit.bic_bilinear


# ---------------------------------------------------------------------------
# Negative control 1: generic bilinear structure is NOT credited
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["D8", "Q8", "S3"])
def test_negative_bilinear_is_not_credited(name):
    """Activations that are a generic bilinear form with independent indices:
    the matrix-product form cannot reach the off-diagonal couplings, so the
    unconstrained bilinear model beats it decisively out of sample and GCR is
    refused. This is the test with real discriminating power -- a coset/Fourier
    account could pass a naive matrix-product fit but fails here."""
    group = resolve_group(name)
    _, matrices = _first_high_degree_block(group)
    activations = _bilinear_activations(group, matrices, n_neurons=16, noise=0.05, seed=2)
    fit = fit_matmul_gcr(activations, matrices, group.cayley_table, n_splits=4, seed=0)
    assert not fit.credits_gcr
    assert fit.bilinear_fve_heldout - fit.mp_fve_heldout > 0.3
    assert fit.mp_minus_bilinear_heldout < 0.0
    assert not fit.mp_favoured


# ---------------------------------------------------------------------------
# Negative control 2: a bare function of ab (ties a failing bilinear) is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["D8", "Q8", "S3"])
def test_negative_function_of_ab_is_not_credited(name):
    """A random per-product lookup is a function of ``a b`` that the restricted
    matrix-product form cannot reproduce. The matrix-product model may even tie
    or beat the (also failing) bilinear model, but it captures only a fraction
    of the function-of-``ab`` ceiling, so the coverage requirement refuses it.
    A matrix-product model that merely ties a failing bilinear form is not
    evidence of GCR, and the instrument encodes exactly that."""
    group = resolve_group(name)
    _, matrices = _first_high_degree_block(group)
    activations = _function_of_ab_activations(group, n_neurons=16, noise=0.05, seed=3)
    fit = fit_matmul_gcr(activations, matrices, group.cayley_table, n_splits=4, seed=0)
    assert fit.ab_fve_heldout > 0.9  # the ceiling is reachable...
    # ...but the matrix product falls below the crediting coverage threshold. The
    # slack is larger for tiny groups (S3 has only 6 products, so one irrep's
    # entries plus the constant already span most of the function-of-ab space) --
    # another face of the low-power regime, and exactly why the coverage bar sits
    # at 0.9 rather than lower.
    assert fit.mp_fraction_of_ab < 0.9
    assert not fit.credits_gcr


# ---------------------------------------------------------------------------
# Identifiability: invariance to the representation's equivalence class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["D8", "Q8", "S3"])
def test_fit_is_invariant_to_conjugation_of_rho(name):
    """The network's internal rho is recoverable only up to equivalence, so the
    fit must be identical for rho and any conjugate ``S rho S^{-1}``. Because the
    coefficients are fit fully freely, the fitted subspace is the conjugation-
    invariant real span of the entries, and every reported statistic is
    unchanged. A naive fit against particular entries would fail this."""
    group = resolve_group(name)
    _, matrices = _first_high_degree_block(group)
    activations = _matrix_product_activations(group, matrices, n_neurons=8, noise=0.05, seed=4)
    rng = np.random.default_rng(7)
    degree = matrices.shape[1]
    conjugator = rng.standard_normal((degree, degree)) + 1j * rng.standard_normal((degree, degree))
    conjugated = np.einsum("ij,gjk,kl->gil", conjugator, matrices, np.linalg.inv(conjugator))
    original = fit_matmul_gcr(activations, matrices, group.cayley_table, n_splits=4, seed=0)
    equivalent = fit_matmul_gcr(activations, conjugated, group.cayley_table, n_splits=4, seed=0)
    assert original.mp_fve_heldout == pytest.approx(equivalent.mp_fve_heldout, abs=1e-9)
    assert original.bilinear_fve_heldout == pytest.approx(equivalent.bilinear_fve_heldout, abs=1e-9)
    assert original.n_params_mp == equivalent.n_params_mp
    assert original.n_params_bilinear == equivalent.n_params_bilinear


# ---------------------------------------------------------------------------
# Degree flags: degree 2 is low-power, degree 3 is clean and high-power
# ---------------------------------------------------------------------------


def test_degree_two_is_flagged_low_power():
    group = resolve_group("D8")
    _, matrices = _first_high_degree_block(group)
    activations = _matrix_product_activations(group, matrices, n_neurons=8, noise=0.05, seed=5)
    fit = fit_matmul_gcr(activations, matrices, group.cayley_table, n_splits=4, seed=0)
    assert fit.irrep_degree == 2
    assert fit.low_power


@pytest.mark.skipif(
    not (_REAL_ARTIFACTS / "smallgroup_27_3.npz").is_file(),
    reason="degree-3 SmallGroup(27,3) artifact not present",
)
def test_degree_three_positive_control_is_high_power():
    """SmallGroup(27,3) has a degree-3 irrep: 9 matrix-product vs 81 bilinear
    complex coefficients, the regime where the constraint is decisive. An exact
    matrix product is credited, cleanly and not flagged low-power."""
    group = load_group(27, 3, directory=_REAL_ARTIFACTS)
    _, matrices = _first_high_degree_block(group)
    assert matrices.shape[1] == 3
    activations = _matrix_product_activations(group, matrices, n_neurons=12, noise=0.05, seed=0)
    fit = fit_matmul_gcr(activations, matrices, group.cayley_table, n_splits=5, seed=0)
    assert not fit.low_power
    assert fit.mp_fve_heldout > 0.95
    assert fit.mp_favoured
    assert fit.credits_gcr
    assert fit.n_params_mp < fit.n_params_bilinear


@pytest.mark.skipif(
    not (_REAL_ARTIFACTS / "smallgroup_27_3.npz").is_file(),
    reason="degree-3 SmallGroup(27,3) artifact not present",
)
def test_degree_three_bilinear_is_refused_decisively():
    """At degree 3 the negative control is unambiguous: bilinear structure the
    matrix product cannot reach leaves an enormous held-out gap."""
    group = load_group(27, 3, directory=_REAL_ARTIFACTS)
    _, matrices = _first_high_degree_block(group)
    activations = _bilinear_activations(group, matrices, n_neurons=12, noise=0.05, seed=1)
    fit = fit_matmul_gcr(activations, matrices, group.cayley_table, n_splits=5, seed=0)
    assert not fit.credits_gcr
    assert fit.bilinear_fve_heldout - fit.mp_fve_heldout > 0.5


# ---------------------------------------------------------------------------
# Feature construction sanity: matrix-product features are the product entries
# ---------------------------------------------------------------------------


def test_matrix_product_features_use_the_homomorphism():
    """The matrix-product design columns are the entries of rho(a b), which equal
    the entries of rho(a) @ rho(b) -- the identity the whole account rests on."""
    group = resolve_group("S3")
    _, matrices = _first_high_degree_block(group)
    order, degree, _ = matrices.shape
    left = np.repeat(np.arange(order), order)
    right = np.tile(np.arange(order), order)
    products = group.cayley_table[left, right]
    features = matrix_product_features(matrices, products)
    assert features.shape == (order * order, 2 * degree**2)
    # First cell (a=0, b=0): identity product, real part is vec(rho(e)).
    expected = matrices[products[0]].reshape(-1)
    np.testing.assert_allclose(features[0, : degree**2], expected.real, atol=1e-12)
    # Homomorphism: rho(a b) == rho(a) @ rho(b) for a sampled cell.
    cell = 5
    direct = matrices[left[cell]] @ matrices[right[cell]]
    np.testing.assert_allclose(features[cell, : degree**2], direct.reshape(-1).real, atol=1e-12)


def test_bilinear_features_strictly_contain_the_matrix_product_span():
    """The matrix-product model is nested in the bilinear model, so the bilinear
    column space must have strictly greater rank at degree >= 2."""
    group = resolve_group("D8")
    _, matrices = _first_high_degree_block(group)
    order = matrices.shape[0]
    left = np.repeat(np.arange(order), order)
    right = np.tile(np.arange(order), order)
    products = group.cayley_table[left, right]
    mp = matrix_product_features(matrices, products)
    bilinear = bilinear_features(matrices, left, right)
    rank_mp = np.linalg.matrix_rank(mp)
    rank_both = np.linalg.matrix_rank(np.concatenate([mp, bilinear], axis=1))
    rank_bilinear = np.linalg.matrix_rank(bilinear)
    assert rank_both == rank_bilinear  # mp span is inside the bilinear span
    assert rank_bilinear > rank_mp


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_fit_rejects_non_finite_activations():
    group = resolve_group("D8")
    _, matrices = _first_high_degree_block(group)
    activations = _matrix_product_activations(group, matrices, n_neurons=4, noise=0.0, seed=0)
    activations[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_matmul_gcr(activations, matrices, group.cayley_table)


def test_fit_rejects_a_non_square_grid():
    group = resolve_group("D8")
    _, matrices = _first_high_degree_block(group)
    with pytest.raises(ValueError, match="square"):
        fit_matmul_gcr(np.zeros((4, 8, 6)), matrices, group.cayley_table)


def test_record_schema_of_a_measured_result():
    group = resolve_group("D8")
    _, matrices = _first_high_degree_block(group)
    activations = _matrix_product_activations(group, matrices, n_neurons=8, noise=0.05, seed=0)
    result = screen_gcr_matmul(activations, group, n_splits=4)
    record = result.to_record()
    assert record["instrument"] == "gcr_matmul"
    assert record["status"] == "low_power"  # D8 is degree 2
    assert record["low_power"] is True
    assert isinstance(record["credits_gcr"], bool)
    assert len(record["fits"]) == 1
    fit_record = record["fits"][0]
    for key in (
        "mp_fve_heldout",
        "bilinear_fve_heldout",
        "ab_fve_heldout",
        "mp_fraction_of_ab",
        "mp_minus_bilinear_heldout",
        "n_params_mp",
        "n_params_bilinear",
        "credits_gcr",
    ):
        assert key in fit_record


# ---------------------------------------------------------------------------
# Post-activation extraction: the re-targeted input tensor
# ---------------------------------------------------------------------------


def _small_transformer(order: int = 8, index: int = 3):
    group = resolve_group("D8")
    config = ProjectConfig(
        device="cpu",
        data={"group": {"order": order, "index": index}},
        model={"arch": "transformer", "d_model": 32, "d_mlp": 64, "n_heads": 4},
        logging={"mode": "disabled"},
    )
    set_seed(0, deterministic=False)
    return group, build_model(config, group)


def test_post_activation_differs_from_mlp_pre():
    """The whole fix: the instrument reads the POST-nonlinearity activation, not
    the linear mlp_pre. For a ReLU model the two must genuinely differ (the
    nonlinearity clips the negatives), and the post tensor must equal ReLU of
    the pre tensor cell for cell -- the model's own activation applied to the
    same grid."""
    group, model = _small_transformer()
    pre = neuron_activations(model, group.order)
    post = post_neuron_activations(model, group.order)
    assert post.shape == pre.shape  # [d_mlp, |G|, |G|]
    assert not np.allclose(post, pre)  # the nonlinearity does something
    np.testing.assert_allclose(post, np.maximum(pre, 0.0), atol=1e-6)  # ReLU of mlp_pre


def test_post_activation_reads_the_cache_key_and_the_fallback_agrees():
    """The extraction reads the model's own mlp_post cache key; reconstructing it
    by applying the model's activation function to mlp_pre (the documented
    fallback for a cache that only exposes mlp_pre) gives the identical tensor,
    so the two extraction routes cannot diverge silently."""
    group, model = _small_transformer()
    from group_algorithm_interp.instruments.gcr_matmul import _mlp_post_from_cache

    tokens = torch.stack(
        [
            torch.arange(group.order).repeat_interleave(group.order),
            torch.arange(group.order).repeat(group.order),
            torch.full((group.order**2,), group.order),
        ],
        dim=1,
    )
    model.eval()
    with torch.no_grad():
        cache = model(tokens, return_cache=True)
    from_cache = _mlp_post_from_cache(cache, model)
    reconstructed = model.activation(cache["mlp_pre"])
    torch.testing.assert_close(from_cache, cache["mlp_post"])
    torch.testing.assert_close(from_cache, reconstructed)
    # And with the cache key stripped, the fallback path reproduces it exactly.
    stripped = dict(cache)
    stripped["mlp_post"] = None
    torch.testing.assert_close(_mlp_post_from_cache(stripped, model), from_cache)


def test_mlp_out_target_is_the_wout_projection_of_post():
    """The secondary target mlp_out = W_out @ mlp_post is the MLP's residual
    contribution: d_model units rather than d_mlp neurons, and exactly the
    post activation pushed through the model's own output projection."""
    group, model = _small_transformer()
    post = post_neuron_activations(model, group.order, target="post")
    mlp_out = post_neuron_activations(model, group.order, target="mlp_out")
    assert mlp_out.shape == (model.d_model, group.order, group.order)
    # Reconstruct W_out @ post over the grid and compare.
    flat_post = post.reshape(post.shape[0], -1)  # [d_mlp, cells]
    expected = (model.W_out.detach().numpy().T @ flat_post).reshape(
        model.d_model, group.order, group.order
    )
    np.testing.assert_allclose(mlp_out, expected, atol=1e-6)


def test_post_activation_rejects_a_bad_target():
    group, model = _small_transformer()
    with pytest.raises(ValueError, match="target must be one of"):
        post_neuron_activations(model, group.order, target="mlp_pre")


# ---------------------------------------------------------------------------
# Integration: drive a real (untrained) model through the post-activation path
# ---------------------------------------------------------------------------


def test_integration_through_the_post_activation_extraction_path():
    """The lighter integration check: a real model's post-nonlinearity MLP
    activation is read over the Cayley grid and screened end to end. An
    untrained model has no matrix-product structure, so it must not credit GCR;
    the point here is that the re-targeted extraction/screen wiring runs and
    yields the documented schema."""
    group, model = _small_transformer()
    result = measure_gcr_matmul(model, group, n_splits=4)
    assert result.defined
    assert len(result.fits) == 1
    assert not result.credits_gcr
    assert 0.0 <= result.fits[0].occupancy <= 1.0
