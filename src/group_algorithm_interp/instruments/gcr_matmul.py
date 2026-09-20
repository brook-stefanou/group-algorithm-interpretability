"""The matrix-product test of the Group Composition via Representations account.

Group Composition via Representations (GCR; Chughtai, Chan and Nanda) makes one
load-bearing, *discriminating* claim: the network forms ``rho(a) rho(b)`` as an
intermediate matrix product and reads it out, so the hidden pre-activations
restricted to irrep ``rho``'s isotypic block are explained by

    A_m[a, b] ~ sum_{i, j, k} c^m_{ij} rho(a)_{ik} rho(b)_{kj}
              = sum_{i, j}    c^m_{ij} [rho(a) rho(b)]_{ij}
              = sum_{i, j}    c^m_{ij} rho(a b)_{ij}.

Both arguments enter as full ``degree x degree`` matrices, combined by matrix
multiplication -- a *contraction over the shared index* ``k``. That shared index
is the whole content of the claim. A coset, Fourier or tensor account produces
activations that are merely *some* function of the product ``a b`` and does not
require the off-diagonal ``rho(a)_{ik}`` cross-terms; a generic bilinear account
allows the two arguments' irrep entries to interact through *independent* index
pairs and so does not require the shared-index contraction either.

This instrument therefore fits three nested-in-capability linear models to the
activation grid and reports the *gaps* between them, which is the only honest
signature -- a high matrix-product fit on its own is meaningless:

* **matrix-product** -- ``A_m[a, b]`` linear in the entries of ``rho(a b)``
  (the shared-index-contracted form, ``d**2`` complex coefficients per neuron);
* **generic bilinear** -- ``A_m[a, b]`` linear in the ``d**4`` products
  ``rho(a)_{ik} rho(b)_{lj}`` with *independent* index pairs ``(i, k)`` and
  ``(l, j)`` -- a strict superset of the matrix-product model that a non-GCR
  bilinear account also satisfies;
* **function of ab** -- the saturated per-product mean, an upper bound any
  account whose output depends only on ``a b`` can reach.

The matrix-product model is *nested* inside the generic bilinear model: setting
the independent-index coefficients to ``c_{ij} delta_{kl}`` recovers it, so the
bilinear model can never fit worse in sample. The discriminating question is
whether, *out of sample and once parameter count is accounted for*, the extra
freedom of the bilinear model buys anything. If it does not -- the constrained
matrix-product form ties or beats the bilinear form on held-out cells and is
favoured by an information criterion -- the shared-index contraction is really
present. If the bilinear form pulls decisively ahead out of sample, the
activations carry bilinear structure that is *not* a matrix product, and GCR is
not credited. A matrix-product model that merely ties a *failing* bilinear model
(both far below the function-of-ab ceiling) has demonstrated nothing: the
coverage requirement ``mp_fraction_of_ab`` guards that case.

Identifiability / basis-invariance
----------------------------------
Two ambiguities would sink a naive fit, and both are dissolved by fitting the
coefficients *fully freely*:

1. The network's internal ``rho`` is recoverable only up to equivalence: the
   artifact's ``rho`` and any conjugate ``S rho(.) S^{-1}`` describe the same
   representation. Under conjugation each entry of ``S rho(a b) S^{-1}`` is a
   fixed complex-linear combination of the entries of ``rho(a b)``, so the real
   linear span of ``{Re rho(a b)_{ij}, Im rho(a b)_{ij}}`` is *unchanged*. We
   fit against that span (all coefficients free), never against particular
   entries, so the reported fit quality is identical for any equivalent ``rho``.
   The same holds for the generic-bilinear span. This is verified numerically in
   the test suite by conjugating ``rho`` with a random invertible ``S``.
2. The readout basis inside the block is arbitrary: a neuron reads some real
   linear functional of the block coordinates. Because ``c^m`` is free, any real
   change of internal basis is absorbed into the fitted coefficients and leaves
   the fit invariant.

We therefore never attempt to *recover* ``rho`` or a canonical basis; we only
ask whether the activations lie in the (basis-invariant) matrix-product span,
and whether the unconstrained bilinear span explains them any better out of
sample. The feature bases are functions of the group's representation matrices
alone -- never of the activations -- so building them on the full grid and then
refitting coefficients per cross-validation fold introduces no leakage.

Degree screen
-------------
A group with no irrep of degree ``>= 2`` cannot distinguish a matrix product
from a scalar bilinear form (every ``rho(g)`` is a scalar, ``rho(a) rho(b)`` is
ordinary multiplication and the shared index is vacuous), so the instrument
returns ``UNDEFINED`` -- a designed outcome, deliberately a string and never a
number. Degree exactly 2 has limited room between the ``d**2 = 4`` and
``d**4 = 16`` coefficient counts, so its results are flagged ``low_power``
rather than clean; degree ``>= 3`` is where the separation is decisive.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..groups.group import FiniteGroup
from ..model import GroupModel
from . import fit_backend
from .device import run_with_device_fallback
from .occupancy import (
    cayley_grid_tokens,
    isotypic_energies,
    population_occupancy,
    restrict_to_nontrivial,
    trivial_block_index,
)

UNDEFINED = "UNDEFINED"

_RANK_TOL = 1e-9
_FVE_FLOOR = 1e-9


# ---------------------------------------------------------------------------
# Feature construction (functions of the group only -- never of activations)
# ---------------------------------------------------------------------------


def _real_features(complex_features: np.ndarray) -> np.ndarray:
    """Stack the real and imaginary parts of complex feature columns.

    ``complex_features`` has shape ``[n_cells, p]``; the result has shape
    ``[n_cells, 2 * p]``. Fitting real coefficients against ``[Re, Im]`` spans
    exactly ``{Re(<c, x>) : c complex}``, the real functions a complex-linear
    read-out of the (possibly complex) irrep entries can produce. Redundant
    columns (e.g. the all-zero imaginary part of a real irrep) are harmless:
    the rank is measured explicitly and least squares uses the minimum-norm
    solution.
    """
    return np.concatenate([complex_features.real, complex_features.imag], axis=1)


def matrix_product_features(irrep_matrices: np.ndarray, products: np.ndarray) -> np.ndarray:
    """Real design columns for the matrix-product model over the Cayley grid.

    The GCR form ``sum_{ij} c_{ij} [rho(a) rho(b)]_{ij}`` equals
    ``sum_{ij} c_{ij} rho(a b)_{ij}`` because ``rho`` is a homomorphism, so the
    features are the flattened entries of ``rho`` evaluated at the product
    ``a b`` of each grid cell. ``irrep_matrices`` is ``[order, d, d]`` complex;
    ``products`` is the flat ``[n_cells]`` array of product indices ``a b``.
    Result: ``[n_cells, 2 * d**2]`` real (before rank reduction).
    """
    order, degree, _ = irrep_matrices.shape
    flat = irrep_matrices.reshape(order, degree * degree)
    return _real_features(flat[products])


def bilinear_features(
    irrep_matrices: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Real design columns for the generic (independent-index) bilinear model.

    Every product ``rho(a)_{ik} rho(b)_{lj}`` with the four indices ranging
    *independently* -- so ``k`` and ``l`` are not tied. This is the alternative a
    non-GCR bilinear account satisfies: a bilinear form in the two arguments'
    irrep coordinates without the shared-index contraction. It has ``d**4``
    complex columns and contains the matrix-product model as the strict subspace
    ``k == l``. ``left``/``right`` are the flat ``[n_cells]`` arrays of ``a`` and
    ``b`` indices. Result: ``[n_cells, 2 * d**4]`` real (before rank reduction).
    """
    order, degree, _ = irrep_matrices.shape
    flat = irrep_matrices.reshape(order, degree * degree)  # [order, d**2]
    left_entries = flat[left]  # [n_cells, d**2]
    right_entries = flat[right]  # [n_cells, d**2]
    outer = left_entries[:, :, None] * right_entries[:, None, :]  # [n_cells, d**2, d**2]
    return _real_features(outer.reshape(outer.shape[0], -1))


# ---------------------------------------------------------------------------
# Cross-validated linear-model fitting
# ---------------------------------------------------------------------------


def _kfold_indices(n_cells: int, n_splits: int, seed: int) -> list[np.ndarray]:
    """A deterministic, near-equal partition of ``range(n_cells)`` into folds."""
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n_cells < n_splits:
        raise ValueError(f"n_cells ({n_cells}) must be >= n_splits ({n_splits})")
    order = np.random.default_rng(seed).permutation(n_cells)
    return [np.sort(fold) for fold in np.array_split(order, n_splits)]


def _design(features: np.ndarray) -> np.ndarray:
    """Prepend an intercept column so every model shares a free constant term.

    The constant is the trivial/DC component (dominant at read position, per the
    occupancy account) and belongs to every model on an equal footing, so it is
    never counted as matrix-product or bilinear structure.
    """
    return np.concatenate([np.ones((features.shape[0], 1)), features], axis=1)


def _cv_sse(
    design: np.ndarray, targets: np.ndarray, folds: list[np.ndarray], ridge: float
) -> float:
    """Pooled held-out sum of squared errors of a linear model, over all folds.

    ``design`` is ``[n_cells, p]`` (functions of the group only), ``targets`` is
    ``[n_cells, n_neurons]``. For each fold the coefficients are refit on the
    other folds' cells and used to predict the held-out cells, so no held-out
    cell informs its own prediction. Coefficients are the only quantity learned
    from the activations; the design is fixed.
    """
    n_cells = design.shape[0]
    all_idx = np.arange(n_cells)
    residual_ss = 0.0
    for fold in folds:
        train = np.setdiff1d(all_idx, fold, assume_unique=True)
        x_train = design[train]
        if ridge > 0.0:
            gram = x_train.T @ x_train + ridge * np.eye(x_train.shape[1])
            coef = np.linalg.solve(gram, x_train.T @ targets[train])
        else:
            coef, *_ = np.linalg.lstsq(x_train, targets[train], rcond=None)
        prediction = design[fold] @ coef
        residual_ss += float(np.square(targets[fold] - prediction).sum())
    return residual_ss


def _heldout_sse(
    design: np.ndarray,
    targets: np.ndarray,
    folds: list[np.ndarray],
    ridge: float,
    fit_device: torch.device | None,
) -> float:
    """Held-out SSE of a linear model, on the numpy-float64-CPU reference path
    or (opt-in) the ``mps``/``cuda`` normal-equations backend.

    ``fit_device`` selects the fitter: ``None`` or a CPU device uses the
    reference :func:`_cv_sse` (numpy ``lstsq`` in float64, unchanged); an
    ``mps`` or ``cuda`` device routes the dominant cross-products through
    :func:`fit_backend.heldout_residual_ss` (float32 normal equations on the
    GPU, centred and ridge-stabilised -- see :mod:`.fit_backend`). The backend
    supplies its own documented Tikhonov stabiliser, so the reference ``ridge``
    (an absolute penalty, ``0`` in the pre-registered runs) applies only to the
    numpy path; the backend is validated against that path to ~1e-3 on the
    reported gaps."""
    if fit_device is not None and fit_device.type in ("mps", "cuda"):
        return fit_backend.heldout_residual_ss(design, targets, folds, device=fit_device)
    return _cv_sse(design, targets, folds, ridge)


def _cv_sse_function_of_ab(
    targets: np.ndarray, products: np.ndarray, folds: list[np.ndarray]
) -> float:
    """Pooled held-out SSE of the saturated ``A_m[a, b] = g(a b)`` model.

    The best predictor of a held-out cell is the training-fold mean of the
    activation over cells with the same product ``a b``; a product absent from
    the training fold falls back to the overall training mean. This is the
    ceiling any function-of-``ab`` account (coset, Fourier, tensor) can reach.
    """
    n_cells = targets.shape[0]
    all_idx = np.arange(n_cells)
    residual_ss = 0.0
    for fold in folds:
        train = np.setdiff1d(all_idx, fold, assume_unique=True)
        global_mean = targets[train].mean(axis=0)
        train_products = products[train]
        prediction = np.empty((fold.size, targets.shape[1]), dtype=np.float64)
        for row, cell in enumerate(fold):
            same = train[train_products == products[cell]]
            prediction[row] = targets[same].mean(axis=0) if same.size else global_mean
        residual_ss += float(np.square(targets[fold] - prediction).sum())
    return residual_ss


def _total_ss(targets: np.ndarray) -> float:
    """Total variance of the targets around each neuron's own mean."""
    return float(np.square(targets - targets.mean(axis=0, keepdims=True)).sum())


def _fve(residual_ss: float, total_ss: float) -> float:
    """Fraction of variance explained; may be negative if worse than the mean."""
    if total_ss <= _FVE_FLOOR:
        return 0.0
    return 1.0 - residual_ss / total_ss


def _bic(residual_ss: float, n_obs: int, n_params: int) -> float:
    """Gaussian BIC ``n ln(RSS / n) + k ln(n)`` for in-sample model comparison.

    ``n_params`` is the total free-coefficient count across all neurons (each
    neuron carries its own coefficients over the shared design), which is what
    penalises the ``d**4``-column bilinear model against the ``d**2``-column
    matrix-product model.
    """
    safe = max(residual_ss, _FVE_FLOOR)
    return n_obs * math.log(safe / n_obs) + n_params * math.log(n_obs)


# ---------------------------------------------------------------------------
# Per-irrep result and the core fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IrrepMatmulFit:
    """The matrix-product-vs-alternatives comparison for one irrep ``rho``.

    All ``*_fve_heldout`` are cross-validated fractions of variance explained,
    pooled over neurons. The load-bearing statistics are the *gaps*:

    ``mp_fraction_of_ab``
        ``mp_fve_heldout / ab_fve_heldout`` -- how much of the variance any
        function of ``a b`` can explain the constrained matrix-product form
        actually captures. Near 1 means the shared-index form leaves nothing on
        the table; near 0 means the activations' product structure is not this
        irrep's matrix product (or not a matrix product at all).

    ``mp_minus_bilinear_heldout``
        ``mp_fve_heldout - bilinear_fve_heldout``. The bilinear model can only
        beat the matrix-product model out of sample if genuine independent-index
        (non-matrix-product) structure is present; if the truth is a matrix
        product, the bilinear model's extra parameters overfit and it does no
        better (typically slightly worse). A clearly negative value falsifies
        the shared-index contraction.

    ``mp_favoured``
        The information criterion prefers the constrained model
        (``bic_mp < bic_bilinear``): a tie in raw held-out fit, achieved with a
        fraction of the parameters, still favours matrix-product here.

    ``credits_gcr``
        The conjunction the instrument treats as crediting GCR for this irrep:
        the matrix-product form captures essentially all the function-of-``ab``
        variance (``mp_fraction_of_ab >= min_coverage``) at a non-trivial
        absolute level (``mp_fve_heldout >= min_abs_fve``), is not beaten by the
        bilinear form out of sample (``mp_minus_bilinear_heldout >= -tie_tol``),
        and is favoured by the information criterion.
    """

    block_index: int
    irrep_index: int
    irrep_degree: int
    occupancy: float
    n_params_mp: int
    n_params_bilinear: int
    n_neurons: int
    n_cells: int
    n_cells_available: int
    row_sampling_applied: bool
    max_rows: int
    sample_seed: int
    mp_fve_heldout: float
    bilinear_fve_heldout: float
    ab_fve_heldout: float
    mp_fraction_of_ab: float
    mp_minus_bilinear_heldout: float
    bic_mp: float
    bic_bilinear: float
    mp_favoured: bool
    credits_gcr: bool
    low_power: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "irrep_index": self.irrep_index,
            "irrep_degree": self.irrep_degree,
            "occupancy": self.occupancy,
            "n_params_mp": self.n_params_mp,
            "n_params_bilinear": self.n_params_bilinear,
            "n_neurons": self.n_neurons,
            "n_cells": self.n_cells,
            "n_cells_available": self.n_cells_available,
            "row_sampling_applied": self.row_sampling_applied,
            "max_rows": self.max_rows,
            "sample_seed": self.sample_seed,
            "mp_fve_heldout": self.mp_fve_heldout,
            "bilinear_fve_heldout": self.bilinear_fve_heldout,
            "ab_fve_heldout": self.ab_fve_heldout,
            "mp_fraction_of_ab": self.mp_fraction_of_ab,
            "mp_minus_bilinear_heldout": self.mp_minus_bilinear_heldout,
            "bic_mp": self.bic_mp,
            "bic_bilinear": self.bic_bilinear,
            "mp_favoured": self.mp_favoured,
            "credits_gcr": self.credits_gcr,
            "low_power": self.low_power,
        }


def fit_matmul_gcr(
    activations: np.ndarray,
    irrep_matrices: np.ndarray,
    cayley_table: np.ndarray,
    *,
    block_index: int = -1,
    irrep_index: int = -1,
    occupancy: float = float("nan"),
    n_splits: int = 5,
    seed: int = 0,
    ridge: float = 0.0,
    min_coverage: float = 0.9,
    min_abs_fve: float = 0.5,
    tie_tol: float = 0.01,
    fit_device: torch.device | None = None,
    max_rows: int = 0,
    sample_seed: int = 0,
) -> IrrepMatmulFit:
    """Fit the three models to one irrep's activation grid and compare them.

    ``activations`` is ``[n_neurons, order, order]`` (as
    :func:`occupancy.neuron_activations` returns) or a single ``[order, order]``
    grid; ``irrep_matrices`` is the ``[order, d, d]`` concrete ``rho`` of the
    irrep under test (degree ``d >= 2``); ``cayley_table`` supplies the products
    ``a b``. The comparison is cross-validated over the ``order**2`` grid cells.

    The verdict thresholds (``min_coverage``, ``min_abs_fve``, ``tie_tol``) are
    exposed so a caller can tighten them; the defaults are the documented
    crediting rule. Degree exactly 2 is flagged ``low_power`` (limited room
    between the ``d**2`` and ``d**4`` coefficient counts), never suppressed.

    ``fit_device`` selects the held-out-FVE fitter: ``None``/CPU (the default)
    uses the numpy-float64-CPU reference; an ``mps`` device routes the
    dominant cross-validated least-squares through the float32
    normal-equations backend (:mod:`.fit_backend`) so a large-group cell runs
    on the Apple GPU. The in-sample BIC ``lstsq`` and the rank are always the
    numpy float64 reference; only the held-out CV fit moves to the device.

    ``max_rows`` (0 = no cap, the default) bounds the number of ``(a, b)`` grid
    cells the fit sees: over that many cells, a fixed-``sample_seed`` uniform
    subsample without replacement of ``max_rows`` cells is drawn *before* the
    designs are built (so the ``d**4``-column bilinear design is never
    materialised for the full ``order**2`` grid), and the cross-validation
    folds, the function-of-``ab`` ceiling and the total variance are all formed
    on the sampled cells -- unbiased estimates of the same quantities on fewer
    rows. Sub-cap (``order**2 <= max_rows``) it is byte-identical to the
    uncapped fit and draws no random numbers.
    """
    grid = np.asarray(activations, dtype=np.float64)
    if grid.ndim == 2:
        grid = grid[None, :, :]
    if grid.ndim != 3:
        raise ValueError("activations must have shape [n_neurons, order, order] or [order, order]")
    order = grid.shape[1]
    if grid.shape[2] != order:
        raise ValueError(f"activation grid must be square, got {grid.shape[1:]}")
    if irrep_matrices.ndim != 3 or irrep_matrices.shape[0] != order:
        raise ValueError(f"irrep_matrices must have shape [{order}, d, d]")
    degree = irrep_matrices.shape[1]
    if degree < 2:
        raise ValueError(
            f"irrep degree {degree} < 2 cannot distinguish a matrix product from a "
            "scalar bilinear form; screen with screen_gcr_matmul first"
        )
    if not np.isfinite(grid).all():
        raise ValueError(
            "activations contain non-finite values; the matrix-product fit is undefined"
        )
    if cayley_table.shape != (order, order):
        raise ValueError(f"cayley_table must have shape [{order}, {order}]")

    n_neurons = grid.shape[0]
    n_cells = order * order
    n_cells_available = n_cells
    left = np.repeat(np.arange(order), order)
    right = np.tile(np.arange(order), order)
    products = np.asarray(cayley_table, dtype=np.int64)[left, right]
    targets = grid.reshape(n_neurons, n_cells).T  # [n_cells, n_neurons]

    # Optional seeded row subsampling: drawn before the designs are built so the
    # dominant (d**4-column bilinear) design is never materialised for the whole
    # order**2 grid. Folds/ceiling/variance below are all formed on the sample.
    row_sampling_applied = bool(max_rows and max_rows > 0 and n_cells > max_rows)
    if row_sampling_applied:
        sample = np.sort(
            np.random.default_rng(sample_seed).choice(n_cells, size=max_rows, replace=False)
        )
        left = left[sample]
        right = right[sample]
        products = products[sample]
        targets = targets[sample]
        n_cells = int(max_rows)

    folds = _kfold_indices(n_cells, n_splits, seed)
    total_ss = _total_ss(targets)

    # Each design (and its lstsq/rank temporaries) is built, used to
    # completion (held-out CV fit and in-sample RSS) and freed before the
    # next one is built, rather than holding both the ``d**2``-column
    # matrix-product design and the much larger ``d**4``-column bilinear
    # design simultaneously for the whole function body. This matters most
    # for the bilinear design on high-degree groups, where ``2 * d**4``
    # columns over ``order**2`` cells dominates this instrument's memory.
    mp_design = _design(matrix_product_features(irrep_matrices, products))
    rank_mp = int(np.linalg.matrix_rank(mp_design, tol=_RANK_TOL))
    mp_fve = _fve(_heldout_sse(mp_design, targets, folds, ridge, fit_device), total_ss)
    mp_coef, *_ = np.linalg.lstsq(mp_design, targets, rcond=None)
    mp_rss = float(np.square(targets - mp_design @ mp_coef).sum())
    del mp_design, mp_coef
    gc.collect()

    bilinear_design = _design(bilinear_features(irrep_matrices, left, right))
    rank_bilinear = int(np.linalg.matrix_rank(bilinear_design, tol=_RANK_TOL))
    bilinear_fve = _fve(_heldout_sse(bilinear_design, targets, folds, ridge, fit_device), total_ss)
    bilinear_coef, *_ = np.linalg.lstsq(bilinear_design, targets, rcond=None)
    bilinear_rss = float(np.square(targets - bilinear_design @ bilinear_coef).sum())
    del bilinear_design, bilinear_coef
    gc.collect()

    ab_fve = _fve(_cv_sse_function_of_ab(targets, products, folds), total_ss)

    fraction = mp_fve / ab_fve if ab_fve > _FVE_FLOOR else 0.0
    gap = mp_fve - bilinear_fve

    n_obs = n_cells * n_neurons
    bic_mp = _bic(mp_rss, n_obs, rank_mp * n_neurons)
    bic_bilinear = _bic(bilinear_rss, n_obs, rank_bilinear * n_neurons)
    mp_favoured = bic_mp < bic_bilinear

    credits_gcr = bool(
        ab_fve > _FVE_FLOOR
        and fraction >= min_coverage
        and mp_fve >= min_abs_fve
        and gap >= -tie_tol
        and mp_favoured
    )

    return IrrepMatmulFit(
        block_index=block_index,
        irrep_index=irrep_index,
        irrep_degree=degree,
        occupancy=occupancy,
        n_params_mp=rank_mp,
        n_params_bilinear=rank_bilinear,
        n_neurons=n_neurons,
        n_cells=n_cells,
        n_cells_available=n_cells_available,
        row_sampling_applied=row_sampling_applied,
        max_rows=int(max_rows),
        sample_seed=int(sample_seed),
        mp_fve_heldout=mp_fve,
        bilinear_fve_heldout=bilinear_fve,
        ab_fve_heldout=ab_fve,
        mp_fraction_of_ab=fraction,
        mp_minus_bilinear_heldout=gap,
        bic_mp=bic_mp,
        bic_bilinear=bic_bilinear,
        mp_favoured=mp_favoured,
        credits_gcr=credits_gcr,
        low_power=(degree == 2),
    )


# ---------------------------------------------------------------------------
# Group-level screen and model-level driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatmulGcrResult:
    """The GCR matrix-product screen for one model on one group.

    ``defined`` is False -- record status ``UNDEFINED`` -- when the group has no
    irrep of degree ``>= 2`` (a scalar representation makes the shared-index
    contraction vacuous). Otherwise ``fits`` holds one :class:`IrrepMatmulFit`
    per degree-``>= 2`` isotypic block, ordered as the group's blocks are;
    ``low_power`` is True when the largest such degree is 2.
    """

    defined: bool
    max_irrep_degree: int
    low_power: bool
    fits: tuple[IrrepMatmulFit, ...]
    reason: str | None = None

    @property
    def credits_gcr(self) -> bool:
        """True when any occupied degree-``>= 2`` irrep credits the matrix product."""
        return any(fit.credits_gcr for fit in self.fits)

    def to_record(self) -> dict[str, Any]:
        if not self.defined:
            return {
                "instrument": "gcr_matmul",
                "status": UNDEFINED,
                "reason": self.reason,
                "max_irrep_degree": self.max_irrep_degree,
            }
        return {
            "instrument": "gcr_matmul",
            "status": "low_power" if self.low_power else "measured",
            "max_irrep_degree": self.max_irrep_degree,
            "low_power": self.low_power,
            "credits_gcr": self.credits_gcr,
            "fits": [fit.to_record() for fit in self.fits],
        }


def _block_occupancies(activations: np.ndarray, group: FiniteGroup) -> np.ndarray:
    """Nontrivial-renormalised isotypic occupancy per block (context only).

    Reuses the occupancy instrument so the ``occupancy`` reported beside each fit
    is the same energy share the spectral core measures, with the trivial/DC
    block removed (its share is an architectural offset, not group structure).
    """
    energies = isotypic_energies(activations, group, argument="left")
    occ = population_occupancy(energies)
    trivial = trivial_block_index(group)
    full = np.zeros(len(group.isotypic_blocks))
    nontrivial = restrict_to_nontrivial(occ, trivial)
    full[[j for j in range(len(group.isotypic_blocks)) if j != trivial]] = nontrivial
    return full


def screen_gcr_matmul(
    activations: np.ndarray,
    group: FiniteGroup,
    *,
    n_splits: int = 5,
    seed: int = 0,
    ridge: float = 0.0,
    min_occupancy: float = 0.0,
    min_coverage: float = 0.9,
    min_abs_fve: float = 0.5,
    tie_tol: float = 0.01,
    fit_device: torch.device | None = None,
    max_rows: int = 0,
    sample_seed: int = 0,
) -> MatmulGcrResult:
    """Run the matrix-product screen over a group's degree-``>= 2`` irreps.

    ``activations`` is ``[n_neurons, order, order]`` from
    :func:`occupancy.neuron_activations`. Returns ``UNDEFINED`` when the group
    is degree-1-dominated. Each degree-``>= 2`` block whose nontrivial occupancy
    is at least ``min_occupancy`` is fitted (``min_occupancy=0`` fits them all
    and lets the caller filter on the reported ``occupancy``); the concrete
    ``rho`` used is the block's first constituent irrep, which spans the same
    real feature space as its complex conjugate.
    """
    degrees = [block.irrep_degree for block in group.isotypic_blocks]
    max_degree = max(degrees)
    if max_degree < 2:
        return MatmulGcrResult(
            defined=False,
            max_irrep_degree=max_degree,
            low_power=False,
            fits=(),
            reason=(
                "every irrep has degree 1: rho(g) is a scalar, rho(a) rho(b) is "
                "ordinary multiplication and the shared-index contraction is "
                "vacuous, so the matrix-product signature is UNDEFINED"
            ),
        )

    occupancies = _block_occupancies(activations, group)
    fits: list[IrrepMatmulFit] = []
    for block_index, block in enumerate(group.isotypic_blocks):
        if block.irrep_degree < 2 or occupancies[block_index] < min_occupancy:
            continue
        irrep_index = block.irrep_indices[0]
        fits.append(
            fit_matmul_gcr(
                activations,
                group.irreps[irrep_index].matrices,
                group.cayley_table,
                block_index=block_index,
                irrep_index=irrep_index,
                occupancy=float(occupancies[block_index]),
                n_splits=n_splits,
                seed=seed,
                ridge=ridge,
                min_coverage=min_coverage,
                min_abs_fve=min_abs_fve,
                tie_tol=tie_tol,
                fit_device=fit_device,
                max_rows=max_rows,
                sample_seed=sample_seed,
            )
        )
    return MatmulGcrResult(
        defined=True,
        max_irrep_degree=max_degree,
        low_power=(max_degree == 2),
        fits=tuple(fits),
    )


_POST_TARGETS = ("post", "mlp_out")


def _mlp_post_from_cache(cache: Any, model: GroupModel) -> torch.Tensor:
    """The post-nonlinearity MLP activation ``mlp_post`` for one forward pass.

    The product structure ``rho(a) rho(b) = rho(a b)`` is realised by the MLP
    *nonlinearity*, so ``mlp_pre`` (linear in the embeddings, essentially an
    additive ``f(a) + g(b)``) carries no matrix-product structure; the signal
    lives in the activation-function *output*. This reads that output straight
    from the model's cache when it exposes an ``mlp_post`` key, and otherwise
    reconstructs it by applying the model's own activation function (ReLU, GeLU,
    ... -- read off the model, never assumed) to ``mlp_pre``. A model with no
    MLP at all (``mlp_pre`` and ``mlp_post`` both absent/None) has no neurons to
    read, so this raises rather than fabricate a tensor.
    """
    post = cache.get("mlp_post")
    if post is not None:
        return post
    pre = cache.get("mlp_pre")
    if pre is None:
        raise ValueError(
            "model has no MLP (use_mlp=false); GCR matrix-product activations are undefined"
        )
    activation = getattr(model, "activation", None)
    if activation is None:
        raise ValueError(
            "model exposes neither a post-activation ('mlp_post') cache key nor an "
            "'activation' function to derive it from 'mlp_pre'"
        )
    return activation(pre)


def post_neuron_activations(
    model: GroupModel,
    order: int,
    *,
    batch_size: int = 8192,
    device: torch.device = torch.device("cpu"),
    target: str = "post",
) -> np.ndarray:
    """The per-unit function on ``G x G`` read *after* the MLP nonlinearity.

    Shape ``[n_units, |G|, |G|]`` with ``A[u, a, b]`` the unit's value at the
    read position (-1) over the full Cayley grid -- the same enumeration
    :func:`occupancy.neuron_activations` uses, so row ``(a, b)`` still aligns
    with ``cayley_table[a, b]``. Unlike that function (which reads the *linear*
    ``mlp_pre``), this reads the post-nonlinearity activation, where the
    ``rho(a) rho(b) = rho(a b)`` matrix product actually lives:

    * ``target="post"`` (default) -- the post-activation neuron activations
      themselves (``d_mlp`` units), the principled Chughtai/Chan/Nanda GCR
      target: ``rho(a b)`` is extracted from the neuron activations;
    * ``target="mlp_out"`` -- the MLP's residual-stream contribution
      ``W_out @ mlp_post`` (``d_model`` units), a cleaner but derived readout of
      the same post-nonlinearity structure. Only available for a model exposing
      ``W_out`` (the transformer); an FC baseline has no such projection.

    ``device`` runs the forward pass there in the model's native dtype
    (float32); the activations returned are moved back to CPU and cast to
    float64 before returning, so the downstream fit stays exactly CPU float64
    regardless of ``device``. The model is moved to ``device`` for the duration
    and always restored to CPU on return (including on an exception); an MPS op
    gap falls back to CPU automatically (:func:`.device.run_with_device_fallback`),
    logged as a ``RuntimeWarning`` since the return is a bare array. Contract:
    this calls ``model.eval()`` and does not restore training mode, matching
    :func:`occupancy.neuron_activations`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if target not in _POST_TARGETS:
        raise ValueError(f"target must be one of {_POST_TARGETS}, got {target!r}")
    tokens = cayley_grid_tokens(order)
    model.eval()

    def _compute(dev: torch.device) -> torch.Tensor:
        moved = dev.type != "cpu"
        if moved:
            model.to(dev)
        try:
            chunks: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, tokens.shape[0], batch_size):
                    batch = tokens[start : start + batch_size].to(dev)
                    cache = model(batch, return_cache=True)
                    units = _mlp_post_from_cache(cache, model)  # [batch, pos, d_mlp]
                    if target == "mlp_out":
                        w_out = getattr(model, "W_out", None)
                        if w_out is None:
                            raise ValueError(
                                "target='mlp_out' requires a model with a 'W_out' MLP output "
                                "projection (the transformer); this model has none"
                            )
                        units = torch.einsum("b p l, l m -> b p m", units, w_out)
                    chunks.append(units[:, -1, :].detach().cpu().to(torch.float64))
            return torch.cat(chunks, dim=0)  # [|G|^2, n_units]
        finally:
            if moved:
                model.to(torch.device("cpu"))

    flat, _note = run_with_device_fallback(_compute, device)
    n_units = flat.shape[1]
    return flat.numpy().reshape(order, order, n_units).transpose(2, 0, 1)


def measure_gcr_matmul(
    model: GroupModel,
    group: FiniteGroup,
    *,
    batch_size: int = 8192,
    device: torch.device = torch.device("cpu"),
    fit_device: torch.device | None = None,
    target: str = "post",
    **screen_kwargs: Any,
) -> MatmulGcrResult:
    """Extract the model's post-nonlinearity activations and run the screen.

    The integration entry point: reads the *post*-nonlinearity MLP activation
    (:func:`post_neuron_activations`, ``target="post"`` by default) over the
    full Cayley grid and screens every degree-``>= 2`` irrep with
    :func:`screen_gcr_matmul`. The post activation, not ``mlp_pre``, is where the
    ``rho(a) rho(b) = rho(a b)`` matrix product lives: ``mlp_pre`` is a linear
    (essentially additive ``f(a) + g(b)``) function of the embeddings, so the
    product structure is created by the nonlinearity and appears only in its
    output and downstream. ``device`` runs the extraction forward pass there
    (native float32); the activations returned are CPU float64 regardless.
    ``fit_device`` is a *separate*, opt-in switch for the held-out-FVE fitter,
    decoupled from the forward-pass ``device`` so the fit does not silently
    change precision when the extraction runs on the GPU: it defaults to
    ``None`` (the numpy-float64-CPU reference), and only an explicit ``mps``
    ``fit_device`` moves the cross-validated least-squares onto the GPU in
    float32 (:mod:`.fit_backend`). ``target`` selects the post-activation
    view (``"post"`` neuron activations, the default and principled target, or
    ``"mlp_out"`` = ``W_out @ mlp_post``). Remaining keyword arguments pass
    straight through to :func:`screen_gcr_matmul`.
    """
    activations = post_neuron_activations(
        model, group.order, batch_size=batch_size, device=device, target=target
    )
    return screen_gcr_matmul(activations, group, fit_device=fit_device, **screen_kwargs)


__all__ = [
    "UNDEFINED",
    "IrrepMatmulFit",
    "MatmulGcrResult",
    "bilinear_features",
    "fit_matmul_gcr",
    "matrix_product_features",
    "measure_gcr_matmul",
    "post_neuron_activations",
    "screen_gcr_matmul",
]
