"""The torch held-out-FVE fitting backend (instruments/fit_backend.py).

The backend's whole warrant is that it reproduces the numpy-float64-CPU
reference fit -- the pooled held-out residual sum of squares and hence the FVE
that ``gcr_matmul._cv_sse`` / ``probes._held_out_fve`` compute -- via float32
normal equations, so the load-bearing *gaps* between two nearly-nested designs
survive the change of fitter. These tests pin that agreement on

* a well-conditioned design (float64 and float32, CPU and -- when present --
  ``mps``): the normal-equations formulation, centering and small ridge must
  match numpy ``lstsq`` to round-off;
* a deliberately near-collinear design (the ill-conditioned regime the
  matrix-product-vs-bilinear comparison actually lives in): float64 to
  round-off, float32 to a tolerance well below the tie tolerances the
  instruments use;
* the exact-fit and total-variance-guard edge cases.

The ``mps`` device path is exercised when the machine has one and skipped
otherwise (CI has no ``mps``), so the CPU-float32 case carries the
device-independent numerical warrant.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from group_algorithm_interp.instruments import fit_backend as fb
from group_algorithm_interp.instruments.gcr_matmul import (
    _cv_sse,
    _design,
    _fve,
    _kfold_indices,
    _total_ss,
)

_HAS_MPS = torch.backends.mps.is_available()
_DEVICES: list[torch.device] = [torch.device("cpu")]
if _HAS_MPS:
    _DEVICES.append(torch.device("mps"))


def _well_conditioned(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p, k = 2048, 30, 12
    design = _design(rng.standard_normal((n, p)))
    coef = rng.standard_normal((design.shape[1], k))
    targets = design @ coef + 0.1 * rng.standard_normal((n, k))
    folds = _kfold_indices(n, 5, 0)
    return design, targets, folds


def _near_collinear(seed: int = 1):
    """A design whose columns are almost linearly dependent -- the regime where
    float32 normal equations are least accurate. A block of columns is a tiny
    perturbation of another block, so ``XtX`` is severely ill-conditioned, as
    the matrix-product design (a strict subspace of the bilinear one) is."""
    rng = np.random.default_rng(seed)
    n, p, k = 2048, 20, 8
    base = rng.standard_normal((n, p))
    near_dup = base + 1e-5 * rng.standard_normal((n, p))
    design = _design(np.concatenate([base, near_dup], axis=1))
    coef = rng.standard_normal((design.shape[1], k))
    targets = design @ coef + 0.05 * rng.standard_normal((n, k))
    folds = _kfold_indices(n, 5, 0)
    return design, targets, folds


@pytest.mark.parametrize("device", _DEVICES, ids=lambda d: d.type)
def test_well_conditioned_matches_numpy(device: torch.device) -> None:
    design, targets, folds = _well_conditioned()
    total = _total_ss(targets)
    fve_ref = _fve(_cv_sse(design, targets, folds, 0.0), total)

    fve = fb.fit_heldout_fve(design, targets, folds, device=device, total_ss=total)
    # float32 (forced on mps) matches the float64 reference to well under the
    # instruments' tie tolerances (>= 1e-2); round-off only.
    assert abs(fve - fve_ref) < 1e-5


def test_cpu_float64_matches_numpy_to_roundoff() -> None:
    design, targets, folds = _well_conditioned()
    total = _total_ss(targets)
    fve_ref = _fve(_cv_sse(design, targets, folds, 0.0), total)
    fve = fb.fit_heldout_fve(
        design, targets, folds, device=torch.device("cpu"), total_ss=total, dtype=torch.float64
    )
    assert abs(fve - fve_ref) < 1e-9


@pytest.mark.parametrize("device", _DEVICES, ids=lambda d: d.type)
def test_near_collinear_gap_survives(device: torch.device) -> None:
    """The ill-conditioned design: float32 normal equations must still land on
    the numpy-lstsq held-out FVE closely enough that a gap between two such
    designs is preserved (the instruments' load-bearing quantity)."""
    design, targets, folds = _near_collinear()
    total = _total_ss(targets)
    fve_ref = _fve(_cv_sse(design, targets, folds, 0.0), total)

    # float64 backend isolates the normal-equations formulation from float32.
    fve64 = fb.fit_heldout_fve(
        design, targets, folds, device=torch.device("cpu"), total_ss=total, dtype=torch.float64
    )
    assert abs(fve64 - fve_ref) < 1e-4

    # float32 (the mps regime) still agrees far inside the 1e-2 tie tolerances.
    fve = fb.fit_heldout_fve(design, targets, folds, device=device, total_ss=total)
    assert abs(fve - fve_ref) < 1e-3


@pytest.mark.parametrize("device", _DEVICES, ids=lambda d: d.type)
def test_residual_ss_mirrors_cv_sse(device: torch.device) -> None:
    design, targets, folds = _well_conditioned(seed=3)
    rss_ref = _cv_sse(design, targets, folds, 0.0)
    rss = fb.heldout_residual_ss(design, targets, folds, device=device)
    assert rss == pytest.approx(rss_ref, rel=1e-4)


def test_exact_fit_is_near_zero_residual() -> None:
    """A target exactly in the design's column space is predicted with
    negligible held-out residual (FVE ~ 1)."""
    rng = np.random.default_rng(4)
    n = 1024
    design = _design(rng.standard_normal((n, 10)))
    coef = rng.standard_normal((design.shape[1], 3))
    targets = design @ coef  # noiseless
    folds = _kfold_indices(n, 5, 0)
    fve = fb.fit_heldout_fve(design, targets, folds, device=torch.device("cpu"))
    assert fve > 1.0 - 1e-4


def test_zero_variance_targets_give_zero_fve() -> None:
    n = 512
    design = _design(np.random.default_rng(5).standard_normal((n, 6)))
    targets = np.ones((n, 2))  # no variance around the mean
    folds = _kfold_indices(n, 5, 0)
    assert fb.fit_heldout_fve(design, targets, folds, device=torch.device("cpu")) == 0.0


def test_one_dimensional_targets_accepted() -> None:
    rng = np.random.default_rng(6)
    n = 512
    design = _design(rng.standard_normal((n, 5)))
    targets = design @ rng.standard_normal(design.shape[1]) + 0.1 * rng.standard_normal(n)
    folds = _kfold_indices(n, 5, 0)
    rss = fb.heldout_residual_ss(design, targets, folds, device=torch.device("cpu"))
    assert np.isfinite(rss) and rss >= 0.0


def test_rejects_bad_shapes_and_ridge() -> None:
    design = np.ones((10, 3))
    folds = [np.array([0, 1])]
    with pytest.raises(ValueError):
        fb.heldout_residual_ss(design, np.ones((9, 2)), folds, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        fb.heldout_residual_ss(
            design, np.ones((10, 2)), folds, device=torch.device("cpu"), ridge=-1.0
        )
    with pytest.raises(ValueError):
        fb.heldout_residual_ss(np.ones(10), np.ones((10, 2)), folds, device=torch.device("cpu"))


@pytest.mark.skipif(not _HAS_MPS, reason="no mps device on this machine")
def test_mps_float64_request_refused() -> None:
    design, targets, folds = _well_conditioned()
    with pytest.raises(ValueError):
        fb.heldout_residual_ss(
            design, targets, folds, device=torch.device("mps"), dtype=torch.float64
        )
