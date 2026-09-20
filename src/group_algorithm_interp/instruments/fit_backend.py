"""A torch fitting backend for the heavy held-out FVE least-squares fits.

Three instruments -- the GCR matrix-product screen (:mod:`.gcr_matmul`), the GCR
character-readout probe (:func:`.probes.gcr_character_readout_instrument`) and
the readout characterisation (:mod:`.readout_characterisation`) -- score linear
models by *held-out* fraction of variance explained (FVE) over the full
``|G|^2`` Cayley grid. Their reference fit is numpy ``lstsq`` in float64 on the
CPU (the minimum-norm least-squares solution refit per cross-validation fold).
For the large groups (order 192/216, degree-``d`` irreps whose bilinear design
has up to ``2 d**4`` columns) that CPU ``lstsq`` -- an SVD of a
``[|G|^2, 2 d**4]`` matrix -- takes over an hour and a half per cell.

This module computes the *same* held-out residual sum of squares (and hence the
same FVE) via the **normal equations** instead of an SVD, so the dominant cost
becomes the ``X_train.T @ X_train`` / ``X_train.T @ Y_train`` matmuls, which run
on Apple's ``mps`` GPU in float32. Only the tiny ``p x p`` coefficient solve is
not a matmul, and it is small enough that ``torch.linalg.solve`` handles it on
device (with a CPU fallback for the solve alone should an ``mps`` op be
missing).

Why this is numerically delicate, and what we do about it
---------------------------------------------------------
The normal equations square the condition number of the design, and float32 has
~7 decimal digits: the matrix-product design is a *strict subspace* of the
bilinear design, so their held-out FVEs are close and ``XtX`` is ill-conditioned
-- exactly the regime where float32 normal equations are least accurate, and
exactly where the load-bearing statistic (the small ``mp_minus_bilinear`` /
character-vs-full-matrix gaps) lives. Two standard stabilisers are applied and
documented here:

* **Column centering.** Each fold's training columns (and targets) are centred
  on their training-fold means before forming ``XtX``; the prediction adds the
  training-target mean back. This is algebraically the intercept-augmented
  least-squares fit (identical predictions), but it removes the large constant
  component from every column before the float32 cross-product, which is where
  catastrophic cancellation would otherwise bite. The instruments' designs
  already carry an explicit intercept column (matrix-product) or are class-axis
  centred (character readout), so centering changes the *parameterisation*, not
  the fitted function -- verified against the numpy path in the tests.
* **Tikhonov ridge.** A small ``lambda * I`` is added to ``XtX`` before the
  solve, with ``lambda = ridge * mean(diag(XtX))`` (relative to the design's own
  scale, so it is dimensionless and design-independent). The default
  ``ridge = 1e-6`` is far below the float32 noise floor of the well-determined
  directions -- so it does not perturb the fit the reference would find -- while
  being large enough to keep the (rank-deficient, redundant-column) designs
  solvable without an SVD. As ``ridge -> 0`` the solution tends to the
  minimum-norm least-squares solution the numpy ``lstsq`` reference returns.

The held-out residual sum is accumulated in float64 on the CPU (the squared
residual is moved off device first), so only the fit itself -- not the score
reduction -- carries float32 error.

This backend is *opt-in*: every instrument keeps its numpy-float64-CPU ``lstsq``
path as the default and reference, and only routes through here when the caller
passes an ``mps`` device. The public entry point is :func:`fit_heldout_fve`;
:func:`heldout_residual_ss` is the drop-in for the instruments' internal
per-fold residual-sum helpers (it mirrors ``gcr_matmul._cv_sse``'s
``(design, targets, folds)`` signature and float return).
"""

from __future__ import annotations

import numpy as np
import torch

# Below this the total variance is treated as zero (matches gcr_matmul._FVE_FLOOR).
_FVE_FLOOR = 1e-9

# Default relative Tikhonov coefficient: lambda = _DEFAULT_RIDGE * mean(diag(XtX)).
_DEFAULT_RIDGE = 1e-6


def _resolve_dtype(device: torch.device, dtype: torch.dtype | None) -> torch.dtype:
    """The compute dtype for the fit on ``device``.

    ``mps`` has no float64, so a fit there is always float32 (an explicit
    float64 request is refused rather than silently downgraded). On the CPU the
    default is float32 too -- matching the ``mps`` numerics so a CPU run is a
    faithful stand-in for testing -- but float64 may be requested to isolate the
    normal-equations formulation from float32 rounding.
    """
    if device.type == "mps":
        if dtype is not None and dtype != torch.float32:
            raise ValueError("mps supports only float32; do not pass dtype != float32 for mps")
        return torch.float32
    return dtype if dtype is not None else torch.float32


def _solve_with_cpu_fallback(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.linalg.solve(a, b)`` on ``a``'s device, falling back to CPU for
    just this small ``p x p`` solve if the device is missing the op.

    The cross-products that dominate the cost stay on the accelerator; only the
    tiny solve is retried on the CPU, and its result is moved back so the caller
    is unaware of the fallback. A ``RuntimeError`` that is not an unsupported-op
    gap propagates -- it is a real failure, not a device limitation."""
    try:
        return torch.linalg.solve(a, b)
    except RuntimeError as exc:
        message = str(exc).lower()
        looks_like_gap = (
            "mps" in message
            or "not currently implemented" in message
            or "not implemented for" in message
            or "not currently supported" in message
        )
        if not looks_like_gap:
            raise
        coef_cpu = torch.linalg.solve(a.cpu(), b.cpu())
        return coef_cpu.to(a.device)


def _complement(all_idx: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Training indices for a held-out ``fold`` (mirrors ``_cv_sse``'s
    ``np.setdiff1d(all_idx, fold, assume_unique=True)``)."""
    return np.setdiff1d(all_idx, fold, assume_unique=True)


def heldout_residual_ss(
    design: np.ndarray,
    targets: np.ndarray,
    folds: list[np.ndarray],
    *,
    device: torch.device,
    ridge: float = _DEFAULT_RIDGE,
    center: bool = True,
    dtype: torch.dtype | None = None,
) -> float:
    """Pooled held-out residual sum of squares of a linear model, via the normal
    equations, on ``device``.

    A drop-in for ``gcr_matmul._cv_sse``: ``design`` is ``[n_rows, p]`` and
    ``targets`` is ``[n_rows, k]`` (``k >= 1`` neurons/columns). For each fold
    the coefficients are refit on the other rows (``train = all \\ fold``) and
    used to predict the held-out rows, and the squared held-out residual is
    accumulated in float64. No held-out row informs its own prediction.

    The fit is centred (per :mod:`this module <group_algorithm_interp.instruments.fit_backend>`)
    when ``center`` and ridge-stabilised with
    ``lambda = ridge * mean(diag(XtX))``. ``dtype`` defaults to float32 (forced
    on ``mps``); float64 may be requested on the CPU to isolate the
    normal-equations formulation from float32 rounding.
    """
    if design.ndim != 2:
        raise ValueError(f"design must be 2-D [n_rows, p], got shape {design.shape}")
    targets2d = targets if targets.ndim == 2 else targets[:, None]
    if targets2d.shape[0] != design.shape[0]:
        raise ValueError(f"design has {design.shape[0]} rows but targets has {targets2d.shape[0]}")
    if ridge < 0.0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")

    compute_dtype = _resolve_dtype(device, dtype)
    n_rows, p = design.shape
    all_idx = np.arange(n_rows)

    x_all = torch.as_tensor(np.ascontiguousarray(design), dtype=compute_dtype, device=device)
    y_all = torch.as_tensor(np.ascontiguousarray(targets2d), dtype=compute_dtype, device=device)
    eye = torch.eye(p, dtype=compute_dtype, device=device)

    residual_ss = 0.0
    for fold in folds:
        fold_idx = np.asarray(fold, dtype=np.int64)
        train = _complement(all_idx, fold_idx)
        train_t = torch.as_tensor(train, dtype=torch.long, device=device)
        fold_t = torch.as_tensor(fold_idx, dtype=torch.long, device=device)

        x_train = x_all.index_select(0, train_t)
        y_train = y_all.index_select(0, train_t)
        x_test = x_all.index_select(0, fold_t)
        y_test = y_all.index_select(0, fold_t)

        if center:
            col_mean = x_train.mean(dim=0, keepdim=True)
            y_mean = y_train.mean(dim=0, keepdim=True)
            x_train = x_train - col_mean
            y_train = y_train - y_mean
            x_test = x_test - col_mean

        xtx = x_train.transpose(0, 1) @ x_train
        xty = x_train.transpose(0, 1) @ y_train
        lam = float(ridge) * float(torch.diagonal(xtx).mean().item())
        coef = _solve_with_cpu_fallback(xtx + lam * eye, xty)

        prediction = x_test @ coef
        if center:
            prediction = prediction + y_mean

        # Accumulate the score in float64 on the CPU: only the *fit* carries
        # float32 error, never the sum-of-squares reduction over |G|^2 cells.
        resid = (y_test - prediction).detach().cpu().to(torch.float64)
        residual_ss += float(torch.square(resid).sum().item())

    return residual_ss


def total_variance(targets: np.ndarray) -> float:
    """Total variance of ``targets`` around each column's own mean, in float64
    (matches ``gcr_matmul._total_ss``)."""
    arr = np.asarray(targets, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return float(np.square(arr - arr.mean(axis=0, keepdims=True)).sum())


def fit_heldout_fve(
    design: np.ndarray,
    targets: np.ndarray,
    folds: list[np.ndarray],
    *,
    device: torch.device,
    ridge: float = _DEFAULT_RIDGE,
    center: bool = True,
    total_ss: float | None = None,
    dtype: torch.dtype | None = None,
) -> float:
    """Held-out fraction of variance explained (FVE) of a linear model, on
    ``device``, via the normal equations.

    The public entry point mirroring the instruments' numpy-float64-CPU FVE:
    ``1 - residual_ss / total_ss`` where ``residual_ss`` is
    :func:`heldout_residual_ss` and ``total_ss`` is the total variance of
    ``targets`` (computed with :func:`total_variance` when not supplied, so the
    caller may instead pass its own convention -- e.g. a held-out-fold total).
    Returns ``0.0`` when the total variance is negligible, matching the numpy
    ``_fve`` guard.
    """
    residual_ss = heldout_residual_ss(
        design, targets, folds, device=device, ridge=ridge, center=center, dtype=dtype
    )
    ss_tot = total_variance(targets) if total_ss is None else float(total_ss)
    if ss_tot <= _FVE_FLOOR:
        return 0.0
    return 1.0 - residual_ss / ss_tot


__all__ = ["fit_heldout_fve", "heldout_residual_ss", "total_variance"]
