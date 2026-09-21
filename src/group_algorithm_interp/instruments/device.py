"""Device selection for the analysis instruments' forward passes.

The heavy instruments (I-15 isotypic-block ablation, the GCR matrix-product
activation extraction, occupancy's neuron-activation sweep, the GCR
character-readout logits) spend almost all their wall time in the model's
*forward* pass, not in the downstream analysis. That analysis is fixed at
CPU float64 -- MPS has no float64 support, and the study does not require
bitwise determinism (training itself ran on GPUs) -- but the forward pass
itself can run on an accelerator in the model's native float32. This module
is the shared, narrow device policy that makes that split safe:

* :func:`resolve_device` turns a CLI ``--device`` spec into a concrete
  ``torch.device``, with ``"auto"`` meaning "MPS if available, else CPU".
  This is deliberately narrower than :func:`group_algorithm_interp.seed.resolve_device`
  (training's device resolution, which prefers CUDA), because these
  instruments usually run locally -- typically on Apple silicon -- and rarely
  see CUDA; a training-style cuda-first ``"auto"`` would silently do the
  wrong thing there. An explicit ``"cuda"`` spec is accepted for the rarer
  case of running these instruments on a CUDA box (e.g. a rented GPU pod for
  a large panel cell): ``"auto"`` itself still never resolves to CUDA, so
  existing callers on Apple silicon see no behaviour change.
* :func:`run_with_device_fallback` runs a forward-pass computation on the
  requested device and, if the device is missing an op the computation
  needs (an MPS gap raised as ``RuntimeError``), transparently retries the
  same computation on CPU rather than crashing the run, returning a
  human-readable note of the fallback alongside the result.

Nothing here forces MPS by default: every instrument function this module
serves keeps ``device=torch.device("cpu")`` as its own default, so importing
this module changes no behaviour until a caller opts in via ``--device``.
"""

from __future__ import annotations

import warnings
from typing import Callable, TypeVar

import torch

_VALID_SPECS = ("auto", "mps", "cuda", "cpu")

T = TypeVar("T")


def resolve_device(spec: str) -> torch.device:
    """Resolve an instrument ``--device`` spec to a concrete ``torch.device``.

    ``spec`` must be one of ``"auto"``, ``"mps"``, ``"cuda"`` or ``"cpu"``.
    ``"auto"`` resolves to MPS when ``torch.backends.mps.is_available()``,
    else CPU -- never CUDA (see the module docstring); requesting CUDA is
    always explicit. Requesting ``"mps"`` or ``"cuda"`` explicitly when it is
    unavailable is not silently downgraded: it raises, so a typo'd or
    genuinely unsupported request fails loudly rather than quietly running on
    CPU with no explanation for a duration that would only make sense on a
    GPU.
    """
    if spec not in _VALID_SPECS:
        raise ValueError(f"device must be one of {_VALID_SPECS}, got {spec!r}")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "--device mps requested but torch.backends.mps.is_available() is False"
            )
        return torch.device("mps")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    # "auto"
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _looks_like_device_op_gap(message: str) -> bool:
    """Heuristic for "this RuntimeError is an unsupported-op/device gap",
    e.g. ``NotImplementedError``-style messages torch raises for MPS
    (``"... not currently implemented for the MPS device"``) or an explicit
    mismatched-device complaint. Anything else (a real numerical bug, a
    shape mismatch) is deliberately not swallowed."""
    lowered = message.lower()
    return (
        "mps" in lowered
        or "not currently implemented" in lowered
        or "not implemented for" in lowered
        or "not currently supported" in lowered
    )


def run_with_device_fallback(
    compute: Callable[[torch.device], T], device: torch.device
) -> tuple[T, str | None]:
    """Run ``compute(device)``; on an MPS op gap, retry once on CPU.

    ``compute`` receives the device it should run its forward pass(es) on,
    and is expected to leave the model it operates on back on CPU when it
    returns or raises (the instrument functions in this package do this via
    ``try/finally``). Returns ``(result, note)``: ``note`` is ``None`` when
    ``device`` is CPU or the computation succeeded on the requested device
    outright, else a human-readable description of the fallback -- callers
    that produce a record dict should carry it as a ``device_fallback``
    field; callers that return a bare array log it via :mod:`warnings`
    instead, since there is no record to carry it in.

    A ``RuntimeError`` that does not look like a device/op gap (see
    :func:`_looks_like_device_op_gap`) is never swallowed -- it propagates,
    since silently falling back on an unrelated bug would hide it.
    """
    if device.type == "cpu":
        return compute(device), None
    try:
        return compute(device), None
    except RuntimeError as exc:
        message = str(exc)
        if not _looks_like_device_op_gap(message):
            raise
        note = (
            f"{device} forward pass hit an unsupported op "
            f"({message.splitlines()[0]!r}); fell back to CPU for this model"
        )
        warnings.warn(note, RuntimeWarning, stacklevel=3)
        return compute(torch.device("cpu")), note


__all__ = ["resolve_device", "run_with_device_fallback"]
