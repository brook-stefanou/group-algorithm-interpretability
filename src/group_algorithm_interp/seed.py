"""Seeding and device selection.

Dependency-light helpers shared by the trainer and tests. Seeding every RNG
(Python, NumPy, torch, CUDA, MPS) up front is what makes a run reproducible from
its manifest alone.

Seeding does not buy bit-reproducibility on CUDA. ``W_E[tokens]`` backpropagates
through ``index_put_(accumulate=True)``, which on CUDA is an ``atomicAdd``; float
addition is not associative, so the reduction order -- and therefore the
gradient -- varies run to run, and the same seed produces slightly different
weights. ``torch.use_deterministic_algorithms(True)`` routes that op to a
deterministic implementation but costs real throughput, so it is a config opt-in
(``deterministic``, default false). A CUDA run is not bit-reproducible by default.

``torch.use_deterministic_algorithms(True)`` also makes the model's first cuBLAS
``einsum`` raise unless ``CUBLAS_WORKSPACE_CONFIG`` is set, because cuBLAS
otherwise picks workspace-dependent reduction splits. cuBLAS reads that variable
when it creates its handle -- the first CUDA BLAS call -- so it must be in the
process environment before then, hence the module-level ``setdefault`` below,
which fires at import time, long before any tensor reaches the GPU. It is inert
unless someone opts in. ``os.environ.setdefault`` keeps an operator's explicit
``CUBLAS_WORKSPACE_CONFIG=:16:8`` (the smaller-workspace alternative) intact.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

from .config import Device

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG a run might touch and (optionally) ask torch for
    deterministic kernels. Call this once, before building the model, so weight
    initialisation is reproducible.

    ``deterministic`` drives ``torch.use_deterministic_algorithms``. It is a pure
    opt-in, defaulting to off, and nothing overrides it -- a CUDA run is not
    bit-reproducible unless the operator asks for it and accepts the throughput
    cost (see the module docstring).

    TF32 is turned on for matmul. It is not a determinism knob (TF32 matmuls are
    reproducible); it trades fp32 mantissa bits for a large speedup on Ampere and
    later, the trade this project wants on rented GPUs. torch's own default leaves
    matmul TF32 off (``fp32_precision="none"``, ``float32_matmul_precision=
    "highest"``), so enabling it takes an explicit call. cudnn's conv TF32 is left
    alone: torch already defaults it on and this project's models have no
    convolutions.

    ``fp32_precision`` is torch's current API for this. Do **not** reintroduce the
    older ``torch.backends.cuda.matmul.allow_tf32`` boolean anywhere: torch tracks
    which of the two APIs a process has used and *raises* ``RuntimeError`` ("mix
    of the legacy and new APIs") on a legacy read after a new-API write, so the
    two cannot be used side by side. This codebase speaks the new dialect only.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    # cudnn.deterministic governs convolution algorithm selection only; this
    # project's models have no convolutions, so it is set for completeness and
    # does nothing on its own. use_deterministic_algorithms is the load-bearing
    # call.
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)


def detect_device() -> str:
    """The best available device, in preference order cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: Device) -> torch.device:
    """Turn the config's device string into a concrete ``torch.device``,
    resolving ``"auto"`` to whatever hardware is actually present."""
    name = detect_device() if device == "auto" else device
    return torch.device(name)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG generator.

    Captures Python, NumPy, and torch (CPU) generators, plus CUDA when available.
    MPS RNG state is not portably retrievable, so it is omitted; re-seeding from
    the config recovers most determinism on Apple silicon.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the RNG generators captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
