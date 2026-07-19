"""Unit tests for deterministic seeding and device selection."""

from __future__ import annotations

import random
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from group_algorithm_interp.seed import (
    capture_rng_state,
    detect_device,
    resolve_device,
    restore_rng_state,
    set_seed,
)

# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


class TestSetSeed:
    def test_deterministic_true(self):
        set_seed(42, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_deterministic_false(self):
        orig_deterministic = torch.backends.cudnn.deterministic
        orig_benchmark = torch.backends.cudnn.benchmark

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        set_seed(42, deterministic=False)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is True

        torch.backends.cudnn.deterministic = orig_deterministic
        torch.backends.cudnn.benchmark = orig_benchmark

    def test_reproducible_random(self):
        set_seed(42)
        seq1 = [random.random() for _ in range(10)]
        set_seed(42)
        seq2 = [random.random() for _ in range(10)]
        assert seq1 == seq2

    def test_reproducible_numpy(self):
        set_seed(42)
        seq1 = np.random.rand(10).tolist()
        set_seed(42)
        seq2 = np.random.rand(10).tolist()
        assert seq1 == seq2

    def test_reproducible_torch(self):
        set_seed(42)
        seq1 = torch.rand(10).tolist()
        set_seed(42)
        seq2 = torch.rand(10).tolist()
        assert seq1 == seq2

    def test_reproducible_weights(self):
        set_seed(42)
        m1 = nn.Linear(16, 16)
        w1 = m1.weight.data.clone()
        b1 = m1.bias.data.clone()

        set_seed(42)
        m2 = nn.Linear(16, 16)
        w2 = m2.weight.data.clone()
        b2 = m2.bias.data.clone()

        assert torch.equal(w1, w2)
        assert torch.equal(b1, b2)

    def test_different_seeds_different_results(self):
        set_seed(42)
        t42 = torch.rand(100)
        set_seed(43)
        t43 = torch.rand(100)
        assert not torch.equal(t42, t43)

    def test_seed_negative_raises_but_leaves_python_random_partially_seeded(self):
        """set_seed(-1) raises ValueError, but non-atomically: ``random.seed(-1)``
        (Python's stdlib RNG, which accepts negative seeds) already ran before
        ``np.random.seed(-1)`` raises, and torch is never reached. This is a
        known defect in seed.py's failure path, out of scope for this pass.
        Pins the ValueError only, not numpy's message text, which is free to
        change across numpy versions and is not this project's contract.
        """
        with pytest.raises(ValueError):
            set_seed(-1)
        state_after_failed_call = random.getstate()
        random.seed(-1)  # mirrors what set_seed(-1) already did before numpy raised
        assert random.getstate() == state_after_failed_call


# ---------------------------------------------------------------------------
# detect_device
# ---------------------------------------------------------------------------


class TestDetectDevice:
    def test_returns_valid_string(self):
        assert detect_device() in {"cuda", "mps", "cpu"}


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_auto(self):
        result = resolve_device("auto")
        expected = torch.device(detect_device())
        assert result == expected

    def test_cpu(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_cuda(self):
        assert resolve_device("cuda") == torch.device("cuda")

    def test_mps(self):
        assert resolve_device("mps") == torch.device("mps")


# ---------------------------------------------------------------------------
# capture_rng_state / restore_rng_state
# ---------------------------------------------------------------------------


class TestRngState:
    def test_capture_has_expected_keys(self):
        state = capture_rng_state()
        for key in ("python", "numpy", "torch", "torch_cuda"):
            assert key in state

    def test_restore_torch_state(self):
        set_seed(42)
        state = capture_rng_state()

        seq1 = [torch.rand(1).item() for _ in range(5)]

        torch.rand(100)
        restore_rng_state(state)

        seq2 = [torch.rand(1).item() for _ in range(5)]
        assert seq1 == seq2

    def test_restore_python_state(self):
        set_seed(42)
        state = capture_rng_state()

        seq1 = [random.random() for _ in range(5)]

        random.random()
        random.random()
        restore_rng_state(state)

        seq2 = [random.random() for _ in range(5)]
        assert seq1 == seq2

    def test_restore_numpy_state(self):
        set_seed(42)
        state = capture_rng_state()

        seq1 = np.random.rand(5).tolist()

        np.random.rand(100)
        restore_rng_state(state)

        seq2 = np.random.rand(5).tolist()
        assert seq1 == seq2

    def test_torch_cuda_none_when_cuda_unavailable(self):
        state = capture_rng_state()
        if not torch.cuda.is_available():
            assert state["torch_cuda"] is None
        else:
            assert state["torch_cuda"] is not None


# ---------------------------------------------------------------------------
# Mock tests for environment-conditional branches (CUDA / MPS)
# ---------------------------------------------------------------------------


class TestSetSeedMocked:
    def test_mps_manual_seed_called_when_mps_available(self):
        with mock.patch.object(torch.backends.mps, "is_available", return_value=True):
            with mock.patch.object(torch.mps, "manual_seed") as mock_mps_seed:
                set_seed(42, deterministic=False)
                mock_mps_seed.assert_called_with(42)


class TestDetectDeviceMocked:
    def test_returns_cuda_when_cuda_available(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            assert detect_device() == "cuda"

    def test_returns_mps_when_cuda_unavailable_and_mps_available(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with mock.patch.object(torch.backends.mps, "is_available", return_value=True):
                assert detect_device() == "mps"

    def test_returns_cpu_when_neither_cuda_nor_mps_available(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with mock.patch.object(torch.backends.mps, "is_available", return_value=False):
                assert detect_device() == "cpu"


class TestRngStateMocked:
    def test_capture_uses_get_rng_state_all_result_when_cuda_available(self):
        """Mocked explicitly so this is meaningful on a non-CUDA machine: the
        real, unmocked ``get_rng_state_all()`` returns ``[]`` when there are
        zero visible CUDA devices, and ``[] is not None`` would pass
        vacuously regardless of whether the capture logic is correct."""
        sentinel = ["fake-per-device-rng-state"]
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.object(torch.cuda, "get_rng_state_all", return_value=sentinel):
                state = capture_rng_state()
                assert state["torch_cuda"] == sentinel

    def test_restore_calls_set_rng_state_all_when_cuda_available_and_state_present(self):
        set_seed(42)
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            state = capture_rng_state()
            with mock.patch.object(torch.cuda, "set_rng_state_all") as mock_set:
                restore_rng_state(state)
                mock_set.assert_called_once_with(state["torch_cuda"])
