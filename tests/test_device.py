"""Device selection for the analysis instruments (instruments/device.py).

Covers the CLI-facing ``resolve_device`` policy (narrower than training's
CUDA-preferring ``auto``: MPS-or-CPU only, explicit ``mps`` fails loudly when
unavailable) and ``run_with_device_fallback``'s CPU fast path, MPS-op-gap
fallback, and refusal to swallow an unrelated ``RuntimeError``.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from group_algorithm_interp.instruments.device import resolve_device, run_with_device_fallback


def test_resolve_device_cpu_is_always_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_rejects_unknown_spec():
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("cuda")


def test_resolve_device_auto_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("mps")


def test_resolve_device_auto_falls_back_to_cpu_without_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_mps_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="mps"):
        resolve_device("mps")


def test_resolve_device_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps") == torch.device("mps")


def test_run_with_device_fallback_cpu_never_calls_compute_twice():
    calls = []

    def compute(dev):
        calls.append(dev)
        return "result"

    result, note = run_with_device_fallback(compute, torch.device("cpu"))
    assert result == "result"
    assert note is None
    assert calls == [torch.device("cpu")]


def test_run_with_device_fallback_succeeds_on_requested_device_without_fallback():
    calls = []

    def compute(dev):
        calls.append(dev)
        return dev

    result, note = run_with_device_fallback(compute, torch.device("mps"))
    assert result == torch.device("mps")
    assert note is None
    assert calls == [torch.device("mps")]


def test_run_with_device_fallback_retries_on_cpu_after_an_mps_op_gap():
    calls = []

    def compute(dev):
        calls.append(dev)
        if dev.type != "cpu":
            raise RuntimeError("aten::foo is not currently implemented for the MPS device")
        return "cpu-result"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, note = run_with_device_fallback(compute, torch.device("mps"))

    assert result == "cpu-result"
    assert note is not None
    assert "fell back to CPU" in note
    assert calls == [torch.device("mps"), torch.device("cpu")]
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_run_with_device_fallback_does_not_swallow_unrelated_runtime_errors():
    def compute(dev):
        raise RuntimeError("shape mismatch: expected [4, 4] got [3, 3]")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        run_with_device_fallback(compute, torch.device("mps"))
